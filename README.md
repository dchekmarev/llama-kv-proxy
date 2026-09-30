<img width="1000" alt="image_" src="https://github.com/user-attachments/assets/0d966dde-f1d8-432f-bad0-aa79a5ccf396" />

# llama-kv-proxy

[![CI](https://github.com/dchekmarev/llama-kv-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/dchekmarev/llama-kv-proxy/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Version](https://img.shields.io/badge/version-0.0.1)](./version.py)

A proxy in front of llama.cpp that makes long-context chat and IDE workflows faster: it manages llama.cpp slots, reuses cached KV context, and restores saved caches from disk when needed. It speaks the OpenAI-compatible Chat Completions API (streaming SSE and non-streaming), so existing clients can connect without changes.

## Why it's needed

llama.cpp provides "slots," each holding a conversation's KV cache, so repeated requests with the same or a very similar prefix skip recomputing the whole prompt and continue from the first mismatching token. In real teams the number of users can easily exceed the number of available slots (e.g., 20 developers but only 4 slots), so naive routing causes random slot reuse and cache overwrites that waste time and GPU/CPU cycles. This proxy steers requests to the right slot, saves evicted caches to disk, and restores them on demand, so long prompts are not recomputed from scratch each time. For 30–60k-token contexts typical of project-wide IDE assistants, restoring a previously cached context takes seconds instead of the minutes a full recompute would.

## Features

- OpenAI-compatible `/v1/chat/completions` (stream and non-stream), plus pass-through of native llama.cpp endpoints
- Slot management: free-first / LRU selection, per-slot async locks, HTTP 503 when all slots stay busy
- Two-tier cache matching: exact per-message prefix hashes, block-LCP fallback
- Disk persistence: llama.cpp slot save/restore plus local `.meta` descriptors — survives restarts
- Automatic cache hygiene: TTL/count/size eviction, superseded-cache cleanup, `.bin` LRU size cap, meta/.bin reconciliation
- Multi-backend and router (`--models-preset`) support with per-model slot pools
- Prometheus `/metrics`: proxy-level `llama_kv_proxy_*` metrics plus backend `/metrics` aggregated across all backends and models with `model`/`backend` labels
- Live dashboard at `/proxy/ui/`: in-flight requests with streaming token tails (reasoning in a separate color), recent history, and per-slot state with the busy request mapped in
- Docker Compose packaging with a healthcheck

## How it works

### Big vs small requests

A request is **big** when its prompt has more than `BIG_THRESHOLD_WORDS` words (each CJK code point counts as its own word).

- **Big** requests get the full cache treatment: match, restore, then save + meta after the response.
- **Small** requests are routed to a free or oldest slot without touching the disk cache.

### Cache matching (two-tier)

For a big request the proxy computes per-message prefix hashes (`sha256(model_id + "\n" + raw_prefix(messages[:k]))`) and block hashes (SHA256 of `WORDS_PER_BLOCK`-word blocks):

1. **Tier 1** — exact message-boundary prefix match against saved metas (`saved_prefix_hashes`, i.e. prompt + response, so continuations match). Ranked by the longest common prefix; ties are broken by the smallest cache size.
2. **Tier 2** — block-based LCP fallback (also matches old metas that only have blocks). Ranked by the fraction of the request covered; ties are broken by the smallest size.

A candidate must cover at least `LCP_TH` of the request. Both tiers filter by `model_id` and `WORDS_PER_BLOCK`.

By default the search first consults an in-memory index of the metas' prefix hashes (`META_INDEX_ENABLED`): a tier-1 hit is an O(n) set lookup instead of a disk scan, validated against the on-disk meta on hit. A miss (or an empty index, or old blocks-only metas that are not indexed) falls back to the two-tier on-disk scan, so results never lose a candidate the legacy path would find.

### Slot selection

- A free slot (never used yet) is preferred; when none are free, the least-recently-used one is taken.
- Each slot is guarded with an async lock; if all slots are busy the request waits up to `ACQUIRE_TIMEOUT` seconds and then gets HTTP 503.
- For a big request with a match, the cache is restored into the chosen slot before the chat. If the backend reports the file missing (404), the stale meta is dropped; other failures keep the meta (a retry can succeed).

### Save and restore from disk

llama.cpp's HTTP server exposes slot save/restore: saving writes a cache file to the `--slot-save-path` directory, and restore loads by file basename (`<key>.bin`). The proxy keeps small local `.meta` files describing cached prefixes for fast lookup, while llama.cpp owns the actual KV `.bin` files. After a big request completes, the proxy saves the slot (prompt + assistant response), writes the meta, and deletes the metas (and `.bin` files) of strict prefixes of the new conversation, so a continuation does not leave stale, shorter caches behind.

### `.bin` cache cleanup

llama.cpp intentionally has no endpoint to delete files from `--slot-save-path` (the `erase` slot action only clears in-memory state; save/restore only write/read), so `.bin` files accumulate unless removed. To bound them, mount the host `--slot-save-path` into the proxy and set `BIN_CACHE_DIR` plus a `BIN_CACHE_MAX_MB` size cap. The proxy then deletes `.bin` files directly:

- on meta eviction/clear (by key);
- by LRU (oldest last-use first, where last-use = the meta timestamp, refreshed on both save and restore) — after every slot save (asynchronously, without delaying the response; a new check does not start until the previous one finishes) and periodically every `BIN_CACHE_INTERVAL_S` seconds.

Orphaned `.bin` files (no matching `.meta`) are removed first. A `.bin` written within the last `BIN_SAVE_GRACE_S` seconds is treated as an in-flight save and skipped. Every `BIN_RECONCILE_INTERVAL_S` seconds the proxy reconciles in both directions: stale metas without a `.bin` and orphaned `.bin` files without a meta are removed.

llama.cpp may also write a `.ckpt` prompt-checkpoint sidecar next to each saved slot blob (see [ggml-org/llama.cpp#24028](https://github.com/ggml-org/llama.cpp/pull/24028)). The proxy treats the sidecar as part of its `.bin`: its size counts toward the `BIN_CACHE_MAX_MB` cap, it is evicted together with the `.bin`, and a `.ckpt` whose `.bin` is gone is dropped as an orphan (fresh ones are protected by the `BIN_SAVE_GRACE_S` window).

### Router backends

A router backend (`llama-server --models-preset`) serves each model on its own child process with its own slots. The proxy discovers per-model slots via `GET /slots?model=X` and keys slot pools by `(backend, model, slot)`, so the same slot id is a different physical slot per model. `n_slots` in `BACKENDS` is only a bootstrap hint, not functional.

### Request forwarding

The proxy forwards the client's request body to the backend verbatim (pass-through) and injects the following fields:

- `model` — from the request, else the backend's model id (TTL-cached), else `MODEL_ID`;
- `cache_prompt` — `true` for big requests (more than `BIG_THRESHOLD_WORDS` words);
- `n_keep` — `-1` (see Limitations);
- slot pinning: `slot_id`/`id_slot` are duplicated in the body root, in `options`, and in the query parameters, so the request is pinned to the selected slot.

## Quick start

1) Start llama.cpp (https://github.com/ggml-org/llama.cpp) with slots and a cache directory:

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/lib/llama-slots --host 0.0.0.0 --port 8080 --swa-full
```

This enables the OpenAI-compatible HTTP server, a pool of 4 slots, and a directory where slot KV caches are saved and restored by basename.

2) Run the proxy next to it:

```bash
git clone https://github.com/dchekmarev/llama-kv-proxy.git
cd llama-kv-proxy
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python3 llama_kv_proxy.py  # or: uvicorn app:app --host 0.0.0.0 --port 8081
```

Run the proxy with a **single worker** (the default). The slot manager keeps per-process state (locks, LRU marks); multiple workers would each track slots independently and could route two requests to the same slot.

Your clients should call the proxy's `/v1/chat/completions` endpoint; the proxy handles matching, slot selection, save/restore, and streaming vs non-streaming automatically:

```bash
# Non-streaming
curl http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}]}'

# Streaming (SSE)
curl -N http://localhost:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

## Docker

```bash
docker compose up -d --build
```

- The container listens on port **8080** (`PORT` env var) and has a healthcheck probing `/proxy/health`.
- KV cache metadata is persisted in `./kv_meta` (mounted volume) — survives container restarts.
- The actual KV cache `.bin` files are managed by llama.cpp under `--slot-save-path`.

The base `docker-compose.yml` does not set `BACKENDS` (defaults to `http://127.0.0.1:8000`). To reach a host-side llama.cpp, copy the override example and adjust it:

```bash
cp docker-compose.override.example.yml docker-compose.override.yml
```

`docker compose up` merges `docker-compose.override.yml` automatically (no extra flags); it is gitignored, so your local settings (backend URL, `BIN_CACHE_DIR` mount, tuning) stay out of the repo. Mounting the host `--slot-save-path` into the container and setting `BIN_CACHE_DIR` enables proxy-side `.bin` cleanup.

Diagnostics:

```bash
docker compose logs -f
docker compose ps
docker inspect llama-kv-proxy --format='{{.State.Health.Status}}'
```

## Configuration

All parameters are environment variables; defaults in parentheses.

| Variable | Default | Description |
|---|---|---|
| `BACKENDS` | — | JSON list of backends, e.g. `[{"url": "http://127.0.0.1:8000", "n_slots": 2}]`. `n_slots` is a bootstrap hint; the proxy discovers the real slots via `GET /slots`. |
| `LLAMA_URL` | `http://127.0.0.1:8000` | Single-backend URL (used when `BACKENDS` is unset). |
| `N_SLOTS` | `2` | Single-backend slot count (used when `BACKENDS` is unset). |
| `WORDS_PER_BLOCK` | `100` | Words per hash block for LCP. |
| `BIG_THRESHOLD_WORDS` | `500` | Prompts longer than this are "big". |
| `LCP_TH` | `0.6` | Minimum share of the request that a cached prefix must cover to be restored. |
| `META_DIR` | `kv_meta` | Directory for local `.meta` descriptors, relative to the app directory. |
| `REQUEST_LOG_DIR` | `kv_reqlog` | Directory for per-request JSON groups (`{timestamp_ms}.{request_id}.{type}.json`: `request`, `response`, `prefix`, `decision`, plus `raw` with the raw SSE for streams), relative to the app directory. Empty disables logging. |
| `REQUEST_LOG_MAX_GROUPS` | `100` | Max number of request groups kept; oldest groups (all their files) are deleted first. `0` disables rotation. |
| `REQUEST_TIMEOUT` | `1500` | HTTP timeout to the backends, seconds. |
| `ACQUIRE_TIMEOUT` | `1500` | Maximum wait for a free slot, seconds. |
| `SAVE_WAIT_TIMEOUT` | `30` | Max seconds a big request waits for an in-flight save of a prefix of its own conversation before giving up on the restore (the previous message's meta lands only after its `.bin` write). `0` disables the wait. |
| `MODEL_ID` | `llama.cpp` | Fallback model id returned by `/v1/models` when the backend is unavailable, and the namespace of a model-less request whose backend id is unknown. It is also the cache namespace of a restore, so two different models sharing it can restore each other's KV: change it when you swap the served model. Client `model` aliases are mapped through the backend preset table instead (see *Model aliases*). |
| `MODEL_ID_TTL` | `60` | Backend model-id cache TTL, seconds. |
| `MODEL_ID_TIMEOUT` | `5` | Model-id fetch timeout, seconds. |
| `UNKNOWN_MODEL_ID_RETRY` | `5` | Retry interval while the model id is unknown, seconds. |
| `METRICS_TIMEOUT` | `5` | Timeout for fetching a backend's `/metrics` during a scrape; a slow backend must not stall the whole `/metrics` response. |
| `SLOT_POLL_INTERVAL_S` | `30` | Interval between backend `GET /slots` polls, seconds. |
| `SLOT_FRESHEN_INTERVAL_S` | `1.0` | On-demand freshen: before picking a slot, the backends serving the request's model are re-polled at most once per this interval, so a slot cut/reload is observed quickly instead of waiting for the periodic poll. `0` disables. |
| `META_TTL_H` | `24` | Age after which `.meta` files are evicted, hours. |
| `META_MAX_FILES` | `1000` | Eviction cap on meta file count. |
| `META_MAX_MB` | `512` | Eviction cap on total meta size, MB. |
| `EVICT_INTERVAL_S` | `3600` | Interval between eviction runs, seconds. |
| `META_INDEX_ENABLED` | `1` | `0` disables the in-memory restore index (every search falls back to the on-disk two-tier scan). |
| `META_INDEX_RECONCILE_INTERVAL_S` | `300` | Interval between index<->disk reconciliations (drops ghost entries from external meta deletions); `0` disables. |
| `BIN_CACHE_DIR` | *(empty)* | Backend `.bin` cache directory (the mounted `--slot-save-path`); empty disables direct `.bin` cleanup. |
| `BIN_CACHE_MAX_MB` | `0` | Max total `.bin` size, MB; `0` disables the size cap. |
| `BIN_CACHE_INTERVAL_S` | `EVICT_INTERVAL_S` | Interval between `.bin` LRU cleanup runs, seconds. |
| `BIN_RECONCILE_INTERVAL_S` | `600` | Interval between meta/.bin reconciliations (both directions); `0` disables. |
| `BIN_SAVE_GRACE_S` | `10` | Grace window: a `.bin` without a meta modified within this window is treated as an in-flight save and skipped. `0` disables the guard. |
| `MIN_BIN_SIZE_VALID` | `1` | Minimum size (MB) of a saved `.bin` to count as a real slot save; smaller files are treated as empty captures and discarded without writing a meta. `0` disables the check. |
| `ERASE_BEFORE_SMALL` | `0` | `1` clears a slot's in-memory KV (`action=erase`) before dispatching a small request, so it does not start on top of another conversation's stale KV. Off by default until verified against the target build. |
| `ERASE_BEFORE_CHAT` | `1` | `1` erases a slot's in-memory KV before any chat that was not preceded by a successful restore, so it does not start on top of a stale or oversized prompt (which can wedge llama.cpp in `PROCESSING_PROMPT`). On by default. |
| `SKIP_RESTORE_SAME_SLOT` | `1` | `1` skips a pre-chat restore when the picked slot already holds the target key's KV (it just saved it, or it was restored there and unused since). On by default. |
| `STUCK_SLOT_THRESHOLD_S` | `1500` | Watchdog: a backend slot reporting `is_processing` for longer than this is presumed wedged and is erased to recover the backend without a restart. `0` disables. |
| `REASONING_IN_KEY` | `0` | `1` includes `reasoning_content`/`reasoning` in the per-message cache-key parts and in the saved assistant response, so distinct reasoning traces get distinct cache keys. Only matters when the backend chat template renders reasoning into the prompt. Off by default (keys byte-identical to the legacy behavior). |
| `UI_ENABLED` | `1` | `0` disables the `/proxy/ui/` dashboard (routes return 404, hooks become no-ops). |
| `UI_HISTORY_MAX` | `200` | Number of finished requests kept in the dashboard history. |
| `UI_TAIL_MAX_CHARS` | `16384` | Max characters of the live token tail kept per request. |
| `UI_PREVIEW_MAX_CHARS` | `2048` | Budget for the prompt preview shown in the tables: the newest whole messages that fit (the conversation tail, not the head). |
| `UI_PROMPT_FULL_MAX_CHARS` | `1048576` | Cap for the full prompt transcript served on demand; kept until the request leaves the history. |
| `PORT` | `8081` | Proxy port. |
| `LOG_LEVEL` | `INFO` | Log level. |

> **Multi-backend note:** when a request omits `model`, the cache key is built from the first backend's model id. If you run multiple backends with different models, requests dispatched to the other backends may use a wrong cache key — keep all backends on the same model, or always send `model` in the request.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible chat endpoint (stream and non-stream). |
| GET | `/v1/models` | Union of the backends' model lists (deduped by id); falls back to `MODEL_ID` when no backend reports any model. |
| GET | `/version` | Proxy name and version (single source of truth: `version.py`). |
| GET | `/proxy/slots` | Aggregated slot state across all backends (state, n_ctx, total_tokens, LRU mark). |
| GET | `/proxy/health` | Backend availability probe plus slot state. |
| GET | `/proxy/ui/` | Live dashboard (self-contained HTML page): active requests with live token tails, recent history, slot grid. |
| GET | `/proxy/ui/state` | JSON snapshot: active requests (with token tails), history, slots with `busy_rid`. |
| GET | `/proxy/ui/request/{rid}` | Full prompt transcript of an in-flight or recent (history) request. |
| GET | `/proxy/ui/events` | SSE stream for the dashboard: initial snapshot, then token batches and start/slot/end events (15 s heartbeat). |
| GET | `/cache/stats` | Cache file count, total size, hit/miss counters. |
| POST | `/cache/clear` | Delete all local meta files (and best-effort purge backend `.bin` files). |
| GET | `/metrics` | Prometheus target: proxy `llama_kv_proxy_*` metrics followed by the merged backend `/metrics` across all active (loaded) models, each backend metric carrying `model` and `backend` labels. `?model=X` filters both halves to a single model; a down backend/model is skipped, never failing the scrape. |
| any | `/{path}` | Forwarded to the first backend as-is (native llama.cpp endpoints: `/slots?model=...`, `/health`, `/tokenize`, …), with the response streamed back. |

## Metrics

`/metrics` serves two halves: the proxy's own `llama_kv_proxy_*` metrics (a dedicated registry, so they never collide with backend names) and the merged backend `llama_server_*` text.

| Metric | Type | Labels | Meaning |
|---|---|---|---|
| `llama_kv_proxy_requests_total` | counter | `model`, `stream`, `outcome` | Chat requests by outcome: `ok`, `error`, `acquire_timeout`, `client_disconnect`. |
| `llama_kv_proxy_request_duration_seconds` | histogram | `model`, `stream` | Total request duration (start to response end). |
| `llama_kv_proxy_ttft_seconds` | histogram | `model`, `stream` | Time to first token; for non-stream it equals the total duration. |
| `llama_kv_proxy_tokens_total` | counter | `model`, `kind` | Backend-reported tokens: `prompt`, `completion`, `cached`. |
| `llama_kv_proxy_restore_total` | counter | `model`, `outcome` | Restore candidate selection for big requests: `hit` / `miss`. |
| `llama_kv_proxy_restore_ratio` | histogram | `model` | Fraction of the prompt covered by the restored cache. |
| `llama_kv_proxy_slot_wait_seconds` | histogram | `model` | Time spent acquiring a backend slot. |
| `llama_kv_proxy_saves_total` | counter | `model`, `outcome` | KV saves after big requests: `ok`, `save_error`, `save_failed`, `empty_capture`. |
| `llama_kv_proxy_save_duration_seconds` | histogram | `model` | Backend slot save duration. |
| `llama_kv_proxy_meta_files` / `llama_kv_proxy_meta_bytes` | gauge | — | Meta files on disk (count / total bytes); refreshed after startup, eviction, reconciliation and `/cache/clear`. |
| `llama_kv_proxy_bin_bytes` | gauge | — | Total `.bin` cache size on disk; refreshed as above. |
| `llama_kv_proxy_slots_total` | gauge | `backend`, `model`, `state` | Backend slots by state from the `/slots` poll. |
| `llama_kv_proxy_backend_up` | gauge | `backend` | Backend reachability from the slot poll (1 up, 0 down). |
| `llama_kv_proxy_stuck_slot_erases_total` | counter | `backend`, `model` | Slots erased by the stuck-slot watchdog. |
| `llama_kv_proxy_evictions_total` | counter | `reason` | Cache entries deleted: `evict`, `clear`, `subsumed`, `lru`, `reconcile`, `bin_clear`, `stale`. |
| `llama_kv_proxy_restore_tier_total` | counter | `tier` | Restore hits by search tier: `index`, `t1_disk`, `t2_blocks`. |
| `llama_kv_proxy_stale_meta_drops_total` | counter | `model` | Stale metas dropped after a restore reported the `.bin` missing. |
| `llama_kv_proxy_inflight_save_waits_total` | counter | `model`, `result` | Continuations that waited for an in-flight save: `hit` / `miss`. |
| `llama_kv_proxy_backend_scrape_failures_total` | counter | `backend` | Backend `/metrics` scrape failures. |

Label cardinality is bounded by design: `model`, `backend`, `stream`, `outcome`, `tier`, `reason` and `state` are small fixed sets; request ids and cache keys never appear in labels.

## Project structure

| Path | Purpose |
|---|---|
| `app/` | FastAPI app: lifespan, request-id middleware, the thin `/v1/chat/completions` endpoint, pass-through, background loops (eviction, slot polling, `.bin` cleanup/reconciliation), non-chat endpoints |
| `chat_flow/` | Chat request pipeline: effective-model resolution, cache key, restore selection, slot acquisition, backend dispatch, streaming reader, save/meta/LRU follow-up |
| `llama_kv_proxy.py` | uvicorn entry point |
| `config.py` | Environment configuration |
| `slot_manager.py` | Slot pools, LRU marks, per-slot locks, restore/save |
| `llama_client.py` | HTTP client for llama.cpp: chat, slot save/restore/erase, models, metrics |
| `hashing/` | Prefix/block hashing, meta files, two-tier matching, eviction |
| `bin_cache.py` | Direct `.bin` cleanup, LRU size cap, meta/.bin reconciliation |
| `metrics.py` | Prometheus aggregation of backend `/metrics` across backends/models |
| `promstats.py` | Proxy-level `llama_kv_proxy_*` metrics (dedicated registry) and storage gauges |
| `request_id.py` | Per-request correlation id (ContextVar + log filter) |
| `reqlog.py` | Request/response/prefix JSON logging with group rotation |
| `version.py` | Single source of truth for the proxy version |
| `tests/` | pytest suite |

## Development

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt mypy

python3 -m pytest tests/ -q   # run the test suite
ruff check .                  # lint
python3 -m mypy app/ chat_flow/ config.py slot_manager.py llama_client.py \
  hashing/ bin_cache.py metrics.py request_id.py version.py llama_kv_proxy.py
```

CI (`.github/workflows/ci.yml`) runs the same checks (ruff, mypy, pytest) on every push and pull request.

## Security

The proxy has **no authentication**. Anyone who can reach the port can read and send chat requests, wipe the cache (`POST /cache/clear`), consume all slots, read the live dashboard (which serves prompt transcripts of recent requests), and reach every native llama.cpp endpoint through the pass-through.

Run it on a trusted network (single host, private LAN, tailnet) or behind an authenticating reverse proxy. Before exposing the port further:

- set `UI_ENABLED=0` if you do not need the dashboard;
- set `REQUEST_LOG_DIR=` (empty) to stop writing full request/response bodies to disk — by default they are logged to `kv_reqlog/` (the last `REQUEST_LOG_MAX_GROUPS` groups are kept).

## Limitations & caveats

- **Single worker only.** The slot manager keeps per-process state (locks, LRU marks); multiple uvicorn workers would each track slots independently and could route two requests to the same slot.
- **`n_keep=-1` NEEDS-VERIFICATION.** The `n_keep=-1` semantics ("keep the whole prompt in the cache") must be verified against the target llama.cpp build before relying on it for cache correctness.
- **Multi-backend:** see the note in Configuration — keep all backends on the same model, or always send `model`.
- **`.bin` cleanup requires a mount.** llama.cpp has no delete endpoint; without mounting `--slot-save-path` and setting `BIN_CACHE_DIR`, `.bin` files are only purged best-effort via the (mostly no-op) `DELETE /slots` fallback.
- **`ERASE_BEFORE_SMALL` is off by default** until verified against the target build (some builds auto-clear on a prompt mismatch, in which case it is redundant).
- **`REASONING_IN_KEY` is off by default.** Enabling it changes the cache key for every message carrying a reasoning field, so previously cached entries stop matching until re-cached (deliberate key churn, only when the flag is turned on).
- **Model aliases.** A client `model` string with no discovered slot pool (e.g. an alias like `default` that llama.cpp resolves to the real model) is mapped to a model id so it shares the slot pool and cache namespace with real-name requests. The mapping uses the backend's own preset table — a name matches a model's `id` or one of its declared `aliases` in `/v1/models`, which is exactly what the backend routes on. That table is served whether or not a model is loaded, so an alias resolves during a backend restart, and it stays unambiguous with several models in the preset (a name two models both claim resolves to nothing). A name outside the preset falls back to the single detected model, which a plain single-model backend may serve regardless of the requested name. When neither applies the name goes upstream unchanged — the backend may still serve it — but the request is proxied without cache treatment, so the alias never becomes a cache namespace: since the key is `sha256(model_id + "\n" + prefix)`, such a namespace would reuse none of the cached prefixes and would outlive the outage in the on-disk metas. That window logs a `model_alias_unresolved` WARNING.
- **Model id discovery.** The backend model id is the loaded model on a router (`llama-server --models-preset`, which reports a per-model `status`), and the first entry on a plain single-model backend (whose entries carry no `status`). A router with nothing loaded resolves to `unknown` rather than to the first preset entry — that entry names a different model, and using it would fork the cache namespace. A pool is never keyed `unknown`: while the id is unresolvable the previously discovered pool is kept, so a backend restart cannot merge every model into one namespace. Alias resolution and the model id share one TTL cache, so a client that always sends an alias costs one `/v1/models` call per `MODEL_ID_TTL`, not one per request.

## Acknowledgments / Inspiration

This project is architecturally inspired by [airnsk/proxycache](https://github.com/airnsk/proxycache). That project ships without a license, and no code from it is carried over into this repository: the proxy, slot-manager, client and hashing modules were independently implemented here, and the entire history consists of code authored in this repository.

## License

MIT — see [LICENSE](./LICENSE).
