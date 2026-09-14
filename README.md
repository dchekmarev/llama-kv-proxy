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
- Prometheus `/metrics` aggregating all backends and models with `model`/`backend` labels
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

If you run into issues using gpt-oss-20b with an IDE like Cline, follow these instructions: https://www.reddit.com/r/CLine/comments/1mtcj2v/making_gptoss_20b_and_cline_work_together/

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
| `REQUEST_TIMEOUT` | `600` | HTTP timeout to the backends, seconds. |
| `ACQUIRE_TIMEOUT` | `300` | Maximum wait for a free slot, seconds. |
| `MODEL_ID` | `llama.cpp` | Fallback model id returned by `/v1/models` when the backend is unavailable. |
| `MODEL_ID_TTL` | `60` | Backend model-id cache TTL, seconds. |
| `MODEL_ID_TIMEOUT` | `5` | Model-id fetch timeout, seconds. |
| `UNKNOWN_MODEL_ID_RETRY` | `5` | Retry interval while the model id is unknown, seconds. |
| `METRICS_TIMEOUT` | `5` | Timeout for fetching a backend's `/metrics` during a scrape; a slow backend must not stall the whole `/metrics` response. |
| `SLOT_POLL_INTERVAL_S` | `30` | Interval between backend `GET /slots` polls, seconds. |
| `META_TTL_H` | `24` | Age after which `.meta` files are evicted, hours. |
| `META_MAX_FILES` | `1000` | Eviction cap on meta file count. |
| `META_MAX_MB` | `512` | Eviction cap on total meta size, MB. |
| `EVICT_INTERVAL_S` | `3600` | Interval between eviction runs, seconds. |
| `BIN_CACHE_DIR` | *(empty)* | Backend `.bin` cache directory (the mounted `--slot-save-path`); empty disables direct `.bin` cleanup. |
| `BIN_CACHE_MAX_MB` | `0` | Max total `.bin` size, MB; `0` disables the size cap. |
| `BIN_CACHE_INTERVAL_S` | `EVICT_INTERVAL_S` | Interval between `.bin` LRU cleanup runs, seconds. |
| `BIN_RECONCILE_INTERVAL_S` | `600` | Interval between meta/.bin reconciliations (both directions); `0` disables. |
| `BIN_SAVE_GRACE_S` | `10` | Grace window: a `.bin` without a meta modified within this window is treated as an in-flight save and skipped. `0` disables the guard. |
| `ERASE_BEFORE_SMALL` | `0` | `1` clears a slot's in-memory KV (`action=erase`) before dispatching a small request, so it does not start on top of another conversation's stale KV. Off by default until verified against the target build. |
| `REASONING_IN_KEY` | `0` | `1` includes `reasoning_content`/`reasoning` in the per-message cache-key parts and in the saved assistant response, so distinct reasoning traces get distinct cache keys. Only matters when the backend chat template renders reasoning into the prompt. Off by default (keys byte-identical to the legacy behavior). |
| `PORT` | `8081` | Proxy port. |
| `LOG_LEVEL` | `INFO` | Log level. |

> **Multi-backend note:** when a request omits `model`, the cache key is built from the first backend's model id. If you run multiple backends with different models, requests dispatched to the other backends may use a wrong cache key — keep all backends on the same model, or always send `model` in the request.

## Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI-compatible chat endpoint (stream and non-stream). |
| GET | `/v1/models` | Backend model list (proxied from the first backend); falls back to `MODEL_ID` when the backend is unavailable. |
| GET | `/version` | Proxy name and version (single source of truth: `version.py`). |
| GET | `/proxy/slots` | Aggregated slot state across all backends (state, n_ctx, total_tokens, LRU mark). |
| GET | `/proxy/health` | Backend availability probe plus slot state. |
| GET | `/cache/stats` | Cache file count, total size, hit/miss counters. |
| POST | `/cache/clear` | Delete all local meta files (and best-effort purge backend `.bin` files). |
| GET | `/metrics` | Prometheus target: merged backend `/metrics` across all active (loaded) models, each metric carrying `model` and `backend` labels. `?model=X` filters to a single model. A down backend/model is skipped, never failing the scrape. |
| any | `/{path}` | Forwarded to the first backend as-is (native llama.cpp endpoints: `/slots?model=...`, `/health`, `/tokenize`, …), with the response streamed back. |

## Project structure

| File | Purpose |
|---|---|
| `app.py` | FastAPI app: lifespan, request-id middleware, the thin `/v1/chat/completions` endpoint, pass-through, background loops (eviction, slot polling, `.bin` cleanup/reconciliation), non-chat endpoints |
| `chat_flow.py` | Chat request pipeline: effective-model resolution, cache key, restore selection, slot acquisition, backend dispatch, streaming reader, save/meta/LRU follow-up |
| `llama_kv_proxy.py` | uvicorn entry point |
| `config.py` | Environment configuration |
| `slot_manager.py` | Slot pools, LRU marks, per-slot locks, restore/save |
| `llama_client.py` | HTTP client for llama.cpp: chat, slot save/restore/erase, models, metrics |
| `hashing.py` | Prefix/block hashing, meta files, two-tier matching, eviction |
| `bin_cache.py` | Direct `.bin` cleanup, LRU size cap, meta/.bin reconciliation |
| `metrics.py` | Prometheus aggregation across backends/models |
| `request_id.py` | Per-request correlation id (ContextVar + log filter) |
| `version.py` | Single source of truth for the proxy version |
| `tests/` | pytest suite |

## Development

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install ruff pytest mypy

python3 -m pytest tests/ -q   # run the test suite
ruff check .                  # lint
python3 -m mypy app.py config.py slot_manager.py llama_client.py \
  hashing.py bin_cache.py metrics.py request_id.py version.py llama_kv_proxy.py \
  chat_flow.py
```

CI (`.github/workflows/ci.yml`) runs the same checks (ruff, mypy, pytest) on every push and pull request.

## Limitations & caveats

- **Single worker only.** The slot manager keeps per-process state (locks, LRU marks); multiple uvicorn workers would each track slots independently and could route two requests to the same slot.
- **`n_keep=-1` NEEDS-VERIFICATION.** The `n_keep=-1` semantics ("keep the whole prompt in the cache") must be verified against the target llama.cpp build before relying on it for cache correctness.
- **Multi-backend:** see the note in Configuration — keep all backends on the same model, or always send `model`.
- **`.bin` cleanup requires a mount.** llama.cpp has no delete endpoint; without mounting `--slot-save-path` and setting `BIN_CACHE_DIR`, `.bin` files are only purged best-effort via the (mostly no-op) `DELETE /slots` fallback.
- **`ERASE_BEFORE_SMALL` is off by default** until verified against the target build (some builds auto-clear on a prompt mismatch, in which case it is redundant).
- **`REASONING_IN_KEY` is off by default.** Enabling it changes the cache key for every message carrying a reasoning field, so previously cached entries stop matching until re-cached (deliberate key churn, only when the flag is turned on).

## Acknowledgments / Inspiration

This project was inspired by and initially built upon concepts from [airnsk/proxycache](https://github.com/airnsk/proxycache).

## License

MIT — see [LICENSE](./LICENSE).
