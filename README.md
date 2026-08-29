<img width="1000"  alt="image_" src="https://github.com/user-attachments/assets/0d966dde-f1d8-432f-bad0-aa79a5ccf396" />

### What this service is

This service is a proxy in front of llama.cpp that makes long‑context chat and IDE workflows much faster by managing llama.cpp slots, reusing cached context, and restoring saved caches from disk when needed. It speaks an OpenAI‑compatible Chat Completions API, so existing clients can connect without changes, including both streaming (SSE) and non‑stream responses depending on request settings.

### Why it’s needed

llama.cpp provides “slots,” each holding a conversation’s KV cache so repeated requests with the same or very similar prefix can skip recomputing the whole prompt and continue from the first mismatching token, which dramatically cuts latency for large prompts. In real teams the number of users can easily exceed the number of available slots (e.g., 20 developers but only 4 slots), so naive routing causes random slot reuse and cache overwrites that waste time and GPU/CPU cycles. This proxy solves that by steering requests to the right slot, saving evicted caches to disk, and restoring them on demand, so long prompts don’t need to be recomputed from scratch each time.

### How requests are balanced and slots are chosen

- Big vs small: A request is “big” when its prompt has more than BIG_THRESHOLD_WORDS words. Big requests get the full cache treatment (restore, save, meta files); small ones are routed to a free or oldest slot without touching the disk cache.
- Restore on demand: For a big request the proxy computes a fast word‑block LCP similarity against the saved .meta descriptors and, if the best match covers at least LCP_TH of the request, restores that cache into the chosen slot — seconds instead of minutes for long contexts, especially in IDE scenarios with 30–60k tokens.
- Free first, oldest when full: The proxy picks a free slot (never used yet) or, when none are free, the least‑recently‑used one.
- Concurrency safety: Each slot is guarded with an async lock; if all slots are busy the request waits up to ACQUIRE_TIMEOUT seconds and then gets HTTP 503.

### Save and restore from disk

llama.cpp’s HTTP server exposes slot save/restore; saving writes a cache file to the directory provided by --slot‑save‑path, and restore loads by file basename (e.g., slotcache_`<key>`.bin), which is exactly how this proxy persists and revives caches across requests and restarts. The proxy keeps small local .meta files describing cached prefixes for fast lookup, while llama.cpp owns the actual KV .bin files under --slot‑save‑path for correctness and performance.

### Request forwarding

The proxy forwards the client's request body to the backend verbatim (pass-through) and injects the following fields:

- `model` — taken from the request, or from the `MODEL_ID` env var if absent;
- `cache_prompt` — `true` for big requests (more than `BIG_THRESHOLD_WORDS` words);
- `n_keep` — `-1` (see below);
- slot pinning: `slot_id`/`id_slot` are duplicated in the body root, in `options`, and in the query parameters, so the request is pinned to the selected slot.

> **NEEDS-VERIFICATION:** the `n_keep=-1` semantics ("keep the whole prompt in the cache") must be verified against the target llama.cpp build before relying on it for cache correctness.

### Quick start

1) Start llama.cpp ( https://github.com/ggml-org/llama.cpp ) with slots and a cache directory:

```bash
llama-server -m ./model.gguf -np 4 --slot-save-path /var/kvcache --host 0.0.0.0 --port 8080 --swa-full
```

This enables the OpenAI‑compatible HTTP server, a pool of 4 slots, and a directory where slot KV caches are saved and restored by basename.

2) Run the proxy next to it:

```bash
git clone https://github.com/airnsk/proxycache.git
cd proxycache
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
python3 proxycache.py  # or: uvicorn app:app --host 0.0.0.0 --port 8081
```

Run the proxy with a **single worker** (the default). The slot manager keeps per‑process state (locks, LRU marks); multiple workers would each track slots independently and could route two requests to the same slot.

Your clients should call the proxy’s /v1/chat/completions endpoint; the proxy will handle similarity, slot selection, save/restore, and streaming vs non‑streaming automatically.

If you run into issues using gpt-oss-20b with an IDE like Cline, follow these instructions: https://www.reddit.com/r/CLine/comments/1mtcj2v/making_gptoss_20b_and_cline_work_together/

### Parameters

All are environment variables; defaults in parentheses.

- BACKENDS: JSON list of backends, e.g., `[{"url": "http://127.0.0.1:8000", "n_slots": 2}]`. If unset, falls back to a single backend from LLAMA_URL (http://127.0.0.1:8000) and N_SLOTS (2).
- WORDS_PER_BLOCK: Words per hash block for LCP (100).
- BIG_THRESHOLD_WORDS: Prompts longer than this are “big” (500).
- LCP_TH: Minimum share of the request that a cached prefix must cover to be restored (0.6).
- META_DIR: Directory for local .meta descriptors, relative to the app directory (kv_meta).
- REQUEST_TIMEOUT: HTTP timeout to the backends in seconds (600).
- ACQUIRE_TIMEOUT: Maximum wait for a free slot in seconds (300).
- MODEL_ID: Model id advertised to clients (llama.cpp).
- MODEL_ID_TTL / MODEL_ID_TIMEOUT / UNKNOWN_MODEL_ID_RETRY: Backend model‑id cache TTL (60s), fetch timeout (5s), and retry interval while the id is unknown (5s).
- SLOT_POLL_INTERVAL_S: Interval between backend GET /slots polls (30).
- META_TTL_H: Age after which .meta files are evicted (24h).
- META_MAX_FILES / META_MAX_MB: Eviction caps on file count (1000) and total size (512 MB).
- EVICT_INTERVAL_S: Interval between eviction runs (3600).
- PORT: Proxy port (8081).
- LOG_LEVEL: Log level (INFO).

> **Multi-backend note:** the cache key is built from the model id of the first backend (`BACKENDS[0]`). If you run multiple backends with different models, requests dispatched to the other backends may use a wrong cache key. Keep all backends on the same model.

### Endpoints

- POST /v1/chat/completions — the OpenAI‑compatible chat endpoint (stream and non‑stream).
- GET /v1/models — the advertised model id.
- GET /slots — aggregated slot state across all backends (state, n_ctx, total_tokens, LRU mark).
- GET /cache/stats — cache file count, total size, hit/miss counters.
- POST /cache/clear — delete all local meta files (and best‑effort purge backend .bin files).

### Tests

```bash
pip install -r requirements-dev.txt
python3 -m pytest tests/ -q
ruff check .
```

### Why this boosts IDE and long‑context productivity

For 30–60k‑token contexts typical in project‑wide IDE assistants, recomputing a full prompt can take minutes, whereas restoring a previously cached context and continuing from the first mismatching token typically takes seconds on llama.cpp, dramatically improving iteration speed for large teams with limited slots.
