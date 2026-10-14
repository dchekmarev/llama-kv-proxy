# core/config.py

"""
Unified configuration for llama-kv-proxy:
- BACKENDS: [{"url": "...", "n_slots": N}]
- WORDS_PER_BLOCK, BIG_THRESHOLD_WORDS, LCP_TH
- PORT, REQUEST_TIMEOUT, CONNECT_TIMEOUT, KEEPALIVE_EXPIRY_S, RETRY_AFTER_S,
  BACKEND_RETRY_ATTEMPTS, BACKEND_RETRY_BACKOFF_S, HOLD_TIMEOUT,
  HEALTH_PROBE_BACKOFF_S, HEALTH_PROBE_MAX_S, MODEL_ID

This module only reads the environment: it creates no directories and
configures no logging at import time. Call init_runtime() at startup.
"""

import json
import os


def parse_backends_env(raw: str | None) -> list[dict]:
    """Parse the BACKENDS env var (JSON list) or fall back to LLAMA_URL/N_SLOTS.

    Raises ValueError with a clear message on broken JSON or a bad N_SLOTS.
    """
    if not raw:
        try:
            n_slots = int(os.getenv("N_SLOTS", "2"))
        except ValueError as e:
            raise ValueError(f"N_SLOTS env must be an integer: {e}") from e
        return [
            {"url": os.getenv("LLAMA_URL", "http://127.0.0.1:8000"), "n_slots": n_slots}
        ]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"BACKENDS env is not valid JSON: {e}") from e
    return parsed


def validate_backends(backends: object) -> None:
    """Validate the backend list; raise ValueError with a clear message.

    A silently empty list used to cause a confusing IndexError later
    (clients[0] / min() over no slots).
    """
    if not isinstance(backends, list) or not backends:
        raise ValueError(
            "BACKENDS config is empty or not a list; provide a JSON list like "
            '[{"url": "http://127.0.0.1:8000", "n_slots": 2}]'
        )
    for i, be in enumerate(backends):
        if not isinstance(be, dict):
            # ValueError on purpose: validate_backends raises a single
            # exception type so callers catch one thing.
            raise ValueError(f"BACKENDS[{i}] must be an object, got {be!r}")  # noqa: TRY004
        url = be.get("url")
        if not isinstance(url, str) or not url:
            raise ValueError(f"BACKENDS[{i}].url is missing or not a string: {be!r}")
        n_slots = be.get("n_slots")
        if not isinstance(n_slots, int) or isinstance(n_slots, bool) or n_slots <= 0:
            raise ValueError(
                    f"BACKENDS[{i}].n_slots must be a positive integer: {be!r}"
            )


def _env_int(name: str, default: int) -> int:
    """Read an integer env var; empty/unset -> default, bad value -> clear error."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as e:
        raise ValueError(f"{name} env must be an integer, got {raw!r}") from e


def _env_float(name: str, default: float) -> float:
    """Read a float env var; empty/unset -> default, bad value -> clear error.

    Always returns a float (the default is coerced too) so callers can rely on
    a consistent type for the interval/timeout values.
    """
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return float(default)
    try:
        return float(raw)
    except ValueError as e:
        raise ValueError(f"{name} env must be a number, got {raw!r}") from e


def _env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var (1/true/yes/on); empty/unset -> default."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


# Backends
BACKENDS_RAW = os.getenv("BACKENDS")
BACKENDS = parse_backends_env(BACKENDS_RAW)

# Words per block for LCP
WORDS_PER_BLOCK = _env_int("WORDS_PER_BLOCK", 100)

# Big request threshold
BIG_THRESHOLD_WORDS = _env_int("BIG_THRESHOLD_WORDS", 500)

# LCP threshold (0..1)
LCP_TH = _env_float("LCP_TH", 0.6)

# Meta dir: anchored to the app directory, not the process cwd, so the cache
# location does not change depending on where the process was started. This
# module lives in the core/ package, so the app directory is its parent: the
# kv_meta/ and kv_reqlog/ dirs sit next to app/, chat_flow/ and friends, not
# inside core/.
APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META_DIR = os.path.join(APP_DIR, os.getenv("META_DIR", "kv_meta"))

# HTTP timeout (read/write/pool). The connect phase is CONNECT_TIMEOUT.
REQUEST_TIMEOUT = _env_float("REQUEST_TIMEOUT", 1500)

# Connect-phase timeout, seconds: a refused connection (or a SYN-dropping
# black hole) must fail fast and surface as a 502 + Retry-After instead of
# holding the request (and its slot) for the whole REQUEST_TIMEOUT. A port
# that accepts and then never answers is the read phase, bounded by
# REQUEST_TIMEOUT.
CONNECT_TIMEOUT = _env_float("CONNECT_TIMEOUT", 10)

# Keep-alive expiry (seconds) of the pooled backend HTTP client. A connection
# left idle longer than the backend's (or a fronting proxy's) idle timeout is
# closed by the far side; reusing it then fails with a spurious "Server
# disconnected without sending a response". Keep this below that idle timeout
# to avoid the stale reuse; a racing close is still covered by the proxy-side
# retry below. See README "Backend unavailable".
KEEPALIVE_EXPIRY_S = max(0.0, _env_float("KEEPALIVE_EXPIRY_S", 5.0))

# Retry-After header value (seconds) on 502 upstream-failure responses: a
# retrying client waits out a backend outage (down, restarting, reloading)
# instead of hammering the proxy. See README "Backend unavailable".
RETRY_AFTER_S = _env_int("RETRY_AFTER_S", 300)

# Proxy-side retry of a retryable backend failure (transport error or upstream
# 5xx) that happens before any response byte reaches the client. Instead of
# bouncing the client with 502 + Retry-After, the proxy re-runs the attempt
# (up to BACKEND_RETRY_ATTEMPTS total, spaced by BACKEND_RETRY_BACKOFF_S with
# exponential growth) and keeps the client connection open. 1 disables the
# retry (the first failure is answered 502 + Retry-After as before). A
# mid-stream failure is never retried: the response is already committed.
BACKEND_RETRY_ATTEMPTS = max(1, _env_int("BACKEND_RETRY_ATTEMPTS", 3))
BACKEND_RETRY_BACKOFF_S = max(0.0, _env_float("BACKEND_RETRY_BACKOFF_S", 0.5))

# Max seconds a request holds its connection (and its slot) while the picked
# backend is unreachable or unhealthy (5xx on /health, e.g. model loading):
# the proxy waits for the backend to come back instead of bouncing the client
# with 502 + Retry-After right away. 0 (or negative) disables the wait (one
# probe, then 502). See README "Backend unavailable".
HOLD_TIMEOUT = max(0.0, _env_float("HOLD_TIMEOUT", 240))

# Backoff for the shared per-backend health prober while a backend stays down.
# The first failed probe is followed by HEALTH_PROBE_BACKOFF_S, doubling each
# probe up to HEALTH_PROBE_MAX_S, to stop hammering /health during a long
# outage. 0 (default) keeps the fixed HOLD_PROBE_INTERVAL_S cadence (prompt
# recovery). MAX_S is ignored while BACKOFF_S is 0. See README "Backend
# unavailable".
HEALTH_PROBE_BACKOFF_S = max(0.0, _env_float("HEALTH_PROBE_BACKOFF_S", 0.0))
HEALTH_PROBE_MAX_S = max(0.0, _env_float("HEALTH_PROBE_MAX_S", 30.0))

# Pause API (POST /proxy/pause): stop dispatching new work, drain in-flight
# committed work, then hold every new request until resume (manual) or until
# the timer elapses. The timer starts once the drain completes. PAUSE_DEFAULT_S
# is used when the request omits `seconds`; PAUSE_MAX_S caps a requested value
# (0 disables the cap). See README "Pause".
PAUSE_DEFAULT_S = _env_float("PAUSE_DEFAULT_S", 60)
PAUSE_MAX_S = max(0.0, _env_float("PAUSE_MAX_S", 0))

# Timeout for waiting on a free slot when all slots are busy.
ACQUIRE_TIMEOUT = _env_float("ACQUIRE_TIMEOUT", 1500)

# Maximum number of requests parked waiting for a free slot, across all models.
# A request that cannot take a slot immediately is failed fast with 503 once
# this many waiters are queued, instead of holding its connection and body.
# 0 disables the cap (unbounded queue). See README "Slot selection".
MAX_WAITING = max(0, _env_int("MAX_WAITING", 64))

# Max seconds a big request waits for an in-flight save of a prefix of its own
# conversation before giving up on the restore. The client treats [DONE] as the
# end of the response and sends the continuation immediately, but the previous
# message's meta only lands after its .bin write finishes; without this wait
# the continuation misses the restore and reprocesses the whole prompt.
# 0 disables the wait.
SAVE_WAIT_TIMEOUT = _env_float("SAVE_WAIT_TIMEOUT", 30)

# Model id
MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Model id cache: TTL (seconds), short timeout for /v1/models,
# and a short retry interval while the id is still unknown.
MODEL_ID_TTL = _env_float("MODEL_ID_TTL", 60)
MODEL_ID_TIMEOUT = _env_float("MODEL_ID_TIMEOUT", 5)
UNKNOWN_MODEL_ID_RETRY = _env_float("UNKNOWN_MODEL_ID_RETRY", 5)

# Short timeout (seconds) for fetching a backend's /metrics during a scrape:
# a slow/down backend must not stall the whole /metrics response.
METRICS_TIMEOUT = _env_float("METRICS_TIMEOUT", 5)

# Interval (seconds) between backend slot-state polls (GET /slots).
SLOT_POLL_INTERVAL_S = _env_float("SLOT_POLL_INTERVAL_S", 30)

# On-demand slot freshen: before a request picks a slot, re-poll the backends
# serving its model so a slot cut/reload is observed within ~this window
# instead of waiting for the periodic poll. Rate-limited per-(backend, model)
# to at most once per interval: a slot topology change is a slow event, so a
# sub-interval re-poll cannot see a different topology. 0 disables the freshen.
SLOT_FRESHEN_INTERVAL_S = _env_float("SLOT_FRESHEN_INTERVAL_S", 1.0)

# Cache eviction: TTL (hours), file count cap, total size cap (MB),
# and the interval (seconds) between periodic eviction runs.
# A cap of 0 disables that limit (same convention as BIN_CACHE_MAX_MB).
META_TTL_H = _env_float("META_TTL_H", 24)
META_MAX_FILES = _env_int("META_MAX_FILES", 1000)
META_MAX_MB = _env_int("META_MAX_MB", 512)
EVICT_INTERVAL_S = _env_float("EVICT_INTERVAL_S", 3600)

# In-memory restore index (the "B+A" design): an in-RAM hash->key index built
# from the metas' per-message prefix hashes, so a restore search is an O(n) set
# lookup (walk the request's prefix hashes from the longest, take the first
# known meta) instead of scanning and JSON-parsing every meta file. On by
# default; the on-disk two-tier scan stays the fallback (index miss, empty
# index, or old blocks-only metas that are not indexed).
META_INDEX_ENABLED = _env_bool("META_INDEX_ENABLED", True)
# Interval (seconds) between index<->disk reconciliations: drops ghost entries
# left by external meta deletions (bin_cache reconcile/clean remove files
# without notifying the index). 0 disables the periodic reconcile.
META_INDEX_RECONCILE_INTERVAL_S = _env_float("META_INDEX_RECONCILE_INTERVAL_S", 300)

# Backend .bin cache directory (mounted from the host --slot-save-path).
# Empty disables direct .bin cleanup.
BIN_CACHE_DIR = os.getenv("BIN_CACHE_DIR", "")
# Max total size (MB) of .bin files; oldest (LRU) are deleted first.
# 0 disables the size cap.
BIN_CACHE_MAX_MB = _env_int("BIN_CACHE_MAX_MB", 0)
# Interval (seconds) between meta/.bin reconciliations (both directions:
# stale metas without a .bin, orphan .bin without a meta). 0 disables it.
BIN_RECONCILE_INTERVAL_S = _env_float("BIN_RECONCILE_INTERVAL_S", 600)
# Grace window (seconds): a .bin file without a meta that was modified within
# this window is treated as an in-flight save (its meta may not have been
# written yet) and is skipped by the LRU cleanup and orphan reconciliation, so
# a just-saved .bin is not deleted before its meta lands. 0 disables the guard.
BIN_SAVE_GRACE_S = _env_float("BIN_SAVE_GRACE_S", 10)
# Minimum size (MB) of a .bin save to count as a real slot save. llama.cpp
# writes only a small header file when the slot's KV cache is empty (e.g. the
# slot was erased mid-generation); recording a meta for such a save poisons the
# cache chain with an empty restore target and deletes the valid shorter
# caches via the subsumed-metas logic. 0 disables the check.
MIN_BIN_SIZE_VALID = _env_int("MIN_BIN_SIZE_VALID", 1)

# Erase a slot's in-memory KV cache (action=erase) before dispatching any chat
# that was NOT preceded by a successful restore. A successful restore already
# sets the slot's prompt to the correct prefix, so it is skipped. Otherwise the
# slot may still hold a stale or oversized prompt from a previous conversation;
# starting a chat on top of it can wedge llama.cpp in PROCESSING_PROMPT (the
# slot never finishes and the server busy-loops). On by default (correctness
# over latency): a big request with no restore hit reprocesses its prompt.
ERASE_BEFORE_CHAT = _env_bool("ERASE_BEFORE_CHAT", True)

# Skip a pre-chat restore when the picked slot already holds the target key's
# KV: it is the slot that just saved that key (a save does not clear the
# slot), or the key was restored there and the slot was not used since. The
# restore would only re-read the .bin and rebuild identical KV. A stale record
# (e.g. a backend restart with the same slot ids) costs at most one full
# re-prefill: the restore is a speed optimization, so a wrong skip never
# changes the output.
SKIP_RESTORE_SAME_SLOT = _env_bool("SKIP_RESTORE_SAME_SLOT", True)

# Watchdog: if a backend slot reports is_processing for longer than this many
# seconds, it is presumed wedged (stuck in prompt processing) and is erased to
# recover the backend without a restart. 0 disables the watchdog.
STUCK_SLOT_THRESHOLD_S = _env_float("STUCK_SLOT_THRESHOLD_S", 1500)

# Include the reasoning fields (reasoning_content / reasoning) in the
# per-message cache-key parts and in the saved assistant response, so two
# requests with identical content but different reasoning traces get distinct
# cache keys. Only matters when the backend chat template renders reasoning
# into the prompt (DeepSeek-R1/Qwen3-style templates). Off by default: when
# off, keys are byte-identical to the legacy behavior (no cache churn).
REASONING_IN_KEY = _env_bool("REASONING_IN_KEY", False)

# Top-level request fields folded into the cache key as a render-context
# leader: they change the PROMPT the backend template renders (tools and
# reasoning instructions land inside the system text), so two requests with
# identical messages but different render params must not share a KV cache. The
# leader is a "ctx:<sha256>" synthetic part prepended to the prefix, so
# differently-rendered conversations diverge at block 0 and never match as
# restore candidates.
RENDER_CTX_FIELDS: tuple[str, ...] = (
    "tools",
    "reasoning_effort",
    "enable_thinking",
    "preserve_thinking",
    "preserve_reasoning",
    "auto_disable_thinking_with_tools",
    "thinking_budget",
    "reasoning_budget_tokens",
    "chat_template_kwargs",
)

# Request/response/prefix logging directory: every /v1/chat/completions
# request writes a group of JSON files (request/response/prefix, plus raw SSE
# for streams) named {timestamp_ms}.{request_id}.{type}.json. Relative paths
# are anchored to the app directory (like META_DIR). Empty disables logging.
REQUEST_LOG_DIR = os.getenv("REQUEST_LOG_DIR", "kv_reqlog")
if REQUEST_LOG_DIR and not os.path.isabs(REQUEST_LOG_DIR):
    REQUEST_LOG_DIR = os.path.join(APP_DIR, REQUEST_LOG_DIR)
# Max number of request groups kept in the directory; oldest groups (all their
# files) are deleted first. 0 disables rotation (keep everything).
REQUEST_LOG_MAX_GROUPS = _env_int("REQUEST_LOG_MAX_GROUPS", 100)

# Live observability dashboard served by the proxy itself under /proxy/ui/
# (same port): in-flight requests with their prompt preview, the live token
# stream per request (SSE), slot states, and recent history. In-memory only.
UI_ENABLED = _env_bool("UI_ENABLED", True)
# Number of finished requests kept in the in-memory history.
UI_HISTORY_MAX = _env_int("UI_HISTORY_MAX", 200)
# Max chars of the generated response kept per request as the live tail.
UI_TAIL_MAX_CHARS = _env_int("UI_TAIL_MAX_CHARS", 16384)
# Max chars of the prompt transcript shown in the request list.
UI_PREVIEW_MAX_CHARS = _env_int("UI_PREVIEW_MAX_CHARS", 2048)
# Max chars of the full prompt transcript kept while a request is in flight
# (fetched on demand by the dashboard; dropped when the request ends).
UI_PROMPT_FULL_MAX_CHARS = _env_int("UI_PROMPT_FULL_MAX_CHARS", 1048576)

# Context-budget admission control: estimate the prompt's token count before
# dispatch and reject a request that would not fit the backend's context, so an
# oversized prompt fails fast instead of holding a slot for a whole prefill.
# MAX_TOKENS is the generation budget reserved per request when the client sends
# neither max_tokens nor n_predict; CONTEXT_ADMISSION_ENABLED turns the check
# off. TOKENIZER_ENABLED/TOKENIZER_MODEL select the optional local Qwen
# tokenizer (dashscope, see requirements-tokenizer.txt); TOKENS_PER_WORD and
# PROMPT_TEMPLATE_MARGIN_TOKENS tune the word-count fallback used when it is off
# or unavailable.
MAX_TOKENS = _env_int("MAX_TOKENS", 32768)
CONTEXT_ADMISSION_ENABLED = _env_bool("CONTEXT_ADMISSION_ENABLED", True)
TOKENIZER_ENABLED = _env_bool("TOKENIZER_ENABLED", True)
TOKENIZER_MODEL = os.getenv("TOKENIZER_MODEL", "qwen-turbo")
TOKENS_PER_WORD = _env_float("TOKENS_PER_WORD", 1.3)
PROMPT_TEMPLATE_MARGIN_TOKENS = _env_int("PROMPT_TEMPLATE_MARGIN_TOKENS", 64)

# Service port
PORT = _env_int("PORT", 8081)

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


def init_runtime() -> None:
    """Validate the config and create the directories the proxy writes to.

    Called from the app lifespan instead of at import time: importing config
    must not touch the filesystem, so tests, linters and scripts can import it
    without side effects. Idempotent.
    """
    validate_backends(BACKENDS)
    os.makedirs(META_DIR, exist_ok=True)
    if REQUEST_LOG_DIR:
        os.makedirs(REQUEST_LOG_DIR, exist_ok=True)
