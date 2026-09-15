# config.py

"""
Unified configuration for llama-kv-proxy:
- BACKENDS: [{"url": "...", "n_slots": N}]
- WORDS_PER_BLOCK, BIG_THRESHOLD_WORDS, LCP_TH
- PORT, REQUEST_TIMEOUT, MODEL_ID
"""

import json
import logging
import os

from request_id import RequestIdFilter


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
validate_backends(BACKENDS)

# Words per block for LCP
WORDS_PER_BLOCK = _env_int("WORDS_PER_BLOCK", 100)

# Big request threshold
BIG_THRESHOLD_WORDS = _env_int("BIG_THRESHOLD_WORDS", 500)

# LCP threshold (0..1)
LCP_TH = _env_float("LCP_TH", 0.6)

# Meta dir: anchored to the app directory, not the process cwd, so the cache
# location does not change depending on where the process was started.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(APP_DIR, os.getenv("META_DIR", "kv_meta"))
os.makedirs(META_DIR, exist_ok=True)

# HTTP timeout
REQUEST_TIMEOUT = _env_float("REQUEST_TIMEOUT", 1500)

# Timeout for waiting on a free slot when all slots are busy.
ACQUIRE_TIMEOUT = _env_float("ACQUIRE_TIMEOUT", 1500)

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

# Cache eviction: TTL (hours), file count cap, total size cap (MB),
# and the interval (seconds) between periodic eviction runs.
# A cap of 0 disables that limit (same convention as BIN_CACHE_MAX_MB).
META_TTL_H = _env_float("META_TTL_H", 24)
META_MAX_FILES = _env_int("META_MAX_FILES", 1000)
META_MAX_MB = _env_int("META_MAX_MB", 512)
EVICT_INTERVAL_S = _env_float("EVICT_INTERVAL_S", 3600)

# Backend .bin cache directory (mounted from the host --slot-save-path).
# Empty disables direct .bin cleanup.
BIN_CACHE_DIR = os.getenv("BIN_CACHE_DIR", "")
# Max total size (MB) of .bin files; oldest (LRU) are deleted first.
# 0 disables the size cap.
BIN_CACHE_MAX_MB = _env_int("BIN_CACHE_MAX_MB", 0)
# Interval (seconds) between .bin LRU cleanup runs.
BIN_CACHE_INTERVAL_S = _env_float("BIN_CACHE_INTERVAL_S", EVICT_INTERVAL_S)
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

# Clear a slot's in-memory KV cache (action=erase) before dispatching a small
# (non-cached) request, so it does not start on top of another conversation's
# stale KV. Some llama.cpp builds auto-clear on a prompt mismatch, in which
# case this is redundant; verify against the target build before relying on it.
# Off by default (no behavior change until verified).
ERASE_BEFORE_SMALL = _env_bool("ERASE_BEFORE_SMALL", False)

# Erase a slot's in-memory KV cache (action=erase) before dispatching any chat
# that was NOT preceded by a successful restore. A successful restore already
# sets the slot's prompt to the correct prefix, so it is skipped. Otherwise the
# slot may still hold a stale or oversized prompt from a previous conversation;
# starting a chat on top of it can wedge llama.cpp in PROCESSING_PROMPT (the
# slot never finishes and the server busy-loops). On by default (correctness
# over latency): a big request with no restore hit reprocesses its prompt.
ERASE_BEFORE_CHAT = _env_bool("ERASE_BEFORE_CHAT", True)

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
if REQUEST_LOG_DIR:
    if not os.path.isabs(REQUEST_LOG_DIR):
        REQUEST_LOG_DIR = os.path.join(APP_DIR, REQUEST_LOG_DIR)
    os.makedirs(REQUEST_LOG_DIR, exist_ok=True)
# Max number of request groups kept in the directory; oldest groups (all their
# files) are deleted first. 0 disables rotation (keep everything).
REQUEST_LOG_MAX_GROUPS = _env_int("REQUEST_LOG_MAX_GROUPS", 100)

# Service port
PORT = _env_int("PORT", 8081)

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


_logging_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once (idempotent).

    Called from the entry point and app startup so logging is set up
    regardless of launch mode (python llama_kv_proxy.py or uvicorn app:app).
    Importing config no longer configures logging as a side effect. The
    request-id filter is attached to the root handlers so every record carries
    the current request's correlation id (empty outside a request).
    """
    global _logging_configured
    if _logging_configured:
        return
    root = logging.getLogger()
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    )
    for handler in root.handlers:
        handler.addFilter(RequestIdFilter())
    _logging_configured = True
