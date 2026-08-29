# config.py

"""
Единая конфигурация для simple_proxycache:
- BACKENDS: [{"url": "...", "n_slots": N}]
- WORDS_PER_BLOCK, BIG_THRESHOLD_WORDS, LCP_TH
- PORT, REQUEST_TIMEOUT, MODEL_ID
"""

import json
import logging
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


# Backends
BACKENDS_RAW = os.getenv("BACKENDS")
BACKENDS = parse_backends_env(BACKENDS_RAW)
validate_backends(BACKENDS)

# Words per block for LCP
WORDS_PER_BLOCK = int(os.getenv("WORDS_PER_BLOCK", "100"))

# Big request threshold
BIG_THRESHOLD_WORDS = int(os.getenv("BIG_THRESHOLD_WORDS", "500"))

# LCP threshold (0..1)
LCP_TH = float(os.getenv("LCP_TH", "0.6"))

# Meta dir: anchored to the app directory, not the process cwd, so the cache
# location does not change depending on where the process was started.
APP_DIR = os.path.dirname(os.path.abspath(__file__))
META_DIR = os.path.join(APP_DIR, os.getenv("META_DIR", "kv_meta"))
os.makedirs(META_DIR, exist_ok=True)

# HTTP timeout
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", "600"))

# Timeout for waiting on a free slot when all slots are busy.
ACQUIRE_TIMEOUT = float(os.getenv("ACQUIRE_TIMEOUT", "300"))

# Model id
MODEL_ID = os.getenv("MODEL_ID", "llama.cpp")

# Model id cache: TTL (seconds), short timeout for /v1/models,
# and a short retry interval while the id is still unknown.
MODEL_ID_TTL = float(os.getenv("MODEL_ID_TTL", "60"))
MODEL_ID_TIMEOUT = float(os.getenv("MODEL_ID_TIMEOUT", "5"))
UNKNOWN_MODEL_ID_RETRY = float(os.getenv("UNKNOWN_MODEL_ID_RETRY", "5"))

# Interval (seconds) between backend slot-state polls (GET /slots).
SLOT_POLL_INTERVAL_S = float(os.getenv("SLOT_POLL_INTERVAL_S", "30"))

# Cache eviction: TTL (hours), file count cap, total size cap (MB),
# and the interval (seconds) between periodic eviction runs.
META_TTL_H = float(os.getenv("META_TTL_H", "24"))
META_MAX_FILES = int(os.getenv("META_MAX_FILES", "1000"))
META_MAX_MB = int(os.getenv("META_MAX_MB", "512"))
EVICT_INTERVAL_S = float(os.getenv("EVICT_INTERVAL_S", "3600"))

# Service port
PORT = int(os.getenv("PORT", "8081"))

# Logs
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL.upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
