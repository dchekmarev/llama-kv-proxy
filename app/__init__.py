# app/__init__.py

"""The proxy's HTTP surface: FastAPI app, routes, lifespan and background jobs.

Route map (the /proxy prefix keeps the native llama.cpp paths free for the
pass-through: /slots, /health, /tokenize, /completion, ... reach the backend):

- POST /v1/chat/completions  the cached chat pipeline (chat_flow)
- GET  /v1/models           the union of the backends' model lists
- GET  /proxy/health        per-backend probe + slot state
- GET  /proxy/slots         the discovered slot pools
- GET  /cache/stats         meta cache files/bytes/hits/misses
- GET,POST /cache/clear     drop the whole cache (metas + backend .bin)
- GET  /metrics             Prometheus text (proxy registry + backends)
- GET  /version             build identifier
- GET  /proxy/ui[/]         the live dashboard (page, state, request, events)
- ANY  /{path:path}         pass-through to the first backend

Everything blocking (meta scans, .bin cleanup) runs in a worker thread so the
event loop keeps serving requests, and every background job is a task that is
cancelled on shutdown.

The implementation lives in private submodules (_asgi, background, middleware,
routes). Every name this package owns - including the private ones and the
config constants read inside the loops - is re-exported here, and the
submodules resolve it through this module at call time, so monkeypatching
`app.<name>` is seen by the internal code exactly as it was when this was a
single flat module. app.state is the one state object the handlers and the
background jobs share: it belongs to the FastAPI instance built in _asgi.
"""

import asyncio
import time

from config import (
    BACKENDS,
    BIN_CACHE_DIR,
    BIN_CACHE_MAX_MB,
    BIN_RECONCILE_INTERVAL_S,
    EVICT_INTERVAL_S,
    LOG_LEVEL,
    META_INDEX_ENABLED,
    META_INDEX_RECONCILE_INTERVAL_S,
    META_MAX_FILES,
    META_MAX_MB,
    META_TTL_H,
    MODEL_ID,
    SLOT_POLL_INTERVAL_S,
    STUCK_SLOT_THRESHOLD_S,
    UI_ENABLED,
    init_runtime,
)
from llama_client import LlamaClient
from logging_setup import setup_logging
from slot_manager import SlotManager

from ._asgi import app, lifespan
from .background import (
    _bin_reconcile_loop,
    _check_stuck_slots,
    _delete_backend_caches,
    _eviction_loop,
    _key_model_pairs,
    _meta_index_reconcile_loop,
    _poll_slots,
    _poll_slots_loop,
    _run_eviction,
    _stuck_slot_first_busy,
)
from .middleware import request_id_middleware
from .routes import (
    cache_clear,
    cache_stats,
    chat,
    health,
    metrics_endpoint,
    models,
    passthrough,
    slots_state,
    ui_events,
    ui_page_route,
    ui_request,
    ui_state,
    version,
)

# Periodic meta-index<->disk reconcile task. A module global (not a lifespan
# local) so shutdown and tests can see it; None when the interval is 0.
_meta_index_reconcile_task: "asyncio.Task[None] | None" = None

__all__ = [
    "BACKENDS",
    "BIN_CACHE_DIR",
    "BIN_CACHE_MAX_MB",
    "BIN_RECONCILE_INTERVAL_S",
    "EVICT_INTERVAL_S",
    "LOG_LEVEL",
    "META_INDEX_ENABLED",
    "META_INDEX_RECONCILE_INTERVAL_S",
    "META_MAX_FILES",
    "META_MAX_MB",
    "META_TTL_H",
    "MODEL_ID",
    "SLOT_POLL_INTERVAL_S",
    "STUCK_SLOT_THRESHOLD_S",
    "UI_ENABLED",
    "LlamaClient",
    "SlotManager",
    "_bin_reconcile_loop",
    "_check_stuck_slots",
    "_delete_backend_caches",
    "_eviction_loop",
    "_key_model_pairs",
    "_meta_index_reconcile_loop",
    "_meta_index_reconcile_task",
    "_poll_slots",
    "_poll_slots_loop",
    "_run_eviction",
    "_stuck_slot_first_busy",
    "app",
    "asyncio",
    "cache_clear",
    "cache_stats",
    "chat",
    "health",
    "init_runtime",
    "lifespan",
    "metrics_endpoint",
    "models",
    "passthrough",
    "request_id_middleware",
    "setup_logging",
    "slots_state",
    "time",
    "ui_events",
    "ui_page_route",
    "ui_request",
    "ui_state",
    "version",
]
