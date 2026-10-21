# chat_flow/__init__.py

"""Chat request orchestration for llama-kv-proxy.

Holds the /v1/chat/completions request pipeline: effective-model resolution,
cache-key computation, restore-candidate selection, slot acquisition, dispatch
to the backend, and the save/meta/LRU follow-up. The FastAPI endpoint in
app/routes.py only parses the JSON body and delegates here.

Additionally:

- acquire_for_request is wrapped in a timeout so it cannot hang forever if a
  slot is never released.
- A retryable backend failure (a stale pooled connection, a restart, a
  transient upstream 5xx) before the response commits does not bounce the
  client with 502 + Retry-After: the attempt is re-run (bounded by
  BACKEND_RETRY_ATTEMPTS, spaced by BACKEND_RETRY_BACKOFF_S) with the same
  correlation id and request group (_retry).
- For non-streaming and streaming big requests the save+meta runs in a
  background task (_BG_SAVE_TASKS), so the .bin write does not add latency
  and survives a client disconnect (the task owns and releases the slot).
- For streaming:
    * reading from llama.cpp happens in a separate background task (the
      reader);
    * the reader pushes chunks into an asyncio.Queue (put with bounded wait,
      so it cannot block forever when the consumer is gone);
    * on a big-completed stream the reader hands off the slot to a detached
      _background_save task (which releases it in its own finally); on any
      other path the reader releases the slot itself;
    * a sentinel None is pushed into the queue (with bounded wait) after the
      slot is released or handed off;
    * on a mid-stream backend error the reader pushes an SSE error event
      (data: {"error": {"message": "..."}}) before the sentinel, so
      the client can distinguish a truncated stream from a normal one; a
      backend failure is marked with a "503 service unavailable" prefix so a
      retrying client (opencode) retries it, while a kill or a stalled
      consumer is not marked retryable;
    * when the generator is closed (client disconnect) the reader is
      cancelled;
    * reader tasks are kept by strong references (_READER_TASKS) so they are
      not GC-collected before their finally runs.

A request is killable by its correlation id while it waits for a slot and
while it generates (_kill). The pipeline races every backend await against the
request's kill token; the streaming reader is cancelled by the kill exactly as
it is by a client disconnect. A killed request answers 499 (non-stream), or
ends its stream with an SSE error event, and never saves a partial answer.

The implementation lives in private submodules (_state, _model, _restore,
_diagnostics, _content, _lru, _save, _stream, _chat, _kill). Every name this
package owns - including the private ones - is re-exported here, and the
submodules resolve it through this module at call time, so monkeypatching
`chat_flow.<name>` is seen by the internal code exactly as it was when this
was a single flat module.
"""

import asyncio
import codecs
import json
import logging
import time
from collections.abc import AsyncGenerator

import httpx
from fastapi.responses import JSONResponse, Response, StreamingResponse

import hashing as hs
from backend.llama_client import RESTORE_MISSING, LlamaClient
from backend.slot_manager import GSlot, QueueFullError, SlotManager
from cache import bin_cache
from core import promstats
from core.config import (
    ACQUIRE_TIMEOUT,
    BACKEND_RETRY_ATTEMPTS,
    BACKEND_RETRY_BACKOFF_S,
    BIG_THRESHOLD_WORDS,
    BIN_CACHE_DIR,
    BIN_CACHE_MAX_MB,
    CONTEXT_ADMISSION_ENABLED,
    HOLD_TIMEOUT,
    LCP_TH,
    MAX_TOKENS,
    MIN_BIN_SIZE_VALID,
    MODEL_ID,
    REASONING_IN_KEY,
    RENDER_CTX_FIELDS,
    RETRY_AFTER_S,
    SAVE_WAIT_TIMEOUT,
    WORDS_PER_BLOCK,
)
from core.request_id import request_id_var
from obs import reqlog
from obs import ui as ui_obs

from ._chat import chat_flow
from ._content import (
    _append_stream_content,
    _assistant_content,
    _log_prefix_group,
    _reasoning_of,
    _render_ctx_of,
    _saved_conversation_values,
    _stream_usage_of,
)
from ._diagnostics import (
    _emit_decision,
    _error_response,
    _new_decision,
    _provider_error_status,
    _record_tokens,
    _set_save_outcome,
    _snapshot_slot,
)
from ._hold import (
    HOLD_PROBE_INTERVAL_S,
    backend_gate,
)
from ._kill import (
    STAGE_GENERATING,
    STAGE_QUEUED,
    KillToken,
    RequestKilled,
)
from ._kill import (
    active as active_kills,
)
from ._kill import (
    bind as bind_kill,
)
from ._kill import (
    kill as kill_request,
)
from ._kill import (
    race as race_kill,
)
from ._kill import (
    reset as reset_kills,
)
from ._kill import (
    unregister as unregister_kill,
)
from ._lru import _purge_backend_files, _schedule_lru_check
from ._model import _health_order, _resolve_effective_model
from ._pause import PausedRetry, race_pause
from ._restore import (
    _find_restore_candidate,
    _finish_inflight_save,
    _register_inflight_save,
    _register_pending_restore,
    _resolve_restore_key,
    _select_restore_candidate,
    _settle_restore,
    _unregister_pending_restore,
    _wait_for_inflight_save,
)
from ._retry import (
    BackendRetry,
    backoff_seconds,
    is_retryable_transport_error,
)
from ._save import _background_save, _save_and_write_meta
from ._state import (
    _BG_SAVE_TASKS,
    _INFLIGHT_SAVES,
    _LRU_TASKS,
    _PENDING_RESTORES,
    _READER_TASKS,
    _RESTORE_ALIAS,
    STREAM_PUT_TIMEOUT,
    STREAM_QUEUE_SIZE,
    _lru_check_in_flight,
    log,
)
from ._stream import start_stream_task

__all__ = [
    "ACQUIRE_TIMEOUT",
    "BACKEND_RETRY_ATTEMPTS",
    "BACKEND_RETRY_BACKOFF_S",
    "BIG_THRESHOLD_WORDS",
    "BIN_CACHE_DIR",
    "BIN_CACHE_MAX_MB",
    "CONTEXT_ADMISSION_ENABLED",
    "HOLD_PROBE_INTERVAL_S",
    "HOLD_TIMEOUT",
    "LCP_TH",
    "MAX_TOKENS",
    "MIN_BIN_SIZE_VALID",
    "MODEL_ID",
    "REASONING_IN_KEY",
    "RENDER_CTX_FIELDS",
    "RESTORE_MISSING",
    "RETRY_AFTER_S",
    "SAVE_WAIT_TIMEOUT",
    "STAGE_GENERATING",
    "STAGE_QUEUED",
    "STREAM_PUT_TIMEOUT",
    "STREAM_QUEUE_SIZE",
    "WORDS_PER_BLOCK",
    "_BG_SAVE_TASKS",
    "_INFLIGHT_SAVES",
    "_LRU_TASKS",
    "_PENDING_RESTORES",
    "_READER_TASKS",
    "_RESTORE_ALIAS",
    "AsyncGenerator",
    "BackendRetry",
    "GSlot",
    "JSONResponse",
    "KillToken",
    "LlamaClient",
    "PausedRetry",
    "QueueFullError",
    "RequestKilled",
    "Response",
    "SlotManager",
    "StreamingResponse",
    "_append_stream_content",
    "_assistant_content",
    "_background_save",
    "_emit_decision",
    "_error_response",
    "_find_restore_candidate",
    "_finish_inflight_save",
    "_health_order",
    "_log_prefix_group",
    "_lru_check_in_flight",
    "_new_decision",
    "_provider_error_status",
    "_purge_backend_files",
    "_reasoning_of",
    "_record_tokens",
    "_register_inflight_save",
    "_register_pending_restore",
    "_render_ctx_of",
    "_resolve_effective_model",
    "_resolve_restore_key",
    "_save_and_write_meta",
    "_saved_conversation_values",
    "_schedule_lru_check",
    "_select_restore_candidate",
    "_set_save_outcome",
    "_settle_restore",
    "_snapshot_slot",
    "_stream_usage_of",
    "_unregister_pending_restore",
    "_wait_for_inflight_save",
    "active_kills",
    "asyncio",
    "backend_gate",
    "backoff_seconds",
    "bin_cache",
    "bind_kill",
    "chat_flow",
    "codecs",
    "hs",
    "httpx",
    "is_retryable_transport_error",
    "json",
    "kill_request",
    "log",
    "logging",
    "promstats",
    "race_kill",
    "race_pause",
    "reqlog",
    "request_id_var",
    "reset_kills",
    "start_stream_task",
    "time",
    "ui_obs",
    "unregister_kill",
]
