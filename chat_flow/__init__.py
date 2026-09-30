# chat_flow/__init__.py

"""Chat request orchestration for llama-kv-proxy.

Holds the /v1/chat/completions request pipeline: effective-model resolution,
cache-key computation, restore-candidate selection, slot acquisition, dispatch
to the backend, and the save/meta/LRU follow-up. The FastAPI endpoint in
app.py only parses the JSON body and delegates here.

Additionally:

- acquire_for_request is wrapped in a timeout so it cannot hang forever if a
  slot is never released.
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
      (data: {"error": "stream interrupted: ..."}) before the sentinel, so
      the client can distinguish a truncated stream from a normal one;
    * when the generator is closed (client disconnect) the reader is
      cancelled;
    * reader tasks are kept by strong references (_READER_TASKS) so they are
      not GC-collected before their finally runs.

The implementation lives in private submodules (_state, _model, _restore,
_diagnostics, _content, _lru, _save, _stream, _chat). Every name this package
owns - including the private ones - is re-exported here, and the submodules
resolve it through this module at call time, so monkeypatching
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

import bin_cache
import hashing as hs
import promstats
import reqlog
import ui as ui_obs
from config import (
    ACQUIRE_TIMEOUT,
    BIG_THRESHOLD_WORDS,
    BIN_CACHE_DIR,
    BIN_CACHE_MAX_MB,
    ERASE_BEFORE_CHAT,
    LCP_TH,
    MIN_BIN_SIZE_VALID,
    MODEL_ID,
    REASONING_IN_KEY,
    RENDER_CTX_FIELDS,
    SAVE_WAIT_TIMEOUT,
    WORDS_PER_BLOCK,
)
from llama_client import RESTORE_MISSING, LlamaClient
from request_id import request_id_var
from slot_manager import GSlot, SlotManager

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
    _new_decision,
    _provider_error_status,
    _record_tokens,
    _set_save_outcome,
    _snapshot_slot,
)
from ._lru import _purge_backend_files, _schedule_lru_check
from ._model import _resolve_effective_model
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
    "BIG_THRESHOLD_WORDS",
    "BIN_CACHE_DIR",
    "BIN_CACHE_MAX_MB",
    "ERASE_BEFORE_CHAT",
    "LCP_TH",
    "MIN_BIN_SIZE_VALID",
    "MODEL_ID",
    "REASONING_IN_KEY",
    "RENDER_CTX_FIELDS",
    "RESTORE_MISSING",
    "SAVE_WAIT_TIMEOUT",
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
    "GSlot",
    "JSONResponse",
    "LlamaClient",
    "Response",
    "SlotManager",
    "StreamingResponse",
    "_append_stream_content",
    "_assistant_content",
    "_background_save",
    "_emit_decision",
    "_find_restore_candidate",
    "_finish_inflight_save",
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
    "asyncio",
    "bin_cache",
    "chat_flow",
    "codecs",
    "hs",
    "httpx",
    "json",
    "log",
    "logging",
    "promstats",
    "reqlog",
    "request_id_var",
    "start_stream_task",
    "time",
    "ui_obs",
]
