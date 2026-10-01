# hashing/__init__.py

"""Cache keys, the meta store and the restore-candidate search.

A conversation is cached under a sha256 key over its rendered prefix, so a
later request with the same prompt prefix can restore the slot's KV cache
instead of re-prefilling it. This package owns:

* the canonical text of a conversation (:func:`raw_prefix`) and its keys
  (:func:`prefix_key_sha256`, :func:`prefix_hashes_from_messages`);
* the block hashes of that text -- the fallback match tier and the legacy
  ``words_per_block``-sized unit llama.cpp caches;
* the on-disk ``<key>.meta.json`` store: atomic writes, subsume-on-continuation
  and TTL/size eviction;
* the restore search: the in-RAM index (:mod:`meta_index`) first, the two-tier
  on-disk scan second.

Everything that touches the disk is synchronous and blocking; the ``*_async``
wrappers run it in a worker thread and keep every index mutation back on the
event loop, so a slow or full disk never stalls the proxy. Module globals
(``META_DIR``, ``META_INDEX_ENABLED``, ``WORDS_PER_BLOCK``, ``_index``) are
read at call time, never captured at import, so the tests can patch them.

The implementation lives in private submodules (_state, _text, _meta_io,
_stats, _search, _index_ops, _evict). Every name this package owns -
including the private ones - is re-exported here, and the submodules resolve
it through this module at call time, so monkeypatching `hashing.<name>` is
seen by the internal code exactly as it was when this was a single flat
module.
"""

import asyncio
import contextlib
import glob
import hashlib
import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Callable
from typing import NamedTuple

from core import promstats
from core.config import META_DIR, META_INDEX_ENABLED, WORDS_PER_BLOCK

from ._evict import (
    _clear_all_meta,
    _MetaFile,
    clear_all_meta_async,
    evict_meta,
    evict_meta_async,
)
from ._index_ops import _index_entries, rebuild_index_async, reconcile_index_async
from ._meta_index import MetaIndex
from ._meta_io import (
    META_SUFFIX,
    _atomic_write,
    _blocks_of,
    _candidate_size,
    _key_of,
    _meta_doc,
    _meta_files,
    _meta_path,
    _prefix_hashes_of,
    _remove_meta_file,
    delete_meta,
    delete_meta_async,
    delete_subsumed_metas,
    delete_subsumed_metas_async,
    scan_all_meta,
    scan_all_meta_async,
    touch_meta,
    write_meta,
    write_meta_async,
)
from ._search import (
    _best_match,
    _is_candidate,
    _lcp_len,
    find_best_restore_candidate,
    find_best_restore_candidate_async,
)
from ._state import _OUTCOME_HIT, _OUTCOME_MISS, _index, log
from ._stats import cache_stats, cache_stats_async, record_hit, record_miss
from ._text import (
    _CJK,
    _TOKEN_RE,
    _all_message_parts,
    _block_hashes,
    _canonical_json,
    _join_prefix,
    _message_part,
    _message_parts,
    _normalized_tool_call,
    _prefix_hashes,
    _reasoning_of,
    _render_ctx_payload,
    block_hashes_from_text,
    prefix_hashes_from_messages,
    prefix_key_sha256,
    raw_prefix,
    render_ctx_digest,
    render_ctx_leader,
    request_prefix_values,
    request_prefix_values_async,
    saved_conversation_values,
    words_from_text,
)

__all__ = [
    "META_DIR",
    "META_INDEX_ENABLED",
    "META_SUFFIX",
    "WORDS_PER_BLOCK",
    "_CJK",
    "_OUTCOME_HIT",
    "_OUTCOME_MISS",
    "_TOKEN_RE",
    "Callable",
    "MetaIndex",
    "NamedTuple",
    "_MetaFile",
    "_all_message_parts",
    "_atomic_write",
    "_best_match",
    "_block_hashes",
    "_blocks_of",
    "_candidate_size",
    "_canonical_json",
    "_clear_all_meta",
    "_index",
    "_index_entries",
    "_is_candidate",
    "_join_prefix",
    "_key_of",
    "_lcp_len",
    "_message_part",
    "_message_parts",
    "_meta_doc",
    "_meta_files",
    "_meta_path",
    "_normalized_tool_call",
    "_prefix_hashes",
    "_prefix_hashes_of",
    "_reasoning_of",
    "_remove_meta_file",
    "_render_ctx_payload",
    "asyncio",
    "block_hashes_from_text",
    "cache_stats",
    "cache_stats_async",
    "clear_all_meta_async",
    "contextlib",
    "delete_meta",
    "delete_meta_async",
    "delete_subsumed_metas",
    "delete_subsumed_metas_async",
    "evict_meta",
    "evict_meta_async",
    "find_best_restore_candidate",
    "find_best_restore_candidate_async",
    "glob",
    "hashlib",
    "json",
    "log",
    "logging",
    "os",
    "prefix_hashes_from_messages",
    "prefix_key_sha256",
    "promstats",
    "raw_prefix",
    "re",
    "rebuild_index_async",
    "reconcile_index_async",
    "record_hit",
    "record_miss",
    "render_ctx_digest",
    "render_ctx_leader",
    "request_prefix_values",
    "request_prefix_values_async",
    "saved_conversation_values",
    "scan_all_meta",
    "scan_all_meta_async",
    "tempfile",
    "time",
    "touch_meta",
    "words_from_text",
    "write_meta",
    "write_meta_async",
]
