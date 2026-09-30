# hashing.py

"""Cache keys, the meta store and the restore-candidate search.

A conversation is cached under a sha256 key over its rendered prefix, so a
later request with the same prompt prefix can restore the slot's KV cache
instead of re-prefilling it. This module owns:

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

import promstats
from config import META_DIR, META_INDEX_ENABLED, WORDS_PER_BLOCK
from meta_index import MetaIndex

log = logging.getLogger(__name__)

META_SUFFIX = ".meta.json"

# In-RAM restore index (see meta_index): only mutated on the event loop, by the
# *_async wrappers and by the search itself.
_index = MetaIndex()

# restore_total outcomes counted by record_hit / record_miss and reported by
# cache_stats() (the counters live in the metrics registry, so /metrics and
# /cache/stats always agree).
_OUTCOME_HIT = "hit"
_OUTCOME_MISS = "miss"


# --- conversation text, tokens and keys ---------------------------------------

# Word tokens: one CJK ideograph, or a run of ASCII letters and digits.
# Everything else -- punctuation such as the ":" of the "<role>:<content>"
# prefix format included -- separates tokens. CJK is split per code point on
# purpose: a space-less Chinese sentence has no whitespace, so whitespace
# tokenization would count a whole paragraph as one word and never cross the
# big-request threshold.
_CJK = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff"
_TOKEN_RE = re.compile(f"[{_CJK}]|[A-Za-z0-9]+")


def words_from_text(text: str) -> list[str]:
    """Approximate the prompt's token count (each CJK code point is a word)."""
    return _TOKEN_RE.findall(text or "")


def prefix_key_sha256(key_material: str) -> str:
    """sha256 of the joined "<model_id>\\n<prefix>" key material.

    Takes the already-joined string: the caller owns the layout.
    """
    return hashlib.sha256(key_material.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    """Deterministic JSON for a key fragment: sorted keys, no whitespace."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
    )


def _render_ctx_payload(render_ctx: dict | str | None) -> str:
    """Canonical, byte-stable form of the render-affecting request params."""
    if not render_ctx:
        return ""
    if isinstance(render_ctx, str):
        return render_ctx
    return _canonical_json(render_ctx)


def render_ctx_leader(render_ctx: dict | str | None) -> str:
    """The synthetic "ctx:<hash>" part that scopes a prefix to its render ctx.

    Tools and reasoning params change the prompt the backend template renders
    (they land inside the system text), so two requests with identical messages
    but different render params must not share a KV cache. The leader is
    prepended to the prefix -- it belongs to no message, so differently
    rendered conversations diverge at block 0 and never match as candidates.

    An empty context yields no leader at all, which keeps the keys of requests
    without render-affecting params byte-identical to the plain ones.
    """
    payload = _render_ctx_payload(render_ctx)
    if not payload:
        return ""
    return "ctx:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def render_ctx_digest(render_ctx: dict | str | None) -> str | None:
    """Digest of the render context for the decision log; None when empty."""
    leader = render_ctx_leader(render_ctx)
    if not leader:
        return None
    return "ctx:" + hashlib.sha256(leader.encode("utf-8")).hexdigest()


def _reasoning_of(msg: dict) -> tuple[object, str]:
    """(value, field name) of a message's reasoning trace.

    Mirrors the save-side extraction (chat_flow._reasoning_of): an explicitly
    present reasoning_content wins over reasoning, including when it is empty.
    """
    value = msg.get("reasoning_content")
    if value is not None:
        return value, "reasoning_content"
    value = msg.get("reasoning")
    if value is not None:
        return value, "reasoning"
    return None, "reasoning_content"


def _normalized_tool_call(call: object) -> object:
    """A tool call with its arguments in one canonical form.

    ``arguments`` arrives either as a JSON string or as an object depending on
    the client, so both are parsed and re-serialized with sorted keys: the same
    call then yields the same key material either way.
    """
    if not isinstance(call, dict):
        return call
    normalized = dict(call)
    function = normalized.get("function")
    if isinstance(function, dict) and "arguments" in function:
        arguments = function["arguments"]
        if isinstance(arguments, str):
            with contextlib.suppress(ValueError):
                arguments = json.loads(arguments)
        normalized["function"] = dict(function, arguments=arguments)
    return normalized


def _message_part(msg: dict, include_reasoning: bool = False) -> str:
    """One message's normalized key material, or "" when it carries none.

    A plain message renders as "<role>:<content stripped>". Everything else
    that changes the rendered prompt (name, tool_calls, tool_call_id and, behind
    include_reasoning, the reasoning trace) is appended as a canonical
    "<field>=<json>" fragment, so two messages that render differently never
    share a key. Absent content contributes nothing: no spurious "None" for the
    assistant tool-call messages, and no part at all for an empty message (which
    would otherwise add a hash and shift the block boundaries).
    """
    fields: list[str] = []
    content = msg.get("content")
    if isinstance(content, str):
        if content.strip():
            fields.append(content.strip())
    elif content is not None:
        fields.append(_canonical_json(content))
    name = msg.get("name")
    if name:
        fields.append(f"name={_canonical_json(name)}")
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        calls = [_normalized_tool_call(c) for c in tool_calls]
        fields.append("tool_calls=" + _canonical_json(calls))
    tool_call_id = msg.get("tool_call_id")
    if tool_call_id:
        fields.append(f"tool_call_id={_canonical_json(tool_call_id)}")
    if include_reasoning:
        value, field = _reasoning_of(msg)
        if value:
            fields.append(f"{field}={_canonical_json(value)}")
    if not fields:
        return ""
    return f"{msg.get('role', '')}:" + ":".join(fields)


def _all_message_parts(
    messages: list[dict] | None, include_reasoning: bool = False
) -> list[str]:
    """One normalized part per message, empty where the message carries nothing.

    Keeping the empties (instead of dropping them) is what lets the prefix
    hashes be walked incrementally: the k-th entry is the k-th message, whether
    or not it changed the prefix.
    """
    return [_message_part(m, include_reasoning) for m in messages or []]


def _message_parts(messages: list[dict] | None, include_reasoning: bool = False) -> list[str]:
    """Every non-empty message part, in order (each message normalized once)."""
    return [part for part in _all_message_parts(messages, include_reasoning) if part]


def _join_prefix(leader: str, parts: list[str]) -> str:
    """The render-context leader followed by the message parts, blank-line joined."""
    if not leader:
        return "\n\n".join(parts)
    return "\n\n".join([leader, *parts]) if parts else leader


def raw_prefix(
    messages: list[dict] | None,
    include_reasoning: bool = False,
    render_ctx: dict | str | None = None,
) -> str:
    """The conversation's canonical text (its hash input)."""
    return _join_prefix(
        render_ctx_leader(render_ctx), _message_parts(messages, include_reasoning)
    )


def _prefix_hashes(parts: list[str], leader: str, model_id: str) -> list[str]:
    """H(model_id + "\\n" + raw_prefix(messages[:k])) for k = 1..n, deduped.

    Incremental: `parts` holds one (possibly empty) entry per message and the
    prefix is grown one part at a time, so no message is normalized twice. The
    first hash is always emitted -- a leading empty message still hashes the
    empty prefix -- while a message that contributes nothing repeats the
    previous hash and is dropped. The last hash is the full-conversation key.
    """
    hashes: list[str] = []
    prefix = leader
    for part in parts:
        if part:
            prefix = f"{prefix}\n\n{part}" if prefix else part
        digest = prefix_key_sha256(model_id + "\n" + prefix)
        if not hashes or hashes[-1] != digest:
            hashes.append(digest)
    return hashes


def prefix_hashes_from_messages(
    messages: list[dict] | None,
    model_id: str,
    include_reasoning: bool = False,
    render_ctx: dict | str | None = None,
) -> list[str]:
    """The per-message prefix hashes of a conversation.

    H(messages[:k]) for k = 1..n, deduped, last element == the conversation key.
    A meta is findable through any of these, so a continuation matches even
    though the cached key is the one of the longer conversation.
    """
    return _prefix_hashes(
        _all_message_parts(messages, include_reasoning),
        render_ctx_leader(render_ctx),
        model_id,
    )


def _block_hashes(words: list[str], words_per_block: int) -> list[str]:
    """One sha256 per words_per_block-word block (the last one may be short)."""
    size = max(1, int(words_per_block))
    return [
        hashlib.sha256(" ".join(words[i : i + size]).encode("utf-8")).hexdigest()
        for i in range(0, len(words), size)
    ]


def block_hashes_from_text(text: str, words_per_block: int = WORDS_PER_BLOCK) -> list[str]:
    """Block hashes of a rendered prefix: sha256 of wpb words joined by a space."""
    return _block_hashes(words_from_text(text), words_per_block)


# --- request / save value bundles -----------------------------------------------


def request_prefix_values(
    messages: list[dict] | None,
    model_id: str,
    words_per_block: int,
    include_reasoning: bool = False,
    render_ctx: dict | str | None = None,
) -> tuple[str, str, list[str], list[str], int]:
    """(prefix, key, blocks, prefix_hashes, n_words) for a request, in one pass.

    Each message is normalized exactly once and the prefix is tokenized exactly
    once (the block hashes and the word count share the tokenization), so the
    values are byte-identical to computing raw_prefix / prefix_key_sha256 /
    block_hashes_from_text / words_from_text / prefix_hashes_from_messages
    separately -- existing cache entries keep matching.
    """
    leader = render_ctx_leader(render_ctx)
    all_parts = _all_message_parts(messages, include_reasoning)
    prefix = _join_prefix(leader, [part for part in all_parts if part])
    words = words_from_text(prefix)
    return (
        prefix,
        prefix_key_sha256(model_id + "\n" + prefix),
        _block_hashes(words, words_per_block),
        _prefix_hashes(all_parts, leader, model_id),
        len(words),
    )


def saved_conversation_values(
    messages: list[dict] | None,
    content: str,
    model_id: str,
    words_per_block: int,
    include_reasoning: bool = False,
    reasoning: str | None = "",
    reasoning_field: str = "reasoning_content",
    render_ctx: dict | str | None = None,
) -> tuple[str, list[str], list[str]]:
    """(prefix, blocks, prefix_hashes) of the conversation to store after a reply.

    The assistant message is appended to the prompt, so the stored conversation
    covers prompt + response and a continuation request (which echoes the
    response back) matches the meta's last prefix hash. The reasoning trace is
    stored under the field name the backend used, for the same reason. An empty
    response keeps the prompt-only values: there is nothing new to store.
    """
    convo = list(messages or [])
    reply = {"role": "assistant", "content": content}
    if content or (include_reasoning and reasoning):
        if include_reasoning and reasoning:
            reply[reasoning_field or "reasoning_content"] = reasoning
        convo.append(reply)
    prefix = raw_prefix(convo, include_reasoning, render_ctx)
    return (
        prefix,
        block_hashes_from_text(prefix, words_per_block),
        prefix_hashes_from_messages(convo, model_id, include_reasoning, render_ctx),
    )


# --- meta store ------------------------------------------------------------------

def _meta_path(key: str) -> str:
    return os.path.join(META_DIR, key + META_SUFFIX)


def _key_of(path: str) -> str:
    """The cache key a meta file belongs to."""
    return os.path.basename(path)[: -len(META_SUFFIX)]


def _meta_files() -> list[str]:
    """Every meta file path in META_DIR (META_DIR is read per call)."""
    return glob.glob(os.path.join(META_DIR, "*" + META_SUFFIX))


def _atomic_write(path: str, doc: dict) -> None:
    """Write doc as JSON through a temp file in the same dir + os.replace.

    A crash (or a full disk) mid-write leaves the previous document intact and
    removes the temp file, so a meta is never half-written and a reader never
    sees a truncated file.
    """
    fd, tmp = tempfile.mkstemp(
        dir=os.path.dirname(path) or ".", prefix=os.path.basename(path) + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise


def _meta_doc(
    key: str,
    prefix: str,
    blocks: list[str],
    words_per_block: int,
    model_id: str,
    prefix_hashes: list[str] | None,
    bin_size: int | None,
    saved_prefix_hashes: list[str] | None,
) -> dict:
    """The meta document for one cached conversation.

    The prompt text itself is not stored, only its size (chars and UTF-8
    bytes), which is all the restore ranking needs; the blocks and the
    per-message prefix hashes are what the search matches on.
    """
    text = prefix or ""
    return {
        "key": key,
        "model_id": model_id,
        "words_per_block": words_per_block,
        "blocks": list(blocks or []),
        "prefix_len": len(text),
        "prefix_bytes": len(text.encode("utf-8")),
        "prefix_hashes": list(prefix_hashes or []),
        "saved_prefix_hashes": list(saved_prefix_hashes or []),
        "bin_size": bin_size,
        "timestamp": time.time(),
    }


def write_meta(
    key: str,
    prefix: str,
    blocks: list[str],
    words_per_block: int,
    model_id: str,
    prefix_hashes: list[str] | None = None,
    bin_size: int | None = None,
    saved_prefix_hashes: list[str] | None = None,
) -> None:
    """(Re)write <key>.meta.json for a just-saved slot cache."""
    _atomic_write(
        _meta_path(key),
        _meta_doc(
            key, prefix, blocks, words_per_block, model_id,
            prefix_hashes, bin_size, saved_prefix_hashes,
        ),
    )


def touch_meta(key: str) -> None:
    """Refresh a meta's timestamp so an actively restored entry survives TTL.

    Best-effort: any failure is swallowed (a stale timestamp only risks an
    early eviction, never a wrong cache hit).
    """
    try:
        path = _meta_path(key)
        with open(path, encoding="utf-8") as f:
            meta = json.load(f)
        if not isinstance(meta, dict):
            return
        meta["timestamp"] = time.time()
        _atomic_write(path, meta)
    except (OSError, ValueError, TypeError):
        log.debug("touch_meta_failed key=%s", key[:16])


def _remove_meta_file(path: str) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def delete_meta(key: str) -> bool:
    """Delete <key>.meta.json. True only when a file was actually removed."""
    return _remove_meta_file(_meta_path(key))


def delete_subsumed_metas(
    new_key: str, new_hashes: list[str], model_id: str
) -> list[str]:
    """Delete the metas the new conversation supersedes; returns their keys.

    A meta is subsumed when its key -- always its full-conversation hash -- is
    one of the new conversation's prefix hashes: that conversation is a strict
    prefix of the new one, and its .bin is superseded by the new save. The new
    key itself and other models' metas are never touched.
    """
    subsumed = set(new_hashes or [])
    if not subsumed:
        return []
    deleted = []
    for meta in scan_all_meta():
        key = meta.get("key")
        if not key or key == new_key or key not in subsumed:
            continue
        if meta.get("model_id") != model_id:
            continue
        if _remove_meta_file(_meta_path(key)):
            deleted.append(key)
    if deleted:
        log.info("subsumed_metas_deleted n=%d", len(deleted))
    return deleted


def scan_all_meta() -> list[dict]:
    """Every readable meta document on disk.

    A file that disappears (or is being written) between the glob and the read
    is skipped, not fatal: the remaining files are still worth matching.
    """
    metas = []
    for path in _meta_files():
        try:
            with open(path, encoding="utf-8") as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(meta, dict):
            metas.append(meta)
    return metas


async def scan_all_meta_async() -> list[dict]:
    """scan_all_meta off the event loop."""
    return await asyncio.to_thread(scan_all_meta)


def cache_stats() -> dict:
    """Meta store state plus the lifetime restore hit/miss counters."""
    files = _meta_files()
    total = 0
    for path in files:
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return {
        "files": len(files),
        "total_bytes": total,
        "hits": int(promstats.counter_sum(promstats.restore_total, outcome=_OUTCOME_HIT)),
        "misses": int(promstats.counter_sum(promstats.restore_total, outcome=_OUTCOME_MISS)),
    }


async def cache_stats_async() -> dict:
    """cache_stats off the event loop."""
    return await asyncio.to_thread(cache_stats)


def record_hit(model: str) -> None:
    """Count a request that restored from the cache."""
    promstats.restore_total.labels(model=model, outcome=_OUTCOME_HIT).inc()
    log.debug("restore_hit model=%s", model)


def record_miss(model: str) -> None:
    """Count a big request with no usable restore candidate."""
    promstats.restore_total.labels(model=model, outcome=_OUTCOME_MISS).inc()
    log.debug("restore_miss model=%s", model)


# --- meta store, off the event loop -----------------------------------------------

async def write_meta_async(
    key: str,
    prefix: str,
    blocks: list[str],
    words_per_block: int,
    model_id: str,
    prefix_hashes: list[str] | None = None,
    bin_size: int | None = None,
    saved_prefix_hashes: list[str] | None = None,
) -> None:
    """write_meta in a worker thread; the index is updated back on the loop."""
    await asyncio.to_thread(
        write_meta,
        key,
        prefix,
        blocks,
        words_per_block,
        model_id,
        prefix_hashes,
        bin_size,
        saved_prefix_hashes,
    )
    if META_INDEX_ENABLED:
        doc = _meta_doc(
            key, prefix, blocks, words_per_block, model_id,
            prefix_hashes, bin_size, saved_prefix_hashes,
        )
        _index.add(doc["key"], doc["model_id"], _candidate_size(doc), _prefix_hashes_of(doc))


async def delete_meta_async(key: str) -> bool:
    """delete_meta in a worker thread; the index is updated back on the loop."""
    removed = await asyncio.to_thread(delete_meta, key)
    if META_INDEX_ENABLED:
        _index.remove(key)
    return removed


async def delete_subsumed_metas_async(
    new_key: str, new_hashes: list[str], model_id: str
) -> list[str]:
    """delete_subsumed_metas in a worker thread, index updates on the loop."""
    deleted = await asyncio.to_thread(delete_subsumed_metas, new_key, new_hashes, model_id)
    if META_INDEX_ENABLED:
        for key in deleted:
            _index.remove(key)
    return deleted


async def request_prefix_values_async(
    messages: list[dict] | None,
    model_id: str,
    words_per_block: int,
    include_reasoning: bool = False,
    render_ctx: dict | str | None = None,
) -> tuple[str, str, list[str], list[str], int]:
    """request_prefix_values off the event loop (the heavy hashing)."""
    return await asyncio.to_thread(
        request_prefix_values, messages, model_id, words_per_block, include_reasoning, render_ctx
    )


# --- restore search ------------------------------------------------------------------

def _lcp_len(request: list[str], candidate: list[str]) -> int:
    """How many leading elements the request and the candidate share."""
    shared = 0
    for want, have in zip(request, candidate):
        if want != have:
            break
        shared += 1
    return shared


def _candidate_size(meta: dict) -> int:
    """Cache size used to break ties between equally long candidates.

    The real .bin size when it is known, else the prompt's UTF-8 byte length,
    else its char length (metas written before prefix_bytes existed), else 0.
    """
    for field in ("bin_size", "prefix_bytes", "prefix_len"):
        value = meta.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        if value > 0:
            return int(value)
    return 0


def _is_candidate(meta: dict, model_id: str, words_per_block: int) -> bool:
    """Both tiers only consider metas of the same model and block size."""
    if meta.get("model_id") != model_id:
        return False
    cached_wpb = meta.get("words_per_block")
    return cached_wpb is None or cached_wpb == words_per_block


def _best_match(
    metas: list[dict], request: list[str], hashes_of: Callable[[dict], list[str]]
) -> tuple[str, int] | None:
    """Longest common prefix over the metas, smallest cache on ties."""
    best_key: str | None = None
    best_shared = 0
    best_size = 0
    for meta in metas:
        key = meta.get("key")
        if not key:
            continue
        shared = _lcp_len(request, hashes_of(meta))
        if not shared:
            continue
        size = _candidate_size(meta)
        if shared > best_shared or (shared == best_shared and size < best_size):
            best_key, best_shared, best_size = key, shared, size
    return (best_key, best_shared) if best_key else None


def _prefix_hashes_of(meta: dict) -> list[str]:
    """The hash list tier 1 (and the index) matches on.

    The response-extended list wins over the prompt-only one, so a continuation
    matches a meta saved for prompt + response.
    """
    return meta.get("saved_prefix_hashes") or meta.get("prefix_hashes") or []


def _blocks_of(meta: dict) -> list[str]:
    return meta.get("blocks") or []


def find_best_restore_candidate(
    req_hashes: list[str] | None,
    req_blocks: list[str] | None,
    words_per_block: int,
    th: float,
    model_id: str,
) -> tuple[str, float] | None:
    """Best restore candidate for a big request, or None. On-disk tiers only.

    Both tiers filter by model_id and words_per_block, and both score a
    candidate by the fraction of the REQUEST it covers (lcp / len(request)), so
    a long partial match beats a short full one. Tier 1 wins outright when it
    matches at all -- including when its best match falls below the threshold,
    in which case the request has no usable cache:

    1. tier 1: per-message prefix hashes, longest common prefix, smallest cache
       on ties. The response-extended hash list wins over the prompt-only one,
       so a continuation matches a meta saved for prompt + response.
    2. tier 2: block hashes, the fallback for metas written before per-message
       hashes existed.
    """
    hashes = list(req_hashes or [])
    blocks = list(req_blocks or [])
    if not hashes and not blocks:
        return None
    metas = [m for m in scan_all_meta() if _is_candidate(m, model_id, words_per_block)]
    if hashes:
        best = _best_match(metas, hashes, _prefix_hashes_of)
        if best is not None:
            ratio = best[1] / len(hashes)
            if ratio >= th:
                promstats.restore_tier_total.labels(tier="t1_disk").inc()
                return best[0], ratio
            return None
    if blocks:
        best = _best_match(metas, blocks, _blocks_of)
        if best is not None and best[1] / len(blocks) >= th:
            ratio = best[1] / len(blocks)
            promstats.restore_tier_total.labels(tier="t2_blocks").inc()
            return best[0], ratio
    return None


async def find_best_restore_candidate_async(
    req_hashes: list[str] | None,
    req_blocks: list[str] | None,
    words_per_block: int,
    th: float,
    model_id: str,
) -> tuple[str, float] | None:
    """find_best_restore_candidate off the event loop, index first.

    With META_INDEX_ENABLED the in-RAM index answers tier 1 in O(n) instead of
    scanning and parsing every meta file. It validates a hit against the
    on-disk meta (dropping ghosts) and ignores words_per_block, so it can be
    more permissive; a miss falls through to the full two-tier disk scan, which
    also covers metas the index does not know (blocks-only legacy entries).
    """
    hashes = list(req_hashes or [])
    if META_INDEX_ENABLED and hashes:
        hit = _index.search(hashes, th, model_id, META_DIR)
        if hit is not None:
            promstats.restore_tier_total.labels(tier="index").inc()
            return hit
    return await asyncio.to_thread(
        find_best_restore_candidate, req_hashes, req_blocks, words_per_block, th, model_id
    )


# --- index maintenance --------------------------------------------------------------

def _index_entries() -> list[dict]:
    """Index entries ({key, model_id, size, hashes}) for every meta on disk."""
    entries = []
    for meta in scan_all_meta():
        key = meta.get("key")
        if not key:
            continue
        entries.append(
            {
                "key": key,
                "model_id": meta.get("model_id") or "",
                "size": _candidate_size(meta),
                "hashes": _prefix_hashes_of(meta),
            }
        )
    return entries


async def rebuild_index_async() -> int:
    """Rebuild the index from disk (startup). Returns the number of metas."""
    entries = await asyncio.to_thread(_index_entries)
    _index.rebuild_from(entries)
    return len(entries)


async def reconcile_index_async() -> list[str]:
    """Drop index entries whose meta file is gone; returns the dropped keys.

    External mutators (bin_cache reconcile/clean) delete meta files without
    notifying the index, so the periodic reconcile turns those ghosts into
    clean misses.
    """
    if not META_INDEX_ENABLED:
        return []
    live = await asyncio.to_thread(lambda: {_key_of(p) for p in _meta_files()})
    return _index.reconcile_keys(live)


# --- eviction --------------------------------------------------------------------------

class _MetaFile(NamedTuple):
    """One meta file on disk, as the eviction pass needs it."""

    path: str
    key: str
    mtime: float
    size: int


def evict_meta(ttl_hours: float, max_files: int, max_mb: float) -> dict:
    """Delete meta files until every cap holds; 0 disables that cap.

    The TTL uses the file mtime (refreshed by write_meta and touch_meta, so
    "oldest" is least recently used); the file-count and byte caps delete oldest
    first. Returns the deleted keys and how many metas remain. Callers pass the
    three settings by keyword.
    """
    alive: list[_MetaFile] = []
    for path in _meta_files():
        try:
            st = os.stat(path)
        except OSError:
            continue
        alive.append(_MetaFile(path, _key_of(path), st.st_mtime, st.st_size))
    deleted: list[str] = []

    def _delete(entry: _MetaFile, reason: str) -> None:
        if _remove_meta_file(entry.path):
            deleted.append(entry.key)
            promstats.evictions_total.labels(reason=reason).inc()

    if ttl_hours > 0:
        cutoff = time.time() - ttl_hours * 3600.0
        for entry in [e for e in alive if e.mtime < cutoff]:
            _delete(entry, "ttl")
        alive = [e for e in alive if e.mtime >= cutoff]
    if max_files > 0:
        while len(alive) > max_files:
            oldest = min(alive, key=lambda e: e.mtime)
            _delete(oldest, "max_files")
            alive.remove(oldest)
    if max_mb > 0:
        limit = max_mb * 1024 * 1024
        total = sum(e.size for e in alive)
        while alive and total > limit:
            oldest = min(alive, key=lambda e: e.mtime)
            total -= oldest.size
            _delete(oldest, "max_mb")
            alive.remove(oldest)
    if deleted:
        log.info("meta_evicted n=%d remaining=%d", len(deleted), len(alive))
    return {"deleted": deleted, "remaining": len(alive)}


async def evict_meta_async(ttl_hours: float, max_files: int, max_mb: float) -> dict:
    """evict_meta in a worker thread (the scan is disk-bound)."""
    return await asyncio.to_thread(evict_meta, ttl_hours, max_files, max_mb)


def _clear_all_meta() -> list[str]:
    """Delete every meta file; the index is emptied by the async wrapper."""
    deleted = []
    for path in _meta_files():
        if _remove_meta_file(path):
            deleted.append(_key_of(path))
    if deleted:
        promstats.evictions_total.labels(reason="clear").inc(len(deleted))
    log.info("meta_cache_cleared n=%d", len(deleted))
    return deleted


async def clear_all_meta_async() -> list[str]:
    """Delete every meta file in a thread and empty the index on the loop."""
    deleted = await asyncio.to_thread(_clear_all_meta)
    if META_INDEX_ENABLED:
        _index.clear()
    return deleted