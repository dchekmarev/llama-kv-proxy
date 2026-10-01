# hashing/_text.py

"""Conversation text, tokens, keys and the request / save value bundles."""

import asyncio
import contextlib
import hashlib
import json
import re

import hashing as hs
from core.config import WORDS_PER_BLOCK

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
    return [hs._message_part(m, include_reasoning) for m in messages or []]


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
    return _block_hashes(hs.words_from_text(text), words_per_block)


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
    words = hs.words_from_text(prefix)
    return (
        prefix,
        prefix_key_sha256(model_id + "\n" + prefix),
        _block_hashes(words, words_per_block),
        _prefix_hashes(all_parts, leader, model_id),
        len(words),
    )


async def request_prefix_values_async(
    messages: list[dict] | None,
    model_id: str,
    words_per_block: int,
    include_reasoning: bool = False,
    render_ctx: dict | str | None = None,
) -> tuple[str, str, list[str], list[str], int]:
    """request_prefix_values off the event loop (the heavy hashing)."""
    return await asyncio.to_thread(
        hs.request_prefix_values, messages, model_id, words_per_block, include_reasoning, render_ctx
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
