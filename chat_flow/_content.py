# chat_flow/_content.py

"""Response-content extraction and the stored-conversation prefix values."""

import asyncio
import json

import chat_flow
import hashing as hs
from obs import reqlog

from . import _state

log = _state.log


def _render_ctx_of(data: dict) -> dict:
    """The render-affecting request params (tools, reasoning_effort, ...).

    These fields change the PROMPT the backend template renders (tools and
    reasoning instructions land inside the system text), so they must be part
    of the cache key: two requests with identical messages but different
    render params must never share a KV cache. Explicit nulls and absent
    fields are omitted, so their canonical form is empty.
    """
    return {
        k: data[k] for k in chat_flow.RENDER_CTX_FIELDS if k in data and data[k] is not None
    }


def _reasoning_of(msg: dict) -> tuple[object, str]:
    """(reasoning value, field name) from a message/delta dict.

    reasoning_content takes precedence over the reasoning field variant; the
    field name is "reasoning_content" when neither is present. Shared by the
    non-stream and stream extraction paths so both stay in parity.
    """
    value = msg.get("reasoning_content")
    if value is not None:
        return value, "reasoning_content"
    value = msg.get("reasoning")
    if value is not None:
        return value, "reasoning"
    return None, "reasoning_content"


def _assistant_content(out: dict) -> tuple[str, str, str]:
    """(Assistant text, reasoning text, reasoning field name) from a non-stream
    chat completion body.

    The reasoning text (reasoning_content, else reasoning) and the field name
    the backend used are only used for the saved-conversation values when
    REASONING_IN_KEY is on; they are always extracted so the caller decides.
    The field name is "reasoning_content" when no reasoning is present.
    """
    choices = out.get("choices") if isinstance(out, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return "", "", "reasoning_content"
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return "", "", "reasoning_content"
    content = message.get("content")
    if content is None:
        content = ""
    content = content if isinstance(content, str) else str(content)
    reasoning, field = chat_flow._reasoning_of(message)
    if reasoning is None:
        reasoning = ""
    return content, (reasoning if isinstance(reasoning, str) else str(reasoning)), field


def _append_stream_content(
    line: str,
    parts: list[str],
    reasoning_parts: list[str] | None = None,
    ui_reason_parts: list[str] | None = None,
) -> str | None:
    """Append assistant text from one SSE line (delta first, message fallback).

    When reasoning_parts is given (REASONING_IN_KEY on), the line's reasoning
    (delta first, message fallback; reasoning_content, else reasoning — same
    priority as the non-stream path) is appended there too, so the save site
    has the response's reasoning trace. ui_reason_parts is the dashboard's
    always-on reasoning sink (independent of REASONING_IN_KEY). Returns the
    reasoning field name the line used ("reasoning_content" or "reasoning"),
    or None when the line carried no reasoning.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices or not isinstance(choices[0], dict):
        return None
    first = choices[0]
    content = None
    reasoning = None
    field = "reasoning_content"
    delta = first.get("delta")
    if isinstance(delta, dict):
        content = delta.get("content")
        reasoning, field = chat_flow._reasoning_of(delta)
    if content is None or reasoning is None:
        message = first.get("message")
        if isinstance(message, dict):
            if content is None:
                content = message.get("content")
            if reasoning is None:
                reasoning, field = chat_flow._reasoning_of(message)
    if isinstance(content, str) and content:
        parts.append(content)
    if isinstance(reasoning, str) and reasoning:
        if reasoning_parts is not None:
            reasoning_parts.append(reasoning)
        if ui_reason_parts is not None:
            ui_reason_parts.append(reasoning)
        return field
    return None


def _stream_usage_of(line: str) -> tuple[dict, dict] | None:
    """(usage, timings) from an SSE line that carries the final usage chunk.

    llama.cpp emits one final SSE chunk with usage/timings (no choices) when
    prompt caching is on. Returns None for content chunks, [DONE] and
    malformed lines. The chunk's timings are kept when present; comma-joined
    keys are not polluted into the assembled content because the usage chunk
    has no delta.
    """
    line = line.strip()
    if not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if not payload or payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    usage = data.get("usage")
    if not isinstance(usage, dict) or not usage:
        return None
    timings = data.get("timings")
    return usage, timings if isinstance(timings, dict) else {}


async def _saved_conversation_values(
    messages: list[dict] | None,
    response_text: str,
    model_id: str,
    fallback_prefix: str,
    fallback_blocks: list[str],
    fallback_hashes: list[str],
    response_reasoning: str = "",
    response_reasoning_field: str = "reasoning_content",
    render_ctx: dict | None = None,
) -> tuple[str, list[str], list[str]]:
    """Prefix values for the stored conversation, or prompt-only fallback.

    response_reasoning_field is the field name the backend used for the
    reasoning trace; the saved assistant message carries it under that name
    so an echoed continuation request matches. render_ctx is the SAME dict
    the request side hashed with, so the saved meta's last prefix hash equals
    the continuation request's key.
    """
    if not response_text and not (chat_flow.REASONING_IN_KEY and response_reasoning):
        return fallback_prefix, fallback_blocks, fallback_hashes
    return await asyncio.to_thread(
        hs.saved_conversation_values,
        messages,
        response_text,
        model_id,
        chat_flow.WORDS_PER_BLOCK,
        chat_flow.REASONING_IN_KEY,
        response_reasoning or None,
        response_reasoning_field,
        render_ctx,
    )


async def _log_prefix_group(
    rid: str,
    ts: str,
    messages: list[dict],
    model_id: str,
    response_text: str,
    response_reasoning: str,
    response_reasoning_field: str,
    prefix: str,
    blocks: list[str],
    prefix_hashes: list[str],
    render_ctx: dict | None = None,
) -> None:
    """Write the group's prefix.json (fire-and-forget, never raises).

    The prefix is the conversation actually stored (prompt + assistant
    response): the text the next message's request will be matched against.
    Used where the saved values are not computed elsewhere (small requests,
    interrupted streams); big completed requests reuse the values the
    background save already computed.
    """
    try:
        saved_prefix, _saved_blocks, saved_hashes = await chat_flow._saved_conversation_values(
            messages,
            response_text,
            model_id,
            prefix,
            blocks,
            prefix_hashes,
            response_reasoning,
            response_reasoning_field,
            render_ctx,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("reqlog_prefix_fail rid=%s: %s", rid, e)
        return
    reqlog.log_file(
        "prefix",
        rid,
        ts,
        {"prefix": saved_prefix, "key": saved_hashes[-1] if saved_hashes else None},
    )
