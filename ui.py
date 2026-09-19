# ui.py

"""Live request observability for the /proxy/ui/ dashboard.

An in-memory registry of in-flight and recently finished chat requests, fed
by hooks in chat_flow.py, plus a small SSE broadcaster that pushes token
batches to connected dashboard viewers. The registry is the single source of
truth for the dashboard; all hook functions are no-ops when UI_ENABLED is off
and never raise: observability must not break the request path.
"""

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field

import config

log = logging.getLogger(__name__)

STATUS_QUEUED = "queued"
STATUS_GENERATING = "generating"
STATUS_DONE = "done"
STATUS_ERROR = "error"
STATUS_CANCELLED = "cancelled"


def _message_line(m) -> str | None:
    """One "[role] content" transcript line, or None for non-dict entries."""
    if not isinstance(m, dict):
        return None
    role = str(m.get("role") or "?")
    content = m.get("content")
    if content is None:
        content = ""
    if not isinstance(content, str):
        try:
            content = json.dumps(content, ensure_ascii=False)
        except (TypeError, ValueError):
            content = str(content)
    return f"[{role}] {content}"


def format_prompt(messages: list | None) -> str:
    """Readable "[role] content" transcript of the request messages."""
    lines = [_message_line(m) for m in messages or []]
    return "\n\n".join(l for l in lines if l is not None)


def format_prompt_tail(messages: list | None, limit: int) -> str:
    """The newest whole messages that fit within limit chars.

    The tail of the conversation is what identifies an in-flight request, so
    the preview shows the last user/assistant/tool turns instead of the (often
    identical) system-prompt head. A transcript that fits is returned whole;
    the newest message is always kept even when it alone exceeds the limit.
    """
    lines = [l for l in (_message_line(m) for m in messages or []) if l is not None]
    out: list[str] = []
    total = 0
    for line in reversed(lines):
        cost = len(line) + (2 if out else 0)
        if out and total + cost > limit:
            break
        out.append(line)
        total += cost
    return "\n\n".join(reversed(out))


@dataclass
class RequestInfo:
    rid: str
    model: str
    stream: bool
    n_words: int
    is_big: bool
    key: str
    status: str = STATUS_QUEUED
    slot: dict | None = None
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    ttft: float | None = None
    n_chars: int = 0
    usage: dict | None = None
    prompt_preview: str = ""
    prompt_full: str | None = None
    tail: str = ""
    tail_reason: str = ""
    error: str | None = None
    # Token batches not yet pushed to viewers (drained by the broadcaster).
    pending: str = ""
    pending_reason: str = ""

    def public(self, with_tail: bool = False) -> dict:
        d: dict = {
            "rid": self.rid,
            "model": self.model,
            "stream": self.stream,
            "n_words": self.n_words,
            "is_big": self.is_big,
            "key": self.key,
            "status": self.status,
            "slot": self.slot,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "ttft": self.ttft,
            "n_chars": self.n_chars,
            "usage": self.usage,
            "prompt_preview": self.prompt_preview,
            "error": self.error,
        }
        if with_tail:
            d["tail"] = self.tail
            d["tail_reason"] = self.tail_reason
        return d


class Registry:
    """In-flight + recent requests, and the fan-out to dashboard viewers."""

    def __init__(self, history_max: int | None = None, tail_max: int | None = None):
        self.active: dict[str, RequestInfo] = {}
        self.history: deque[RequestInfo] = deque(
            maxlen=history_max if history_max is not None else config.UI_HISTORY_MAX
        )
        self._tail_max = tail_max if tail_max is not None else config.UI_TAIL_MAX_CHARS
        self._subs: set[asyncio.Queue] = set()
        self._broadcaster: asyncio.Task | None = None

    # -- hooks (called from chat_flow; never raise) -------------------------

    def start(
        self,
        rid: str,
        *,
        model: str,
        stream: bool,
        n_words: int,
        is_big: bool,
        key: str,
        messages: list | None,
    ) -> None:
        if not rid:
            return
        full = format_prompt(messages)
        if len(full) > config.UI_PROMPT_FULL_MAX_CHARS:
            full = full[: config.UI_PROMPT_FULL_MAX_CHARS] + "\n…[truncated]"
        self.active[rid] = RequestInfo(
            rid=rid,
            model=model,
            stream=stream,
            n_words=n_words,
            is_big=is_big,
            key=key[:16],
            prompt_preview=format_prompt_tail(messages, config.UI_PREVIEW_MAX_CHARS),
            prompt_full=full,
        )
        self._emit({"type": "start", "rid": rid, "info": self.active[rid].public()})

    def slot(self, rid: str, be_id: int, model: str, slot_id: int) -> None:
        info = self.active.get(rid)
        if info is None:
            return
        info.slot = {"backend": be_id, "model": model, "id": slot_id}
        info.status = STATUS_GENERATING
        self._emit({"type": "slot", "rid": rid, "slot": info.slot})

    def ttft(self, rid: str, ttft: float) -> None:
        info = self.active.get(rid)
        if info is not None and info.ttft is None:
            info.ttft = ttft

    def usage(self, rid: str, usage: dict) -> None:
        info = self.active.get(rid)
        if info is not None and info.usage is None:
            info.usage = usage

    def tokens(self, rid: str, content: str, reasoning: str) -> None:
        info = self.active.get(rid)
        if info is None:
            return
        if content:
            info.n_chars += len(content)
            info.tail = (info.tail + content)[-self._tail_max :]
            info.pending += content
        if reasoning:
            info.tail_reason = (info.tail_reason + reasoning)[-self._tail_max :]
            info.pending_reason += reasoning

    def end(self, rid: str, *, status: str = STATUS_DONE, error: str | None = None) -> None:
        info = self.active.pop(rid, None)
        if info is None:
            return
        info.status = status
        info.ended_at = time.time()
        info.error = error
        # prompt_full is kept: finished requests stay inspectable in the
        # history until they are evicted from the deque.
        self.history.appendleft(info)
        self._emit({"type": "end", "rid": rid, "info": info.public()})

    # -- viewer side ---------------------------------------------------------

    def snapshot(self) -> dict:
        return {
            "active": [i.public(with_tail=True) for i in self.active.values()],
            "history": [i.public() for i in self.history],
        }

    def _find(self, rid: str) -> RequestInfo | None:
        info = self.active.get(rid)
        if info is None:
            for i in self.history:
                if i.rid == rid:
                    info = i
                    break
        return info

    def full_prompt(self, rid: str) -> str | None:
        """Full prompt of an in-flight or recently finished request."""
        info = self._find(rid)
        return info.prompt_full if info is not None else None

    def response_tail(self, rid: str) -> tuple[str, str] | None:
        """(content, reasoning) tail of an in-flight or recent request."""
        info = self._find(rid)
        if info is None:
            return None
        return info.tail, info.tail_reason

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        self._subs.add(q)
        self._ensure_broadcaster()
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subs.discard(q)
        if not self._subs and self._broadcaster is not None:
            self._broadcaster.cancel()
            self._broadcaster = None

    def _ensure_broadcaster(self) -> None:
        if self._broadcaster is None or self._broadcaster.done():
            self._broadcaster = asyncio.create_task(self._broadcast_loop())

    async def _broadcast_loop(self) -> None:
        """Push pending token batches to all viewers every 200 ms."""
        try:
            while True:
                await asyncio.sleep(0.2)
                events = self._drain_pending()
                if not events:
                    continue
                for q in list(self._subs):
                    try:
                        q.put_nowait(events)
                    except asyncio.QueueFull:
                        pass
        except asyncio.CancelledError:
            pass

    def _drain_pending(self) -> list[dict]:
        events: list[dict] = []
        for rid, info in list(self.active.items()):
            if info.pending or info.pending_reason:
                events.append(
                    {
                        "type": "tokens",
                        "rid": rid,
                        "content": info.pending,
                        "reasoning": info.pending_reason,
                    }
                )
                info.pending = ""
                info.pending_reason = ""
        return events

    def _emit(self, event: dict) -> None:
        if not self._subs:
            return
        for q in list(self._subs):
            try:
                q.put_nowait([event])
            except asyncio.QueueFull:
                pass


registry = Registry()


def _guard(op: str):
    """Wrap a registry hook: no-op when the UI is off, never raises."""

    def wrapper(fn):
        def inner(*args, **kwargs):
            if not config.UI_ENABLED:
                return
            try:
                fn(*args, **kwargs)
            except Exception:  # noqa: BLE001
                log.warning("ui_hook_failed op=%s", op, exc_info=True)

        return inner

    return wrapper


@_guard("start")
def req_start(rid, *, model, stream, n_words, is_big, key, messages) -> None:
    registry.start(rid, model=model, stream=stream, n_words=n_words,
                   is_big=is_big, key=key, messages=messages)


@_guard("slot")
def req_slot(rid, be_id, model, slot_id) -> None:
    registry.slot(rid, be_id, model, slot_id)


@_guard("ttft")
def req_ttft(rid, ttft) -> None:
    registry.ttft(rid, ttft)


@_guard("usage")
def req_usage(rid, usage) -> None:
    registry.usage(rid, usage)


@_guard("tokens")
def req_tokens(rid, content, reasoning) -> None:
    registry.tokens(rid, content, reasoning)


@_guard("end")
def req_end(rid, *, status=STATUS_DONE, error=None) -> None:
    registry.end(rid, status=status, error=error)
