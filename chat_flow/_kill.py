# chat_flow/_kill.py

"""Operator-initiated cancellation of a request in flight or in the queue.

A request has to be killable from two stages, and neither of them is
addressable by id on its own: the slot queue holds bare futures
(`SlotManager._waiters`) with no request id, and a generating request is a
task nobody holds a reference to. This module keeps the one map that makes
both reachable: correlation id -> KillToken.

The token carries a single asyncio.Event, the cooperative kill signal that
every await point of the pipeline races against, plus an optional abort
callable for the streaming reader, whose task the kill cancels directly
(the reader's finally already owns the slot release and the SSE error event,
exactly as it does for a client disconnect).

The registry is touched only from the event loop, like every other piece of
per-request state in this package, so it needs no lock.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

log = logging.getLogger(__name__)

# The two stages a request can be killed in. "queued" is waiting for a slot,
# "generating" is running on one (including the background save, which the
# caller no longer awaits).
STAGE_QUEUED = "queued"
STAGE_GENERATING = "generating"

T = TypeVar("T")


class RequestKilled(Exception):
    """Raised inside the request pipeline when its token was killed."""


class KillToken:
    """The kill handle of one request."""

    __slots__ = ("_abort", "event", "killed", "model", "rid", "stage")

    def __init__(self, rid: str, model: str) -> None:
        self.rid = rid
        self.model = model
        self.stage = STAGE_QUEUED
        self.event = asyncio.Event()
        self.killed = False
        self._abort: Callable[[], None] | None = None

    def bind_abort(self, abort: Callable[[], None]) -> None:
        """Register the immediate abort of a stage the token cannot race.

        Fired at once when the kill already landed: the stages run back to back
        and a kill in between must not be lost.
        """
        self._abort = abort
        if self.killed:
            abort()

    def kill(self) -> None:
        """Signal the kill. Idempotent: the abort runs once."""
        if self.killed:
            return
        self.killed = True
        self.event.set()
        if self._abort is not None:
            self._abort()


# correlation id -> token, for the requests that are still running.
_TOKENS: dict[str, KillToken] = {}


def bind(rid: str, model: str) -> KillToken:
    """Make `rid` killable and return its token.

    A re-bind of a live id (a client reusing its X-Request-ID) replaces the
    token: the old request keeps running but can no longer be killed, which is
    better than two requests sharing one handle.
    """
    token = KillToken(rid, model)
    _TOKENS[rid] = token
    return token


def unregister(rid: str) -> None:
    """Drop the handle of a request that is over. Idempotent."""
    _TOKENS.pop(rid, None)


def kill(rid: str) -> str | None:
    """Kill the request `rid`; return the stage it died in, None if unknown.

    A request that is already gone is unknown: the handle is dropped as soon
    as the request ends, so a stale kill can never hit a request that reused
    the id.
    """
    token = _TOKENS.get(rid)
    if token is None:
        return None
    token.kill()
    return token.stage


def active() -> list[dict[str, Any]]:
    """The killable requests, oldest first (for GET /proxy/requests)."""
    return [
        {"rid": t.rid, "model": t.model, "stage": t.stage}
        for t in sorted(_TOKENS.values(), key=lambda t: t.rid)
    ]


def reset() -> None:
    """Forget every handle (tests only)."""
    _TOKENS.clear()


async def race(awaitable: Awaitable[T], token: KillToken | None) -> T:
    """Await `awaitable`, unless `token` is killed first.

    The kill wins by cancelling the work and raising RequestKilled, so the
    caller unwinds through its own cleanup (the slot release lives there, not
    here). The work is awaited to completion after the cancel: its finally
    blocks are what give the slot back, and they only run once the task has
    been stepped again.
    """
    if token is None:
        return await awaitable
    work = asyncio.ensure_future(awaitable)
    killer = asyncio.ensure_future(token.event.wait())
    try:
        done, _pending = await asyncio.wait(
            {work, killer}, return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        raise RequestKilled(token.rid)
    finally:
        killer.cancel()
        if not work.done():
            work.cancel()
        await asyncio.gather(work, return_exceptions=True)
