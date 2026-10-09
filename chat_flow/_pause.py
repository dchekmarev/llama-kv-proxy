# chat_flow/_pause.py

"""Pause-aware interruption of pre-commit pipeline work.

A pause covers a backend restart, so a request that has not committed a
response yet must not survive it: its model id, prefix hashes and slot pin
would be stale. Every await that runs before the commit point (slot
acquisition, the backend hold, the restore and the dispatch) is therefore
raced against the pause signal here. When the pause wins, the work is
cancelled, :class:`PausedRetry` unwinds the attempt through its own cleanup
(which releases the slot), and the pipeline's retry loop re-runs the attempt
from scratch once the pause is over.

The signal is the controller's pause event, so this mirrors
:func:`chat_flow._kill.race` for the pause case and composes with it:
``race_kill(race_pause(work), token)``.
"""

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

from core import pause as pause_mod

T = TypeVar("T")


class PausedRetry(Exception):
    """Raised inside an attempt when a pause landed before the response.

    The attempt owns no response bytes yet, so it can unwind and be re-run
    from scratch after the pause ends.
    """


async def race_pause(awaitable: Awaitable[T]) -> T:
    """Await `awaitable`, unless a pause is requested first.

    On a pause the work is cancelled and :class:`PausedRetry` is raised: the
    caller's own cleanup runs (the slot release lives there, not here) and
    the work is awaited to completion so its finally blocks finish.
    """
    work = asyncio.ensure_future(awaitable)
    if pause_mod.pause_event().is_set():
        # The pause landed before the work got a chance to run: do not let a
        # non-yielding awaitable (e.g. a no-restore slot acquire) slip past.
        work.cancel()
        await asyncio.gather(work, return_exceptions=True)
        raise PausedRetry()
    waiter = asyncio.ensure_future(pause_mod.pause_event().wait())
    try:
        done, _pending = await asyncio.wait(
            {work, waiter}, return_when=asyncio.FIRST_COMPLETED
        )
        if work in done:
            return work.result()
        raise PausedRetry()
    finally:
        waiter.cancel()
        if not work.done():
            work.cancel()
        await asyncio.gather(work, waiter, return_exceptions=True)
