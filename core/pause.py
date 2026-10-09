# core/pause.py

"""Process-wide pause of the proxy (POST /proxy/pause).

Pausing stops the proxy from dispatching new work and holds every request that
arrives while paused until the pause ends. It exists so an operator can restart
or reload a backend without racing in-flight requests: after the pause request
the proxy drains the work that cannot be transparently re-dispatched (committed
SSE streams, background saves, pass-through) and only then starts the timer.

Lifecycle:

* ``request(seconds)`` marks the pause requested at once, so admission closes
  and new requests hold; it then waits for the drain to finish and activates.
  The timer starts at activation. A second call while active extends the
  deadline (``now + seconds``).
* ``wait_not_paused()`` is the admission hold: it returns at once when running,
  otherwise it blocks until a manual ``resume()`` or the timer elapses.
* ``activity_begin`` / ``activity_end`` bracket a unit of committed work (a
  held backend slot, an open pass-through) so the drain knows what is left.

The controller is touched only from the event loop, like the rest of the
per-process state, so it needs no lock: every state transition runs to the next
``await`` without interruption. Auto-expiry is lazy (checked on access and from
``wait_not_paused``), so an idle proxy needs no timer task.

This is the bottom layer of the import graph: it imports only ``config`` and
``promstats`` and depends on nothing in ``backend``/``chat_flow``/``app``.
"""

import asyncio
import time
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from . import config, promstats

# phases in the snapshot
PHASE_OFF = "off"
PHASE_DRAINING = "draining"
PHASE_ACTIVE = "active"


def clamp_seconds(seconds: float | None) -> float:
    """Resolve a requested pause duration against the configured default/cap.

    ``None`` (the request omitted it) falls back to ``PAUSE_DEFAULT_S``. A
    positive ``PAUSE_MAX_S`` caps the value. The result is always positive.
    """
    if seconds is None or seconds <= 0:
        seconds = config.PAUSE_DEFAULT_S
    if config.PAUSE_MAX_S > 0:
        seconds = min(seconds, config.PAUSE_MAX_S)
    return float(seconds)


class PauseController:
    """The pause state machine (one process-wide instance)."""

    def __init__(self) -> None:
        self._requested = False
        self._paused = False
        self._reason: str | None = None
        self._seconds = 0.0
        self._requested_at: float | None = None
        self._activated_at: float | None = None
        self._expires_at: float | None = None
        self._activated_wall: float | None = None
        self._expires_wall: float | None = None
        self._activity = 0
        self._resumed = asyncio.Event()
        self._resumed.set()
        self._activated = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()
        # Set while a pause is requested (the signal pre-commit work races
        # against so it can unwind and be re-dispatched after resume).
        self._pause_event = asyncio.Event()

    # --- admission ---------------------------------------------------------

    def pause_event(self) -> asyncio.Event:
        """The event that is set while a pause is requested."""
        return self._pause_event

    def is_paused(self) -> bool:
        """True while admission is closed (draining or active)."""
        self._expire_if_due()
        return self._requested

    async def wait_not_paused(self) -> None:
        """Block until the proxy is running again (admission hold).

        Re-checks after every wake: a resume followed by a fresh pause must
        not let a waiter slip into the new pause.
        """
        while True:
            self._expire_if_due()
            if not self._requested:
                return
            if self._paused and self._expires_at is not None:
                remaining = self._expires_at - time.monotonic()
                if remaining <= 0:
                    self._do_resume("expire")
                    continue
                try:
                    await asyncio.wait_for(self._resumed.wait(), timeout=remaining)
                except TimeoutError:
                    continue
            else:
                # Draining: wake on a resume, or when the pause activates so
                # the loop can switch to the timeout-bounded wait.
                await self._wait_for_change()

    async def _wait_for_change(self) -> None:
        """Wait for a resume or the activation of the requested pause."""
        active = asyncio.ensure_future(self._activated.wait())
        resumed = asyncio.ensure_future(self._resumed.wait())
        try:
            await asyncio.wait({active, resumed}, return_when=asyncio.FIRST_COMPLETED)
        finally:
            active.cancel()
            resumed.cancel()
            await asyncio.gather(active, resumed, return_exceptions=True)

    # --- lifecycle ---------------------------------------------------------

    async def request(self, seconds: float, reason: str | None = None) -> dict[str, Any]:
        """Request (or extend) a pause and return the activated snapshot.

        Marks the pause requested at once (new requests start holding), waits
        for the drain to complete, then activates. The timer starts at
        activation.
        """
        if not self._requested:
            self._requested = True
            self._requested_at = time.monotonic()
            self._resumed.clear()
            self._activated.clear()
            self._pause_event.set()
            promstats.paused.set(1)
        if reason is not None:
            self._reason = reason
        self._seconds = float(seconds)

        if self._activity > 0:
            await self._drained.wait()

        now = time.monotonic()
        if not self._paused:
            self._paused = True
            self._activated_at = now
            self._activated_wall = time.time()
            if self._requested_at is not None:
                promstats.pause_drain_seconds.observe(now - self._requested_at)
            promstats.pause_total.labels(action="activate").inc()
            self._activated.set()
        else:
            promstats.pause_total.labels(action="extend").inc()
        self._expires_at = now + self._seconds
        self._expires_wall = time.time() + self._seconds
        return self.snapshot()

    def resume(self) -> dict[str, Any]:
        """End a pause (manual resume). Idempotent."""
        if self._requested:
            self._do_resume("resume")
        return self.snapshot()

    def _expire_if_due(self) -> None:
        if (
            self._paused
            and self._expires_at is not None
            and time.monotonic() >= self._expires_at
        ):
            self._do_resume("expire")

    def _do_resume(self, action: str) -> None:
        self._requested = False
        self._paused = False
        self._reason = None
        self._seconds = 0.0
        self._requested_at = None
        self._activated_at = None
        self._expires_at = None
        self._activated_wall = None
        self._expires_wall = None
        promstats.paused.set(0)
        promstats.pause_total.labels(action=action).inc()
        self._pause_event.clear()
        self._activated.clear()
        self._resumed.set()

    # --- drain accounting --------------------------------------------------

    def activity_begin(self) -> None:
        """Mark a unit of committed work (a held slot, an open pass-through)."""
        self._activity += 1
        self._drained.clear()

    def activity_end(self) -> None:
        """Mark a unit of committed work as over."""
        if self._activity > 0:
            self._activity -= 1
        if self._activity <= 0:
            self._activity = 0
            self._drained.set()

    @contextmanager
    def activity(self) -> Iterator[None]:
        """``activity_begin`` / ``activity_end`` around a block."""
        self.activity_begin()
        try:
            yield
        finally:
            self.activity_end()

    # --- reporting ---------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """The current pause state (for the API, health and the dashboard)."""
        self._expire_if_due()
        if not self._requested:
            phase = PHASE_OFF
        elif not self._paused:
            phase = PHASE_DRAINING
        else:
            phase = PHASE_ACTIVE
        return {
            "paused": self._requested,
            "activated": self._paused,
            "phase": phase,
            "reason": self._reason,
            "seconds": self._seconds,
            "activity": self._activity,
            "activated_at": self._activated_wall,
            "expires_at": self._expires_wall,
        }

    def reset(self) -> None:
        """Forget every pause state (tests only)."""
        self._requested = False
        self._paused = False
        self._reason = None
        self._seconds = 0.0
        self._requested_at = None
        self._activated_at = None
        self._expires_at = None
        self._activated_wall = None
        self._expires_wall = None
        self._activity = 0
        self._resumed = asyncio.Event()
        self._resumed.set()
        self._activated = asyncio.Event()
        self._drained = asyncio.Event()
        self._drained.set()
        self._pause_event = asyncio.Event()


_controller = PauseController()


async def request(seconds: float, reason: str | None = None) -> dict[str, Any]:
    """Request (or extend) a pause. See :meth:`PauseController.request`."""
    return await _controller.request(seconds, reason)


def resume() -> dict[str, Any]:
    """End a pause. See :meth:`PauseController.resume`."""
    return _controller.resume()


def snapshot() -> dict[str, Any]:
    """The current pause state."""
    return _controller.snapshot()


def is_paused() -> bool:
    """True while admission is closed."""
    return _controller.is_paused()


def pause_event() -> asyncio.Event:
    """The event that is set while a pause is requested."""
    return _controller.pause_event()


async def wait_not_paused() -> None:
    """Block until the proxy is running again (admission hold)."""
    await _controller.wait_not_paused()


def activity_begin() -> None:
    """Mark a unit of committed work as started."""
    _controller.activity_begin()


def activity_end() -> None:
    """Mark a unit of committed work as over."""
    _controller.activity_end()


def activity() -> AbstractContextManager[None]:
    """``activity_begin`` / ``activity_end`` around a block."""
    return _controller.activity()


def reset() -> None:
    """Forget every pause state (tests only)."""
    _controller.reset()
