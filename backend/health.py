# backend/health.py

"""Shared per-backend health: the single source of truth for liveness.

One :class:`BackendHealth` lives on the :class:`~backend.slot_manager.SlotManager`
(``sm.health``) and is written by every producer -- the periodic slot poll and
the hold's prober -- and read by every consumer: model/alias/slot selection
(``chat_flow._model``), the HOLD (``chat_flow._hold``) and the health endpoint.

It sits in ``backend/`` because that is the lowest layer both ``chat_flow``
(hold, selection) and ``app`` (poll, routes) may import; putting it in
``chat_flow`` would invert the layering for the poll.

Probing is single-flight: while a backend is down, the first waiter starts one
prober task and every other waiter parks on the same recovery event, so N
concurrent holds issue one probe per interval instead of N. A waiter that gives
up (budget/kill) detaches in a ``finally`` and the prober is cancelled once the
last waiter leaves, leaving no task or event waiter behind.

Cadence: while down the prober re-probes every ``interval`` (the hold's
``HOLD_PROBE_INTERVAL_S``) so recovery is seen within ~1-2s. With
``HEALTH_PROBE_BACKOFF_S`` set the delay doubles on every failed probe up to
``HEALTH_PROBE_MAX_S``, which stops hammering ``/health`` during a long outage
at the cost of a slower recovery.
"""

import asyncio
from collections.abc import Awaitable, Callable

from core import config, promstats

Probe = Callable[[], Awaitable[bool]]


class BackendHealth:
    """Per-backend up/down verdict, recovery signal and single-flight prober."""

    def __init__(
        self, backoff_s: float | None = None, max_s: float | None = None
    ) -> None:
        self._backoff_s = (
            config.HEALTH_PROBE_BACKOFF_S if backoff_s is None else max(0.0, backoff_s)
        )
        cap = config.HEALTH_PROBE_MAX_S if max_s is None else max(0.0, max_s)
        self._max_s = max(self._backoff_s, cap)
        self._up: dict[int, bool] = {}
        self._events: dict[int, asyncio.Event] = {}
        self._waiters: dict[int, int] = {}
        self._probers: dict[int, asyncio.Task[None]] = {}

    # --- state ----------------------------------------------------------------

    def up(self, backend_index: int) -> bool:
        """The last verdict; a never-recorded backend counts as up."""
        return self._up.get(backend_index, True)

    def set_up(self, backend_index: int, up: bool) -> None:
        """Record a liveness verdict, mirror it to the gauge and wake waiters.

        The prometheus gauge is kept consistent here so every producer (poll
        and prober) feeds the same signal; waiters parked on recovery wake as
        soon as any producer sees the backend up.
        """
        self._up[backend_index] = up
        promstats.backend_up.labels(backend=str(backend_index)).set(1.0 if up else 0.0)
        event = self._events.get(backend_index)
        if event is not None:
            if up:
                event.set()
            else:
                event.clear()

    def waiters(self, backend_index: int) -> int:
        """How many requests are currently parked on this backend."""
        return self._waiters.get(backend_index, 0)

    def prober_running(self, backend_index: int) -> bool:
        """Whether a prober task is currently probing this backend."""
        task = self._probers.get(backend_index)
        return task is not None and not task.done()

    # --- waiting --------------------------------------------------------------

    async def wait_up(
        self,
        backend_index: int,
        timeout: float,
        probe: Probe,
        interval: float = 1.0,
    ) -> bool:
        """Wait until the backend is up or ``timeout`` elapses.

        Returns the verdict at the end (True when up, False on timeout). The
        first waiter starts the shared prober; a nested waiter only parks.
        Cancelling this coroutine detaches cleanly.
        """
        if self.up(backend_index):
            return True
        event = self._events.setdefault(backend_index, asyncio.Event())
        self._waiters[backend_index] = self._waiters.get(backend_index, 0) + 1
        self._ensure_prober(backend_index, probe, interval)
        try:
            try:
                await asyncio.wait_for(event.wait(), timeout)
            except TimeoutError:
                pass
            return self.up(backend_index)
        finally:
            self._waiters[backend_index] -= 1
            if self._waiters[backend_index] <= 0 and not self.up(backend_index):
                task = self._stop_prober(backend_index)
                current = asyncio.current_task()
                if task is not None and current is not None and current.cancelling() == 0:
                    await asyncio.gather(task, return_exceptions=True)

    # --- prober ---------------------------------------------------------------

    def _ensure_prober(self, backend_index: int, probe: Probe, interval: float) -> None:
        if self.prober_running(backend_index):
            return
        self._probers[backend_index] = asyncio.create_task(
            self._probe_loop(backend_index, probe, interval)
        )

    def _stop_prober(self, backend_index: int) -> asyncio.Task[None] | None:
        task = self._probers.pop(backend_index, None)
        if task is not None and not task.done():
            task.cancel()
        return task

    async def _probe_loop(self, backend_index: int, probe: Probe, interval: float) -> None:
        delay = self._backoff_s or interval
        try:
            while not self.up(backend_index):
                if await probe():
                    self.set_up(backend_index, True)
                    return
                self.set_up(backend_index, False)
                await asyncio.sleep(delay)
                if self._backoff_s > 0.0:
                    delay = min(delay * 2.0, self._max_s)
        finally:
            if self._probers.get(backend_index) is asyncio.current_task():
                self._probers.pop(backend_index, None)
