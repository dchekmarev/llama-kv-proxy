# chat_flow/_hold.py

"""Backend hold: wait for the picked backend to serve before dispatching.

A dead or loading backend used to bounce the client immediately with
502 + Retry-After; the client then retried on its own backoff. The hold
accepts the connection and waits (bounded by HOLD_TIMEOUT) for the backend to
come back, so a short outage is a long TTFT instead of a failed turn. It only
runs before any response bytes are sent; a mid-stream death still ends with
the SSE error event.

The wait rides the shared per-backend health state
(:class:`backend.health.BackendHealth`, ``sm.health``): while a backend is
down a single prober probes it on behalf of every concurrent hold and all
waiters wake on the shared recovery event, so N held requests do not issue N
probe loops against ``/health``. Kill/pause handling stays out of the health
layer: the caller wraps this gate with ``race_kill``/``race_pause``.
"""

import asyncio
import time

import chat_flow as chat_flow_pkg
from backend.health import BackendHealth
from backend.llama_client import LlamaClient
from backend.slot_manager import SlotManager

from ._kill import KillToken

# Probe cadence while holding: one /health per second is cheap and keeps the
# recovery latency under two seconds.
HOLD_PROBE_INTERVAL_S = 1.0


async def backend_gate(
    client: LlamaClient,
    kill_token: KillToken | None,
    sm: SlotManager | None = None,
) -> bool:
    """Wait until the backend serves or HOLD_TIMEOUT elapses.

    Returns True when the backend is serving, False when the hold budget is
    exhausted (the caller answers 502 + Retry-After). A kill during the hold
    raises RequestKilled, which the caller's existing kill path turns into a
    499. `kill_token` may be None (the pass-through is not killable).

    With `sm` the wait uses the shared per-backend health state, so concurrent
    holds of the same down backend share one prober; without it (no shared
    state) the legacy per-request probe loop runs.
    """
    deadline = time.monotonic() + chat_flow_pkg.HOLD_TIMEOUT
    index = sm.client_index(client) if sm is not None else None
    if sm is not None and index is not None:
        return await _gated_wait(sm.health, index, client, kill_token, deadline)
    while True:
        healthy = await chat_flow_pkg.race_kill(client.probe(), kill_token)
        if healthy or time.monotonic() >= deadline:
            return healthy
        await chat_flow_pkg.race_kill(
            asyncio.sleep(chat_flow_pkg.HOLD_PROBE_INTERVAL_S), kill_token
        )


async def _gated_wait(
    health: BackendHealth,
    index: int,
    client: LlamaClient,
    kill_token: KillToken | None,
    deadline: float,
) -> bool:
    """The shared-state hold: probe once, then park on the recovery event."""
    if time.monotonic() >= deadline:
        # No hold budget (HOLD_TIMEOUT=0): the legacy single probe, then 502.
        healthy = await chat_flow_pkg.race_kill(client.probe(), kill_token)
        health.set_up(index, healthy)
        return healthy
    if not health.up(index):
        # Known down: join the shared prober instead of probing per request.
        return await chat_flow_pkg.race_kill(
            _park(health, index, client, deadline), kill_token
        )
    healthy = await chat_flow_pkg.race_kill(client.probe(), kill_token)
    health.set_up(index, healthy)
    if healthy:
        return True
    return await chat_flow_pkg.race_kill(
        _park(health, index, client, deadline), kill_token
    )


async def _park(
    health: BackendHealth, index: int, client: LlamaClient, deadline: float
) -> bool:
    return await health.wait_up(
        index,
        deadline - time.monotonic(),
        client.probe,
        interval=chat_flow_pkg.HOLD_PROBE_INTERVAL_S,
    )
