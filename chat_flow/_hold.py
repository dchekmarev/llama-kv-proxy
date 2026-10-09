# chat_flow/_hold.py

"""Backend hold: wait for the picked backend to serve before dispatching.

A dead or loading backend used to bounce the client immediately with
502 + Retry-After; the client then retried on its own backoff. The hold
accepts the connection and waits (bounded by HOLD_TIMEOUT, probing every
HOLD_PROBE_INTERVAL_S) for the backend to come back, so a short outage is a
long TTFT instead of a failed turn. It only runs before any response bytes
are sent; a mid-stream death still ends with the SSE error event.
"""

import asyncio
import time

import chat_flow as chat_flow_pkg
from backend.llama_client import LlamaClient

from ._kill import KillToken

# Probe cadence while holding: one /health per second is cheap and keeps the
# recovery latency under two seconds.
HOLD_PROBE_INTERVAL_S = 1.0


async def backend_gate(client: LlamaClient, kill_token: KillToken | None) -> bool:
    """Wait until the backend serves or HOLD_TIMEOUT elapses.

    Returns True when the backend is serving, False when the hold budget is
    exhausted (the caller answers 502 + Retry-After). A kill during the hold
    raises RequestKilled, which the caller's existing kill path turns into a
    499. `kill_token` may be None (the pass-through is not killable).
    """
    deadline = time.monotonic() + chat_flow_pkg.HOLD_TIMEOUT
    while True:
        healthy = await chat_flow_pkg.race_kill(client.probe(), kill_token)
        if healthy or time.monotonic() >= deadline:
            return healthy
        await chat_flow_pkg.race_kill(
            asyncio.sleep(chat_flow_pkg.HOLD_PROBE_INTERVAL_S), kill_token
        )
