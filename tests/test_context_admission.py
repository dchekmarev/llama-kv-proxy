# tests/test_context_admission.py

"""Context-budget admission control in the slot pool.

With llama.cpp ``--parallel N --kv-unified`` the N slots share ONE context:
admitting N requests by slot count alone lets them collectively exceed that
context and the backend answers 500 "Context size has been exceeded". A
per-(backend, model) budget gate keeps the sum of the in-flight requests'
estimated prompt+generation tokens within the shared context.

The budget rides on the slot lifecycle: a request reserves its estimate while
it holds a slot, release returns exactly that amount, and a request that fits
no empty pool is admitted best effort (parking it would only time it out).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from backend import slot_manager as sm_module
from backend.llama_client import LlamaClient
from backend.slot_manager import SlotManager
from core import promstats
from core.promstats import admission_oversized_total

_RESERVED = "llama_kv_proxy_context_reserved_tokens"


def _reserved(backend: str = "0", model: str = "m") -> float | None:
    """The published reserved-token gauge for one pool (None when unset)."""
    return promstats.REGISTRY.get_sample_value(
        _RESERVED, {"backend": backend, "model": model}
    )


def _manager(monkeypatch, n_backends: int = 1, n_slots: int = 2) -> SlotManager:
    """A SlotManager with n_backends mocked clients (for per-backend tests)."""
    monkeypatch.setattr(
        sm_module,
        "BACKENDS",
        [{"url": f"http://be{i}", "n_slots": n_slots} for i in range(n_backends)],
    )
    manager = SlotManager()
    clients = []
    for _ in range(n_backends):
        c = MagicMock()
        c.probe = AsyncMock(return_value=True)
        c.save_slot = AsyncMock(return_value=True)
        c.restore_slot = AsyncMock(return_value=True)
        c.erase_slot = AsyncMock(return_value=True)
        c.get_model_ctx = AsyncMock(return_value=None)
        c.is_router = AsyncMock(return_value=False)
        c.get_slots = AsyncMock(return_value=None)
        clients.append(c)
    manager.set_clients(clients)
    return manager


def _pool(manager: SlotManager, backend: int, model: str, n_ctx: int, n_slots: int = 2):
    manager.set_backend_slots(
        backend, model, [{"id": i, "n_ctx": n_ctx} for i in range(n_slots)]
    )


# --- the gate ---------------------------------------------------------------


async def test_budget_blocks_second_request_until_release(sm):
    """Two free slots but one 100-token context: 60 reserved + 60 requested
    does not fit, so the second request parks instead of taking the free slot."""
    _pool(sm, 0, "m", n_ctx=100, n_slots=2)
    ga, _, _, _ = await sm.acquire_for_request("m", need_tokens=60)

    second = asyncio.create_task(sm.acquire_for_request("m", need_tokens=60))
    done, _ = await asyncio.wait({second}, timeout=0.05)
    assert not done, "60 must not fit while 60 is reserved in a 100-token context"

    sm.release(ga)
    gb, lockb, _, _ = await asyncio.wait_for(second, timeout=1.0)
    assert lockb.locked(), "the freed budget must let the second request acquire"
    sm.release(gb)
    assert not lockb.locked()


async def test_oversized_request_is_admitted_and_counted(sm):
    """A request larger than the whole context is admitted best effort onto a
    free slot (it can never fit, so waiting is pointless) and counted once."""
    _pool(sm, 0, "m", n_ctx=100, n_slots=2)
    ga, _, _, _ = await sm.acquire_for_request("m", need_tokens=60)

    gb, lockb, _, _ = await sm.acquire_for_request("m", need_tokens=200)

    assert gb != ga, "the oversized request must take the free slot"
    assert promstats.counter_sum(admission_oversized_total, model="m") == 1.0
    sm.release(ga)
    sm.release(gb)
    assert not lockb.locked()


async def test_oversized_parks_when_no_slot_is_free(sm):
    """An oversized request is only waived onto a FREE slot; with every slot
    busy it parks like any other and is counted when it finally acquires."""
    _pool(sm, 0, "m", n_ctx=100, n_slots=1)
    ga, _, _, _ = await sm.acquire_for_request("m", need_tokens=10)

    second = asyncio.create_task(sm.acquire_for_request("m", need_tokens=200))
    done, _ = await asyncio.wait({second}, timeout=0.05)
    assert not done, "no free slot: the oversized request must park"
    assert promstats.counter_sum(admission_oversized_total, model="m") == 0.0

    sm.release(ga)
    gb, _, _, _ = await asyncio.wait_for(second, timeout=1.0)
    assert promstats.counter_sum(admission_oversized_total, model="m") == 1.0
    sm.release(gb)


async def test_release_wakes_oldest_waiter_that_fits(sm):
    """A freed slot must go to the oldest waiter that FITS, not merely the
    oldest: a too-big waiter that re-parks must not starve a smaller one that
    could have taken the slot, while the slot and budget sit idle."""
    _pool(sm, 0, "m", n_ctx=100, n_slots=2)
    gx, _, _, _ = await sm.acquire_for_request("m", need_tokens=30)
    gw, _, _, _ = await sm.acquire_for_request("m", need_tokens=30)

    big = asyncio.create_task(sm.acquire_for_request("m", need_tokens=80))
    small = asyncio.create_task(sm.acquire_for_request("m", need_tokens=20))
    for _ in range(20):
        await asyncio.sleep(0.01)  # let both park, big first
    assert len(sm._waiters.get("m", ())) == 2, "both waiters must be parked"

    sm.release(gx)  # reserved 30: big needs 110 (no fit), small needs 50 (fits)

    gs, _, _, _ = await asyncio.wait_for(small, timeout=1.0)
    assert not big.done(), "the too-big oldest waiter must stay parked"
    assert len(sm._waiters.get("m", ())) == 1, "only the big waiter remains"

    sm.release(gs)
    sm.release(gw)  # reserved 0: the big waiter now fits
    gb, _, _, _ = await asyncio.wait_for(big, timeout=1.0)
    sm.release(gb)
    assert not sm._waiters.get("m")


async def test_need_none_is_ungated(sm):
    """`need_tokens=None` disables admission: two requests take two free slots
    regardless of how tiny the context is, and reserve nothing."""
    _pool(sm, 0, "m", n_ctx=1, n_slots=2)

    g1, _, _, _ = await sm.acquire_for_request("m")
    g2, _, _, _ = await sm.acquire_for_request("m")

    assert g1 != g2
    assert sm._reserved.get((0, "m"), 0) == 0
    assert promstats.counter_sum(admission_oversized_total, model="m") == 0.0
    sm.release(g1)
    sm.release(g2)


async def test_unknown_total_ctx_is_ungated(sm):
    """Slots without a usable n_ctx leave the pool ungated: a huge request is
    admitted without the oversized counter firing."""
    sm.set_backend_slots(0, "m", [{"id": 0}, {"id": 1}])  # no n_ctx at all

    g1, _, _, _ = await sm.acquire_for_request("m", need_tokens=10**9)
    g2, _, _, _ = await sm.acquire_for_request("m", need_tokens=10**9)

    assert g1 != g2
    assert promstats.counter_sum(admission_oversized_total, model="m") == 0.0
    sm.release(g1)
    sm.release(g2)


# --- reservation accounting -------------------------------------------------


async def test_release_returns_the_reservation(sm):
    """The reserved gauge rises on acquire and drops back to zero on release,
    after which the same request fits again."""
    _pool(sm, 0, "m", n_ctx=100, n_slots=1)
    g, _, _, _ = await sm.acquire_for_request("m", need_tokens=60)
    assert _reserved() == 60.0

    sm.release(g)
    assert _reserved() == 0.0
    assert sm._reserved.get((0, "m"), 0) == 0

    g2, lock2, _, _ = await sm.acquire_for_request("m", need_tokens=60)
    assert g2 == g
    sm.release(g2)
    assert not lock2.locked()


async def test_reservation_survives_slot_cut_while_held(sm):
    """A slot removed from the pool while held keeps its reservation until the
    owner releases it (mirroring the lock/LRU bookkeeping)."""
    _pool(sm, 0, "m", n_ctx=100, n_slots=2)
    g, _, _, _ = await sm.acquire_for_request("m", need_tokens=60)
    assert _reserved() == 60.0

    sm.set_backend_slots(0, "m", [{"id": 1, "n_ctx": 100}])  # cut the held slot

    assert _reserved() == 60.0, "reservation must outlive the pool replacement"
    sm.release(g)
    assert _reserved() == 0.0
    assert g not in sm._slot_reserved


async def test_double_release_is_ignored(sm):
    """release is 'exactly once', but a stray second call must not drive the
    pool negative or raise."""
    _pool(sm, 0, "m", n_ctx=100, n_slots=1)
    g, _, _, _ = await sm.acquire_for_request("m", need_tokens=60)
    sm.release(g)
    sm.release(g)

    assert _reserved() == 0.0
    assert sm._reserved.get((0, "m"), 0) == 0


async def test_budget_is_per_backend(monkeypatch):
    """Two backends each hold their own 100-token context: one 100-token
    request per backend fits, routed to the respective pools."""
    manager = _manager(monkeypatch, n_backends=2, n_slots=1)
    _pool(manager, 0, "m", n_ctx=100, n_slots=1)
    _pool(manager, 1, "m", n_ctx=100, n_slots=1)

    g0, _, _, _ = await manager.acquire_for_request("m", need_tokens=100)
    g1, _, _, _ = await manager.acquire_for_request("m", need_tokens=100)

    assert {g0[0], g1[0]} == {0, 1}, "each backend admits its own 100-token request"
    manager.release(g0)
    manager.release(g1)


# --- /v1/models fallback ----------------------------------------------------


async def test_freshen_learns_ctx_from_models_fallback(sm):
    """When /slots carries no n_ctx, freshen fills total_ctx from
    /v1/models meta.n_ctx, so the pool is gated after the refresh."""
    client = sm.backends[0]["client"]
    client.get_slots = AsyncMock(return_value=[{"id": 0}, {"id": 1}])
    client.get_model_ctx = AsyncMock(return_value=100)

    await sm.freshen_model("m")

    assert sm._total_ctx[(0, "m")] == 100
    ga, _, _, _ = await sm.acquire_for_request("m", need_tokens=60)
    second = asyncio.create_task(sm.acquire_for_request("m", need_tokens=60))
    done, _ = await asyncio.wait({second}, timeout=0.05)
    assert not done, "the learned context must gate the pool"
    sm.release(ga)
    gb, _, _, _ = await asyncio.wait_for(second, timeout=1.0)
    sm.release(gb)


def _models_resp(payload: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = 200
    r.raise_for_status = MagicMock()
    r.json = MagicMock(return_value=payload)
    return r


def _client(payload) -> LlamaClient:
    c = LlamaClient("http://be")
    c.client = MagicMock()
    if isinstance(payload, Exception):
        c.client.get = AsyncMock(side_effect=payload)
    else:
        c.client.get = AsyncMock(return_value=_models_resp(payload))
    return c


async def test_get_model_ctx_matches_by_id():
    c = _client(
        {
            "data": [
                {"id": "other", "meta": {"n_ctx": 10}},
                {"id": "m", "meta": {"n_ctx": 4096}},
            ]
        },
    )
    assert await c.get_model_ctx("m") == 4096


async def test_get_model_ctx_single_entry_fallback():
    """A plain server's only entry is used when no id matches."""
    c = _client({"data": [{"id": "served", "meta": {"n_ctx": 8192}}]})
    assert await c.get_model_ctx("whatever") == 8192


async def test_get_model_ctx_single_loaded_entry_fallback():
    """A router with no id match falls back to its single loaded entry."""
    c = _client(
        {
            "data": [
                {"id": "a", "status": {"value": "unloaded"}, "meta": {"n_ctx": 1}},
                {"id": "b", "status": {"value": "loaded"}, "meta": {"n_ctx": 2048}},
            ]
        },
    )
    assert await c.get_model_ctx("nope") == 2048


async def test_get_model_ctx_none_on_ambiguous_listing():
    """Several entries and no id match: nothing may be guessed."""
    c = _client(
        {
            "data": [
                {"id": "a", "status": {"value": "unloaded"}, "meta": {"n_ctx": 1}},
                {"id": "b", "status": {"value": "unloaded"}, "meta": {"n_ctx": 2}},
            ]
        },
    )
    assert await c.get_model_ctx("nope") is None


async def test_get_model_ctx_defensive_shapes():
    """A missing meta, a non-int n_ctx, a non-positive n_ctx and a failed
    fetch all degrade to None instead of raising."""
    assert await _client({"data": [{"id": "m"}]}).get_model_ctx("m") is None
    assert await _client({"data": [{"id": "m", "meta": {}}]}).get_model_ctx("m") is None
    assert (
        await _client({"data": [{"id": "m", "meta": {"n_ctx": "big"}}]}).get_model_ctx(
            "m"
        )
        is None
    )
    assert (
        await _client({"data": [{"id": "m", "meta": {"n_ctx": 0}}]}).get_model_ctx("m")
        is None
    )
    assert await _client(RuntimeError("down")).get_model_ctx("m") is None
