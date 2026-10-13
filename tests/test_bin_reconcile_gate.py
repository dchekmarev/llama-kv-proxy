# tests/test_bin_reconcile_gate.py

"""The .bin reconcile is gated on backend liveness.

A down or restarting backend can have a transiently empty or half-written
slot-save directory; a reconcile run then would delete every meta whose .bin
is momentarily missing. The reconcile pass must skip while any backend fails
its probe, so a restart never turns an outage into a wiped cache index.
"""

import asyncio

import pytest

import app as app_module
from app import background
from app.background import _backends_serving


class _Stop(Exception):
    """Raised from the mocked asyncio.sleep to end the loop under test."""


class FakeClient:
    def __init__(self, up):
        self._up = up

    async def probe(self):
        return self._up


@pytest.mark.asyncio
async def test_backends_serving_requires_all_up(monkeypatch):
    monkeypatch.setattr(
        app_module.app.state, "clients", [FakeClient(True)], raising=False
    )
    assert await _backends_serving() is True

    monkeypatch.setattr(
        app_module.app.state,
        "clients",
        [FakeClient(True), FakeClient(False)],
        raising=False,
    )
    assert await _backends_serving() is False

    monkeypatch.setattr(
        app_module.app.state, "clients", [FakeClient(False)], raising=False
    )
    assert await _backends_serving() is False


@pytest.mark.asyncio
async def test_backends_serving_empty_is_not_serving(monkeypatch):
    monkeypatch.setattr(app_module.app.state, "clients", [], raising=False)
    assert await _backends_serving() is False


@pytest.mark.asyncio
async def test_reconcile_skipped_while_backend_down(monkeypatch):
    calls = []

    async def _sleep_then_stop(seconds):
        if getattr(_sleep_then_stop, "done", False):
            raise _Stop
        _sleep_then_stop.done = True

    monkeypatch.setattr(asyncio, "sleep", _sleep_then_stop)
    monkeypatch.setattr(app_module, "BIN_RECONCILE_INTERVAL_S", 0.01)
    monkeypatch.setattr(app_module, "BIN_CACHE_DIR", "/bin_cache")
    monkeypatch.setattr(
        app_module.app.state, "clients", [FakeClient(False)], raising=False
    )
    monkeypatch.setattr(
        background.bin_cache,
        "reconcile_bin_cache",
        lambda dir: calls.append(dir) or {"deleted_metas": [], "deleted_bins": []},
    )

    with pytest.raises(_Stop):
        await app_module._bin_reconcile_loop()

    assert calls == []


@pytest.mark.asyncio
async def test_reconcile_runs_while_backend_up(monkeypatch):
    calls = []

    async def _sleep_then_stop(seconds):
        if getattr(_sleep_then_stop, "done", False):
            raise _Stop
        _sleep_then_stop.done = True

    monkeypatch.setattr(asyncio, "sleep", _sleep_then_stop)
    monkeypatch.setattr(app_module, "BIN_RECONCILE_INTERVAL_S", 0.01)
    monkeypatch.setattr(app_module, "BIN_CACHE_DIR", "/bin_cache")
    monkeypatch.setattr(
        app_module.app.state, "clients", [FakeClient(True)], raising=False
    )
    monkeypatch.setattr(
        background.bin_cache,
        "reconcile_bin_cache",
        lambda dir: calls.append(dir) or {"deleted_metas": [], "deleted_bins": []},
    )

    with pytest.raises(_Stop):
        await app_module._bin_reconcile_loop()

    assert calls == ["/bin_cache"]
