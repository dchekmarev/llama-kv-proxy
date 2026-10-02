# tests/conftest.py

"""Shared pytest fixtures for the llama-kv-proxy test suite.

These are the fixtures that were copy-pasted across many test files. A test
file may still define its own ``sm`` / ``meta_dir`` to override the shared
version (e.g. to add extra client mocks or point at a different META_DIR);
pytest resolves the nearest-scope fixture, so local definitions win.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import chat_flow
import hashing as hs
from backend import slot_manager as sm_module
from backend.slot_manager import SlotManager
from core import promstats
from obs import ui as ui_obs


@pytest.fixture(autouse=True)
def _clean_prom_registry():
    """Isolate the proxy metrics registry between tests."""
    promstats.reset()
    yield
    promstats.reset()


@pytest.fixture(autouse=True)
def _clean_meta_index():
    """Reset the shared restore index between tests.

    The index flag is on by default, so metas written during a test (via
    write_meta_async) land in the module-level singleton; without this reset
    they would leak into the next test's search (whose META_DIR is a different
    temp dir).
    """
    hs._index.clear()
    yield
    hs._index.clear()


@pytest.fixture(autouse=True)
def _clean_ui_registry():
    """Reset the live-dashboard registry between tests.

    Every chat_flow run registers its request in the module-level registry;
    without this reset requests would leak across tests. A broadcaster task
    left over from a previous test's event loop is cancelled as well.
    """
    reg = ui_obs.registry
    reg.active.clear()
    reg.history.clear()
    task = reg._broadcaster
    if task is not None:
        task.cancel()
        reg._broadcaster = None
    yield
    reg.active.clear()
    reg.history.clear()
    task = reg._broadcaster
    if task is not None:
        task.cancel()
        reg._broadcaster = None


@pytest.fixture(autouse=True)
def _clean_kill_registry():
    """Reset the kill registry between tests.

    Every chat_flow run binds a kill token for its request id; without this
    reset a token left behind by a test (a killed one is dropped explicitly, a
    finished one by the pipeline) would make the next test's request id
    unkillable.
    """
    chat_flow.reset_kills()
    yield
    chat_flow.reset_kills()


@pytest.fixture()
def meta_dir(tmp_path, monkeypatch):
    """Point the hashing module's META_DIR at a fresh temp dir."""
    monkeypatch.setattr(hs, "META_DIR", str(tmp_path))
    return tmp_path


@pytest.fixture()
def sm(monkeypatch):
    """A SlotManager with one backend and a fully-mocked client."""
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    client.restore_slot = AsyncMock(return_value=True)
    client.erase_slot = AsyncMock(return_value=True)
    client.get_model_id_cached = AsyncMock(return_value="m1")
    # No preset alias table: a client model name maps to nothing here.
    client.resolve_model_id_cached = AsyncMock(return_value=None)
    client.get_loaded_model = AsyncMock(return_value="m1")
    client.chat_completions = AsyncMock(return_value={"choices": []})
    # On-demand freshen: a plain backend whose re-poll reports no change (None
    # keeps the existing pool), so the chat_flow hook is exercised without
    # disturbing a test's set-up pool.
    client.is_router = AsyncMock(return_value=False)
    client.get_slots = AsyncMock(return_value=None)
    manager.set_clients([client])
    return manager
