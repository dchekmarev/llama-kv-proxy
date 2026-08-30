# tests/conftest.py

"""Shared pytest fixtures for the llama-kv-proxy test suite.

These are the fixtures that were copy-pasted across many test files. A test
file may still define its own ``sm`` / ``meta_dir`` to override the shared
version (e.g. to add extra client mocks or point at a different META_DIR);
pytest resolves the nearest-scope fixture, so local definitions win.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

import hashing as hs
import slot_manager as sm_module
from slot_manager import SlotManager


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
    client.get_model_id_cached = AsyncMock(return_value="m1")
    client.get_loaded_model = AsyncMock(return_value="m1")
    client.chat_completions = AsyncMock(return_value={"choices": []})
    manager.set_clients([client])
    return manager
