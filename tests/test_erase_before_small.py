# tests/test_erase_before_small.py

"""A small (non-cached) request is never preceded by a restore, so with
ERASE_BEFORE_CHAT on it clears the slot's KV cache before dispatching (it must
not start on top of another conversation's stale/oversized prompt). With the
flag off, the slot is left untouched. The big-request / restore-outcome matrix
lives in test_erase_before_chat.py."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import chat_flow
import slot_manager as sm_module
from slot_manager import SlotManager


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


@pytest.fixture()
def sm(monkeypatch):
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
    manager.set_clients([client])
    return manager


async def _small_chat(sm, monkeypatch, flag: bool):
    client = sm.backends[0]["client"]
    app_module.app.state.sm = sm
    app_module.app.state.clients = [client]
    monkeypatch.setattr(chat_flow, "ERASE_BEFORE_CHAT", flag)
    data = {"messages": [{"role": "user", "content": "hi"}], "stream": False}
    await app_module.chat(FakeRequest(data))
    return client


@pytest.mark.asyncio
async def test_small_request_erases_slot_when_flag_on(sm, monkeypatch):
    client = await _small_chat(sm, monkeypatch, flag=True)
    client.erase_slot.assert_awaited_once()


@pytest.mark.asyncio
async def test_small_request_keeps_slot_when_flag_off(sm, monkeypatch):
    client = await _small_chat(sm, monkeypatch, flag=False)
    client.erase_slot.assert_not_awaited()
