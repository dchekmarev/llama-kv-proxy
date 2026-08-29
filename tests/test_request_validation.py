# tests/test_request_validation.py

"""P3-5: 400 on invalid JSON body / non-object body, and roles included in
the cache key (same content under different roles must not collide)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

import app as app_module
import hashing as hs
import slot_manager as sm_module
from slot_manager import SlotManager


class FakeRequest:
    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class BrokenRequest:
    """A request whose body is not valid JSON."""

    async def json(self):
        raise ValueError("bad json")


@pytest.fixture()
def sm(monkeypatch):
    monkeypatch.setattr(sm_module, "BACKENDS", [{"url": "http://be", "n_slots": 2}])
    manager = SlotManager()
    client = MagicMock()
    client.save_slot = AsyncMock(return_value=True)
    client.restore_slot = AsyncMock(return_value=True)
    client.get_model_id_cached = AsyncMock(return_value="m1")
    client.get_loaded_model = AsyncMock(return_value="m1")
    client.chat_completions = AsyncMock(return_value={"choices": []})
    manager.set_clients([client])
    app_module.app.state.sm = manager
    app_module.app.state.clients = [client]
    return manager


async def test_invalid_json_returns_400(sm):
    """A body that fails JSON parsing becomes HTTP 400, not 500."""
    resp = await app_module.chat(BrokenRequest())

    assert resp.status_code == 400


async def test_non_dict_json_returns_400(sm):
    """A valid JSON body that is not an object becomes HTTP 400."""
    resp = await app_module.chat(FakeRequest([1, 2, 3]))

    assert resp.status_code == 400


def test_raw_prefix_includes_roles():
    """Roles are part of the prefix text."""
    msgs = [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hello"},
    ]

    assert hs.raw_prefix(msgs) == "system:be nice\n\nuser:hello"


def test_same_content_different_roles_do_not_collide():
    """The same content under different roles produces different keys."""
    a = hs.raw_prefix([{"role": "system", "content": "x"}])
    b = hs.raw_prefix([{"role": "user", "content": "x"}])

    assert a != b
    assert hs.prefix_key_sha256("m\n" + a) != hs.prefix_key_sha256("m\n" + b)
