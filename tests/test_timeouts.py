# tests/test_timeouts.py

"""P2-2: control-plane timeouts must be configurable via env, not hardcoded
in app.py."""

import importlib

import pytest

import chat_flow
import config


def test_acquire_timeout_default():
    assert config.ACQUIRE_TIMEOUT == pytest.approx(1500.0)


def test_chat_flow_uses_config_acquire_timeout():
    """chat_flow must take the timeout from config, not hardcode its own copy."""
    assert chat_flow.ACQUIRE_TIMEOUT == config.ACQUIRE_TIMEOUT


def test_acquire_timeout_from_env(monkeypatch):
    monkeypatch.setenv("ACQUIRE_TIMEOUT", "42")
    importlib.reload(config)
    try:
        assert config.ACQUIRE_TIMEOUT == pytest.approx(42.0)
    finally:
        importlib.reload(config)
