# tests/test_config.py

"""P2-1: backend config must be validated at startup with a clear error
instead of silently degrading to an empty list (later IndexError)."""

import os

import pytest

import config


def test_parse_backends_env_broken_json_raises():
    with pytest.raises(ValueError, match="not valid JSON"):
        config.parse_backends_env("[oops")


def test_parse_backends_env_valid_json():
    parsed = config.parse_backends_env(
        '[{"url": "http://be1", "n_slots": 2}, {"url": "http://be2", "n_slots": 1}]'
    )
    assert parsed == [
        {"url": "http://be1", "n_slots": 2},
        {"url": "http://be2", "n_slots": 1},
    ]


def test_parse_backends_env_default(monkeypatch):
    monkeypatch.setenv("LLAMA_URL", "http://custom:9999")
    monkeypatch.setenv("N_SLOTS", "3")
    parsed = config.parse_backends_env(None)
    assert parsed == [{"url": "http://custom:9999", "n_slots": 3}]


def test_parse_backends_env_bad_n_slots_raises(monkeypatch):
    monkeypatch.setenv("N_SLOTS", "abc")
    with pytest.raises(ValueError, match="N_SLOTS"):
        config.parse_backends_env(None)


def test_validate_rejects_empty_list():
    with pytest.raises(ValueError, match="empty"):
        config.validate_backends([])


def test_validate_rejects_non_list():
    with pytest.raises(ValueError, match="list"):
        config.validate_backends({"url": "http://be"})


def test_validate_rejects_missing_url():
    with pytest.raises(ValueError, match="url"):
        config.validate_backends([{"n_slots": 2}])


def test_validate_rejects_bad_n_slots():
    with pytest.raises(ValueError, match="n_slots"):
        config.validate_backends([{"url": "http://be", "n_slots": 0}])


def test_validate_accepts_valid():
    config.validate_backends([{"url": "http://be", "n_slots": 2}])


def test_meta_dir_is_absolute_and_in_app_dir():
    """META_DIR must not depend on the process cwd."""
    assert os.path.isabs(config.META_DIR), (
        f"META_DIR must be absolute, got {config.META_DIR!r}"
    )
    app_dir = os.path.dirname(os.path.abspath(config.__file__))
    assert config.META_DIR.startswith(app_dir), (
        f"META_DIR must live in the app dir {app_dir!r}, got {config.META_DIR!r}"
    )
