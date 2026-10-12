# tests/test_config.py

"""P2-1: backend config must be validated at startup with a clear error
instead of silently degrading to an empty list (later IndexError)."""

import importlib
import os

import pytest

from core import config


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


def test_bin_cache_config_attrs_exist():
    """The .bin cleanup config values are loaded with the right types."""
    assert isinstance(config.BIN_CACHE_DIR, str)
    assert isinstance(config.BIN_CACHE_MAX_MB, int)


def test_meta_dir_is_absolute_and_in_app_dir():
    """META_DIR must not depend on the process cwd."""
    assert os.path.isabs(config.META_DIR), (
        f"META_DIR must be absolute, got {config.META_DIR!r}"
    )
    # config lives in the core/ package; the app dir is its parent.
    pkg_dir = os.path.dirname(os.path.abspath(config.__file__))
    app_dir = os.path.dirname(pkg_dir)
    assert config.META_DIR.startswith(app_dir + os.sep), (
        f"META_DIR must live in the app dir {app_dir!r}, got {config.META_DIR!r}"
    )


def test_env_int_default_when_unset(monkeypatch):
    monkeypatch.delenv("WORDS_PER_BLOCK", raising=False)
    assert config._env_int("WORDS_PER_BLOCK", 100) == 100


def test_env_int_parses_value(monkeypatch):
    monkeypatch.setenv("WORDS_PER_BLOCK", "250")
    assert config._env_int("WORDS_PER_BLOCK", 100) == 250


def test_env_int_bad_value_raises_with_name(monkeypatch):
    monkeypatch.setenv("WORDS_PER_BLOCK", "abc")
    with pytest.raises(ValueError, match="WORDS_PER_BLOCK"):
        config._env_int("WORDS_PER_BLOCK", 100)


def test_env_float_default_and_parse(monkeypatch):
    monkeypatch.delenv("LCP_TH", raising=False)
    assert config._env_float("LCP_TH", 0.6) == 0.6
    monkeypatch.setenv("LCP_TH", "0.8")
    assert config._env_float("LCP_TH", 0.6) == 0.8


def test_env_float_bad_value_raises_with_name(monkeypatch):
    monkeypatch.setenv("LCP_TH", "xyz")
    with pytest.raises(ValueError, match="LCP_TH"):
        config._env_float("LCP_TH", 0.6)


def test_env_bool_truthy_and_falsy(monkeypatch):
    for v in ("1", "true", "yes", "on"):
        monkeypatch.setenv("ERASE_BEFORE_CHAT", v)
        assert config._env_bool("ERASE_BEFORE_CHAT", False) is True
    for v in ("0", "false", "no", "off"):
        monkeypatch.setenv("ERASE_BEFORE_CHAT", v)
        assert config._env_bool("ERASE_BEFORE_CHAT", True) is False
    monkeypatch.delenv("ERASE_BEFORE_CHAT", raising=False)
    assert config._env_bool("ERASE_BEFORE_CHAT", True) is True


def test_init_runtime_creates_dirs(monkeypatch, tmp_path):
    """init_runtime creates the meta and request-log directories."""
    meta = tmp_path / "kv_meta"
    reqlog = tmp_path / "kv_reqlog"
    monkeypatch.setattr(config, "META_DIR", str(meta))
    monkeypatch.setattr(config, "REQUEST_LOG_DIR", str(reqlog))
    config.init_runtime()
    assert meta.is_dir()
    assert reqlog.is_dir()


def test_init_runtime_skips_empty_request_log_dir(monkeypatch, tmp_path):
    """An empty REQUEST_LOG_DIR disables the directory, no mkdir."""
    meta = tmp_path / "kv_meta"
    monkeypatch.setattr(config, "META_DIR", str(meta))
    monkeypatch.setattr(config, "REQUEST_LOG_DIR", "")
    config.init_runtime()
    assert meta.is_dir()


def test_init_runtime_validates_backends(monkeypatch):
    """Backends are validated at startup, not at import time."""
    monkeypatch.setattr(config, "BACKENDS", [])
    with pytest.raises(ValueError, match="empty"):
        config.init_runtime()


def test_context_admission_defaults():
    """Context-budget admission knobs carry their documented defaults."""
    assert config.MAX_TOKENS == 32768
    assert config.CONTEXT_ADMISSION_ENABLED is True
    assert config.TOKENIZER_ENABLED is True
    assert config.TOKENIZER_MODEL == "qwen-turbo"
    assert config.TOKENS_PER_WORD == pytest.approx(1.3)
    assert config.PROMPT_TEMPLATE_MARGIN_TOKENS == 64


def test_max_tokens_from_env(monkeypatch):
    monkeypatch.setenv("MAX_TOKENS", "4096")
    importlib.reload(config)
    try:
        assert config.MAX_TOKENS == 4096
    finally:
        monkeypatch.delenv("MAX_TOKENS")
        importlib.reload(config)


def test_context_admission_enabled_from_env(monkeypatch):
    monkeypatch.setenv("CONTEXT_ADMISSION_ENABLED", "false")
    importlib.reload(config)
    try:
        assert config.CONTEXT_ADMISSION_ENABLED is False
    finally:
        monkeypatch.delenv("CONTEXT_ADMISSION_ENABLED")
        importlib.reload(config)


def test_tokenizer_enabled_from_env(monkeypatch):
    monkeypatch.setenv("TOKENIZER_ENABLED", "off")
    importlib.reload(config)
    try:
        assert config.TOKENIZER_ENABLED is False
    finally:
        monkeypatch.delenv("TOKENIZER_ENABLED")
        importlib.reload(config)


def test_tokenizer_model_from_env(monkeypatch):
    monkeypatch.setenv("TOKENIZER_MODEL", "qwen-max")
    importlib.reload(config)
    try:
        assert config.TOKENIZER_MODEL == "qwen-max"
    finally:
        monkeypatch.delenv("TOKENIZER_MODEL")
        importlib.reload(config)


def test_tokens_per_word_from_env(monkeypatch):
    monkeypatch.setenv("TOKENS_PER_WORD", "1.5")
    importlib.reload(config)
    try:
        assert config.TOKENS_PER_WORD == pytest.approx(1.5)
    finally:
        monkeypatch.delenv("TOKENS_PER_WORD")
        importlib.reload(config)


def test_prompt_template_margin_from_env(monkeypatch):
    monkeypatch.setenv("PROMPT_TEMPLATE_MARGIN_TOKENS", "128")
    importlib.reload(config)
    try:
        assert config.PROMPT_TEMPLATE_MARGIN_TOKENS == 128
    finally:
        monkeypatch.delenv("PROMPT_TEMPLATE_MARGIN_TOKENS")
        importlib.reload(config)


def test_backend_retry_defaults():
    """The proxy-side retry and keep-alive knobs carry their defaults."""
    assert config.KEEPALIVE_EXPIRY_S == pytest.approx(5.0)
    assert config.BACKEND_RETRY_ATTEMPTS == 3
    assert config.BACKEND_RETRY_BACKOFF_S == pytest.approx(0.5)


def test_backend_retry_attempts_floor_is_one(monkeypatch):
    """0 (or negative) attempts still means one attempt, never zero."""
    monkeypatch.setenv("BACKEND_RETRY_ATTEMPTS", "0")
    importlib.reload(config)
    try:
        assert config.BACKEND_RETRY_ATTEMPTS == 1
    finally:
        monkeypatch.delenv("BACKEND_RETRY_ATTEMPTS")
        importlib.reload(config)
