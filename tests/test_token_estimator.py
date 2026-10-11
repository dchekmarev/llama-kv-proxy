# tests/test_token_estimator.py

"""The prompt-token estimator: the optional dashscope tokenizer path, the
deterministic word-count fallback and the async wrapper. No dashscope and no
network are needed -- the module's tokenizer getter is monkeypatched."""

import math

import pytest

from hashing import tokens


class _FakeTokenizer:
    """A tokenizer stub whose ``encode`` returns a fixed token list."""

    def __init__(self, encoded: list[int]):
        self.encoded = encoded
        self.seen: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.seen.append(text)
        return self.encoded


@pytest.fixture(autouse=True)
def _fresh_tokenizer():
    tokens.reset_tokenizer()
    yield
    tokens.reset_tokenizer()


def _expected_fallback(text: str) -> int:
    words = len(tokens.words_from_text(text))
    return math.ceil(words * tokens.TOKENS_PER_WORD) + tokens.PROMPT_TEMPLATE_MARGIN_TOKENS


def test_estimate_uses_tokenizer(monkeypatch):
    """A working tokenizer wins; the count is len(encode) + template margin."""
    fake = _FakeTokenizer([10, 20, 30])
    monkeypatch.setattr(tokens, "_get_tokenizer", lambda: fake)
    assert (
        tokens.estimate_prompt_tokens("hello")
        == 3 + tokens.PROMPT_TEMPLATE_MARGIN_TOKENS
    )
    assert fake.seen == ["hello"]


def test_estimate_falls_back_when_tokenizer_unavailable(monkeypatch):
    """A failing tokenizer construction (e.g. dashscope absent) -> word estimate."""

    def _boom() -> tokens._Tokenizer:
        raise ImportError("no dashscope")

    monkeypatch.setattr(tokens, "_get_tokenizer", _boom)
    text = "one two three four"
    assert tokens.estimate_prompt_tokens(text) == _expected_fallback(text)


def test_estimate_falls_back_when_encode_raises(monkeypatch):
    """An encode failure is caught too, not only a construction failure."""

    class _Bad:
        def encode(self, text: str) -> list[int]:
            raise RuntimeError("encode failed")

    monkeypatch.setattr(tokens, "_get_tokenizer", lambda: _Bad())
    text = "alpha beta"
    assert tokens.estimate_prompt_tokens(text) == _expected_fallback(text)


def test_estimate_falls_back_when_len_raises(monkeypatch):
    """A malformed encode result (len() fails) must fall back, not raise:
    the "Never raises" contract covers the len() call, not just encode()."""

    class _NoLen:
        def __len__(self) -> int:
            raise TypeError("no length")

    class _Bad:
        def encode(self, text: str) -> object:
            return _NoLen()

    monkeypatch.setattr(tokens, "_get_tokenizer", lambda: _Bad())
    text = "gamma delta"
    assert tokens.estimate_prompt_tokens(text) == _expected_fallback(text)


def test_disabled_tokenizer_forces_fallback(monkeypatch):
    """TOKENIZER_ENABLED=false must not even look at the tokenizer."""
    monkeypatch.setattr(tokens, "TOKENIZER_ENABLED", False)

    def _unexpected() -> tokens._Tokenizer:
        raise AssertionError("tokenizer must not be used when disabled")

    monkeypatch.setattr(tokens, "_get_tokenizer", _unexpected)
    text = "a b c d"
    assert tokens.estimate_prompt_tokens(text) == _expected_fallback(text)


def test_fallback_logged_once(monkeypatch, caplog):
    """The fallback warning fires once per process, not per call."""

    def _boom() -> tokens._Tokenizer:
        raise RuntimeError("nope")

    monkeypatch.setattr(tokens, "_get_tokenizer", _boom)
    with caplog.at_level("WARNING", logger="hashing"):
        tokens.estimate_prompt_tokens("a b c")
        tokens.estimate_prompt_tokens("d e f")
    assert caplog.text.count("using word estimate") == 1


def test_reset_tokenizer_clears_state():
    """reset_tokenizer drops the cached tokenizer and the logged flag."""
    tokens._TOKENIZER = _FakeTokenizer([1])
    tokens._FALLBACK_LOGGED = True
    tokens.reset_tokenizer()
    assert tokens._TOKENIZER is None
    assert tokens._FALLBACK_LOGGED is False


async def test_async_wrapper_matches_sync(monkeypatch):
    """The async wrapper returns the same value as the sync estimator."""
    fake = _FakeTokenizer([1, 2, 3, 4, 5])
    monkeypatch.setattr(tokens, "_get_tokenizer", lambda: fake)
    assert await tokens.estimate_prompt_tokens_async("hi") == (
        tokens.estimate_prompt_tokens("hi")
    )


def test_package_reexports_estimator():
    """The hashing package re-exports the public estimator API."""
    import hashing as hs

    assert hs.estimate_prompt_tokens is tokens.estimate_prompt_tokens
    assert hs.estimate_prompt_tokens_async is tokens.estimate_prompt_tokens_async
    assert hs.reset_tokenizer is tokens.reset_tokenizer
