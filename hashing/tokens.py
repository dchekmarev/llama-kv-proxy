# hashing/tokens.py

"""Prompt-token estimation for context-budget admission control.

Admission control needs the prompt's token count BEFORE dispatching: the
backend's own ``n_prompt_tokens`` only arrives after a (possibly long) prefill,
too late to reject an oversized request. Two estimators are on offer:

* the local Qwen tokenizer from the optional ``dashscope`` package
  (``requirements-tokenizer.txt``), which encodes the prompt in-process with
  the tokenizer family the served Qwen models use -- no per-request round trip
  to a tokenizer endpoint and no network dependency at admission time. It is
  loaded lazily and cached for the process lifetime.
* a deterministic word-count fallback (``ceil(words * TOKENS_PER_WORD)``) used
  when ``TOKENIZER_ENABLED`` is off, ``dashscope`` is not installed, or
  encoding fails. The fallback is logged once, not on every call.

Both add ``PROMPT_TEMPLATE_MARGIN_TOKENS``: the raw prefix the proxy hashes
omits the chat template's special tokens (BOS/EOS and the per-role/per-turn
markers), so a small constant margin keeps the estimate on the safe side of
the real rendered prompt length.

The estimator is synchronous (the tokenizer encode is CPU-bound and can block
for a large prompt); callers on the event loop use
``estimate_prompt_tokens_async``, which runs it in a worker thread. Importing
this module never imports ``dashscope``.
"""

import asyncio
import math
from typing import Protocol

from core.config import (
    PROMPT_TEMPLATE_MARGIN_TOKENS,
    TOKENIZER_ENABLED,
    TOKENIZER_MODEL,
    TOKENS_PER_WORD,
)

from ._state import log
from ._text import words_from_text


class _Tokenizer(Protocol):
    """The one method of the dashscope tokenizer this module relies on."""

    def encode(self, text: str) -> list[int]: ...


# Lazy process-wide tokenizer cache; None until the first estimate. Created on
# first use, which may be a worker thread (estimate_prompt_tokens_async runs
# the sync estimator via asyncio.to_thread); a race there can at worst build
# the tokenizer twice, since the load is idempotent. Reset by reset_tokenizer().
_TOKENIZER: _Tokenizer | None = None
# The fallback warning fires once per process (reset by reset_tokenizer), so a
# permanently unavailable tokenizer does not spam the log once per request.
_FALLBACK_LOGGED = False


def _get_tokenizer() -> _Tokenizer:
    """Create and cache the dashscope tokenizer for TOKENIZER_MODEL.

    Imports dashscope lazily so importing hashing does not require the
    optional dependency. Raises if dashscope is missing or creation fails; the
    caller turns that into the word-count fallback.
    """
    global _TOKENIZER
    tokenizer = _TOKENIZER
    if tokenizer is None:
        from dashscope import get_tokenizer  # type: ignore[import-not-found]

        tokenizer = get_tokenizer(TOKENIZER_MODEL)
        _TOKENIZER = tokenizer
    return tokenizer


def _fallback_estimate(text: str) -> int:
    """Deterministic word-count estimate, template margin included."""
    words = len(words_from_text(text))
    return math.ceil(words * TOKENS_PER_WORD) + PROMPT_TEMPLATE_MARGIN_TOKENS


def estimate_prompt_tokens(text: str) -> int:
    """Estimate ``text``'s prompt token count, template margin included.

    Uses the cached local tokenizer when ``TOKENIZER_ENABLED`` is on; otherwise
    (disabled, dashscope absent, or an encode failure) falls back to the
    word-count estimate. Never raises.
    """
    global _FALLBACK_LOGGED
    if not TOKENIZER_ENABLED:
        return _fallback_estimate(text)
    try:
        tokens = _get_tokenizer().encode(text)
        return len(tokens) + PROMPT_TEMPLATE_MARGIN_TOKENS
    except Exception as e:  # noqa: BLE001
        if not _FALLBACK_LOGGED:
            log.warning("prompt tokenizer unavailable, using word estimate: %s", e)
            _FALLBACK_LOGGED = True
        return _fallback_estimate(text)


async def estimate_prompt_tokens_async(text: str) -> int:
    """estimate_prompt_tokens off the event loop (the encode is CPU-bound)."""
    return await asyncio.to_thread(estimate_prompt_tokens, text)


def reset_tokenizer() -> None:
    """Drop the cached tokenizer and the once-only fallback flag (test hook)."""
    global _TOKENIZER, _FALLBACK_LOGGED
    _TOKENIZER = None
    _FALLBACK_LOGGED = False
