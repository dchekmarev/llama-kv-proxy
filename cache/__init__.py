# cache/__init__.py

"""The on-disk KV cache: the backend ``.bin`` files themselves.

:mod:`cache.bin_cache` owns everything that happens to those files directly:
the total-size cap applied in LRU order (taken from the meta timestamps),
orphan reclamation, and the ``.ckpt`` sidecars llama.cpp leaves behind.
Everything that happens to the *meta* documents that describe those files
belongs to :mod:`hashing` instead.
"""

from . import bin_cache

__all__ = ["bin_cache"]