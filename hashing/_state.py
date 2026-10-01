# hashing/_state.py

"""Package-level mutable state, shared constants and the package logger.

Every object below is created exactly once and re-exported by
hashing/__init__.py, so hashing.<name> and the submodules always see the same
object.
"""

import logging

from ._meta_index import MetaIndex

log = logging.getLogger("hashing")

# restore_total outcomes counted by record_hit / record_miss and reported by
# cache_stats() (the counters live in the metrics registry, so /metrics and
# /cache/stats always agree).
_OUTCOME_HIT = "hit"
_OUTCOME_MISS = "miss"

# In-RAM restore index (see meta_index): only mutated on the event loop, by the
# *_async wrappers and by the search itself.
_index = MetaIndex()
