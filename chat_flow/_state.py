# chat_flow/_state.py

"""Package-level mutable state, shared constants and the package logger.

Every container below is created exactly once and re-exported by
chat_flow/__init__.py, so chat_flow.<name> and the submodules always see the
same object.
"""

import asyncio
import logging

log = logging.getLogger("chat_flow")

STREAM_QUEUE_SIZE = 16
# Bounded wait on queue push: if the consumer disappears (client
# disconnected and the generator is closed), the reader must not block on
# put forever — otherwise its finally (and the slot release) would never run.
STREAM_PUT_TIMEOUT = 60.0

# Strong references to running reader tasks: the event loop keeps only
# weak references to tasks; without this a reader could be GC-collected
# mid-execution and skip its finally (slot release, response aclose).
_READER_TASKS: "set[asyncio.Task]" = set()

# Strong references to in-flight LRU check tasks (same GC reason as above)
# and a guard against concurrent checks racing on the same files.
_LRU_TASKS: "set[asyncio.Task]" = set()
_lru_check_in_flight = False

# Strong references to in-flight non-stream background save tasks (same GC
# reason as _READER_TASKS): the .bin write must not delay the JSON response.
_BG_SAVE_TASKS: "set[asyncio.Task]" = set()

# In-flight saves: request key -> completion event. The client treats [DONE]
# as the end of the response and sends the continuation immediately, but the
# previous message's meta only lands after its .bin write finishes. The
# previous request is a strict prefix of the continuation's request, so its
# key is among the continuation's prefix hashes: a continuation that misses
# the restore search can detect the relevant in-flight save and wait for it
# instead of reprocessing the whole prompt.
_INFLIGHT_SAVES: dict[str, asyncio.Event] = {}

# Keys that big requests have selected for restore and are now waiting to
# acquire a slot for (refcount of waiters per key). A concurrent save that
# subsumes such a key deletes its .bin; the waiter must restore the longer
# cache that superseded it instead (substitution), or the restore fails with
# "file not found" and the whole prompt is reprocessed.
_PENDING_RESTORES: dict[str, int] = {}
# Deleted-key -> replacement-key for pending restores. Newer saves only alias
# keys that are still awaited; once no waiter depends on a key its alias is
# dropped (see _unregister_pending_restore).
_RESTORE_ALIAS: dict[str, str] = {}
