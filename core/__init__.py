# core/__init__.py

"""Cross-cutting infrastructure: configuration, logging, ids, version, metrics.

Everything here is shared by the other packages and depends on nothing in
them, so this is the bottom layer of the import graph:

* :mod:`core.config` -- the single source of truth for every tunable, read
  from the environment by :func:`core.config.init_runtime`;
* :mod:`core.request_id` -- the per-request correlation id (a ContextVar)
  and the log filter that stamps it on every record;
* :mod:`core.logging_setup` -- the root logger configuration;
* :mod:`core.version` -- the single source of truth for ``__version__``;
* :mod:`core.promstats` -- the proxy's own Prometheus registry.

The submodules are public and imported through this package
(``from core import config``), never as top-level modules, so a value that
the tests patch (``config.META_DIR``, ``promstats.BIN_CACHE_DIR``) is
patched on the very object the rest of the proxy reads it from.
"""

from . import config, logging_setup, promstats, request_id, version

__all__ = ["config", "logging_setup", "promstats", "request_id", "version"]