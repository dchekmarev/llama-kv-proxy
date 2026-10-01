# backend/__init__.py

"""Integration with the llama.cpp backends: the HTTP client and the slot pools.

* :mod:`backend.llama_client` -- one :class:`LlamaClient` per backend URL:
  wire paths, router/preset awareness, the resolved-model-id cache and the
  mapping of backend errors onto HTTP status codes.
* :mod:`backend.slot_manager` -- the exclusive slot pools over those clients
  (FIFO acquisition, the :class:`GSlot` identity handed to the rest of the
  proxy, KV bookkeeping and LRU marks).

The submodules are public and imported through this package
(``from backend import slot_manager``), never as top-level modules: the tests
patch module attributes (``slot_manager.BACKENDS``, ``slot_manager.time``) and
every consumer must read the very same object.
"""

from . import llama_client, slot_manager

__all__ = ["llama_client", "slot_manager"]