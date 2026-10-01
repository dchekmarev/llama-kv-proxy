# app/_asgi.py

"""The ASGI application object and the lifespan that runs its background jobs.

The FastAPI instance is built here so that routes.py and middleware.py can
register onto it at import time while app/__init__.py stays a plain
re-export module: importing them after the instance was built would be a
module-level import that is not at the top of the file.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI

import app as app_pkg
import hashing
from llama_client import LlamaClient

log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Build the backend clients, start the background jobs, close on exit."""
    app_pkg.setup_logging(app_pkg.LOG_LEVEL)
    app_pkg.init_runtime()
    clients: list[LlamaClient] = []
    tasks: list[asyncio.Task[None]] = []
    try:
        for be in app_pkg.BACKENDS:
            # Appended one by one: a constructor that raises must not hide the
            # clients already built from the finally below.
            clients.append(app_pkg.LlamaClient(be["url"]))
        sm = app_pkg.SlotManager()
        sm.set_clients(clients)
        application.state.clients = clients
        application.state.sm = sm

        if app_pkg.META_INDEX_ENABLED:
            indexed = await hashing.rebuild_index_async()
            log.info("meta_index_rebuilt metas=%d", indexed)

        tasks = [
            asyncio.create_task(app_pkg._poll_slots_loop()),
            asyncio.create_task(app_pkg._eviction_loop()),
            asyncio.create_task(app_pkg._bin_reconcile_loop()),
        ]
        # The reconcile only exists to maintain the in-RAM index, so it needs
        # both the index and a non-zero interval.
        if app_pkg.META_INDEX_ENABLED and app_pkg.META_INDEX_RECONCILE_INTERVAL_S > 0:
            app_pkg._meta_index_reconcile_task = asyncio.create_task(
                app_pkg._meta_index_reconcile_loop()
            )
            tasks.append(app_pkg._meta_index_reconcile_task)
        else:
            app_pkg._meta_index_reconcile_task = None
        yield
    finally:
        # Reached on shutdown and on a startup failure alike, so a start that
        # dies halfway cannot leak the clients it already opened.
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        app_pkg._meta_index_reconcile_task = None
        for client in clients:
            with suppress(Exception):
                await client.close()
        log.info("shutdown_complete")


app: FastAPI = FastAPI(title="llama-kv-proxy", lifespan=lifespan)
