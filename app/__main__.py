# app/__main__.py

"""Uvicorn entry point (``python -m app``).

IMPORTANT: run with a single worker (the default). The slot manager keeps
per-process state (locks, LRU marks); multiple workers would each track
slots independently and could route two requests to the same slot.
"""

import uvicorn

from core.config import LOG_LEVEL, PORT
from core.logging_setup import setup_logging

from ._asgi import app

if __name__ == "__main__":
    setup_logging(LOG_LEVEL)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())
