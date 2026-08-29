# proxycache.py

"""
Точка запуска uvicorn.

IMPORTANT: run with a single worker (the default). The slot manager keeps
per-process state (locks, LRU marks); multiple workers would each track
slots independently and could route two requests to the same slot.
"""

import uvicorn

from app import app
from config import LOG_LEVEL, PORT

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level=LOG_LEVEL.lower())
