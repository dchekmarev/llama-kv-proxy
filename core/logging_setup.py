# core/logging_setup.py

"""Root logger configuration for the proxy.

Lives outside config.py so importing config stays free of side effects and
logging concerns. Called from the entry point and from the app lifespan so
logging is set up regardless of the launch mode (python llama_kv_proxy.py or
uvicorn app:app).
"""

import logging

from .request_id import RequestIdFilter

_logging_configured = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once (idempotent).

    The request-id filter is attached to the root handlers so every record
    carries the current request's correlation id (empty outside a request).
    """
    global _logging_configured
    if _logging_configured:
        return
    root = logging.getLogger()
    logging.basicConfig(
        level=level.upper(),
        format="%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s",
    )
    for handler in root.handlers:
        handler.addFilter(RequestIdFilter())
    _logging_configured = True
