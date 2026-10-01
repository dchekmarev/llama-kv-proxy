# tests/test_logging_setup.py

"""Root logging setup lives outside config, so it is configured once and stays
idempotent no matter which launch mode called it."""

import logging

from core.logging_setup import setup_logging


def test_setup_logging_is_idempotent():
    """Calling setup_logging twice must not stack duplicate handlers."""
    setup_logging("INFO")
    n = len(logging.getLogger().handlers)
    setup_logging("DEBUG")
    assert len(logging.getLogger().handlers) == n


def test_setup_logging_attaches_request_id_filter():
    """Every root handler carries the request-id filter, so records correlate."""
    from core.request_id import RequestIdFilter

    setup_logging("INFO")
    for handler in logging.getLogger().handlers:
        assert any(isinstance(f, RequestIdFilter) for f in handler.filters)
