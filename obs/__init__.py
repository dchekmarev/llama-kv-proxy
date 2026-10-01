# obs/__init__.py

"""Observability: everything the proxy says about itself.

* :mod:`obs.metrics` -- the ``/metrics`` scrape target: the proxy's own
  registry (see :mod:`core.promstats`) followed by the backend ``/metrics``
  merged across every active backend and model.
* :mod:`obs.reqlog` -- the per-request JSON groups on disk: request,
  response, prefix, decision and the raw SSE of a stream.
* :mod:`obs.ui` -- the live request registry and the SSE broadcaster behind
  the dashboard.
* :mod:`obs.ui_page` -- the dashboard itself, one self-contained HTML
  document with no external assets.
"""

from . import metrics, reqlog, ui, ui_page

__all__ = ["metrics", "reqlog", "ui", "ui_page"]