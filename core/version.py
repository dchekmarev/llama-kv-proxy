# core/version.py

"""Single source of truth for the proxy version.

Imported by the app (the /version endpoint and /proxy/health) so operators can
identify the running build. pyproject.toml reads the version from here
(dynamic = ["version"]), so this literal is the only place it exists.
"""

__version__ = "0.0.1"
