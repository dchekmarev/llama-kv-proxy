# version.py

"""Single source of truth for the proxy version.

Imported by the app (the /version endpoint and /proxy/health) so operators can
identify the running build. Keep in sync with the Dockerfile VERSION arg.
"""

__version__ = "0.0.1"
