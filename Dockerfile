# For full reproducibility pin to a digest, e.g. python:3.11-slim@sha256:...
FROM python:3.11-slim

# Keep in sync with version.py (the single source of truth).
ARG VERSION=0.0.1
LABEL version=${VERSION}

# Run as a non-root user matching the host user (pass via build args) so mounted
# volumes (kv_meta, /bin_cache) are writable without chown. Defaults to 1000:1000.
ARG UID=1000
ARG GID=1000

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Create a user/group matching the host UID/GID (skip if already present) and
# drop privileges. USER accepts numeric IDs, so this works even if the IDs were
# pre-existing in the base image.
RUN set -eux; \
    if ! getent group ${GID} >/dev/null; then groupadd -g ${GID} appuser; fi; \
    if ! getent passwd ${UID} >/dev/null; then useradd -u ${UID} -g ${GID} -d /app -s /usr/sbin/nologin appuser; fi; \
    chown -R ${UID}:${GID} /app
USER ${UID}:${GID}

EXPOSE 8080

# Probe the port the app actually listens on (PORT env, default 8081).
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=20s \
    CMD python -c "import json,os,sys,urllib.request; p=os.environ.get('PORT','8081'); d=json.load(urllib.request.urlopen(f'http://localhost:{p}/proxy/health')); sys.exit(0 if d.get('ok') else 1)"

CMD ["python", "llama_kv_proxy.py"]
