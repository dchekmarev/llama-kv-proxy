FROM python:3.11-slim

# Keep in sync with version.py (the single source of truth).
ARG VERSION=0.0.1
LABEL version=${VERSION}

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

EXPOSE 8080

CMD ["python", "llama_kv_proxy.py"]
