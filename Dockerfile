# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# -------- Stage 1: dependencies (build venv) --------
FROM ${PYTHON_IMAGE} AS dependencies

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    GGML_CUDA=OFF

# System libs for building and for Python wheels that link to audio backends
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        pkg-config \
        # PortAudio/ALSA for sounddevice (mic runtime and import check)
        libasound2-dev \
        libportaudio2 \
        portaudio19-dev \
        # libsndfile for soundfile (file-fallback decode)
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Install Python deps
COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
 && python -m pip install --requirement /tmp/requirements.txt \
 && python -m pip check

# Sanity import check including pywhispercpp and soundfile
RUN python - <<'PY'
import fastapi, uvicorn, numpy, sounddevice, soundfile
import pywhispercpp, faster_whisper
print("Imports OK")
PY


# -------- Stage 2: runtime --------
FROM ${PYTHON_IMAGE} AS runtime

ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HOME=/home/slaidshow \
    XDG_CACHE_HOME=/app/models/cache \
    HF_HOME=/app/models/huggingface \
    HF_HUB_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TRANSFORMERS_OFFLINE=1 \
    CONFIG_PATH=/app/config/generated.yaml \
    APP_BIND_HOST=0.0.0.0 \
    APP_BIND_PORT=8080 \
    APP_FASTER_WHISPER_DEVICE=cpu \
    APP_FASTER_WHISPER_COMPUTE_TYPE=int8 \
    APP_FASTER_WHISPER_DOWNLOAD_ROOT=/app/models/faster-whisper \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2

# Runtime system libs: audio + BLAS/OpenMP
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libstdc++6 \
        libgomp1 \
        # audio stack for mic + file decoding
        libasound2 \
        libportaudio2 \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 slaidshow \
    && useradd --uid 10001 --gid slaidshow \
         --create-home --shell /usr/sbin/nologin slaidshow

# Bring venv from builder
COPY --from=dependencies /opt/venv /opt/venv

WORKDIR /app

# Copy source with a restrictive context; avoid models/secrets in layers
RUN --mount=type=bind,source=.,target=/source,readonly \
    tar --create --file=- --directory=/source \
        --exclude='.git' \
        --exclude='.env' \
        --exclude='.env.*' \
        --exclude='*.pem' \
        --exclude='*.key' \
        --exclude='.venv' \
        --exclude='venv' \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.pytest_cache' \
        --exclude='.ruff_cache' \
        --exclude='models' \
        --exclude='recordings' \
        --exclude='outputs' \
        --exclude='config/hardware.json' \
        --exclude='config/generated.yaml' \
        . \
    | tar --extract --file=- --directory=/app --no-same-owner

# Verify entrypoint and critical imports (now within runtime image)
RUN test -f /app/app.py \
 && mkdir -p /app/config /app/models /app/recordings /app/outputs \
 && chown -R slaidshow:slaidshow /app \
 && python - <<'PY'
import fastapi, uvicorn, numpy, sounddevice, soundfile
import pywhispercpp, faster_whisper
print("Runtime imports OK")
PY

USER slaidshow:slaidshow

EXPOSE 8080

# Simple HTTP health; does not verify model availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD python - <<'PY' || exit 1
import os, urllib.request
port=os.environ.get('APP_BIND_PORT','8080')
with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=8) as r:
    pass
print("healthy")
PY

STOPSIGNAL SIGTERM

CMD ["sh", "-c", "exec python -m uvicorn app:app --host \"${APP_BIND_HOST}\" --port \"${APP_BIND_PORT}\" --workers 1 --timeout-graceful-shutdown 30"]
