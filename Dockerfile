# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=python:3.12-slim-bookworm

# Build native dependencies without retaining compilers in the runtime image.
FROM ${PYTHON_IMAGE} AS dependencies

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2 \
    GGML_CUDA=OFF

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        git \
        libasound2-dev \
        libportaudio2 \
        portaudio19-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt

RUN python -m pip install --requirement /tmp/requirements.txt \
    && python -m pip check


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

# CUDA belongs to the separate Ollama and ComfyUI services.
# PortAudio and ALSA support host audio devices passed through by Compose.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libasound2 \
        libgomp1 \
        libportaudio2 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 slaidshow \
    && useradd --uid 10001 --gid slaidshow \
        --create-home --shell /usr/sbin/nologin slaidshow

COPY --from=dependencies /opt/venv /opt/venv

WORKDIR /app

# Copy source without storing local secrets, environments or model assets
# in image layers. A restrictive .dockerignore should additionally limit
# the build context sent to the builder.
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

RUN test -f /app/app.py \
    && mkdir -p /app/config /app/models /app/recordings /app/outputs \
    && chown -R slaidshow:slaidshow /app \
    && python -c "import fastapi, uvicorn, sounddevice, faster_whisper"

USER slaidshow:slaidshow

EXPOSE 8080

# This checks the HTTP endpoint, not model availability or GPU readiness.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('APP_BIND_PORT', '8080') + '/health', timeout=8).close()"

STOPSIGNAL SIGTERM

# One worker owns the microphone, show state and in-process scheduling.
# Runtime overrides remain possible through the corresponding variables.
CMD ["sh", "-c", "exec python -m uvicorn app:app --host \"${APP_BIND_HOST}\" --port \"${APP_BIND_PORT}\" --workers 1 --timeout-graceful-shutdown 30"]
