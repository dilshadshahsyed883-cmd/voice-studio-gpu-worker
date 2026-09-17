FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG CHATTERBOX_COMMIT=5de7a54aa4e5e2baadb0182dde554908b48b85c2
ARG INDICF5_COMMIT=13f7c4d627cc10111aea8fe9c0039462cacacdc7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    TORCH_HOME=/models/torch \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gateway.txt requirements-chatterbox.txt requirements-indicf5.txt /tmp/

# Share the CUDA-enabled torch/torchaudio from the PyTorch base image, but isolate
# incompatible Transformers versions in separate virtual environments.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-gateway.txt \
    && python -m venv --system-site-packages /opt/venvs/chatterbox \
    && /opt/venvs/chatterbox/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venvs/chatterbox/bin/pip install -r /tmp/requirements-chatterbox.txt \
    && /opt/venvs/chatterbox/bin/pip install --no-deps "git+https://github.com/resemble-ai/chatterbox.git@${CHATTERBOX_COMMIT}" \
    && python -m venv --system-site-packages /opt/venvs/indicf5 \
    && /opt/venvs/indicf5/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venvs/indicf5/bin/pip install -r /tmp/requirements-indicf5.txt \
    && /opt/venvs/indicf5/bin/pip install --no-deps "git+https://github.com/AI4Bharat/IndicF5.git@${INDICF5_COMMIT}"

COPY app /app
RUN mkdir -p /models/huggingface /models/torch /tmp/voice-studio \
    && chmod 0777 /models/huggingface /models/torch /tmp/voice-studio

EXPOSE 8000
# Salad Container Gateway routes traffic over IPv6, so the public gateway process
# must listen on ::. Keep the internal engine subprocesses on loopback IPv4.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -g -fsS "http://[::1]:${PORT}/healthz" || exit 1

CMD ["bash","-lc","uvicorn main:app --host :: --port ${PORT} --workers 1"]
