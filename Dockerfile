# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG INDICF5_COMMIT=13f7c4d627cc10111aea8fe9c0039462cacacdc7

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_CACHE=/opt/hf-cache/hub \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub \
    HF_MODULES_CACHE=/opt/hf-cache/modules \
    TORCH_HOME=/opt/torch-cache \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gateway.txt requirements-indicf5.txt /tmp/

# Install every Python/system dependency at image-build time.
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-gateway.txt \
    && python -m venv --system-site-packages /opt/venvs/indicf5 \
    && /opt/venvs/indicf5/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venvs/indicf5/bin/pip install -r /tmp/requirements-indicf5.txt \
    && /opt/venvs/indicf5/bin/pip install --no-deps "git+https://github.com/AI4Bharat/IndicF5.git@${INDICF5_COMMIT}"

# IndicF5 upstream currently has a known Vocos/PyTorch meta-tensor issue on
# recent CUDA/PyTorch stacks. Apply the minimal upstream-compatible fix during
# the image build so the vocoder can materialize parameters before loading its
# state dict. Fail the build if the expected upstream code is no longer present.
RUN /opt/venvs/indicf5/bin/python - <<'PY'
from pathlib import Path
import inspect
import f5_tts.infer.utils_infer as u

p = Path(inspect.getsourcefile(u))
text = p.read_text()
needle = """        vocoder = Vocos.from_hparams(config_path)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
"""
replacement = """        vocoder = Vocos.from_hparams(config_path)
        if any(param.is_meta for param in vocoder.parameters()):
            vocoder = vocoder.to_empty(device="cpu")
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
"""
if "vocoder = vocoder.to_empty(device=\"cpu\")" not in text:
    if needle not in text:
        raise SystemExit(f"IndicF5 Vocos patch precondition not found in {p}")
    p.write_text(text.replace(needle, replacement, 1))
print(f"patched IndicF5 Vocos loader: {p}")
PY

COPY app /app
RUN mkdir -p /opt/hf-cache/hub /opt/hf-cache/modules /opt/torch-cache /opt/models /tmp/voice-studio \
    && chmod 0777 /tmp/voice-studio

# IndicF5 is gated on Hugging Face. The HF token is used only as a BuildKit
# secret to fetch the model during the GitHub Actions build; it is never copied
# into an image layer. The public Vocos vocoder is bundled at the same time.
RUN --mount=type=secret,id=hf_token \
    /opt/venvs/indicf5/bin/python - <<'PY'
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download

token_path = Path("/run/secrets/hf_token")
if not token_path.is_file() or not token_path.read_text().strip():
    raise SystemExit("HF_TOKEN build secret is required to bundle gated ai4bharat/IndicF5")
token = token_path.read_text().strip()
cache = "/opt/hf-cache/hub"

api = HfApi(token=token)
indic_info = api.model_info("ai4bharat/IndicF5")
indic_sha = indic_info.sha
vocos_info = HfApi().model_info("charactr/vocos-mel-24khz")
vocos_sha = vocos_info.sha

snapshot_download(
    repo_id="ai4bharat/IndicF5",
    revision=indic_sha,
    cache_dir=cache,
    token=token,
)
snapshot_download(
    repo_id="charactr/vocos-mel-24khz",
    revision=vocos_sha,
    cache_dir=cache,
)

Path("/opt/models/bundle.json").write_text(json.dumps({
    "engine": "indicf5",
    "offline_bundle": True,
    "indicf5_repo": "ai4bharat/IndicF5",
    "indicf5_revision": indic_sha,
    "vocoder_repo": "charactr/vocos-mel-24khz",
    "vocoder_revision": vocos_sha,
}, indent=2))
PY

# Prove at build time that all runtime-critical model/vocoder files resolve
# locally, then lock normal runtime into offline mode.
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    /opt/venvs/indicf5/bin/python /app/verify_bundle.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    INDICF5_OFFLINE_BUNDLE=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -g -fsS "http://[::1]:${PORT}/healthz" || exit 1

# Fail closed if the self-contained model bundle is incomplete. No model,
# tokenizer, vocoder, Python package, or other runtime dependency is fetched here.
CMD ["bash","-lc","/opt/venvs/indicf5/bin/python /app/verify_bundle.py && exec uvicorn main:app --host :: --port ${PORT} --workers 1"]
