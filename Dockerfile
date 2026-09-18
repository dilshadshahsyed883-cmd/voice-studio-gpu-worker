# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

ARG INDICF5_COMMIT=13f7c4d627cc10111aea8fe9c0039462cacacdc7
ARG INDICF5_MODEL_REVISION=ba85abedf18dc479a447eaa0eccbd76ab78a47d5
ARG VOCOS_MODEL_REVISION=0feb3fdd929bcd6649e0e7c5a688cf7dd012ef21

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache \
    HF_HUB_CACHE=/opt/hf-cache/hub \
    HUGGINGFACE_HUB_CACHE=/opt/hf-cache/hub \
    HF_MODULES_CACHE=/opt/hf-cache/modules \
    TORCH_HOME=/opt/torch-cache \
    TORCHDYNAMO_DISABLE=1 \
    INDICF5_REFERENCE_MIN_SECONDS=2 \
    INDICF5_REFERENCE_MAX_SECONDS=15 \
    INDICF5_CHUNK_MAX_BYTES=180 \
    ENGINE_CHUNK_TIMEOUT=240 \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg git curl ca-certificates python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-gateway.txt requirements-indicf5.txt /tmp/

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r /tmp/requirements-gateway.txt \
    && python -m venv --system-site-packages /opt/venvs/indicf5 \
    && /opt/venvs/indicf5/bin/pip install --upgrade pip setuptools wheel \
    && /opt/venvs/indicf5/bin/pip install -r /tmp/requirements-indicf5.txt \
    && /opt/venvs/indicf5/bin/pip install --no-deps "git+https://github.com/AI4Bharat/IndicF5.git@\${INDICF5_COMMIT}"

# Keep the known Vocos/PyTorch meta-tensor compatibility fix in the image.
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
if 'vocoder = vocoder.to_empty(device="cpu")' not in text:
    if needle not in text:
        raise SystemExit(f"IndicF5 Vocos patch precondition not found in {p}")
    p.write_text(text.replace(needle, replacement, 1))
print(f"patched IndicF5 Vocos loader: {p}")
PY

COPY app /app
RUN mkdir -p /opt/hf-cache/hub /opt/hf-cache/modules /opt/torch-cache /opt/models /tmp/voice-studio \
    && chmod 0777 /tmp/voice-studio

# Bundle exact immutable model revisions. The HF token is build-only and never
# copied into the image.
RUN --mount=type=secret,id=hf_token \
    INDICF5_MODEL_REVISION="\${INDICF5_MODEL_REVISION}" \
    VOCOS_MODEL_REVISION="\${VOCOS_MODEL_REVISION}" \
    /opt/venvs/indicf5/bin/python - <<'PY'
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

token_path = Path("/run/secrets/hf_token")
if not token_path.is_file() or not token_path.read_text().strip():
    raise SystemExit("HF_TOKEN build secret is required to bundle gated ai4bharat/IndicF5")
token = token_path.read_text().strip()
cache = "/opt/hf-cache/hub"
indic_sha = os.environ["INDICF5_MODEL_REVISION"]
vocos_sha = os.environ["VOCOS_MODEL_REVISION"]

indic_snapshot = Path(snapshot_download(
    repo_id="ai4bharat/IndicF5",
    revision=indic_sha,
    cache_dir=cache,
    token=token,
))
snapshot_download(
    repo_id="charactr/vocos-mel-24khz",
    revision=vocos_sha,
    cache_dir=cache,
)

# Version 4: remove upstream runtime torch.compile calls. The previous runtime
# spawned persistent Inductor workers and could keep a 3090 busy for tens of
# minutes on short jobs.
model_py = indic_snapshot / "model.py"
text = model_py.read_text()
old_vocoder = '        self.vocoder = torch.compile(load_vocoder(vocoder_name="vocos", is_local=False, device=device))'
new_vocoder = '        self.vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=device)'
old_model = """        self.ema_model = torch.compile(load_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device
            )
        )"""
new_model = """        self.ema_model = load_model(
                DiT,
                dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
                mel_spec_type="vocos",
                vocab_file=vocab_path,
                device=device
            )"""
if old_vocoder not in text:
    raise SystemExit("IndicF5 v4 patch precondition failed for vocoder torch.compile")
if old_model not in text:
    raise SystemExit("IndicF5 v4 patch precondition failed for model torch.compile")
text = text.replace(old_vocoder, new_vocoder, 1).replace(old_model, new_model, 1)
if "torch.compile(" in text:
    raise SystemExit("IndicF5 v4 patch incomplete: torch.compile remains in model.py")
model_py.write_text(text)

# Unqualified helper lookups resolve to the exact bundled revisions offline.
for repo_id, sha in (
    ("ai4bharat/IndicF5", indic_sha),
    ("charactr/vocos-mel-24khz", vocos_sha),
):
    repo_cache = Path(cache) / ("models--" + repo_id.replace("/", "--"))
    refs = repo_cache / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(sha)

Path("/opt/models/bundle.json").write_text(json.dumps({
    "engine": "indicf5",
    "offline_bundle": True,
    "runtime_mode": "eager_v4",
    "torch_compile": False,
    "indicf5_repo": "ai4bharat/IndicF5",
    "indicf5_revision": indic_sha,
    "vocoder_repo": "charactr/vocos-mel-24khz",
    "vocoder_revision": vocos_sha,
}, indent=2))
PY

RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
    /opt/venvs/indicf5/bin/python /app/verify_bundle.py

ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    INDICF5_OFFLINE_BUNDLE=1 \
    INDICF5_RUNTIME_MODE=eager_v4

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD curl -g -fsS "http://[::1]:\${PORT}/healthz" || exit 1

CMD ["bash","-lc","/opt/venvs/indicf5/bin/python /app/verify_bundle.py && exec uvicorn main:app --host :: --port \${PORT} --workers 1"]
