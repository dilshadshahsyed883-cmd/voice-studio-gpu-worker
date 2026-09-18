from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

ENGINE = "indicf5"
PORT = int(os.getenv("ENGINE_PORT", "9102"))
HF_CACHE_DIR = Path(os.getenv("HF_HUB_CACHE", "/opt/hf-cache/hub"))
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/voice-studio")
TMP.mkdir(parents=True, exist_ok=True)
REFERENCE_MIN_SECONDS = float(os.getenv("INDICF5_REFERENCE_MIN_SECONDS", "2"))
REFERENCE_MAX_SECONDS = float(os.getenv("INDICF5_REFERENCE_MAX_SECONDS", "15"))
app = FastAPI(title="Voice Studio IndicF5 engine")

_model = None
_model_lock = threading.Lock()
INDICF5_LANGS = {"as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"}


class Request(BaseModel):
    job_id: str
    text: str = Field(min_length=1, max_length=2000)
    language: str
    engine: str
    preset: str = "narration"
    speed: float = Field(default=1.0, ge=0.7, le=1.3)
    settings: dict[str, Any] = Field(default_factory=dict)
    reference_text: str = ""
    sample_wav_base64: str
    output_format: str = "wav"


def lang_base(value: str) -> str:
    return value.strip().lower().replace("_", "-").split("-", 1)[0]


def _disable_torch_compile() -> None:
    # Build-time source patch is the primary guard. This runtime identity
    # replacement is a second safety boundary against future upstream code.
    if getattr(torch, "_voice_studio_compile_disabled", False):
        return

    def eager_identity(model, *args, **kwargs):
        return model

    torch.compile = eager_identity  # type: ignore[assignment]
    torch._voice_studio_compile_disabled = True  # type: ignore[attr-defined]


def load_model():
    global _model
    if _model is not None:
        return _model

    with _model_lock:
        if _model is not None:
            return _model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available")

        _disable_torch_compile()
        from transformers import AutoModel

        bundle_path = Path("/opt/models/bundle.json")
        if not bundle_path.is_file():
            raise RuntimeError("IndicF5 bundle manifest is missing")
        bundle = json.loads(bundle_path.read_text())
        revision = str(bundle.get("indicf5_revision") or "").strip()
        if not revision:
            raise RuntimeError("IndicF5 bundled revision is missing")
        if bundle.get("runtime_mode") != "eager_v4":
            raise RuntimeError("IndicF5 bundle runtime mode is not eager_v4")

        _model = AutoModel.from_pretrained(
            "ai4bharat/IndicF5",
            revision=revision,
            trust_remote_code=True,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=True,
        )
        return _model


def decode_reference(value: str, job_id: str) -> tuple[Path, float]:
    raw = base64.b64decode(value, validate=True)
    if not raw or len(raw) > 12 * 1024 * 1024:
        raise ValueError("reference audio size is invalid")

    path = TMP / f"{job_id}-reference.wav"
    path.write_bytes(raw)
    try:
        info = sf.info(path)
        duration = float(info.frames / info.samplerate) if info.samplerate else 0.0
    except Exception:
        path.unlink(missing_ok=True)
        raise ValueError("reference audio is not a readable WAV")

    if duration < REFERENCE_MIN_SECONDS or duration > REFERENCE_MAX_SECONDS:
        path.unlink(missing_ok=True)
        raise ValueError(
            f"IndicF5 reference audio is {duration:.2f}s; use a clean "
            f"{REFERENCE_MIN_SECONDS:g}-{REFERENCE_MAX_SECONDS:g}s WAV with its exact transcript"
        )
    return path, duration


def wav_bytes(audio: np.ndarray, sr: int, speed: float, job_id: str) -> bytes:
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    if abs(speed - 1.0) < 0.01:
        out = io.BytesIO()
        sf.write(out, audio, sr, format="WAV", subtype="PCM_16")
        return out.getvalue()

    src = TMP / f"{job_id}-src.wav"
    dst = TMP / f"{job_id}-speed.wav"
    sf.write(src, audio, sr, subtype="PCM_16")
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(src),
                "-filter:a",
                f"atempo={speed:.4f}",
                str(dst),
            ],
            check=True,
            timeout=120,
        )
        return dst.read_bytes()
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def generate_indicf5(req: Request, ref: Path) -> tuple[np.ndarray, int]:
    lang = lang_base(req.language)
    if lang not in INDICF5_LANGS:
        raise ValueError(f"language {lang} is not supported by IndicF5")
    if not req.reference_text.strip():
        raise ValueError("IndicF5 requires reference_text")

    # v4 receives one small duration-safe chunk from the outer worker.
    model = load_model()
    audio = model(req.text, ref_audio_path=str(ref), ref_text=req.reference_text)
    arr = np.asarray(audio)
    if arr.dtype == np.int16:
        arr = arr.astype(np.float32) / 32768.0
    else:
        arr = arr.astype(np.float32)
    return arr.squeeze(), 24000


@app.on_event("startup")
def warm_model() -> None:
    load_model()


@app.get("/healthz")
def healthz():
    return {
        "ok": True,
        "engine": ENGINE,
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": _model is not None,
        "torch_compile": False,
        "runtime_mode": "eager_v4",
        "reference_max_seconds": REFERENCE_MAX_SECONDS,
    }


@app.post("/generate")
def generate(req: Request):
    if req.engine != ENGINE:
        raise HTTPException(400, f"request engine {req.engine} does not match process engine {ENGINE}")

    ref = None
    try:
        ref, _duration = decode_reference(req.sample_wav_base64, req.job_id)
        audio, sr = generate_indicf5(req, ref)
        return Response(content=wav_bytes(audio, sr, req.speed, req.job_id), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(500, str(exc)[:4000])
    finally:
        if ref:
            ref.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level=os.getenv("LOG_LEVEL", "info"))
