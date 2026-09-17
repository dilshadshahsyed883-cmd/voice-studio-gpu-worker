from __future__ import annotations

import base64
import io
import os
import re
import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

ENGINE = os.environ["ENGINE_NAME"].strip().lower()
PORT = int(os.getenv("ENGINE_PORT", "9101"))
CACHE_DIR = Path(os.getenv("HF_HOME", "/models/huggingface"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)
TMP = Path("/tmp/voice-studio")
TMP.mkdir(parents=True, exist_ok=True)
app = FastAPI(title=f"Voice Studio {ENGINE} engine")

_model = None
_model_lock = threading.Lock()

CHATTERBOX_LANGS = {"ar","da","de","el","en","es","fi","fr","he","hi","it","ja","ko","ms","nl","no","pl","pt","ru","sv","sw","tr","zh"}
INDICF5_LANGS = {"as","bn","gu","hi","kn","ml","mr","or","pa","ta","te"}

class Request(BaseModel):
    job_id: str
    text: str = Field(min_length=1, max_length=60000)
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


def split_text(text: str, limit: int = 800) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return [text]
    sentences = re.split(r"(?<=[.!?।！？؟])\s+", text)
    chunks, buf = [], ""
    for sentence in sentences:
        if not sentence:
            continue
        parts = [sentence[i:i+limit] for i in range(0, len(sentence), limit)] if len(sentence) > limit else [sentence]
        for part in parts:
            candidate = (buf + " " + part).strip()
            if buf and len(candidate) > limit:
                chunks.append(buf)
                buf = part
            else:
                buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def load_model():
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is not None:
            return _model
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is not available")
        if ENGINE == "chatterbox":
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            _model = ChatterboxMultilingualTTS.from_pretrained(device="cuda", t3_model="v3")
        elif ENGINE == "indicf5":
            from transformers import AutoModel
            _model = AutoModel.from_pretrained("ai4bharat/IndicF5", trust_remote_code=True, cache_dir=str(CACHE_DIR))
        else:
            raise RuntimeError(f"unsupported engine {ENGINE}")
        return _model


def decode_reference(value: str, job_id: str) -> Path:
    raw = base64.b64decode(value, validate=True)
    if not raw or len(raw) > 12 * 1024 * 1024:
        raise ValueError("reference audio size is invalid")
    path = TMP / f"{job_id}-reference.wav"
    path.write_bytes(raw)
    return path


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
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-i",str(src),"-filter:a",f"atempo={speed:.4f}",str(dst)], check=True, timeout=120)
        return dst.read_bytes()
    finally:
        src.unlink(missing_ok=True)
        dst.unlink(missing_ok=True)


def concat(parts: list[np.ndarray], sr: int) -> np.ndarray:
    if len(parts) == 1:
        return parts[0]
    silence = np.zeros(int(sr * 0.14), dtype=np.float32)
    joined = []
    for i, part in enumerate(parts):
        if i:
            joined.append(silence)
        joined.append(np.asarray(part, dtype=np.float32).squeeze())
    return np.concatenate(joined)


def generate_chatterbox(req: Request, ref: Path) -> tuple[np.ndarray, int]:
    lang = lang_base(req.language)
    if lang not in CHATTERBOX_LANGS:
        raise ValueError(f"language {lang} is not supported by Chatterbox")
    model = load_model()
    settings = req.settings or {}
    defaults = {"fast": (0.4,0.45), "narration": (0.5,0.5), "expressive": (0.75,0.3)}
    ex, cfg = defaults.get(req.preset, defaults["narration"])
    kwargs = {
        "audio_prompt_path": str(ref),
        "exaggeration": float(settings.get("exaggeration", ex)),
        "cfg_weight": float(settings.get("cfg_weight", cfg)),
        "temperature": float(settings.get("temperature", 0.8)),
        "repetition_penalty": float(settings.get("repetition_penalty", 1.2)),
        "min_p": float(settings.get("min_p", 0.05)),
        "top_p": float(settings.get("top_p", 1.0)),
    }
    parts = []
    for chunk in split_text(req.text, 700):
        wav = model.generate(chunk, language_id=lang, **kwargs)
        parts.append(wav.squeeze().detach().cpu().numpy().astype(np.float32))
    return concat(parts, int(model.sr)), int(model.sr)


def generate_indicf5(req: Request, ref: Path) -> tuple[np.ndarray, int]:
    lang = lang_base(req.language)
    if lang not in INDICF5_LANGS:
        raise ValueError(f"language {lang} is not supported by IndicF5")
    if not req.reference_text.strip():
        raise ValueError("IndicF5 requires reference_text")
    model = load_model()
    parts = []
    for chunk in split_text(req.text, 650):
        audio = model(chunk, ref_audio_path=str(ref), ref_text=req.reference_text)
        arr = np.asarray(audio)
        if arr.dtype == np.int16:
            arr = arr.astype(np.float32) / 32768.0
        else:
            arr = arr.astype(np.float32)
        parts.append(arr.squeeze())
    return concat(parts, 24000), 24000


@app.get("/healthz")
def healthz():
    return {"ok": True, "engine": ENGINE, "cuda_available": torch.cuda.is_available(), "model_loaded": _model is not None}


@app.post("/generate")
def generate(req: Request):
    if req.engine != ENGINE:
        raise HTTPException(400, f"request engine {req.engine} does not match process engine {ENGINE}")
    ref = None
    try:
        ref = decode_reference(req.sample_wav_base64, req.job_id)
        if ENGINE == "chatterbox":
            audio, sr = generate_chatterbox(req, ref)
        elif ENGINE == "indicf5":
            audio, sr = generate_indicf5(req, ref)
        else:
            raise RuntimeError(f"unsupported engine {ENGINE}")
        return Response(content=wav_bytes(audio, sr, req.speed, req.job_id), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(500, str(exc)[:1000])
    finally:
        if ref:
            ref.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level=os.getenv("LOG_LEVEL","info"))
