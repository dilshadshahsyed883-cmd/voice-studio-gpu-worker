from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

APP_VERSION = "0.1.0"
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "").strip()
ENGINE_TIMEOUT = float(os.getenv("ENGINE_START_TIMEOUT", "180"))
REQUEST_TIMEOUT = float(os.getenv("ENGINE_REQUEST_TIMEOUT", "600"))
BASE_DIR = Path("/app")
ENGINE_PORTS = {"chatterbox": 9101, "indicf5": 9102}
ENGINE_PYTHON = {
    "chatterbox": "/opt/venvs/chatterbox/bin/python",
    "indicf5": "/opt/venvs/indicf5/bin/python",
}

app = FastAPI(title="Voice Studio GPU Worker", version=APP_VERSION)


class GenerateRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    profile_id: str = Field(default="", max_length=128)
    profile_name: str = Field(default="", max_length=160)
    text: str = Field(min_length=1, max_length=60000)
    language: str = Field(min_length=1, max_length=24)
    engine: Literal["chatterbox", "indicf5"]
    preset: Literal["fast", "narration", "expressive"] = "narration"
    speed: float = Field(default=1.0, ge=0.7, le=1.3)
    settings: dict[str, Any] = Field(default_factory=dict)
    reference_language: str = Field(default="auto", max_length=24)
    reference_text: str = Field(default="", max_length=5000)
    sample_wav_base64: str = Field(min_length=4, max_length=18000000)
    output_format: Literal["wav"] = "wav"


class EngineManager:
    def __init__(self) -> None:
        self.engine: str | None = None
        self.proc: subprocess.Popen | None = None
        self.lock = asyncio.Lock()

    def _stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        self.proc = None
        self.engine = None

    async def ensure(self, engine: str) -> str:
        async with self.lock:
            if self.engine == engine and self.proc and self.proc.poll() is None:
                return f"http://127.0.0.1:{ENGINE_PORTS[engine]}"
            self._stop()
            port = ENGINE_PORTS[engine]
            env = os.environ.copy()
            env["ENGINE_NAME"] = engine
            env["ENGINE_PORT"] = str(port)
            cmd = [ENGINE_PYTHON[engine], str(BASE_DIR / "engine_server.py")]
            self.proc = subprocess.Popen(cmd, env=env)
            self.engine = engine
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + ENGINE_TIMEOUT
            async with httpx.AsyncClient(timeout=2.0) as client:
                while time.monotonic() < deadline:
                    if self.proc.poll() is not None:
                        code = self.proc.returncode
                        self._stop()
                        raise RuntimeError(f"{engine} engine process exited with code {code}")
                    try:
                        r = await client.get(url + "/healthz")
                        if r.status_code == 200:
                            return url
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)
            self._stop()
            raise RuntimeError(f"{engine} engine did not become healthy in time")

manager = EngineManager()


def _auth(authorization: str | None) -> None:
    if not WORKER_TOKEN:
        return
    if authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "service": "voice-studio-gpu-worker",
        "version": APP_VERSION,
        "active_engine": manager.engine,
        "worker_token_configured": bool(WORKER_TOKEN),
    }


@app.get("/readyz")
async def readyz():
    return {"ok": True, "active_engine": manager.engine}


@app.post("/v1/clone/generate")
async def generate(req: GenerateRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if req.engine == "indicf5" and not req.reference_text.strip():
        raise HTTPException(status_code=400, detail="IndicF5 requires reference_text")
    try:
        base = await manager.ensure(req.engine)
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(base + "/generate", json=req.model_dump())
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text[:1000])
        return Response(content=r.content, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:1000])
