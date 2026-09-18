from __future__ import annotations

import asyncio
import hashlib
import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

APP_VERSION = "0.3.0-indicf5-async"
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "").strip()
ENGINE_TIMEOUT = float(os.getenv("ENGINE_START_TIMEOUT", "180"))
REQUEST_TIMEOUT = float(os.getenv("ENGINE_REQUEST_TIMEOUT", "900"))
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "3600"))
BASE_DIR = Path("/app")
ENGINE = "indicf5"
ENGINE_PORT = 9102
ENGINE_PYTHON = "/opt/venvs/indicf5/bin/python"
JOB_DIR = Path("/tmp/voice-studio/jobs")
JOB_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Voice Studio IndicF5 GPU Worker", version=APP_VERSION)


class GenerateRequest(BaseModel):
    job_id: str = Field(min_length=1, max_length=128)
    profile_id: str = Field(default="", max_length=128)
    profile_name: str = Field(default="", max_length=160)
    text: str = Field(min_length=1, max_length=60000)
    language: str = Field(min_length=1, max_length=24)
    engine: Literal["indicf5"] = "indicf5"
    preset: Literal["fast", "narration", "expressive"] = "narration"
    speed: float = Field(default=1.0, ge=0.7, le=1.3)
    settings: dict[str, Any] = Field(default_factory=dict)
    reference_language: str = Field(default="auto", max_length=24)
    reference_text: str = Field(default="", max_length=5000)
    sample_wav_base64: str = Field(min_length=4, max_length=18000000)
    output_format: Literal["wav"] = "wav"


class EngineManager:
    def __init__(self) -> None:
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

    async def ensure(self) -> str:
        async with self.lock:
            if self.proc and self.proc.poll() is None:
                return f"http://127.0.0.1:{ENGINE_PORT}"
            self._stop()
            env = os.environ.copy()
            env["ENGINE_NAME"] = ENGINE
            env["ENGINE_PORT"] = str(ENGINE_PORT)
            self.proc = subprocess.Popen([ENGINE_PYTHON, str(BASE_DIR / "engine_server.py")], env=env)
            url = f"http://127.0.0.1:{ENGINE_PORT}"
            deadline = time.monotonic() + ENGINE_TIMEOUT
            async with httpx.AsyncClient(timeout=2.0) as client:
                while time.monotonic() < deadline:
                    if self.proc.poll() is not None:
                        code = self.proc.returncode
                        self._stop()
                        raise RuntimeError(f"IndicF5 engine process exited with code {code}")
                    try:
                        r = await client.get(url + "/healthz")
                        if r.status_code == 200:
                            return url
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)
            self._stop()
            raise RuntimeError("IndicF5 engine did not become healthy in time")


@dataclass
class AsyncJob:
    job_id: str
    fingerprint: str
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""
    result_path: str = ""
    payload: GenerateRequest | None = None

    def public(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "status": self.status,
            "engine": ENGINE,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result_available": bool(self.status == "completed" and self.result_path),
            "error": self.error if self.status == "failed" else "",
        }


manager = EngineManager()
jobs: dict[str, AsyncJob] = {}
queue: asyncio.Queue[str] = asyncio.Queue()
runner_task: asyncio.Task | None = None


def _auth(authorization: str | None) -> None:
    if not WORKER_TOKEN:
        return
    if authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _fingerprint(req: GenerateRequest) -> str:
    data = req.model_dump()
    # Hash the reference audio instead of duplicating it in the canonical JSON.
    sample = data.pop("sample_wav_base64")
    data["sample_sha256"] = hashlib.sha256(sample.encode("ascii")).hexdigest()
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _safe_result_path(job_id: str) -> Path:
    digest = hashlib.sha256(job_id.encode("utf-8")).hexdigest()[:24]
    return JOB_DIR / f"{digest}.wav"


def _purge_old_jobs() -> None:
    cutoff = time.time() - JOB_TTL_SECONDS
    for job_id, job in list(jobs.items()):
        if job.status not in {"completed", "failed"} or job.updated_at >= cutoff:
            continue
        if job.result_path:
            Path(job.result_path).unlink(missing_ok=True)
        jobs.pop(job_id, None)


async def _run_one(job: AsyncJob) -> None:
    req = job.payload
    if req is None:
        raise RuntimeError("job payload is unavailable")
    job.status = "loading_engine"
    job.updated_at = time.time()
    base = await manager.ensure()
    job.status = "generating"
    job.updated_at = time.time()
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
        r = await client.post(base + "/generate", json=req.model_dump())
    if r.status_code >= 400:
        detail = r.text[:4000].strip()
        raise RuntimeError(f"IndicF5 engine HTTP {r.status_code}: {detail}")
    if not r.content:
        raise RuntimeError("IndicF5 returned an empty audio response")
    result = _safe_result_path(job.job_id)
    result.write_bytes(r.content)
    job.result_path = str(result)
    job.status = "completed"
    job.updated_at = time.time()


async def _runner() -> None:
    while True:
        job_id = await queue.get()
        try:
            job = jobs.get(job_id)
            if job is None or job.status != "queued":
                continue
            try:
                await _run_one(job)
            except Exception as exc:
                job.status = "failed"
                job.error = str(exc)[:4000]
                job.updated_at = time.time()
            finally:
                # Never retain reference audio after execution.
                job.payload = None
        finally:
            queue.task_done()


@app.on_event("startup")
async def startup() -> None:
    global runner_task
    if runner_task is None or runner_task.done():
        runner_task = asyncio.create_task(_runner())


@app.on_event("shutdown")
async def shutdown() -> None:
    global runner_task
    if runner_task is not None:
        runner_task.cancel()
        try:
            await runner_task
        except asyncio.CancelledError:
            pass
        runner_task = None
    manager._stop()


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "service": "voice-studio-gpu-worker",
        "version": APP_VERSION,
        "engine": ENGINE,
        "engine_process_running": bool(manager.proc and manager.proc.poll() is None),
        "worker_token_configured": bool(WORKER_TOKEN),
        "async_jobs": True,
    }


@app.get("/readyz")
async def readyz():
    return {
        "ok": True,
        "engine": ENGINE,
        "accepting_jobs": True,
        "queued_jobs": queue.qsize(),
    }


@app.post("/v1/jobs", status_code=202)
async def submit_job(req: GenerateRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)
    _purge_old_jobs()
    if not req.reference_text.strip():
        raise HTTPException(status_code=400, detail="IndicF5 requires reference_text")
    fp = _fingerprint(req)
    existing = jobs.get(req.job_id)
    if existing is not None:
        if existing.fingerprint != fp:
            raise HTTPException(status_code=409, detail="job_id already exists with a different payload")
        return existing.public()
    job = AsyncJob(job_id=req.job_id, fingerprint=fp, payload=req)
    jobs[job.job_id] = job
    await queue.put(job.job_id)
    return job.public()


@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    _purge_old_jobs()
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job.public()


@app.get("/v1/jobs/{job_id}/result")
async def job_result(job_id: str, authorization: str | None = Header(default=None)):
    _auth(authorization)
    _purge_old_jobs()
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job.status == "failed":
        raise HTTPException(status_code=409, detail=job.error or "job failed")
    if job.status != "completed" or not job.result_path:
        raise HTTPException(status_code=409, detail=f"job is {job.status}")
    path = Path(job.result_path)
    if not path.is_file():
        raise HTTPException(status_code=410, detail="job result expired")
    return FileResponse(path, media_type="audio/wav", filename=f"{job_id}.wav")


# Compatibility endpoint. VPS3 production will use the async job endpoints above.
@app.post("/v1/clone/generate")
async def generate(req: GenerateRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)
    if not req.reference_text.strip():
        raise HTTPException(status_code=400, detail="IndicF5 requires reference_text")
    try:
        base = await manager.ensure()
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            r = await client.post(base + "/generate", json=req.model_dump())
        if r.status_code >= 400:
            raise HTTPException(status_code=r.status_code, detail=r.text[:4000])
        return Response(content=r.content, media_type="audio/wav")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)[:4000])
