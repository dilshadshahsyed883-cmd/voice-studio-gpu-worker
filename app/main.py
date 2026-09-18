from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import re
import signal
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

APP_VERSION = "0.4.0-indicf5-eager"
WORKER_TOKEN = os.getenv("WORKER_TOKEN", "").strip()
ENGINE_TIMEOUT = float(os.getenv("ENGINE_START_TIMEOUT", "180"))
CHUNK_TIMEOUT = float(os.getenv("ENGINE_CHUNK_TIMEOUT", "240"))
CHUNK_MAX_BYTES = int(os.getenv("INDICF5_CHUNK_MAX_BYTES", "180"))
REFERENCE_MIN_SECONDS = float(os.getenv("INDICF5_REFERENCE_MIN_SECONDS", "2"))
REFERENCE_MAX_SECONDS = float(os.getenv("INDICF5_REFERENCE_MAX_SECONDS", "15"))
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
        self.ready = False

    def is_running(self) -> bool:
        return bool(self.proc and self.proc.poll() is None)

    def is_ready(self) -> bool:
        return self.ready and self.is_running()

    def _stop(self) -> None:
        proc = self.proc
        self.proc = None
        self.ready = False
        if not proc or proc.poll() is not None:
            return

        # Engine gets its own process group. Always stop the whole group so
        # PyTorch/Inductor/compiler children can never survive a restart.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=8)
            return
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    async def ensure(self) -> str:
        async with self.lock:
            url = f"http://127.0.0.1:{ENGINE_PORT}"
            if self.is_ready():
                return url

            if not self.is_running():
                self._stop()
                env = os.environ.copy()
                env["ENGINE_NAME"] = ENGINE
                env["ENGINE_PORT"] = str(ENGINE_PORT)
                env["TORCHDYNAMO_DISABLE"] = "1"
                self.proc = subprocess.Popen(
                    [ENGINE_PYTHON, str(BASE_DIR / "engine_server.py")],
                    env=env,
                    start_new_session=True,
                )
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
                            data = r.json()
                            if data.get("ok") is True and data.get("model_loaded") is True:
                                self.ready = True
                                return url
                    except Exception:
                        pass
                    await asyncio.sleep(1.0)

            self._stop()
            raise RuntimeError("IndicF5 engine/model did not become ready in time")


@dataclass
class AsyncJob:
    job_id: str
    fingerprint: str
    status: str = "queued"
    stage: str = "queued"
    current_chunk: int = 0
    total_chunks: int = 0
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str = ""
    result_path: str = ""
    payload: GenerateRequest | None = None

    def public(self) -> dict[str, Any]:
        pct = 0
        if self.total_chunks:
            pct = min(100, int((self.current_chunk / self.total_chunks) * 100))
        if self.status == "completed":
            pct = 100
        return {
            "job_id": self.job_id,
            "status": self.status,
            "stage": self.stage,
            "engine": ENGINE,
            "current_chunk": self.current_chunk,
            "total_chunks": self.total_chunks,
            "progress_percent": pct,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "result_available": bool(self.status == "completed" and self.result_path),
            "error": self.error if self.status == "failed" else "",
        }


manager = EngineManager()
jobs: dict[str, AsyncJob] = {}
queue: asyncio.Queue[str] = asyncio.Queue()
runner_task: asyncio.Task | None = None
warm_task: asyncio.Task | None = None
warm_error = ""


def _auth(authorization: str | None) -> None:
    if not WORKER_TOKEN:
        return
    if authorization != f"Bearer {WORKER_TOKEN}":
        raise HTTPException(status_code=401, detail="unauthorized")


def _fingerprint(req: GenerateRequest) -> str:
    data = req.model_dump()
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


def _split_piece(piece: str, max_bytes: int) -> list[str]:
    piece = piece.strip()
    if not piece:
        return []
    if len(piece.encode("utf-8")) <= max_bytes:
        return [piece]

    out: list[str] = []
    buf = ""
    for word in re.split(r"(\s+)", piece):
        if not word:
            continue
        candidate = buf + word
        if buf and len(candidate.encode("utf-8")) > max_bytes:
            out.append(buf.strip())
            buf = word.lstrip()
            if len(buf.encode("utf-8")) <= max_bytes:
                continue

        if len(buf.encode("utf-8")) > max_bytes:
            small = ""
            for ch in buf:
                if small and len((small + ch).encode("utf-8")) > max_bytes:
                    out.append(small)
                    small = ch
                else:
                    small += ch
            buf = small

    if buf.strip():
        out.append(buf.strip())
    return out


def _split_text(text: str, max_bytes: int) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    pieces = re.split(r"(?<=[.!?।！？؟;:؛])\s+|(?<=[,，、])\s*", text)
    chunks: list[str] = []
    buf = ""
    for piece in pieces:
        for part in _split_piece(piece, max_bytes):
            candidate = (buf + " " + part).strip()
            if buf and len(candidate.encode("utf-8")) > max_bytes:
                chunks.append(buf)
                buf = part
            else:
                buf = candidate
    if buf:
        chunks.append(buf)
    return chunks


def _reference_wav_duration(value: str) -> float | None:
    try:
        raw = base64.b64decode(value, validate=True)
        with wave.open(io.BytesIO(raw), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.getnframes()
            return (frames / rate) if rate else None
    except Exception:
        return None


def _join_wav_parts(parts: list[bytes], silence_seconds: float = 0.12) -> bytes:
    if not parts:
        raise RuntimeError("IndicF5 produced no audio chunks")

    params = None
    all_frames: list[bytes] = []
    silence = b""
    for idx, raw in enumerate(parts):
        with wave.open(io.BytesIO(raw), "rb") as wf:
            current = (
                wf.getnchannels(),
                wf.getsampwidth(),
                wf.getframerate(),
                wf.getcomptype(),
            )
            if params is None:
                params = current
                channels, width, rate, comptype = params
                if comptype != "NONE":
                    raise RuntimeError("IndicF5 returned compressed WAV unexpectedly")
                silence_frames = int(rate * silence_seconds)
                silence = b"\x00" * silence_frames * channels * width
            elif current != params:
                raise RuntimeError("IndicF5 chunk WAV formats do not match")
            if idx:
                all_frames.append(silence)
            all_frames.append(wf.readframes(wf.getnframes()))

    channels, width, rate, _ = params
    out = io.BytesIO()
    with wave.open(out, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(width)
        wf.setframerate(rate)
        wf.setcomptype("NONE", "not compressed")
        for frames in all_frames:
            wf.writeframes(frames)
    return out.getvalue()


async def _warm_engine() -> None:
    global warm_error
    try:
        warm_error = ""
        await manager.ensure()
    except Exception as exc:
        warm_error = str(exc)[:1000]


async def _generate_chunk(base: str, payload: dict[str, Any], chunk_index: int, total: int) -> bytes:
    timeout = httpx.Timeout(CHUNK_TIMEOUT)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            r = await client.post(base + "/generate", json=payload)
    except httpx.TimeoutException as exc:
        manager._stop()
        raise RuntimeError(
            f"IndicF5 chunk {chunk_index}/{total} exceeded {int(CHUNK_TIMEOUT)}s watchdog; engine was terminated"
        ) from exc

    if r.status_code >= 400:
        detail = r.text[:4000].strip()
        raise RuntimeError(f"IndicF5 chunk {chunk_index}/{total} HTTP {r.status_code}: {detail}")
    if not r.content:
        raise RuntimeError(f"IndicF5 chunk {chunk_index}/{total} returned empty audio")
    return r.content


async def _run_one(job: AsyncJob) -> None:
    req = job.payload
    if req is None:
        raise RuntimeError("job payload is unavailable")

    chunks = _split_text(req.text, CHUNK_MAX_BYTES)
    if not chunks:
        raise RuntimeError("text produced no generation chunks")

    job.total_chunks = len(chunks)
    job.status = "loading_engine"
    job.stage = "loading_model"
    job.updated_at = time.time()
    base = await manager.ensure()

    parts: list[bytes] = []
    for index, chunk in enumerate(chunks, start=1):
        job.status = "generating"
        job.stage = f"chunk_{index}_of_{len(chunks)}"
        job.current_chunk = index - 1
        job.updated_at = time.time()

        payload = req.model_dump()
        payload["job_id"] = hashlib.sha256(f"{job.job_id}:{index}".encode("utf-8")).hexdigest()[:24]
        payload["text"] = chunk
        audio = await _generate_chunk(base, payload, index, len(chunks))
        parts.append(audio)

        job.current_chunk = index
        job.updated_at = time.time()

    job.stage = "assembling"
    joined = _join_wav_parts(parts)
    result = _safe_result_path(job.job_id)
    result.write_bytes(joined)

    job.result_path = str(result)
    job.status = "completed"
    job.stage = "completed"
    job.current_chunk = job.total_chunks
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
                job.stage = "failed"
                job.error = str(exc)[:4000]
                job.updated_at = time.time()
            finally:
                job.payload = None
        finally:
            queue.task_done()


@app.on_event("startup")
async def startup() -> None:
    global runner_task, warm_task
    if runner_task is None or runner_task.done():
        runner_task = asyncio.create_task(_runner())
    if warm_task is None or warm_task.done():
        warm_task = asyncio.create_task(_warm_engine())


@app.on_event("shutdown")
async def shutdown() -> None:
    global runner_task, warm_task
    for task in (runner_task, warm_task):
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    runner_task = None
    warm_task = None
    manager._stop()


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "service": "voice-studio-gpu-worker",
        "version": APP_VERSION,
        "engine": ENGINE,
        "engine_process_running": manager.is_running(),
        "engine_ready": manager.is_ready(),
        "worker_token_configured": bool(WORKER_TOKEN),
        "async_jobs": True,
        "torch_compile": False,
        "chunk_max_bytes": CHUNK_MAX_BYTES,
        "chunk_timeout_seconds": CHUNK_TIMEOUT,
        "reference_max_seconds": REFERENCE_MAX_SECONDS,
    }


@app.get("/readyz")
async def readyz():
    if not manager.is_ready():
        return JSONResponse(
            status_code=503,
            content={
                "ok": False,
                "engine": ENGINE,
                "accepting_jobs": False,
                "stage": "loading_model" if not warm_error else "engine_error",
                "error": warm_error,
            },
        )
    return {
        "ok": True,
        "engine": ENGINE,
        "accepting_jobs": True,
        "engine_warm": True,
        "queued_jobs": queue.qsize(),
    }


@app.post("/v1/jobs", status_code=202)
async def submit_job(req: GenerateRequest, authorization: str | None = Header(default=None)):
    _auth(authorization)
    _purge_old_jobs()
    if not req.reference_text.strip():
        raise HTTPException(status_code=400, detail="IndicF5 requires reference_text")

    duration = _reference_wav_duration(req.sample_wav_base64)
    if duration is not None and not (REFERENCE_MIN_SECONDS <= duration <= REFERENCE_MAX_SECONDS):
        raise HTTPException(
            status_code=422,
            detail=(
                f"IndicF5 reference audio is {duration:.2f}s; use a clean "
                f"{REFERENCE_MIN_SECONDS:g}-{REFERENCE_MAX_SECONDS:g}s WAV with its exact transcript"
            ),
        )

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


@app.post("/v1/clone/generate")
async def synchronous_generation_disabled(
    req: GenerateRequest,
    authorization: str | None = Header(default=None),
):
    _auth(authorization)
    raise HTTPException(
        status_code=410,
        detail="Synchronous IndicF5 generation is disabled in v4; submit to /v1/jobs and poll status",
    )
