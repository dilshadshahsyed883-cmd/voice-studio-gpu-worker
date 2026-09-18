from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download

CACHE_DIR = Path(os.getenv("HF_HUB_CACHE", "/opt/hf-cache/hub"))
MANIFEST = Path("/opt/models/bundle.json")

REQUIRED = (
    ("ai4bharat/IndicF5", "config.json"),
    ("ai4bharat/IndicF5", "model.py"),
    ("ai4bharat/IndicF5", "model.safetensors"),
    ("ai4bharat/IndicF5", "checkpoints/vocab.txt"),
    ("charactr/vocos-mel-24khz", "config.yaml"),
    ("charactr/vocos-mel-24khz", "pytorch_model.bin"),
)


def main() -> int:
    if not MANIFEST.is_file():
        raise RuntimeError(f"missing bundle manifest: {MANIFEST}")

    data = json.loads(MANIFEST.read_text())
    if data.get("engine") != "indicf5":
        raise RuntimeError("bundle manifest engine mismatch")
    if data.get("indicf5_repo") != "ai4bharat/IndicF5":
        raise RuntimeError("bundle manifest IndicF5 repo mismatch")
    if data.get("runtime_mode") != "eager_v4":
        raise RuntimeError("bundle manifest runtime mode must be eager_v4")

    resolved = []
    revisions = {
        "ai4bharat/IndicF5": data.get("indicf5_revision"),
        "charactr/vocos-mel-24khz": data.get("vocoder_revision"),
    }
    resolved_paths = {}

    for repo_id, filename in REQUIRED:
        revision = revisions.get(repo_id)
        if not revision:
            raise RuntimeError(f"missing pinned revision for {repo_id}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision,
            cache_dir=str(CACHE_DIR),
            local_files_only=True,
        )
        p = Path(path)
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"missing bundled file: {repo_id}/{filename}")
        resolved_paths[(repo_id, filename)] = p
        resolved.append({"repo": repo_id, "file": filename, "bytes": p.stat().st_size})

    model_py = resolved_paths[("ai4bharat/IndicF5", "model.py")].read_text()
    if "torch.compile(" in model_py:
        raise RuntimeError("IndicF5 model.py still contains torch.compile; eager_v4 patch was not applied")
    if "self.vocoder = load_vocoder(" not in model_py:
        raise RuntimeError("IndicF5 eager vocoder patch not found")
    if "self.ema_model = load_model(" not in model_py:
        raise RuntimeError("IndicF5 eager model patch not found")

    # Verify unqualified helper lookups also resolve entirely offline.
    for repo_id, filename in REQUIRED:
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=str(CACHE_DIR),
            local_files_only=True,
        )
        p = Path(path)
        if not p.is_file() or p.stat().st_size <= 0:
            raise RuntimeError(f"offline main-ref resolution failed: {repo_id}/{filename}")

    print(json.dumps({
        "ok": True,
        "engine": "indicf5",
        "runtime_mode": "eager_v4",
        "torch_compile": False,
        "offline_bundle": True,
        "cache_dir": str(CACHE_DIR),
        "required_files": resolved,
    }, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"IndicF5 bundle verification failed: {exc}", file=sys.stderr)
        raise
