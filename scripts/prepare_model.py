"""Download the exact approved model revision; never select the latest weights.

Competition data and submission files are deliberately outside this script.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import HfApi, snapshot_download
from project_contract import MODEL_ID, MODEL_REVISION, MODEL_WEIGHTS


def main() -> None:
    info = HfApi().model_info(MODEL_ID, revision=MODEL_REVISION)
    if info.sha != MODEL_REVISION:
        raise RuntimeError("The approved model revision could not be verified")

    available = {entry.rfilename for entry in info.siblings}
    if MODEL_WEIGHTS not in available:
        raise RuntimeError(f"Pinned revision does not provide {MODEL_WEIGHTS}")

    destination = ROOT / "artifacts/checkpoints/pretrained"
    snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=destination,
        allow_patterns=[
            "config.json", MODEL_WEIGHTS, "tokenizer.json",
            "tokenizer_config.json", "special_tokens_map.json", "README.md",
        ],
        max_workers=2,
    )
    if not (destination / MODEL_WEIGHTS).is_file():
        raise RuntimeError(f"Expected checkpoint missing: {MODEL_WEIGHTS}")

    provenance = ROOT / "artifacts/provenance/backbone.json"
    provenance.parent.mkdir(parents=True, exist_ok=True)
    # Preserve this schema to keep existing experiment cache provenance valid.
    provenance.write_text(
        json.dumps({"model_id": MODEL_ID, "revision": MODEL_REVISION}, indent=2),
        encoding="utf-8",
    )
    print(f"Pinned backbone verified: {MODEL_ID}@{MODEL_REVISION}")


if __name__ == "__main__":
    main()
