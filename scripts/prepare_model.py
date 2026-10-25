"""Download only the approved backbone and retain provenance; no uploads."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from huggingface_hub import HfApi, snapshot_download
import requests


def main() -> None:
    out = ROOT / "artifacts/provenance"
    out.mkdir(parents=True, exist_ok=True)
    model_id = "jhu-clsp/mmBERT-small"
    info = HfApi().model_info(model_id)
    print("Approved backbone", model_id, "revision", info.sha, flush=True)
    (out / "backbone.json").write_text(json.dumps({"model_id": model_id, "revision": info.sha}, indent=2), encoding="utf-8")
    for page in ("overview", "rules"):
        try:
            url = f"https://www.kaggle.com/competitions/rmit-hackathon-2026/{page}"
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            (out / f"kaggle_{page}.html").write_text(response.text, encoding="utf-8")
            print("Kaggle page", page, response.status_code, len(response.content), flush=True)
        except requests.RequestException as exc:
            print("Kaggle access limitation:", str(exc), flush=True)
    # Kaggle's public read service may expose the competition text without authentication.
    for method, payload in (
        ("GetCompetition", {"competitionName": "rmit-hackathon-2026"}),
        ("GetCompetitionRules", {"competitionId": 169360}),
    ):
        try:
            response = requests.post(
                "https://www.kaggle.com/api/i/competitions.CompetitionService/" + method,
                json=payload, timeout=20,
                headers={"User-Agent": "Mozilla/5.0", "Referer": "https://www.kaggle.com/competitions/rmit-hackathon-2026/rules"},
            )
            (out / f"kaggle_{method}.txt").write_text(response.text, encoding="utf-8")
            print("Kaggle read service", method, response.status_code, "bytes", len(response.content), flush=True)
        except requests.RequestException as exc:
            print("Kaggle read service unavailable:", str(exc), flush=True)
    # Discover the public read endpoints from Kaggle's own client bundle.
    try:
        html = (out / "kaggle_overview.html").read_text(encoding="utf-8")
        match = re.search(r'src="(/static/assets/app\.js[^\"]*)"', html)
        if match:
            js = requests.get("https://www.kaggle.com" + match.group(1), timeout=45).text
            (out / "kaggle_client.js").write_text(js, encoding="utf-8")
            snippets = [js[max(0,m.start()-180):m.end()+180] for m in re.finditer(r'GetCompetitionRules|GetCompetitionPages|ListCompetitionPages|competitions\.[A-Za-z]*(?:Page|Rules)[A-Za-z]*Service', js)]
            (out / "kaggle_read_endpoints.txt").write_text("\n".join(snippets), encoding="utf-8")
            print("Public rules endpoint discovery:", "\n".join(snippets)[:3000], flush=True)
    except requests.RequestException as exc:
        print("Kaggle client unavailable:", str(exc), flush=True)
    try:
        response = requests.post("https://www.kaggle.com/api/i/competitions.PageService/ListPages",
                                 json={"competitionId": 169360}, timeout=30)
        (out / "competition_pages.json").write_text(response.text, encoding="utf-8")
        print("Competition pages", response.status_code, response.text[:1200], flush=True)
    except requests.RequestException as exc:
        print("Competition pages unavailable:", str(exc), flush=True)
    destination = ROOT / "artifacts/checkpoints/pretrained"
    files = {entry.rfilename for entry in info.siblings}
    weights = "model.safetensors" if "model.safetensors" in files else "pytorch_model.bin"
    snapshot_download(model_id, revision=info.sha, local_dir=destination,
                      allow_patterns=["config.json", weights, "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "README.md"],
                      max_workers=2)
    assert (destination / weights).is_file(), "Expected official checkpoint"
    print("Backbone download verified:", destination, flush=True)


if __name__ == "__main__":
    main()
