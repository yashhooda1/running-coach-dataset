"""
push_model_card.py — upload MODEL_CARD.md as the adapter repo's README.

    python push_model_card.py --repo hoodarunner/running-coach-qwen3b-lora
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--card", default="MODEL_CARD.md")
ap.add_argument("--dry-run", action="store_true")
a = ap.parse_args()

card = Path(a.card)
if not card.exists():
    raise SystemExit(f"{card} not found — put it in this folder first")

print(f"{card} -> README.md on {a.repo}  ({card.stat().st_size/1024:.1f} KB)")
if a.dry_run:
    raise SystemExit("dry run, nothing uploaded")

HfApi().upload_file(
    path_or_fileobj=str(card),
    path_in_repo="README.md",
    repo_id=a.repo,
    repo_type="model",
)
print(f"https://huggingface.co/{a.repo}")
