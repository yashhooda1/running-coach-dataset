"""
push_to_hub.py — publish the dataset repo.

    pip install huggingface_hub datasets
    hf auth login
    python push_to_hub.py --repo hoodarunner/running-coach-sft

Uploads the JSONL splits, the normalized seed plans, the generator source (so
the corpus is reproducible from the card alone), and the dataset card as README.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi


SOURCE_FILES = ["vdot.py", "parse_plans.py", "strava.py",
                "generate_dataset.py", "build.py", "eval_paces.py",
                "generate_eval.py"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. hoodarunner/running-coach-sft")
    ap.add_argument("--data", default="data")
    ap.add_argument("--private", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    root = Path(__file__).parent
    data = Path(a.data)
    planned = []

    for name in ["train.jsonl", "validation.jsonl", "reference.jsonl",
                 "seed_plans.json", "stats.json", "calibration.json"]:
        f = data / name
        if f.exists():
            planned.append((f, f"data/{name}"))
    for name in SOURCE_FILES:
        f = root / name
        if f.exists():
            planned.append((f, f"src/{name}"))
    planned.append((root / "DATASET_CARD.md", "README.md"))

    for local, remote in planned:
        print(f"{local} -> {remote}  ({local.stat().st_size/1024:.1f} KB)")
    if a.dry_run:
        print("\ndry run, nothing uploaded")
        return

    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    for local, remote in planned:
        api.upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                        repo_id=a.repo, repo_type="dataset")
        print(f"uploaded {remote}")
    print(f"\nhttps://huggingface.co/datasets/{a.repo}")


if __name__ == "__main__":
    main()
