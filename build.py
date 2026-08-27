"""
build.py — one command to produce the whole dataset directory.

    python build.py --n-profiles 300

Produces:
    data/seed_plans.json      normalized form of the four source documents
    data/reference.jsonl      those four as a held-out style reference (not trained on)
    data/train.jsonl          synthetic SFT corpus
    data/validation.jsonl     profile-disjoint split
    data/stats.json           counts for the dataset card
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

from generate_dataset import SYSTEM, build


def reference_split(seed_plans_path: Path, uploads: Path, out: Path):
    """The four source documents, kept whole, as an eval/style reference."""
    plans = json.loads(seed_plans_path.read_text())
    rows = []
    for pl in plans:
        raw = (uploads / pl["source_file"]).read_text(encoding="utf-8").strip()
        goal = pl["goal"]["raw"] or "a return to competitive training"
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Write me a training plan for {goal}."},
                {"role": "assistant", "content": raw},
            ],
            "task": "plan_generation",
            "profile_id": pl["plan_id"],
            "layout": pl["layout"],
            "session_count": pl["session_count"],
            "source": "author-retained assistant output",
            "note": ("Reference only. Paces here are not machine-verified; use "
                     "for style comparison and eval, not for training."),
        })
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"reference   {len(rows):5d} rows -> {out}")
    return len(rows)


def merge_reviews(out: Path, val_frac: float = 0.1):
    """Fold the Strava-derived reviews into the splits.

    Real weeks are consecutive and autocorrelated, so the split is taken as a
    contiguous tail rather than at random: a shuffled split would put week 41
    in train and week 42 in validation, which are nearly the same week.
    """
    src = out / "reviews.jsonl"
    if not src.exists():
        return
    rows = [json.loads(l) for l in src.open(encoding="utf-8")]
    cut = int(len(rows) * (1 - val_frac))
    for split, chunk in [("train", rows[:cut]), ("validation", rows[cut:])]:
        with (out / f"{split}.jsonl").open("a", encoding="utf-8") as f:
            for r in chunk:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"merged {len(rows)} training_review rows "
          f"({cut} train / {len(rows)-cut} validation, contiguous split)")


def validate(path: Path) -> dict:
    tasks, bad = Counter(), 0
    n = 0
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        n += 1
        m = row["messages"]
        roles = [x["role"] for x in m]
        if roles != ["system", "user", "assistant"]:
            bad += 1
        if not m[-1]["content"].strip():
            bad += 1
        tasks[row.get("task", "?")] += 1
    assert bad == 0, f"{bad} malformed rows in {path}"
    return {"rows": n, "tasks": dict(tasks)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-profiles", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--uploads", default="sources")
    ap.add_argument("--out", default="data")
    ap.add_argument("--strava", default="sources/activities.csv")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    here = Path(__file__).parent
    subprocess.run([sys.executable, str(here / "parse_plans.py"), "--in", a.uploads,
                    "--out", str(out / "seed_plans.json")], check=True)
    strava = Path(a.strava)
    if strava.exists():
        subprocess.run([sys.executable, str(here / "strava.py"), "--csv", str(strava),
                        "--out", str(out)], check=True)
    else:
        print(f"no Strava export at {strava}; skipping review split")
    print()
    n_ref = reference_split(out / "seed_plans.json", Path(a.uploads),
                            out / "reference.jsonl")
    build(a.n_profiles, a.seed, out, val_frac=0.1)

    merge_reviews(out)
    stats = {
        "train": validate(out / "train.jsonl"),
        "validation": validate(out / "validation.jsonl"),
        "reference": validate(out / "reference.jsonl"),
        "n_profiles": a.n_profiles,
        "seed": a.seed,
    }
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print("\n" + json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
