"""
eval_paces.py — does the model's arithmetic survive fine-tuning?

Loss going down tells you the model learned the register. It does not tell you
whether the threshold pace it just prescribed follows from the athlete's PR.
This scores that directly: pull every pace out of the generated text, compare
against the VDOT ground truth for that profile, and report error in seconds per
mile.

Self-test (should score near-perfect, since the references are generated
from the same model):

    python eval_paces.py --generations data/validation.jsonl --field reference

Against a fine-tune, write generations to JSONL with a "generated" field:

    python eval_paces.py --generations runs/qwen7b.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
from collections import defaultdict
from pathlib import Path

from vdot import (DISTANCES_M, MILE_M, paces_from_vdot, parse_time,
                  race_time_from_vdot)

# "6:23/mi", "6:23/mile", "6:23 per mile"
PACE_RE = re.compile(r"(\d{1,2}:\d{2})\s*(?:/\s*mi(?:le)?|\s*per\s*mile)")
TOL_S = 8.0  # seconds per mile; beyond this the zone is a different workout


def truth(vdot: float, goal_vdot: float | None = None,
          goal_distance: str | None = None) -> dict:
    """Legitimate paces in a response come from two fitness levels, not one.

    Current fitness sets the training zones; the goal sets race-pace work. Heat
    guidance deliberately shifts threshold slower. All of those are correct
    output, so all of them belong in the reference set.
    """
    out = {}
    for tag, v in [("cur", vdot), ("goal", goal_vdot)]:
        if v is None:
            continue
        p = paces_from_vdot(v)
        out.update({
            f"{tag}_easy_fast": p.easy_fast, f"{tag}_easy_slow": p.easy_slow,
            f"{tag}_marathon": p.marathon, f"{tag}_threshold": p.threshold,
            f"{tag}_interval": p.interval, f"{tag}_repetition": p.repetition,
        })
    base = paces_from_vdot(vdot)
    out["heat_threshold"] = base.threshold + 12   # dew point > 70F guidance
    if goal_vdot and goal_distance in DISTANCES_M:
        # goal race pace itself sits between zones and is valid output
        d = DISTANCES_M[goal_distance]
        out["goal_race_pace"] = race_time_from_vdot(goal_vdot, d) / (d / MILE_M)
    return out


def score_row(text: str, vdot: float, goal_vdot: float | None = None,
              goal_distance: str | None = None):
    """Every pace in the text should be within tolerance of some real zone."""
    t = truth(vdot, goal_vdot, goal_distance)
    found = [parse_time(m.group(1)) for m in PACE_RE.finditer(text)]
    errs, unmatched = [], 0
    for f in found:
        e = min(abs(f - v) for v in t.values())
        errs.append(e)
        if e > TOL_S:
            unmatched += 1
    return {"n_paces": len(found), "errors": errs, "off_zone": unmatched}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", required=True)
    ap.add_argument("--field", default="generated",
                    help="'generated' for model output, 'reference' to self-test")
    a = ap.parse_args()

    per_task = defaultdict(lambda: {"errors": [], "off": 0, "n": 0, "rows": 0})
    rows = 0
    for line in Path(a.generations).open(encoding="utf-8"):
        row = json.loads(line)
        text = (row["messages"][-1]["content"] if a.field == "reference"
                else row.get("generated", ""))
        if not text or "vdot" not in row:
            continue
        s = score_row(text, row["vdot"],
                      row.get("effective_goal_vdot") or row.get("goal_vdot"),
                      row.get("goal_distance"))
        b = per_task[row.get("task", "?")]
        b["errors"] += s["errors"]
        b["off"] += s["off_zone"]
        b["n"] += s["n_paces"]
        b["rows"] += 1
        rows += 1

    print(f"scored {rows} generations\n")
    print(f"{'task':22s} {'rows':>5s} {'paces':>6s} {'median err':>11s} "
          f"{'p90 err':>8s} {'off-zone':>9s}")
    all_err, all_off, all_n = [], 0, 0
    for task, b in sorted(per_task.items()):
        if not b["errors"]:
            print(f"{task:22s} {b['rows']:5d} {0:6d} {'-':>11s} {'-':>8s} {'-':>9s}")
            continue
        med = st.median(b["errors"])
        p90 = sorted(b["errors"])[int(0.9 * len(b["errors"])) - 1]
        pct = 100 * b["off"] / b["n"]
        print(f"{task:22s} {b['rows']:5d} {b['n']:6d} {med:9.1f}s {p90:7.1f}s "
              f"{pct:8.1f}%")
        all_err += b["errors"]
        all_off += b["off"]
        all_n += b["n"]

    if all_err:
        print(f"\n{'OVERALL':22s} {rows:5d} {all_n:6d} "
              f"{st.median(all_err):9.1f}s "
              f"{sorted(all_err)[int(0.9*len(all_err))-1]:7.1f}s "
              f"{100*all_off/all_n:8.1f}%")
        print(f"\noff-zone = pace more than {TOL_S:.0f}s/mi from any real "
              f"training zone for that athlete")


if __name__ == "__main__":
    main()
