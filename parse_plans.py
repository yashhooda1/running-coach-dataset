"""
parse_plans.py — normalize the four seed plan documents into one schema.

The four files disagree on structure: week ranges with bullets, phases with
"Day N", phases with weekday headers on their own line, and absolute dates.
Everything lands in the same shape so they can serve as style reference and as
a human-validated holdout set.

Usage:
    python parse_plans.py --in /mnt/user-data/uploads --out data/seed_plans.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday",
            "Friday", "Saturday", "Sunday"]

SESSION_RULES = [
    # Order matters. "8 x 400m @ mile pace (90 sec rest)" is an interval
    # session, not a rest day, so only a leading "Rest" counts as rest.
    ("rest",      r"^\s*rest\b"),
    ("race",      r"\brace day\b|race the\b"),
    ("interval",  r"\bintervals?\b|\d+\s*x\s*\d+(?:\.\d+)?\s*(?:m|mile|k)\b|track workout|vo2"),
    ("repetition", r"\bstrides?\b|hill repeats?|\bfartlek\b"),
    ("tempo",     r"\btempo\b|threshold"),
    ("long",      r"\blong run\b"),
    ("cross",     r"cross-?train|elliptical|cycling|swimming"),
    ("strength",  r"strength|flexibility|stretching|\bcore\b"),
    ("nutrition", r"nutrition|hydration|\bfuel|\bsleep\b"),
    ("mileage_target", r"(weekly )?mileage|miles per week|\bmpw\b"),
    ("easy",      r"\beasy\b|recovery run|medium long"),
    ("rest",      r"\brest day\b|active recovery"),
]

DIST_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-\s*(\d+(?:\.\d+)?)\s*)?miles?\b", re.I)
PACE_RE = re.compile(r"(\d:\d{2})\s*(?:-\s*(\d:\d{2}))?\s*(?:/|\s)?(?:mile|mi)\b", re.I)
REP_RE = re.compile(r"(\d+)\s*x\s*(\d+(?:\.\d+)?)\s*(m|mile|miles|min|k)\b", re.I)
GOAL_RE = re.compile(
    r"sub[-\s]?(\d+(?::\d{2})*)[-\s]*(?:minute\s*)?"
    r"(8k|5k|10k|mile|half[- ]?marathon|marathon|half)?", re.I)


def classify(text: str) -> str:
    t = text.lower()
    for name, pattern in SESSION_RULES:
        if re.search(pattern, t):
            return name
    return "other"


def extract_distance(text: str):
    m = DIST_RE.search(text)
    if not m:
        return None
    lo = float(m.group(1))
    hi = float(m.group(2)) if m.group(2) else lo
    return {"low_mi": lo, "high_mi": hi}


def extract_reps(text: str):
    out = []
    for m in REP_RE.finditer(text):
        out.append({"reps": int(m.group(1)),
                    "value": float(m.group(2)),
                    "unit": m.group(3).lower()})
    return out or None


def extract_paces(text: str):
    out = []
    for m in PACE_RE.finditer(text):
        out.append(m.group(0).strip())
    return out or None


def make_session(day_label, raw, week_index, phase):
    return {
        "week": week_index,
        "phase": phase,
        "day": day_label,
        "type": classify(raw),
        "raw": raw.strip(),
        "distance": extract_distance(raw),
        "reps": extract_reps(raw),
        "paces_mentioned": extract_paces(raw),
    }


# ---------------------------------------------------------------- parsers

def parse_dated(text: str):
    """'Monday, February 03, 2025: <session>' — one line per session."""
    sessions, notes = [], []
    week, seen = 0, None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(%s),\s*([A-Za-z]+ \d{1,2}, \d{4}):\s*(.+)$"
                     % "|".join(WEEKDAYS), line)
        if not m:
            notes.append(line)
            continue
        weekday, date, body = m.groups()
        if seen is None or (weekday == "Monday" and seen != date):
            week += 1
        seen = date
        s = make_session(weekday, body, week, "unspecified")
        s["date"] = date
        sessions.append(s)
    return sessions, notes


def parse_day_numbered(text: str):
    """'Weeks 1-4: Base Building' + 'Day 1: Easy run, 5 miles'."""
    sessions, notes = [], []
    phase, week_lo, week_hi = "unspecified", 1, 1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        wm = re.match(r"^Weeks?\s*(\d+)\s*[-–]\s*(\d+)\s*:\s*(.+)$", line, re.I)
        if wm:
            week_lo, week_hi, phase = int(wm.group(1)), int(wm.group(2)), wm.group(3).strip()
            continue
        if re.match(r"^Race Week", line, re.I):
            phase, week_lo, week_hi = "Race Week", week_hi + 1, week_hi + 1
            continue
        dm = re.match(r"^(Day \d+|Race Day)\s*:\s*(.+)$", line, re.I)
        if dm:
            s = make_session(dm.group(1), dm.group(2), week_lo, phase)
            s["applies_to_weeks"] = [week_lo, week_hi]
            sessions.append(s)
            continue
        notes.append(line)
    return sessions, notes


def parse_weekday_blocks(text: str):
    """'Week 1-4: <phase>' + 'Monday: Easy Run' with detail on the next line."""
    sessions, notes = [], []
    phase, week_lo, week_hi = "unspecified", 1, 1
    lines = [l.strip() for l in text.splitlines()]
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line:
            i += 1
            continue
        wm = re.match(r"^Weeks?\s*(\d+)\s*[-–]\s*(\d+)\s*:\s*(.+)$", line, re.I)
        if wm:
            week_lo, week_hi, phase = int(wm.group(1)), int(wm.group(2)), wm.group(3).strip()
            i += 1
            continue
        dm = re.match(r"^(%s)\s*:\s*(.*)$" % "|".join(WEEKDAYS), line)
        if dm:
            weekday, head = dm.group(1), dm.group(2).strip()
            body = [head] if head else []
            j = i + 1
            while j < len(lines) and lines[j] and not re.match(
                    r"^(%s)\s*:|^Weeks?\s*\d|^Taper|^Key Focus" % "|".join(WEEKDAYS), lines[j]):
                body.append(lines[j])
                j += 1
            s = make_session(weekday, " — ".join(body), week_lo, phase)
            s["applies_to_weeks"] = [week_lo, week_hi]
            sessions.append(s)
            i = j
            continue
        notes.append(line)
        i += 1
    return sessions, notes


def parse_prose_weeks(text: str):
    """'**Week 1-2: Recovery and Base Building**' + bullet prescriptions."""
    sessions, notes = [], []
    phase, week_lo, week_hi = "unspecified", 1, 1
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        wm = re.match(r"^\*{0,2}Weeks?\s*(\d+)\s*[-–]\s*(\d+)\s*:\s*(.+?)\*{0,2}$", line, re.I)
        if wm:
            week_lo, week_hi, phase = int(wm.group(1)), int(wm.group(2)), wm.group(3).strip()
            continue
        if line.startswith("-"):
            body = line.lstrip("- ").strip()
            s = make_session("unspecified", body, week_lo, phase)
            s["applies_to_weeks"] = [week_lo, week_hi]
            s["prescription_style"] = "guideline"
            sessions.append(s)
            continue
        notes.append(line.strip("*"))
    return sessions, notes


def detect_parser(text: str):
    if re.search(r"^(?:%s),\s*[A-Za-z]+ \d{1,2}, \d{4}:" % "|".join(WEEKDAYS),
                 text, re.M):
        return "dated", parse_dated
    if re.search(r"^Day \d+:", text, re.M):
        return "day_numbered", parse_day_numbered
    if re.search(r"^(?:%s):" % "|".join(WEEKDAYS), text, re.M):
        return "weekday_blocks", parse_weekday_blocks
    return "prose_weeks", parse_prose_weeks


def extract_goal(text: str, filename: str):
    blob = f"{filename} {text[:800]}"
    m = GOAL_RE.search(blob)
    if not m:
        return {"raw": None, "target": None, "distance": None}
    goal = {"raw": m.group(0).strip(),
            "target": m.group(1),
            "distance": (m.group(2) or "").lower().replace("-", " ").strip() or None}
    if goal["distance"] in ("half marathon", "half"):
        goal["distance"] = "half"
    return goal


def parse_file(path: Path):
    text = path.read_text(encoding="utf-8")
    style, fn = detect_parser(text)
    sessions, notes = fn(text)
    return {
        "plan_id": path.stem.lower().replace(" ", "_"),
        "source_file": path.name,
        "layout": style,
        "provenance": "assistant-generated coaching output, retained by the author",
        "goal": extract_goal(text, path.stem.replace("_", " ")),
        "session_count": len(sessions),
        "sessions": sessions,
        "notes": [n for n in notes if len(n) > 20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="indir", default="sources")
    ap.add_argument("--out", dest="out", default="data/seed_plans.json")
    a = ap.parse_args()

    plans = [parse_file(p) for p in sorted(Path(a.indir).glob("*.txt"))]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plans, indent=2), encoding="utf-8")

    for p in plans:
        kinds = {}
        for s in p["sessions"]:
            kinds[s["type"]] = kinds.get(s["type"], 0) + 1
        print(f"{p['source_file']:46s} layout={p['layout']:15s} "
              f"sessions={p['session_count']:3d} goal={p['goal']['raw']}")
        print(f"{'':46s} {kinds}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
