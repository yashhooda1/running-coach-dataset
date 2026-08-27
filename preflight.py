"""
preflight.py — run before `git add`. Refuses to pass if something that should
not be public is about to become public.

    python preflight.py

Checks:
  1. the raw Strava export is ignored (GPS-derived filenames, activity names,
     private notes — strava.py strips these at load, the raw CSV does not)
  2. no HF/GitHub/AWS-shaped token is sitting in a tracked file
  3. the repo is not inside OneDrive, which corrupts .git
  4. .gitignore covers the generated artifacts
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path.cwd()
fail, warn = [], []

# ---------------------------------------------------------------- 1. secrets
TOKEN_PATTERNS = [
    (r"hf_[A-Za-z0-9]{34,}", "Hugging Face token"),
    (r"gh[pousr]_[A-Za-z0-9]{36,}", "GitHub token"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key"),
    (r"sk-[A-Za-z0-9]{32,}", "OpenAI-style key"),
]

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".jsonl", ".yml", ".yaml",
                 ".ipynb", ".cfg", ".toml", ".bat", ".sh"}

def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout

in_repo = (ROOT / ".git").exists()
files = ([Path(p) for p in git("ls-files").split("\n") if p] if in_repo
         else [p for p in ROOT.rglob("*") if p.is_file()])

for f in files:
    if f.suffix.lower() not in TEXT_SUFFIXES or not f.exists():
        continue
    if ".git" in f.parts or "__pycache__" in f.parts:
        continue
    try:
        body = f.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        continue
    for pat, label in TOKEN_PATTERNS:
        if re.search(pat, body):
            fail.append(f"{label} found in {f} — rotate it, then remove it")

# ------------------------------------------------------- 2. the Strava export
csv = ROOT / "sources" / "activities.csv"
if csv.exists():
    if not in_repo:
        warn.append("sources/activities.csv present; .gitignore must cover it "
                    "before you init (it does in the shipped .gitignore)")
    else:
        tracked = git("ls-files", "sources/activities.csv").strip()
        if tracked:
            fail.append("sources/activities.csv is TRACKED BY GIT. This is your "
                        "raw Strava export: activity names, private notes, "
                        "GPS-derived filenames. Run:\n"
                        "      git rm --cached sources/activities.csv")
        else:
            print("  ok   sources/activities.csv present and untracked")

# --------------------------------------------------------------- 3. OneDrive
if "OneDrive" in str(ROOT):
    warn.append(f"repo is inside OneDrive ({ROOT}). OneDrive's file locking "
                "corrupts .git objects. Move it somewhere like C:\\dev\\ "
                "before running git init.")

# -------------------------------------------------------------- 4. gitignore
gi = ROOT / ".gitignore"
if not gi.exists():
    fail.append(".gitignore missing")
else:
    body = gi.read_text(encoding="utf-8")
    for needed in ["sources/activities.csv", "data/*.jsonl", "generations/"]:
        if needed not in body:
            warn.append(f".gitignore does not mention {needed}")

# ------------------------------------------------------------------- report
print()
for w in warn:
    print(f"  WARN  {w}")
for f_ in fail:
    print(f"  FAIL  {f_}")

if fail:
    print("\nDo not commit yet.")
    sys.exit(1)
print("\npreflight passed" + (" (with warnings)" if warn else ""))
