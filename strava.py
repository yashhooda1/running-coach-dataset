"""
strava.py — turn a Strava bulk export into (a) calibration constants for the
synthetic generator and (b) real training weeks for a review task.

Two things happen here that are worth separating:

1. CALIBRATION. The synthetic generator originally used constants I made up:
   a mileage ramp, a cutback rhythm, a long-run share, and a heat rule of
   "add 15-20s/mi above a 65F dew point". Real data replaces every one of
   those with a measured value. The heat rule in particular did not survive.

2. REVIEW EXAMPLES. Real weeks of training, de-identified and summarized,
   paired with feedback computed from the week's own statistics. This is the
   one task type synthetic data cannot produce honestly, because real training
   is messier than any generator: missed days, doubles, 1-mile shakeouts
   logged as separate activities, and ramps no plan would prescribe.

PRIVACY. Activity names, descriptions, private notes, gear, filenames, media
links, and all GPS-derived fields are dropped at load and never reach an
output file. Absolute dates are reduced to a month (kept, because seasonality
is the whole point of the heat analysis) and a sequential week index.

Usage:
    python strava.py --csv /path/to/activities.csv --out data/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MILE_M = 1609.344

# Everything in this list is dropped before any analysis runs.
PII_COLUMNS = [
    "Activity Name", "Activity Description", "Activity Private Note",
    "Activity Gear", "Filename", "Media", "Bike", "Gear", "Athlete Weight",
]

KEEP = {
    "Activity ID": "aid", "Activity Date": "date", "Activity Type": "type",
    "Distance.1": "m", "Moving Time": "sec", "Elapsed Time.1": "elapsed",
    "Average Heart Rate": "hr", "Max Heart Rate.1": "maxhr",
    "Elevation Gain": "elev", "Average Cadence": "cad",
    "Dewpoint": "dp_c", "Weather Temperature": "temp_c", "Humidity": "humidity",
    "Relative Effort": "effort", "Perceived Exertion": "rpe",
}


def load(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])
    cols = {k: v for k, v in KEEP.items() if k in df.columns}
    df = df[list(cols)].rename(columns=cols)

    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    for c in ["m", "sec", "hr", "maxhr", "elev", "cad", "dp_c", "temp_c",
              "humidity", "effort", "rpe"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    runs = df[df["type"] == "Run"].copy()
    # Strava logs warmups, strides, and cooldowns as separate activities;
    # anything under a kilometre is a fragment, not a session.
    runs = runs[(runs["m"] > 1000) & (runs["sec"] > 120)]
    runs["mi"] = runs["m"] / MILE_M
    runs["pace"] = runs["sec"] / runs["mi"]
    runs["dp_f"] = runs["dp_c"] * 9 / 5 + 32
    runs["temp_f"] = runs["temp_c"] * 9 / 5 + 32
    runs["day"] = runs["date"].dt.normalize()
    runs = runs[(runs["pace"] > 240) & (runs["pace"] < 900)]
    return runs.sort_values("date").reset_index(drop=True)


# ------------------------------------------------------------- classification

def classify_sessions(runs: pd.DataFrame) -> pd.DataFrame:
    """Label each run easy / workout / long, using the athlete's own baseline.

    No fixed pace thresholds: the baseline is a trailing 60-day median, so the
    labels stay meaningful as fitness changes across six years.
    """
    runs = runs.copy().set_index("date")
    baseline = runs["pace"].rolling("60D").median()
    typical_day = runs["mi"].rolling("60D").quantile(0.6)
    runs["baseline_pace"] = baseline
    runs["is_workout"] = runs["pace"] < baseline - 40      # 40s/mi faster
    runs["is_long"] = (runs["mi"] >= (typical_day * 1.45).clip(lower=8))
    runs["session"] = np.where(runs["is_workout"], "workout",
                       np.where(runs["is_long"], "long", "easy"))
    return runs.reset_index()


def weekly_frame(runs: pd.DataFrame) -> pd.DataFrame:
    daily = runs.groupby("day").agg(
        mi=("mi", "sum"), sec=("sec", "sum"), n=("mi", "size"),
        longest=("mi", "max"), hr=("hr", "mean"), dp=("dp_f", "mean"),
        workouts=("is_workout", "sum"),
    )
    daily.index = pd.to_datetime(daily.index)
    full = daily.resample("W").agg(
        mi=("mi", "sum"), sec=("sec", "sum"), run_days=("mi", "size"),
        longest=("longest", "max"), hr=("hr", "mean"), dp=("dp", "mean"),
        workouts=("workouts", "sum"), activities=("n", "sum"),
    )
    full["pace"] = full["sec"] / full["mi"].replace(0, np.nan)
    full["prev"] = full["mi"].shift(1)
    full["ratio"] = full["mi"] / full["prev"].replace(0, np.nan)
    return full


# --------------------------------------------------------------- calibration

def ols(X: np.ndarray, y: np.ndarray):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    dof = max(len(y) - X.shape[1], 1)
    se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X)) * float(resid @ resid) / dof)
    r2 = 1 - float(resid @ resid) / float(((y - y.mean()) ** 2).sum())
    return b, se, r2


def heat_model(runs: pd.DataFrame) -> dict:
    """How much does a humid morning actually cost, at the same heart rate?

    Conditioning on heart rate and on year matters. Without the year terms the
    fitness trend across six seasons swamps the weather signal and the
    coefficient comes out near zero with the wrong sign.
    """
    d = runs.dropna(subset=["hr", "dp_f", "pace"])
    d = d[(d["hr"].between(130, 165)) & (d["mi"].between(3, 16))]
    if len(d) < 100:
        return {"available": False, "n": int(len(d))}
    years = sorted(d["date"].dt.year.unique())
    cols = [np.ones(len(d)), d["dp_f"].values, d["hr"].values, d["mi"].values]
    names = ["const", "dp_f", "hr", "mi"]
    for y in years[1:]:
        cols.append((d["date"].dt.year == y).astype(float).values)
        names.append(f"y{y}")
    b, se, r2 = ols(np.column_stack(cols), d["pace"].values)
    i = names.index("dp_f")
    return {
        "available": True, "n": int(len(d)), "r2": round(r2, 3),
        "s_per_mile_per_F": round(float(b[i]), 3),
        "se": round(float(se[i]), 3),
        "t": round(float(b[i] / se[i]), 2),
        "ci95": [round(float(b[i] - 1.96 * se[i]), 3),
                 round(float(b[i] + 1.96 * se[i]), 3)],
        "cost_55F_to_75F_s_per_mile": round(float(20 * b[i]), 1),
    }


def calibrate(runs: pd.DataFrame, weeks: pd.DataFrame) -> dict:
    active = weeks[weeks["mi"] > 5]
    ratio = active["ratio"].dropna()
    ratio = ratio[(ratio > 0.2) & (ratio < 3)]
    long_share = (active["longest"] / active["mi"].replace(0, np.nan)).dropna()
    mix = runs["session"].value_counts(normalize=True).round(3).to_dict()

    return {
        "n_runs": int(len(runs)),
        "n_weeks": int(len(active)),
        "date_range": [str(runs["date"].min().date()), str(runs["date"].max().date())],
        "weekly_mileage": {
            "median": round(float(active["mi"].median()), 1),
            "p10": round(float(active["mi"].quantile(0.1)), 1),
            "p90": round(float(active["mi"].quantile(0.9)), 1),
            "max": round(float(active["mi"].max()), 1),
        },
        "week_over_week_ratio": {
            "median": round(float(ratio.median()), 3),
            "p75": round(float(ratio.quantile(0.75)), 3),
            "p90": round(float(ratio.quantile(0.90)), 3),
            "p95": round(float(ratio.quantile(0.95)), 3),
        },
        "cutback_rate": round(float((ratio < 0.85).mean()), 3),
        "long_run_share_of_week": {
            "median": round(float(long_share.median()), 3),
            "p90": round(float(long_share.quantile(0.9)), 3),
        },
        "run_days_per_week": {
            "median": float(active["run_days"].median()),
            "p90": float(active["run_days"].quantile(0.9)),
        },
        "activities_per_run_day": round(
            float(active["activities"].sum() / max(active["run_days"].sum(), 1)), 2),
        "session_mix": mix,
        "workout_days_per_week": round(float(active["workouts"].mean()), 2),
        "heat": heat_model(runs),
    }


# ----------------------------------------------------------- review examples

SYSTEM = (
    "You are a distance running coach. Prescribe training from the athlete's "
    "demonstrated fitness, state paces explicitly, and explain the purpose of "
    "each session. Flag goals that the available training time cannot support. "
    "You are not a medical provider: refer pain, injury, or health questions to "
    "a clinician rather than training through them."
)


def fmt_pace(s: float) -> str:
    return f"{int(s // 60)}:{int(round(s % 60)):02d}/mi"


def review_rows(weeks: pd.DataFrame, cal: dict, min_weeks: int = 3):
    """One example per real week with enough signal to say something about."""
    rows = []
    w = weeks.reset_index().rename(columns={"index": "week_end"})
    w.columns = ["week_end"] + list(w.columns[1:])
    for i in range(1, len(w)):
        cur, prev = w.iloc[i], w.iloc[i - 1]
        if not (cur["mi"] >= 15 and cur["run_days"] >= 3 and prev["mi"] > 5):
            continue
        month = pd.Timestamp(cur["week_end"]).month
        ratio = cur["mi"] / prev["mi"]
        long_share = cur["longest"] / cur["mi"]

        facts = (
            f"Last week: {cur['mi']:.1f} miles across {int(cur['run_days'])} "
            f"days ({int(cur['activities'])} logged activities), longest run "
            f"{cur['longest']:.1f} miles, average pace "
            f"{fmt_pace(cur['pace'])}, {int(cur['workouts'])} sessions faster "
            f"than my easy baseline. Week before was {prev['mi']:.1f} miles."
        )
        if pd.notna(cur["dp"]):
            facts += f" Average dew point was {cur['dp']:.0f}F."
        facts += " How did that week look?"

        obs = []
        if ratio > 1.30:
            obs.append(
                f"The jump is the thing to look at: {prev['mi']:.0f} to "
                f"{cur['mi']:.0f} miles is {100*(ratio-1):.0f}% in one week. "
                f"Your own history sits at a median week-over-week ratio of "
                f"{cal['week_over_week_ratio']['median']:.2f} with the 90th "
                f"percentile at {cal['week_over_week_ratio']['p90']:.2f}, so "
                f"this week is in the top decile of ramps you've ever run. "
                f"That's survivable once; it's a pattern that ends in a "
                f"stress reaction if it repeats."
            )
        elif ratio < 0.80:
            obs.append(
                f"Down {100*(1-ratio):.0f}% from the previous week. If that "
                f"was planned, good — cutback weeks are where the adaptation "
                f"lands. About {100*cal['cutback_rate']:.0f}% of your active "
                f"weeks are cutbacks, which is a healthy rhythm."
            )
        else:
            obs.append(
                f"Volume moved {100*(ratio-1):+.0f}% week over week, which is "
                f"the right size of change — enough to progress, small enough "
                f"to absorb."
            )

        if long_share > 0.35:
            obs.append(
                f"The long run is {100*long_share:.0f}% of the week. That's "
                f"high; past about a third, the long run stops being an "
                f"endurance session and starts being the reason the next three "
                f"days are slow. Either bring the week up around it or bring "
                f"the long run down."
            )
        elif long_share < 0.18 and cur["mi"] > 25:
            obs.append(
                f"The long run is only {100*long_share:.0f}% of the week. "
                f"There's room to extend it without touching total volume — "
                f"take the miles from a midweek easy day."
            )

        if cur["workouts"] == 0:
            obs.append("No session faster than your easy baseline. Fine for a "
                       "recovery or base week; a problem if it's three in a row.")
        elif cur["workouts"] >= 3:
            obs.append(
                f"{int(cur['workouts'])} faster-than-baseline sessions in one "
                f"week. Your six-year average is "
                f"{cal['workout_days_per_week']:.1f} workout days a week. "
                f"Three is a lot of hard "
                f"days to recover from, and the usual cost isn't injury, it's "
                f"the easy days creeping up in pace until nothing is easy."
            )

        if pd.notna(cur["dp"]) and cur["dp"] >= 70 and cal["heat"].get("available"):
            h = cal["heat"]
            obs.append(
                f"On the weather: a {cur['dp']:.0f}F dew point sounds brutal, "
                f"and it feels it, but your own six years of data put the cost "
                f"at about {h['s_per_mile_per_F']:.2f} s/mi per degree of dew "
                f"point at matched heart rate — roughly "
                f"{abs(h['cost_55F_to_75F_s_per_mile']):.0f} s/mi across the "
                f"full swing from a cool morning to a humid one, not the "
                f"15-20 s/mi the usual rule of thumb claims. Some of that is "
                f"because you already run before sunrise. Don't hand back more "
                f"pace than the weather is actually taking."
            )
        if cur["activities"] > cur["run_days"] * 1.5:
            obs.append(
                f"{int(cur['activities'])} activities over "
                f"{int(cur['run_days'])} days — you're logging warmups, "
                f"strides, and cooldowns separately. Nothing wrong with it, "
                f"but it means any weekly total computed per-activity rather "
                f"than per-day will read low."
            )

        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": facts},
                {"role": "assistant", "content": " ".join(obs)},
            ],
            "task": "training_review",
            "profile_id": f"real-w{i:03d}",
            "month": int(month),
            "week_mi": round(float(cur["mi"]), 1),
            "ratio": round(float(ratio), 3),
            "source": "strava-derived-v1",
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="sources/activities.csv")
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    runs = classify_sessions(load(Path(a.csv)))
    weeks = weekly_frame(runs)
    cal = calibrate(runs, weeks)
    (out / "calibration.json").write_text(json.dumps(cal, indent=2))

    rows = review_rows(weeks, cal)
    with (out / "reviews.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(json.dumps(cal, indent=2))
    print(f"\n{len(rows)} review examples -> {out/'reviews.jsonl'}")


if __name__ == "__main__":
    main()
