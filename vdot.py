"""
vdot.py — Daniels/Gilbert VDOT model.

This is the arithmetic layer. Every pace that appears in the generated dataset
comes from here, never from a template string, so the corpus can't teach the
model wrong numbers.

References:
  VO2 (ml/kg/min) at velocity v (m/min):
      VO2 = -4.60 + 0.182258*v + 0.000104*v^2
  Fraction of VO2max sustainable for t minutes:
      %max = 0.8 + 0.1894393*e^(-0.012778t) + 0.2989558*e^(-0.1932605t)
  VDOT = VO2(race) / %max(race duration)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

MILE_M = 1609.344

DISTANCES_M = {
    "1500m": 1500.0,
    "mile": MILE_M,
    "3000m": 3000.0,
    "2mile": 2 * MILE_M,
    "5k": 5000.0,
    "8k": 8000.0,
    "5mile": 5 * MILE_M,
    "10k": 10000.0,
    "15k": 15000.0,
    "10mile": 10 * MILE_M,
    "half": 21097.5,
    "marathon": 42195.0,
}

# Fraction of VDOT that defines each training intensity.
# Tuned so the outputs land on the published Daniels tables at VDOT 45/55/65.
INTENSITY = {
    "E_slow": 0.62,   # easy, slow end
    "E_fast": 0.72,   # easy, fast end
    "M": 0.842,       # marathon pace
    "T": 0.884,       # threshold / tempo
    "I": 0.977,       # interval (VO2max)
    "R": 1.058,       # repetition (speed)
}


def vo2_at_velocity(v_m_min: float) -> float:
    return -4.60 + 0.182258 * v_m_min + 0.000104 * v_m_min**2


def velocity_at_vo2(vo2: float) -> float:
    a, b, c = 0.000104, 0.182258, -4.60 - vo2
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


def pct_max_for_duration(t_min: float) -> float:
    return (
        0.8
        + 0.1894393 * math.exp(-0.012778 * t_min)
        + 0.2989558 * math.exp(-0.1932605 * t_min)
    )


def vdot_from_race(distance_m: float, time_s: float) -> float:
    t_min = time_s / 60.0
    v = distance_m / t_min
    return vo2_at_velocity(v) / pct_max_for_duration(t_min)


def race_time_from_vdot(vdot: float, distance_m: float) -> float:
    """Invert the VDOT relation by bisection. Returns seconds."""
    lo, hi = 60.0, 6 * 3600.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if vdot_from_race(distance_m, mid) > vdot:
            lo = mid  # too fast a time implies a higher VDOT -> go slower
        else:
            hi = mid
    return (lo + hi) / 2


def riegel(time_s: float, from_m: float, to_m: float, exp: float = 1.06) -> float:
    """Riegel cross-check. Diverges from VDOT past ~2x distance extrapolation."""
    return time_s * (to_m / from_m) ** exp


def pace_s_per_mile(v_m_min: float) -> float:
    return MILE_M / v_m_min * 60.0


def pace_for_intensity(vdot: float, key: str) -> float:
    """Seconds per mile at the given Daniels intensity."""
    v = velocity_at_vo2(INTENSITY[key] * vdot)
    return pace_s_per_mile(v)


def fmt_time(seconds: float) -> str:
    seconds = round(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def fmt_rep(seconds: float) -> str:
    """Rep splits under a minute read better as '41s' than '0:41'."""
    return f"{round(seconds)}s" if seconds < 60 else fmt_time(seconds)


def fmt_pace(s_per_mile: float) -> str:
    return fmt_time(s_per_mile) + "/mi"


def parse_time(text: str) -> float:
    """'18:15' | '1:24:30' | '4:58' -> seconds."""
    parts = [float(p) for p in text.strip().split(":")]
    out = 0.0
    for p in parts:
        out = out * 60 + p
    return out


@dataclass
class PaceSet:
    vdot: float
    easy_slow: float
    easy_fast: float
    marathon: float
    threshold: float
    interval: float
    repetition: float

    def as_strings(self) -> dict:
        return {
            "vdot": round(self.vdot, 1),
            "easy": f"{fmt_time(self.easy_fast)}-{fmt_time(self.easy_slow)}/mi",
            "marathon": fmt_pace(self.marathon),
            "threshold": fmt_pace(self.threshold),
            "interval": fmt_pace(self.interval),
            "repetition": fmt_pace(self.repetition),
            "interval_400": fmt_rep(self.interval * 0.25),
            "interval_800": fmt_time(self.interval * 0.5),
            "interval_1000": fmt_time(self.interval * 1000 / MILE_M),
            "interval_1200": fmt_time(self.interval * 1200 / MILE_M),
            "rep_200": fmt_rep(self.repetition * 200 / MILE_M),
            "rep_400": fmt_rep(self.repetition * 0.25),
        }

    def dict(self) -> dict:
        return asdict(self)


def paces_from_vdot(vdot: float) -> PaceSet:
    return PaceSet(
        vdot=vdot,
        easy_slow=pace_for_intensity(vdot, "E_slow"),
        easy_fast=pace_for_intensity(vdot, "E_fast"),
        marathon=pace_for_intensity(vdot, "M"),
        threshold=pace_for_intensity(vdot, "T"),
        interval=pace_for_intensity(vdot, "I"),
        repetition=pace_for_intensity(vdot, "R"),
    )


def paces_from_race(distance_key: str, time_text: str) -> PaceSet:
    d = DISTANCES_M[distance_key]
    return paces_from_vdot(vdot_from_race(d, parse_time(time_text)))


def equivalent_races(vdot: float, keys=None) -> dict:
    keys = keys or ["mile", "5k", "8k", "10k", "half", "marathon"]
    return {k: fmt_time(race_time_from_vdot(vdot, DISTANCES_M[k])) for k in keys}


if __name__ == "__main__":
    for label, dist, t in [
        ("Yash 5k PR", "5k", "18:15"),
        ("Yash mile PR", "mile", "4:58"),
        ("Yash half PR", "half", "1:24:00"),
        ("sub-3 marathon", "marathon", "2:59:59"),
        ("sub-30 8k", "8k", "29:59"),
        ("sub-1:20 half", "half", "1:19:59"),
    ]:
        v = vdot_from_race(DISTANCES_M[dist], parse_time(t))
        p = paces_from_vdot(v).as_strings()
        print(f"{label:16s} {t:>8s}  VDOT {v:5.1f}  E {p['easy']}  T {p['threshold']}  I {p['interval']}  R400 {p['rep_400']}")
        print(f"{'':16s} equivalents: {equivalent_races(v)}")
