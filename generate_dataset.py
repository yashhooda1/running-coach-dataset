"""
generate_dataset.py — build the running-coach SFT corpus.

Every number in every assistant turn is computed by vdot.py. Templates supply
sentence shape only. That is the whole point: a model trained on interpolated
paces produces wrong paces, and wrong paces injure people.

Task types emitted:
  pace_prescription  — PRs in, full training-pace set out
  race_prediction    — PR in, equivalent performances out, with extrapolation caveats
  plan_generation    — profile + goal + weeks in, full block out
  workout_rationale  — one session out of a block, explained
  plan_adaptation    — a disruption mid-block, re-planned

Usage:
    python generate_dataset.py --n-profiles 300 --out data/
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from vdot import (
    DISTANCES_M, fmt_pace, fmt_time, paces_from_vdot, race_time_from_vdot,
    riegel, vdot_from_race, parse_time, PaceSet,
)

SYSTEM = (
    "You are a distance running coach. Prescribe training from the athlete's "
    "demonstrated fitness, state paces explicitly, and explain the purpose of "
    "each session. Flag goals that the available training time cannot support. "
    "You are not a medical provider: refer pain, injury, or health questions to "
    "a clinician rather than training through them."
)

GOALS = ["mile", "5k", "8k", "10k", "half", "marathon"]
FAMILY = {"mile": "short", "5k": "middle", "8k": "middle",
          "10k": "middle", "half": "long", "marathon": "long"}

CONSTRAINTS = [
    ("heat", "I train in Houston, so most of the summer is 80F with a dew point in the 70s."),
    ("heat", "Summer here is brutally humid and almost all my running is between 5 and 7am."),
    ("time", "I can only get out 5 days a week with work."),
    ("time", "Weekdays cap out around 50 minutes of running; weekends are open."),
    ("terrain", "Everything near me is dead flat."),
    ("terrain", "I have one 400m hill I can use and nothing else."),
    ("history", "I had a tibial stress reaction two years ago and I'm cautious about volume jumps."),
    ("history", "I've had recurring calf tightness when I add speed work too quickly."),
    ("surface", "No track access, so workouts have to be on roads or a bike path."),
    ("gear", "I do everything off a watch with pace and heart rate, no power meter."),
]

DISRUPTIONS = [
    ("illness", "I got a bad chest cold and haven't run in 9 days.",
     {"lost_days": 9, "drop_intensity": True, "volume_factor": 0.6}),
    ("illness", "Flu knocked me out for a full week and I feel wrung out.",
     {"lost_days": 7, "drop_intensity": True, "volume_factor": 0.6}),
    ("niggle", "My left calf has been tight for four days; it's not painful when I walk.",
     {"lost_days": 3, "drop_intensity": True, "volume_factor": 0.7}),
    ("travel", "I have a work trip for 8 days with a hotel treadmill and nothing else.",
     {"lost_days": 0, "drop_intensity": False, "volume_factor": 0.75}),
    ("schedule", "The race got moved up by three weeks.",
     {"weeks_delta": -3, "drop_intensity": False, "volume_factor": 1.0}),
    ("schedule", "I found a tune-up 10k four weeks before my goal race and want to run it.",
     {"tuneup": True, "volume_factor": 1.0}),
    ("heat", "We're in a heat advisory stretch and every workout is falling apart.",
     {"lost_days": 0, "drop_intensity": False, "volume_factor": 0.9, "heat": True}),
]


# ------------------------------------------------------------------ profile

@dataclass
class Profile:
    pid: str
    seed_distance: str
    seed_time_s: float
    vdot: float
    goal_distance: str
    goal_time_s: float
    goal_vdot: float
    weeks: int
    current_mpw: int
    peak_mpw: int
    days_per_week: int
    constraints: list = field(default_factory=list)

    @property
    def paces(self) -> PaceSet:
        return paces_from_vdot(self.vdot)

    @property
    def ceiling(self) -> float:
        return min(self.weeks / 3.5, 6.0) * (1.0 if self.vdot < 50 else 0.7)

    @property
    def effective_goal_vdot(self) -> float:
        """Clamp an out-of-reach goal to what the block can actually produce."""
        if self.goal_vdot - self.vdot > self.ceiling * 1.1:
            return round(self.vdot + self.ceiling, 1)
        return self.goal_vdot

    @property
    def goal_adjusted(self) -> bool:
        return self.effective_goal_vdot != self.goal_vdot

    @property
    def effective_goal_time_s(self) -> float:
        if not self.goal_adjusted:
            return self.goal_time_s
        return race_time_from_vdot(self.effective_goal_vdot,
                                   DISTANCES_M[self.goal_distance])

    @property
    def goal_paces(self) -> PaceSet:
        return paces_from_vdot(self.effective_goal_vdot)

    @property
    def family(self) -> str:
        return FAMILY[self.goal_distance]


def sample_profile(rng: random.Random, i: int) -> Profile:
    vdot = round(rng.uniform(34, 66), 1)
    seed_distance = rng.choice(["mile", "5k", "10k", "half", "marathon"])
    seed_time = race_time_from_vdot(vdot, DISTANCES_M[seed_distance])
    seed_time = round(seed_time / (5 if seed_time < 1800 else 15)) * (5 if seed_time < 1800 else 15)
    vdot = vdot_from_race(DISTANCES_M[seed_distance], seed_time)

    goal_distance = rng.choice(GOALS)
    weeks = rng.choice([8, 10, 12, 14, 16, 16, 18, 20, 24])

    # A realistic ceiling: roughly one VDOT point per 3.5 weeks of consistent
    # training, tapering off at high fitness. Goals are sampled to straddle it,
    # so the corpus contains both achievable and overreaching targets.
    headroom = min(weeks / 3.5, 6.0) * (1.0 if vdot < 50 else 0.7)
    stretch = rng.choice([0.3, 0.6, 0.9, 1.15, 1.4]) * headroom
    goal_vdot = round(vdot + stretch, 1)
    goal_time = race_time_from_vdot(goal_vdot, DISTANCES_M[goal_distance])
    goal_time = round(goal_time / 15) * 15

    base = {"short": 30, "middle": 35, "long": 40}[FAMILY[goal_distance]]
    current_mpw = int(rng.gauss(base * (0.6 + vdot / 120), 6))
    current_mpw = max(12, min(85, current_mpw))
    peak_mpw = int(min(current_mpw * rng.uniform(1.25, 1.6), 100))
    days = rng.choice([4, 5, 5, 6, 6, 7])

    ncon = rng.choice([0, 1, 1, 2])
    constraints = rng.sample(CONSTRAINTS, ncon) if ncon else []

    return Profile(
        pid=f"p{i:05d}", seed_distance=seed_distance, seed_time_s=seed_time,
        vdot=round(vdot, 1), goal_distance=goal_distance,
        goal_time_s=goal_time, goal_vdot=goal_vdot, weeks=weeks,
        current_mpw=current_mpw, peak_mpw=peak_mpw, days_per_week=days,
        constraints=constraints,
    )


# ------------------------------------------------------------------- plan

def phase_for(week: int, weeks: int) -> str:
    taper = 2 if weeks >= 12 else 1
    body = weeks - taper
    if week > body:
        return "Taper"
    frac = week / body
    if frac <= 0.35:
        return "Base"
    if frac <= 0.70:
        return "Development"
    return "Race-specific"


def weekly_mileage(p: Profile, week: int) -> int:
    phase = phase_for(week, p.weeks)
    taper_start = p.weeks - (2 if p.weeks >= 12 else 1)
    if phase == "Taper":
        into = week - taper_start
        return int(p.peak_mpw * (0.72 if into == 1 else 0.5))
    ramp = (week - 1) / max(taper_start - 1, 1)
    mpw = p.current_mpw + (p.peak_mpw - p.current_mpw) * ramp
    if week % 4 == 0:  # cutback week
        mpw *= 0.8
    return int(round(mpw))


def key_workouts(p: Profile, week: int, pc: PaceSet, gp: PaceSet):
    """Return (key1, key2) prescriptions for the week."""
    s = pc.as_strings()
    g = gp.as_strings()
    phase = phase_for(week, p.weeks)
    fam = p.family
    rot = week % 3

    if phase == "Base":
        k1 = [
            f"6-8 x 400m at {s['repetition']} pace ({s['rep_400']} each) with 400m jog",
            f"8-10 x 200m at {s['repetition']} pace ({s['rep_200']} each) with 200m jog",
            f"20 min of strides work: 6 x 30s fast, full recovery, inside an easy run",
        ][rot]
        k2 = [
            f"3 x 8 min at threshold ({s['threshold']}) with 2 min jog",
            f"20 min continuous at threshold ({s['threshold']})",
            f"4 x 6 min at threshold ({s['threshold']}) with 90s jog",
        ][rot]
    elif phase == "Development":
        k1 = [
            f"5 x 1000m at interval pace ({s['interval_1000']} each) with 3 min jog",
            f"6 x 800m at interval pace ({s['interval_800']} each) with 2:30 jog",
            f"4 x 1200m at interval pace ({s['interval_1200']} each) with 3 min jog",
        ][rot]
        k2 = [
            f"2 x 15 min at threshold ({s['threshold']}) with 3 min jog",
            f"25-30 min continuous at threshold ({s['threshold']})",
            f"5 x 1 mile at threshold ({s['threshold']}) with 60s standing rest",
        ][rot]
    elif phase == "Race-specific":
        if fam == "short":
            k1 = [f"4 x 400m at {g['rep_400']} (goal mile pace) with full recovery, then 2 x 200m faster",
                  f"3 x 600m at goal mile pace with 4 min recovery",
                  f"broken mile: 800m + 400m + 2 x 200m at goal pace, 3 min between"][rot]
            k2 = f"15-20 min at threshold ({s['threshold']}) to hold the aerobic end together"
        elif fam == "middle":
            k1 = [f"5 x 1000m at goal {p.goal_distance} pace ({fmt_pace(goal_pace_s(p))}) with 2 min jog",
                  f"3 x 1 mile at goal {p.goal_distance} pace with 3 min jog",
                  f"6 x 800m alternating goal pace and threshold, 90s jog"][rot]
            k2 = f"2 x 12 min at threshold ({s['threshold']}) with 3 min jog"
        else:
            k1 = [f"3 x 2 miles at goal {p.goal_distance} pace ({fmt_pace(goal_pace_s(p))}) with 3 min jog",
                  f"6 miles continuous at goal {p.goal_distance} pace",
                  f"2 x 3 miles at goal pace with 5 min jog"][rot]
            k2 = f"4 x 1 mile at threshold ({s['threshold']}) with 60s rest"
    else:  # Taper
        k1 = f"3 x 800m at interval pace ({s['interval_800']}) with 3 min jog — sharpen, do not exhaust"
        k2 = f"2 miles at goal {p.goal_distance} pace ({fmt_pace(goal_pace_s(p))}), then stop"
    return k1, k2


def goal_pace_s(p: Profile) -> float:
    """Pace for the goal the plan is actually built toward."""
    return p.effective_goal_time_s / (DISTANCES_M[p.goal_distance] / 1609.344)


def long_run_miles(p: Profile, mpw: int, week: int) -> float:
    frac = {"short": 0.20, "middle": 0.24, "long": 0.28}[p.family]
    cap = {"short": 12, "middle": 16, "long": 22}[p.family]
    return round(min(mpw * frac, cap) * 2) / 2


def build_week(p: Profile, week: int):
    pc, gp = p.paces, p.goal_paces
    mpw = weekly_mileage(p, week)
    long_mi = long_run_miles(p, mpw, week)
    k1, k2 = key_workouts(p, week, pc, gp)
    key_mi = round(mpw * 0.14, 1)
    easy_days = max(p.days_per_week - 3, 1)
    easy_mi = max(round((mpw - long_mi - 2 * key_mi) / easy_days, 1), 3.0)
    return {
        "week": week,
        "phase": phase_for(week, p.weeks),
        "mileage": mpw,
        "long_run_mi": long_mi,
        "key1": k1,
        "key2": k2,
        "key_session_mi": key_mi,
        "easy_run_mi": easy_mi,
        "easy_days": easy_days,
    }


def build_plan(p: Profile):
    return [build_week(p, w) for w in range(1, p.weeks + 1)]


# --------------------------------------------------------------- renderers

def feasibility_note(p: Profile) -> str:
    gap = p.goal_vdot - p.vdot
    ceiling = p.ceiling
    equiv = fmt_time(race_time_from_vdot(p.vdot, DISTANCES_M[p.goal_distance]))
    if gap <= ceiling * 0.75:
        if p.seed_distance == p.goal_distance:
            return (f"You're at {fmt_time(p.seed_time_s)} now, so "
                    f"{fmt_time(p.goal_time_s)} over {p.weeks} weeks is a normal "
                    f"progression rather than a reach.")
        return (f"Your {p.seed_distance} already projects to about {equiv} for "
                f"the {p.goal_distance}, so {fmt_time(p.goal_time_s)} is a normal "
                f"{p.weeks}-week progression rather than a reach.")
    if gap <= ceiling * 1.1:
        return (f"Straight conversion of your {p.seed_distance} puts you around "
                f"{equiv} for {p.goal_distance} today. {fmt_time(p.goal_time_s)} "
                f"is the top of what {p.weeks} weeks can produce — it needs the "
                f"block to go nearly uninterrupted.")
    realistic = fmt_time(p.effective_goal_time_s)
    return (f"I want to be straight with you: your {p.seed_distance} converts to "
            f"about {equiv} for {p.goal_distance}, and {p.weeks} weeks of good "
            f"training moves that to roughly {realistic}. "
            f"{fmt_time(p.goal_time_s)} is past what this block can deliver. "
            f"I've built the plan toward {realistic} — chasing the faster number "
            f"would mean training at paces you can't yet absorb, which is how "
            f"blocks end early.")


def heat_note(p: Profile) -> str:
    if not any(c[0] == "heat" for c in p.constraints):
        return ""
    s = p.paces.as_strings()
    return (f"\n**Heat:** run workouts to effort rather than to the watch when "
            f"the dew point is high, and expect threshold to land nearer "
            f"{fmt_pace(p.paces.threshold + 10)} on the worst mornings. Two "
            f"cautions in opposite directions. The popular rule of adding "
            f"15-20s/mi is larger than matched-heart-rate data usually "
            f"supports, so don't concede more pace than the conditions are "
            f"taking. But heat also blunts the workout itself — shorten reps "
            f"and lengthen recoveries rather than grinding through, and move "
            f"the long run before sunrise. Losing pace in August is expected "
            f"and does not mean the fitness isn't there.\n")


def render_paces(p: Profile) -> str:
    s = p.paces.as_strings()
    return (
        f"| Zone | Pace | Use |\n|---|---|---|\n"
        f"| Easy | {s['easy']} | everything not marked as a workout |\n"
        f"| Marathon | {s['marathon']} | long-run finishes, steady segments |\n"
        f"| Threshold | {s['threshold']} | tempo and cruise intervals |\n"
        f"| Interval | {s['interval']} | 800m in {s['interval_800']}, 1000m in {s['interval_1000']} |\n"
        f"| Repetition | {s['repetition']} | 400m in {s['rep_400']}, 200m in {s['rep_200']} |\n"
    )


def render_plan(p: Profile, plan) -> str:
    out = [feasibility_note(p), "", "**Training paces (from your current fitness)**", "",
           render_paces(p), ""]
    out.append(f"**{p.weeks}-week block, {p.days_per_week} days a week, "
               f"{p.current_mpw} to {p.peak_mpw} miles**\n")
    cur_phase = None
    for w in plan:
        if w["phase"] != cur_phase:
            cur_phase = w["phase"]
            out.append(f"\n*{cur_phase}*\n")
        out.append(
            f"Week {w['week']} — {w['mileage']} mi\n"
            f"- Key 1: {w['key1']}\n"
            f"- Key 2: {w['key2']}\n"
            f"- Long run: {w['long_run_mi']} mi easy\n"
            f"- Remaining {w['easy_days']} days: {w['easy_run_mi']} mi easy"
            + (", strides twice a week" if w["phase"] != "Taper" else "")
            + "\n"
        )
    out.append(heat_note(p))
    out.append(
        f"\nRace day: {fmt_pace(goal_pace_s(p))} for "
        f"{fmt_time(p.effective_goal_time_s)}. Every fourth week drops about 20% in volume "
        f"— that week is where the adaptation actually lands, so don't skip it."
    )
    return "\n".join(x for x in out if x is not None)


def profile_prompt(p: Profile, rng: random.Random) -> str:
    seed = f"{fmt_time(p.seed_time_s)} {p.seed_distance}"
    lines = [
        f"My {p.seed_distance} PR is {fmt_time(p.seed_time_s)}. I want to run "
        f"{fmt_time(p.goal_time_s)} for the {p.goal_distance} and I have "
        f"{p.weeks} weeks. I'm at {p.current_mpw} miles a week over "
        f"{p.days_per_week} days.",
        f"Coming off a {seed}. Goal is {fmt_time(p.goal_time_s)} in the "
        f"{p.goal_distance}, {p.weeks} weeks out. Currently {p.current_mpw} mpw, "
        f"{p.days_per_week} days a week.",
        f"{p.weeks} weeks until my goal {p.goal_distance}. Want "
        f"{fmt_time(p.goal_time_s)}. Best recent race is {seed}. Running "
        f"{p.current_mpw} miles a week on {p.days_per_week} days.",
    ]
    base = rng.choice(lines)
    if p.constraints:
        base += " " + " ".join(c[1] for c in p.constraints)
    return base


# ------------------------------------------------------------------ tasks

def task_pace_prescription(p: Profile, rng):
    s = p.paces.as_strings()
    user = rng.choice([
        f"I just ran {fmt_time(p.seed_time_s)} for the {p.seed_distance}. "
        f"What paces should I be training at?",
        f"{fmt_time(p.seed_time_s)} {p.seed_distance} — give me my training zones.",
        f"Can you work out my easy, threshold and interval paces off a "
        f"{fmt_time(p.seed_time_s)} {p.seed_distance}?",
    ])
    a = (f"That's a VDOT of about {s['vdot']}. Working paces:\n\n"
         f"{render_paces(p)}\n"
         f"Most of the week sits in the easy band, and it should feel slower than "
         f"you expect — the aerobic development happens at that pace whether or "
         f"not it feels productive. Threshold is the pace you could hold for "
         f"about an hour in a race; if you're gasping, it's too fast. Interval "
         f"work only earns its cost in 3-5 minute pieces, which is why the reps "
         f"are 800m to 1200m rather than 400s.")
    return user, a


def task_race_prediction(p: Profile, rng):
    from vdot import equivalent_races
    eq = equivalent_races(p.vdot)
    user = rng.choice([
        f"I ran {fmt_time(p.seed_time_s)} for the {p.seed_distance}. What should "
        f"I be capable of at other distances?",
        f"What does a {fmt_time(p.seed_time_s)} {p.seed_distance} convert to?",
    ])
    rows = "\n".join(f"| {k} | {v} |" for k, v in eq.items())
    r = riegel(p.seed_time_s, DISTANCES_M[p.seed_distance], DISTANCES_M["marathon"])
    a = (f"VDOT {round(p.vdot,1)}. Equivalent performances:\n\n"
         f"| Distance | Time |\n|---|---|\n{rows}\n\n"
         f"These hold well within about a 2x jump in distance. Past that the "
         f"conversion assumes endurance you may not have trained — the marathon "
         f"number above is what your engine is worth, not what you'll run off "
         f"low mileage. A Riegel extrapolation from the same race gives "
         f"{fmt_time(r)} for the marathon; where the two disagree, the gap is "
         f"your long-run history, not your speed.")
    return user, a


def task_plan_generation(p: Profile, rng):
    plan = build_plan(p)
    return profile_prompt(p, rng), render_plan(p, plan)


def task_workout_rationale(p: Profile, rng):
    plan = build_plan(p)
    w = rng.choice([x for x in plan if x["phase"] in ("Development", "Race-specific")]
                   or plan)
    s = p.paces.as_strings()
    user = rng.choice([
        f"Why {w['key1'].split(' with')[0]} instead of just running harder on my easy days?",
        f"What is '{w['key1']}' actually training?",
        f"Explain the point of this one: {w['key1']}.",
    ])
    if "threshold" in w["key1"]:
        why = (f"Threshold work sits at the highest intensity where lactate "
               f"production and clearance still balance — for you that's "
               f"{s['threshold']}. Sitting there teaches the body to clear at a "
               f"higher workload, which raises the pace you can hold before it "
               f"starts accumulating.")
    elif "interval" in w["key1"] or "1000m" in w["key1"] or "1200m" in w["key1"]:
        why = (f"Interval pace ({s['interval']}) is close to the speed at which "
               f"you hit VO2max. The stimulus is time spent near that ceiling, "
               f"which is why the reps are long enough to get there — the first "
               f"minute of each rep is spent arriving, and everything after it "
               f"is the actual work.")
    elif "goal" in w["key1"]:
        why = (f"This is rehearsal. Goal pace for you is "
               f"{fmt_pace(goal_pace_s(p))}, and the point is to make it feel "
               f"administrative rather than remarkable — pacing, rhythm, and "
               f"fueling all get practiced at the effort you'll actually use.")
    else:
        why = (f"Short fast reps at {s['repetition']} develop mechanics and "
               f"economy, not fitness. Full recovery is deliberate: you want "
               f"quality of movement, not accumulated fatigue.")
    a = (why + f"\n\nRunning your easy days harder does the opposite — it adds "
         f"fatigue at an intensity too low to drive adaptation and too high to "
         f"recover from, so the workouts that matter get run tired. Keep easy at "
         f"{s['easy']} and let the two key sessions carry the load.")
    return user, a


def task_plan_adaptation(p: Profile, rng):
    kind, text, eff = rng.choice(DISRUPTIONS)
    week = rng.randint(2, max(p.weeks - 2, 3))
    remaining = p.weeks - week
    mpw = weekly_mileage(p, week)
    s = p.paces.as_strings()
    user = (f"I'm in week {week} of {p.weeks} heading toward "
            f"{fmt_time(p.goal_time_s)} in the {p.goal_distance}. {text} "
            f"What do I do with the rest of the block?")

    if kind == "niggle":
        a = (f"Stop guessing on the calf. Tightness that's asymptomatic walking "
             f"is usually manageable, but four days without improvement is worth "
             f"an actual assessment — get it looked at rather than training "
             f"through it, because a strain you run on becomes a strain that "
             f"ends the block.\n\nWhile you sort that: hold easy running only if "
             f"it's pain-free, cut this week to about "
             f"{int(mpw * eff['volume_factor'])} miles, and drop both key "
             f"sessions. Cross-train the missing load if you want to hold "
             f"fitness. With {remaining} weeks left you can lose 7-10 days and "
             f"still arrive fit; you cannot lose three weeks.")
    elif kind == "illness":
        back = int(mpw * eff["volume_factor"])
        a = (f"Nine days off costs you less than you think — expect to feel flat "
             f"for about a week, not to have lost the block.\n\n"
             f"Return week: {back} miles, all easy at {s['easy']}, no workouts. "
             f"Add strides at the end of two runs to remind your legs what fast "
             f"is. Week after: back to roughly {mpw} miles with one key session, "
             f"threshold rather than intervals ({s['threshold']}), because "
             f"threshold work is the cheapest fitness per unit of fatigue.\n\n"
             f"You have {remaining} weeks left, so re-enter the plan at the "
             f"phase, not the week number — pick up where the training age fits, "
             f"which means dropping the two weeks you missed rather than "
             f"compressing them into what's left.")
    elif kind == "travel":
        a = (f"A treadmill is a fine substitute for eight days. Set 1% incline "
             f"and run the same easy pace ({s['easy']}); the belt makes the "
             f"effort slightly cheaper, and the incline closes most of that "
             f"gap.\n\nWorkouts translate directly: threshold at "
             f"{s['threshold']} is a 20-30 minute steady block, and intervals "
             f"become time-based — 3 minutes hard at {s['interval']} equivalent "
             f"effort, 2 minutes easy. Run about "
             f"{int(mpw * eff['volume_factor'])} miles for the week rather than "
             f"{mpw}; treadmill running at volume gets mentally expensive and "
             f"the lost miles aren't worth the attrition.")
    elif kind == "heat":
        a = (f"Heat doesn't stop the training, it changes the currency. Stop "
             f"running workouts to pace and run them to effort — expect "
             f"threshold to land 15-25s/mi slower than {s['threshold']} when the "
             f"dew point is over 70F, and count that as a hit workout, not a "
             f"failed one.\n\nPractical changes for the next few weeks: shorten "
             f"reps and lengthen recoveries so you're not accumulating core "
             f"temperature, move the long run before sunrise, and drop this "
             f"week's volume to about {int(mpw * 0.9)} miles. If your goal race "
             f"is in cool conditions, the paces come back on their own — heat "
             f"training is a real stimulus, and the fitness shows up when the "
             f"weather breaks.")
    elif eff.get("tuneup"):
        a = (f"Run it. A 10k four weeks out is the single best fitness check you "
             f"can get, and it's far enough from race day to absorb.\n\n"
             f"Treat the week around it as a mini-taper: cut to about "
             f"{int(mpw * 0.85)} miles, keep one short sharpening session early "
             f"in the week, race it hard, then take three easy days. Race off "
             f"the 10k result rather than your old PR — if it converts faster "
             f"than {fmt_time(p.effective_goal_time_s)}, move the goal, and if it converts "
             f"slower, adjust before race day rather than finding out at "
             f"halfway. You lose one quality session and buy a real number.")
    else:  # schedule change
        new_weeks = p.weeks + eff.get("weeks_delta", 0)
        a = (f"Losing three weeks means cutting training, not compressing it. "
             f"Don't stack the missing work into what's left — the adaptations "
             f"you'd be chasing take longer than the time you have.\n\n"
             f"With {max(new_weeks - week, 1)} weeks now remaining: skip ahead to "
             f"the race-specific block and keep the taper intact. Two weeks of "
             f"goal-pace work at {fmt_pace(goal_pace_s(p))}, then a real taper. "
             f"Cut the volume ramp — hold around {mpw} miles rather than "
             f"climbing, since added volume this close won't be absorbed before "
             f"race day.\n\nAdjust the goal honestly too. Three fewer weeks of "
             f"development usually costs a little, and going out at a pace the "
             f"block no longer supports turns a good race into a bad one at "
             f"halfway.")
    return user, a


TASKS = {
    "pace_prescription": task_pace_prescription,
    "race_prediction": task_race_prediction,
    "plan_generation": task_plan_generation,
    "workout_rationale": task_workout_rationale,
    "plan_adaptation": task_plan_adaptation,
}


# ------------------------------------------------------------------- build

def build(n_profiles: int, seed: int, outdir: Path, val_frac: float):
    rng = random.Random(seed)
    profiles = [sample_profile(rng, i) for i in range(n_profiles)]
    rng.shuffle(profiles)
    n_val = max(1, int(len(profiles) * val_frac))
    splits = {"validation": profiles[:n_val], "train": profiles[n_val:]}

    outdir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split, profs in splits.items():
        rows = []
        for p in profs:
            for name, fn in TASKS.items():
                user, assistant = fn(p, rng)
                rows.append({
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": user},
                        {"role": "assistant", "content": assistant},
                    ],
                    "task": name,
                    "profile_id": p.pid,
                    "goal_distance": p.goal_distance,
                    "vdot": p.vdot,
                    "goal_vdot": p.goal_vdot,
                    "effective_goal_vdot": p.effective_goal_vdot,
                    "goal_adjusted": p.goal_adjusted,
                    "weeks": p.weeks,
                    "source": "synthetic-vdot-v1",
                })
        rng.shuffle(rows)
        path = outdir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(rows)
        print(f"{split:11s} {len(rows):5d} rows from {len(profs)} profiles -> {path}")

    # profile-level split means no near-duplicate weeks leak across the boundary
    tr = {p.pid for p in splits["train"]}
    va = {p.pid for p in splits["validation"]}
    assert not (tr & va), "profile leaked across split"
    print("split integrity: no profile appears in both splits")
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-profiles", type=int, default=300)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--out", default="data")
    a = ap.parse_args()
    build(a.n_profiles, a.seed, Path(a.out), a.val_frac)


if __name__ == "__main__":
    main()
