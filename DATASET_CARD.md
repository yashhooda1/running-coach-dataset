---
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
tags:
  - running
  - coaching
  - sports-science
  - synthetic
  - instruction-tuning
pretty_name: Running Coach SFT
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
  - config_name: reference
    data_files:
      - split: reference
        path: data/reference.jsonl
---

# Running Coach SFT

Instruction-tuning data for a distance-running coaching assistant. Every pace,
split, and race-equivalent in the corpus is computed from a Daniels/Gilbert VDOT
implementation rather than written into a template, so the numbers are internally
consistent across all 1,500 examples.

## Why this exists

Coaching corpora scraped from forums and blogs teach a model the register of
coaching without the arithmetic underneath it. A model that interpolates training
paces produces paces that are wrong by 10-30 seconds per mile, which is the
difference between a threshold run and a hard tempo, and repeated over a training
block that difference is how people get hurt. Here the numbers come from a model
and the templates supply sentence shape only.

## Splits

| Split | Rows | Contents |
|---|---|---|
| `train` | 1,350 | synthetic, 270 athlete profiles × 5 tasks |
| `validation` | 150 | synthetic, 30 profiles, disjoint from train |
| `reference` | 4 | source documents, held out — see Provenance |

`train` and `validation` each contain two kinds of row: synthetic examples
generated from the VDOT model, and `training_review` examples derived from a
real six-year Strava history (1,648 runs, 2017-2023). The synthetic rows are
split by athlete profile; the review rows are split as a **contiguous tail**,
because consecutive real weeks are autocorrelated and a shuffled split would put
week 41 in train and week 42 in validation.

Splitting is at the **profile** level, not the example level. Five tasks derived
from one athlete profile share paces and plan structure; splitting at the row
level would leak them across the boundary and make validation loss optimistic.

## Tasks

| Task | Input | Output |
|---|---|---|
| `pace_prescription` | a race result | full training-pace set with usage notes |
| `race_prediction` | a race result | equivalent performances, with extrapolation caveats |
| `plan_generation` | fitness, goal, weeks, mileage, constraints | full training block |
| `workout_rationale` | one session from a block | physiological purpose and why not to run easy days harder |
| `plan_adaptation` | a mid-block disruption | revised plan |
| `training_review` | a real week's summary statistics | feedback computed from that week against the athlete's own six-year baseline |

## Schema

```json
{
  "messages": [
    {"role": "system", "content": "You are a distance running coach. ..."},
    {"role": "user", "content": "My 5k PR is 18:15. I want to run ..."},
    {"role": "assistant", "content": "..."}
  ],
  "task": "plan_generation",
  "profile_id": "p00042",
  "goal_distance": "half",
  "vdot": 55.4,
  "goal_vdot": 58.1,
  "weeks": 16,
  "source": "synthetic-vdot-v1"
}
```

## How the synthetic data is constructed

1. Sample a VDOT in [34, 66] and a seed race distance; round the resulting time
   to a realistic granularity and recompute VDOT from the rounded time.
2. Sample a goal distance, a horizon of 8-24 weeks, current and peak mileage,
   days per week, and zero to two training constraints (heat, no track, flat
   terrain, injury history, limited weekday time).
3. Sample the goal to straddle a realistic improvement ceiling of roughly one
   VDOT point per 3.5 weeks, discounted above VDOT 50. **About a third of
   profiles carry an unreachable goal on purpose**, and the assistant response
   says so and builds toward what the block can deliver instead. A coaching
   model that validates every target is not a useful coaching model.
4. Build the block: phase boundaries from the horizon, a mileage ramp with a
   cutback every fourth week, two key sessions rotating through a phase-specific
   library, and a long run capped by goal distance.
5. Render five task views of the same profile.

## Calibration against real training data

The synthetic generator originally used constants chosen by hand. A Strava bulk
export of 1,648 runs over six years replaced them with measured values:

| Constant | Hand-picked | Measured |
|---|---|---|
| week-over-week mileage ratio, p90 | ~1.10 (linear ramp) | **1.52** |
| cutback week frequency | every 4th week (0.25) | **0.28** of active weeks |
| long-run share of weekly volume | 0.24-0.28 | **0.24** median, 0.39 p90 |
| easy / workout / long session mix | not modeled | **0.84 / 0.08 / 0.07** |
| heat penalty | "add 15-20 s/mi above a 65F dew point" | **0.24 s/mi per F** (see below) |

### The heat rule did not survive

Regressing pace on dew point while controlling for heart rate, distance, and
year-level fitness (n=462 aerobic-band runs) gives **0.235 s/mi per degree F**
(SE 0.066, t=3.6, 95% CI 0.11 to 0.36). That is about **5 s/mi** across the full
swing from a cool morning to a humid one — not the 15-20 s/mi the common rule of
thumb prescribes. The reverse specification (heart rate on dew point, converted
through the HR-pace slope) independently gives 0.31 s/mi per F.

The year fixed effects are load-bearing. Without them the fitness trend across
six seasons swamps the weather signal and the coefficient comes out near zero
with the wrong sign — cold-weather runs are also race-season runs.

Two caveats on generalizing it: this is one athlete, and that athlete runs
before sunrise year-round, so the summer runs are already sampled at the coolest
hour of the day. Someone running at noon would pay more. For that reason the
measured coefficient is quoted **only in `training_review` rows**, where it is
the athlete's own data; the synthetic `plan_generation` rows give effort-based
heat guidance without asserting a number.

## Provenance

The `reference` split contains four training-plan documents written by an AI
assistant in earlier chat sessions and retained by the author. They are included
as a **style and evaluation reference and are not intended as training data**:
their paces are not machine-verified and at least one contains internally
inconsistent prescriptions. `data/seed_plans.json` holds a normalized parse of
the same four documents (99 sessions, classified by type, with distances, rep
structures, and mentioned paces extracted).

The `training_review` rows derive from the dataset author's own Strava export,
published with their consent. Before any analysis runs, the loader drops activity
names, descriptions, private notes, gear, filenames, media links, and athlete
weight. No GPS coordinates, routes, or absolute dates reach an output file:
weeks carry a sequential index and a month number, the latter kept because
seasonality is the point of the heat analysis. The published rows contain
aggregate weekly statistics only, never individual activities.

No scraped forum or blog content and no proprietary coaching material is
included.

## Intended use

Fine-tuning small chat models to give structured, arithmetically consistent
running-training guidance; benchmarking whether a model's prescribed paces
actually follow from the fitness it was given.

## Limitations and risks

- **Not medical advice, and not a substitute for a coach.** The system prompt
  directs refusal to clinical questions, but a fine-tuned model will still
  produce plausible prescriptions for athletes it should be declining.
- **Single physiological model.** Everything derives from Daniels/Gilbert VDOT.
  Pfitzinger, Canova, polarized, and Norwegian-method approaches are absent, so
  a model trained only on this corpus will present one methodology as the whole
  field.
- **Template diversity is bounded.** Sentence shapes come from a fixed library;
  linguistic variety is far lower than in human-written coaching text. Mix with
  general instruction data to avoid style collapse.
- **Constraints are shallow.** Heat, terrain, and injury history change a few
  sentences, not the structure of the block. Real coaching restructures.
- **VDOT extrapolation degrades past about a 2x distance jump.** Marathon
  predictions from a mile time appear in the corpus with caveats attached, but
  the underlying number is still weakly supported.
- **No strength work, nutrition periodization, or return-to-run protocols.**

## Reproduction

```bash
python build.py --n-profiles 300 --seed 11
```

`vdot.py` is self-checking: run it directly to print paces and race equivalents
against known table anchors.
