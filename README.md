# running-coach-dataset

Build a running-coach instruction-tuning corpus, publish it to the Hugging Face
Hub, fine-tune on HF hardware, and evaluate the one thing that matters — whether
the model's prescribed paces actually follow from the athlete's fitness.

```
vdot.py             Daniels/Gilbert VDOT model. Self-checking: `python vdot.py`
strava.py           ingest a Strava export: de-identify, calibrate, build reviews
parse_plans.py      normalize the four seed plan documents into one schema
generate_dataset.py profiles -> plans -> five task views, all paced off vdot.py
build.py            one command for the whole data/ directory
push_to_hub.py      publish the dataset repo
train_sft.py        TRL + LoRA, sized for `hf jobs`
generate_eval.py    run a model over validation, emit generations
eval_paces.py       pace-accuracy scoring
DATASET_CARD.md     uploaded as the repo README
```

## Quickstart

```bash
python build.py --n-profiles 300
python eval_paces.py --generations data/validation.jsonl --field reference   # 0.0% off-zone
python push_to_hub.py --repo <user>/running-coach-sft --dry-run
```

Then train without a local GPU:

```bash
hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \
  train_sft.py --dataset <user>/running-coach-sft \
  --base Qwen/Qwen2.5-7B-Instruct \
  --push <user>/running-coach-qwen7b-lora
```

## Measure the baseline before you train

```bash
python generate_eval.py --base Qwen/Qwen2.5-7B-Instruct --out runs/base.jsonl
python eval_paces.py --generations runs/base.jsonl
```

The base model's off-zone rate is the only thing that makes the post-training
number mean anything. Run it first; it takes one GPU-hour and it is the
difference between "12% off-zone" and "12% off-zone, down from 41%".

## The design decision worth defending

Templates supply sentence shape. `vdot.py` supplies every number. Nothing in the
corpus contains a pace that was typed by hand, because a corpus with hand-typed
paces teaches a model to interpolate them, and interpolated training paces are
wrong by enough to matter.

The same reasoning is why `eval_paces.py` exists and why the reference split
scores 0.0% off-zone before any training happens. That zero is the floor. Any
nonzero rate after fine-tuning is the model drifting off the arithmetic, and it
is the number to report — validation loss will look fine either way.

## Deliberate properties of the corpus

- **Roughly a third of profiles carry an unreachable goal.** The assistant says
  so and builds toward what the block can actually deliver. A coaching model
  that ratifies every target is worse than no model.
- **Splits are profile-disjoint.** Five tasks from one athlete share paces and
  structure; a row-level split leaks them and makes eval loss lie.
- **The clinical boundary is in the system prompt and in the responses.** The
  injury-adaptation examples refer out to a clinician rather than prescribing
  through pain.
- **The four source documents are held out, not trained on.** Their paces are
  not machine-verified and at least one is internally inconsistent.
- **Real data overrides invented constants.** Six years of Strava history set
  the ramp rates, cutback rhythm, long-run share, and session mix. The heat
  rule I had written by hand — "add 15-20 s/mi above a 65F dew point" — came
  out at 0.24 s/mi per degree in the data, about 5 s/mi across a full dew point
  swing, and was removed rather than kept as a plausible-sounding number.

## Honest limitation

If the goal is a coach you would actually run off, the stronger architecture is
a small model with `vdot.py` exposed as a tool plus retrieval over plan
templates — the numbers stay exact by construction instead of being approximated
by weights. This repo is the fine-tuning path; the two are not mutually
exclusive, and the same `vdot.py` serves both.

# Results

Add this to `README.md`, after "The design decision worth defending".

---

## Results

Qwen2.5-3B-Instruct, LoRA on 1,527 examples, scored on 150 profile-disjoint
validation rows with identical prompts and greedy decoding.

| | paces emitted | median err | p90 err | off-zone |
|---|---|---|---|---|
| reference (corpus ground truth) | 1,066 | 0.2s | 0.5s | **0.0%** |
| base model | 349 | 7.4s | 79.6s | **47.3%** |
| + fine-tune | 1,242 | 5.5s | 54.3s | **39.3%** |

The 0.0% is the floor the eval was built to establish. Everything else is
measured against it.

### The aggregate hides the finding

| task | paces base → tuned | off-zone base → tuned |
|---|---|---|
| pace_prescription | 180 → 156 | 50.6% → 47.4% |
| plan_adaptation | 31 → 148 | 19.4% → 26.4% ↑ |
| plan_generation | 70 → 875 | 55.7% → 39.3% ↓ |
| race_prediction | 63 → 3 | 42.9% → 66.7% (metric bug, see below) |
| workout_rationale | 5 → 60 | 40.0% → 48.3% ↑ |

Three of five tasks got worse. The overall number improves because
`plan_generation` — the only substantial improver — grew from 20% to 70% of all
emitted paces. Excluding `race_prediction`, the volume-weighted rate improves
48.3% → 39.2%, while the unweighted per-task mean is flat at 41.4% → 40.3%.
Both are reported because they disagree.

### What improved was format compliance

Pace counts against the reference: `workout_rationale` 5 → 60 against a
reference of 60. `plan_generation` 70 → 875 against 814. The model learned
almost exactly where a coach puts numbers.

What it did not learn is which numbers. 39.3% off-zone means two in five
prescribed paces are more than 8 s/mi from any real zone for that athlete. The
fine-tune learned the shape of coaching and not the arithmetic under it — it
became more confident without becoming much more correct.

Final training loss was 0.07 at 96.7% token accuracy, so "undertrained" doesn't
explain it. `assistant_only_loss` was active, so masking doesn't either.

### Known bug in the eval

`race_prediction`'s correct output is a table of race *times*, not per-mile
paces; the reference emits 0 paces for it. The tuned model correctly stopped
emitting paces there (63 → 3) and the eval scores that as a regression. Either
exclude the row or extend `eval_paces.py` to score times for this task.

### Why this is the useful result

This is the empirical case for the architecture worth shipping: expose
`vdot.py` as a tool and let the model handle prose. A model that writes
correctly-shaped plans containing wrong paces is more dangerous than one that
obviously refuses to commit — and without an eval measuring the arithmetic
directly, the fine-tune's fluency would have read as success. Validation loss
looked excellent throughout.

Cost, end to end: about $4 of A10G time — 74 minutes training, 53 minutes
generation, plus a 2.5-hour baseline pass on a free Colab T4.
