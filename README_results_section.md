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
