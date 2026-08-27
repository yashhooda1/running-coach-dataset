---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
license: apache-2.0
datasets:
  - hoodarunner/running-coach-sft
language:
  - en
tags:
  - running
  - coaching
  - lora
  - sports-science
pipeline_tag: text-generation
---

# running-coach-qwen3b-lora

LoRA adapter for Qwen2.5-3B-Instruct, fine-tuned on
[hoodarunner/running-coach-sft](https://huggingface.co/datasets/hoodarunner/running-coach-sft)
(1,527 examples) to give structured distance-running training guidance.

**Read the evaluation before using this.** The result is not a straightforward
win, and the interesting part is *which* thing improved.

## What this model learned, and what it didn't

Every pace in the training corpus is computed from a Daniels/Gilbert VDOT
implementation, so the corpus itself scores **0.0% off-zone** — the reference
floor. The evaluation asks whether the fine-tuned model's prescribed paces
actually follow from the athlete's stated fitness.

`off-zone` = a pace more than 8 s/mi from any legitimate training zone for that
athlete (easy, marathon, threshold, interval, repetition, goal race pace, or the
heat-adjusted threshold). 8 s/mi is roughly the width that separates one zone
from the next, so beyond it the model has prescribed a different workout than
the one it named.

### Headline

| | paces emitted | median err | p90 err | off-zone |
|---|---|---|---|---|
| reference (ground truth) | 1,066 | 0.2s | 0.5s | **0.0%** |
| Qwen2.5-3B-Instruct (base) | 349 | 7.4s | 79.6s | **47.3%** |
| + this adapter | 1,242 | 5.5s | 54.3s | **39.3%** |

Identical prompts, greedy decoding, 150 validation rows, profile-disjoint from
training.

### Per task, which is where the story is

| task | paces base → tuned | off-zone base → tuned | |
|---|---|---|---|
| pace_prescription | 180 → 156 | 50.6% → 47.4% | −3.2 |
| plan_adaptation | 31 → 148 | 19.4% → 26.4% | **+7.0 worse** |
| plan_generation | 70 → 875 | 55.7% → 39.3% | **−16.4 better** |
| race_prediction | 63 → 3 | 42.9% → 66.7% | see note |
| workout_rationale | 5 → 60 | 40.0% → 48.3% | **+8.3 worse** |

**Three of five tasks got worse.** The aggregate improves because
`plan_generation` — the one task that improved substantially — went from 20% of
emitted paces to 70% of them.

Two defensible aggregates, both reported because they disagree:

| (excluding race_prediction) | base | tuned | delta |
|---|---|---|---|
| per-pace, volume-weighted | 48.3% | 39.2% | −9.1 |
| per-task, unweighted mean | 41.4% | 40.3% | −1.1 |

Per pace, the model is meaningfully better. Per task, it is flat.

### The race_prediction row is a metric bug, not a regression

That task's correct output is a table of race *times*, not per-mile paces — the
ground-truth reference emits 0 paces for it. The base model wrongly produced 63
paces there; the tuned model learned to stop, emitting 3. The eval scores that
correct behaviour as a 66.7% failure over n=3. Exclude this row, or extend
`eval_paces.py` to score times on this task.

### What actually improved: format, not arithmetic

Pace counts against the ground-truth reference:

| task | reference | base | tuned |
|---|---|---|---|
| pace_prescription | 150 | 180 | 156 |
| plan_adaptation | 42 | 31 | 148 |
| plan_generation | 814 | 70 | **875** |
| workout_rationale | 60 | 5 | **60** |

The model learned almost exactly where a coach puts numbers —
`workout_rationale` matches the reference to the pace. The base model wrote
training blocks that mostly declined to say what pace to run anything at; this
one writes blocks that look right.

But 39.3% off-zone against a 0.0% floor means **two in five prescribed paces are
still more than 8 s/mi from any real zone for that athlete.** The fine-tune
learned the shape of coaching nearly perfectly and the arithmetic underneath it
barely at all. It became more confident without becoming much more correct.

"It needed more training" is not available as an explanation: final training loss
was 0.07 with 96.7% mean token accuracy. The model fit the corpus tightly.
`assistant_only_loss` was active, so this is not a masking artifact either.

## Intended use

Research and demonstration. Specifically, as evidence for a claim about
architecture: for numerically grounded domains, exposing the calculation as a
tool beats trying to compress it into weights. The same
[`vdot.py`](https://huggingface.co/datasets/hoodarunner/running-coach-sft/blob/main/src/vdot.py)
that generated the corpus can serve as that tool.

**Not for actual training prescription.** At two in five paces off-zone, using
this to plan real running would put an athlete at threshold when it said easy.

## Limitations

- **Not medical advice.** The system prompt directs clinical questions to a
  clinician, but the model will still produce confident prescriptions.
- **One physiological model.** Daniels/Gilbert VDOT only. Pfitzinger, Canova,
  polarized and Norwegian methods are absent.
- **Bounded template diversity.** Trained on a template-generated corpus; the
  register is narrow and it will read repetitively.
- **The evaluation covers paces, not coaching quality.** A plan can be
  arithmetically perfect and still be bad training.

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = "Qwen/Qwen2.5-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(base, device_map="auto")
model = PeftModel.from_pretrained(model, "hoodarunner/running-coach-qwen3b-lora")
tok = AutoTokenizer.from_pretrained(base)
```

## Training

LoRA r=32, alpha=64, dropout=0.05 on all attention and MLP projections. 4-bit
NF4 base, 3 epochs over 1,527 examples, lr 1e-4 cosine with 3% warmup, effective
batch 16, max sequence 2048, `assistant_only_loss=True`. 288 optimizer steps,
74 minutes on a single A10G.

## Reproducing the evaluation

```bash
hf download hoodarunner/running-coach-evals generations/base.jsonl  --repo-type dataset --local-dir .
hf download hoodarunner/running-coach-evals generations/tuned.jsonl --repo-type dataset --local-dir .
python eval_paces.py --generations generations/base.jsonl
python eval_paces.py --generations generations/tuned.jsonl
```

Both generation files are published at
[hoodarunner/running-coach-evals](https://huggingface.co/datasets/hoodarunner/running-coach-evals).
Scripts are in the `src/` folder of the dataset repo.
