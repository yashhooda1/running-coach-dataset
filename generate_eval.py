"""
generate_eval.py — run a model over the validation split and write generations
in the shape eval_paces.py expects.

This closes the loop. eval_paces.py scores a `generated` field; nothing else in
the repo produces one, so without this the fine-tune can only be judged by
validation loss, which says nothing about whether the paces are right.

Run the base model FIRST, before training. That number is the comparison that
matters: a fine-tune that lands at 12% off-zone is a success if the base model
was at 40%, and a failure if the base was at 8%. Without the baseline the
after-number means nothing.

    # baseline, before any training
    python generate_eval.py --base Qwen/Qwen2.5-7B-Instruct \
        --split data/validation.jsonl --out runs/base.jsonl
    python eval_paces.py --generations runs/base.jsonl

    # after training
    python generate_eval.py --base Qwen/Qwen2.5-7B-Instruct \
        --adapter <user>/running-coach-qwen7b-lora \
        --split data/validation.jsonl --out runs/tuned.jsonl
    python eval_paces.py --generations runs/tuned.jsonl

The inline `# /// script` block below only takes effect under `uv run` or
`hf jobs uv run`. Calling this with a plain `python generate_eval.py` uses
whatever is in your system site-packages and will fail on a missing torch,
accelerate, or peft. Use one of these two:

    uv run generate_eval.py ...          # local, uv builds an isolated env
    hf jobs uv run ... generate_eval.py  # remote GPU

On HF Jobs, so nothing runs locally. Note --push-to: a job's disk is thrown
away when the job finishes, so without it the generations are lost and you pay
for a run that produced nothing.

    hf jobs uv run --flavor a10g-large --timeout 2h --secrets HF_TOKEN \
        generate_eval.py --base Qwen/Qwen2.5-7B-Instruct \
        --adapter <user>/running-coach-qwen7b-lora \
        --split <user>/running-coach-sft --out runs/tuned.jsonl \
        --push-to <user>/running-coach-evals
"""

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch",
#   "transformers>=4.44",
#   "peft>=0.12",
#   "datasets",
#   "accelerate",
# ]
# ///

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_rows(spec: str, split_name: str = "validation"):
    """Accept either a local JSONL path or a Hub dataset id."""
    p = Path(spec)
    if p.exists():
        return [json.loads(l) for l in p.open(encoding="utf-8")]
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    f = hf_hub_download(spec, f"data/{split_name}.jsonl", repo_type="dataset")
    return list(load_dataset("json", data_files=f)["train"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", default=None,
                    help="LoRA repo or path; omit for a baseline run")
    ap.add_argument("--split", default="data/validation.jsonl")
    ap.add_argument("--out", default="runs/generations.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--max-new", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run without a GPU; only sane with a small --limit")
    ap.add_argument("--push-to", default=None,
                    help="Hub dataset repo to upload the generations to. "
                         "REQUIRED on HF Jobs: the job's filesystem is "
                         "ephemeral and anything written locally is discarded "
                         "when the job ends.")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_rows(a.split)
    # training_review rows carry no vdot, so eval_paces skips them anyway
    rows = [r for r in rows if "vdot" in r and r.get("vdot") is not None]
    if a.limit:
        rows = rows[: a.limit]
    print(f"{len(rows)} scoreable rows")

    tok = AutoTokenizer.from_pretrained(a.base, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if not torch.cuda.is_available() and not a.allow_cpu:
        raise SystemExit(
            "No CUDA device. A 7B model on CPU would take days for 150 "
            "generations. Run this on HF Jobs instead:\n\n"
            "  hf jobs uv run --flavor a10g-large --timeout 2h "
            "--secrets HF_TOKEN \\\n"
            "    generate_eval.py --base <base> --split <user>/<dataset> "
            "--out runs/base.jsonl\n\n"
            "Or pass --allow-cpu --limit 5 to smoke-test the plumbing.")

    # transformers renamed torch_dtype -> dtype; support both
    try:
        model = AutoModelForCausalLM.from_pretrained(
            a.base, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            a.base, torch_dtype=torch.bfloat16, device_map="auto")
    if a.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.adapter)
        print(f"adapter: {a.adapter}")
    else:
        print("baseline run, no adapter")
    model.eval()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Greedy by default. Sampling would make the off-zone rate a function of
    # the seed, and the point of this number is that it's comparable run to run.
    gen_kwargs = dict(max_new_tokens=a.max_new, pad_token_id=tok.pad_token_id)
    if a.temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=a.temperature)
    else:
        gen_kwargs.update(do_sample=False)

    with out.open("w", encoding="utf-8") as f:
        for i in range(0, len(rows), a.batch):
            chunk = rows[i : i + a.batch]
            prompts = [
                tok.apply_chat_template(r["messages"][:-1], tokenize=False,
                                        add_generation_prompt=True)
                for r in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True,
                      truncation=True, max_length=2048).to(model.device)
            with torch.no_grad():
                ids = model.generate(**enc, **gen_kwargs)
            for r, seq in zip(chunk, ids):
                text = tok.decode(seq[enc["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
                # carry the metadata through untouched; eval_paces needs
                # vdot, effective_goal_vdot, goal_distance and task
                f.write(json.dumps({**r, "generated": text},
                                   ensure_ascii=False) + "\n")
            print(f"  {min(i+a.batch, len(rows))}/{len(rows)}", flush=True)

    print(f"\nwrote {out}")

    if a.push_to:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(a.push_to, repo_type="dataset", private=True,
                        exist_ok=True)
        remote = f"generations/{out.name}"
        api.upload_file(path_or_fileobj=str(out), path_in_repo=remote,
                        repo_id=a.push_to, repo_type="dataset")
        print(f"uploaded to https://huggingface.co/datasets/{a.push_to}"
              f"/blob/main/{remote}")
        print(f"pull it back with:\n"
              f"  hf download {a.push_to} {remote} --repo-type dataset "
              f"--local-dir .")
    else:
        print(f"now: python eval_paces.py --generations {out}")


if __name__ == "__main__":
    main()
