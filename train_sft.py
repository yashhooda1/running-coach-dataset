"""
train_sft.py — LoRA SFT, version-tolerant.

Complete file. Overwrite whatever you have; do not patch this one.

Everything learned the hard way is already in here:
  - splits loaded as individual files, so the reference split's extra columns
    can't trigger a schema CastError
  - dtype chosen from the device, because T4 and older cards have no bfloat16
  - SFTConfig fields filtered against whatever the installed TRL declares,
    since fields it used to inherit from TrainingArguments come and go
  - dtype / torch_dtype kwarg rename handled both ways

On HF Jobs:

    hf jobs uv run --flavor a10g-large --timeout 3h --secrets HF_TOKEN \
      --with torch --with transformers --with trl --with peft \
      --with datasets --with accelerate --with bitsandbytes \
      --name train-3b train_sft.py \
      --dataset <user>/running-coach-sft \
      --base Qwen/Qwen2.5-3B-Instruct \
      --push <user>/running-coach-qwen3b-lora
"""

# /// script
# requires-python = ">=3.10,<3.13"
# dependencies = [
#   "torch",
#   "transformers>=4.44",
#   "trl>=0.11",
#   "peft>=0.12",
#   "datasets",
#   "bitsandbytes",
#   "accelerate",
# ]
# ///

from __future__ import annotations

import argparse
import dataclasses

import torch
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


def load_split(repo: str, name: str):
    """Load one split file directly.

    load_dataset(repo) would try to reconcile train, validation and reference
    under one schema; reference carries three extra columns and it raises.
    """
    f = hf_hub_download(repo, f"data/{name}.jsonl", repo_type="dataset")
    return load_dataset("json", data_files=f)["train"]


def build_config(a, bf16_ok: bool) -> SFTConfig:
    wanted = dict(
        output_dir="out",
        num_train_epochs=a.epochs,
        learning_rate=a.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        per_device_train_batch_size=a.batch,
        gradient_accumulation_steps=a.accum,
        gradient_checkpointing=True,
        max_seq_length=a.max_seq,
        packing=False,               # plans are long; packing would splice them
        bf16=bf16_ok,
        fp16=not bf16_ok,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        report_to="none",
        # train on the coach's turn only, not on the athlete's question
        assistant_only_loss=True,
    )

    valid = {f.name for f in dataclasses.fields(SFTConfig)}

    # known renames — dropping max_seq_length silently would truncate plans
    if "max_seq_length" not in valid and "max_length" in valid:
        wanted["max_length"] = wanted.pop("max_seq_length")
    if "eval_strategy" not in valid and "evaluation_strategy" in valid:
        wanted["evaluation_strategy"] = wanted.pop("eval_strategy")

    dropped = sorted(k for k in wanted if k not in valid)
    if dropped:
        print(f"SFTConfig on this TRL does not accept {dropped} — dropping")
    if "assistant_only_loss" in dropped:
        print("NOTE: training on the full sequence, not just the coach's turn")

    return SFTConfig(**{k: v for k, v in wanted.items() if k in valid})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hoodarunner/running-coach-sft")
    ap.add_argument("--base", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--push", default=None, help="model repo for the adapter")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--no-4bit", action="store_true")
    a = ap.parse_args()

    train = load_split(a.dataset, "train")
    val = load_split(a.dataset, "validation")
    print(f"train {len(train)}  validation {len(val)}")

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    dt = torch.bfloat16 if bf16_ok else torch.float16
    print(f"dtype: {dt}")

    quant = None if a.no_4bit else BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=dt,
        bnb_4bit_use_double_quant=True,
    )

    load = dict(quantization_config=quant, device_map="auto",
                attn_implementation="sdpa")
    try:
        model = AutoModelForCausalLM.from_pretrained(a.base, dtype=dt, **load)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=dt, **load)

    peft_config = LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    trainer = SFTTrainer(
        model=model,
        args=build_config(a, bf16_ok),
        train_dataset=train,
        eval_dataset=val,
        processing_class=tok,
        peft_config=peft_config,
    )
    trainer.train()
    trainer.save_model("out/final")

    if a.push:
        trainer.model.push_to_hub(a.push)
        tok.push_to_hub(a.push)
        print(f"https://huggingface.co/{a.push}")


if __name__ == "__main__":
    main()
