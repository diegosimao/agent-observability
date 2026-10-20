"""LoRA fine-tuning of a small model on curated production traces.

    uv run --extra finetune python finetune/train_lora.py
    uv run --extra finetune python finetune/train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --epochs 4

LoRA (Hu et al., 2021, arXiv:2106.09685) trains only low-rank matrices on the
attention/MLP projections: ~1% of the parameters, fits on a laptop GPU.

The loss is computed ONLY on assistant turns (tool calls and the final answer).
The system prompt, the question and the tool results are context: training on
them would teach the model to "make up" API responses.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def tokenize(example: dict, tokenizer, max_len: int) -> dict:
    msgs, tools = example["messages"], example["tools"]
    full = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False)

    # [start, end) character spans of the text that belong to assistant turns
    spans = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        before = tokenizer.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=False)
        # the Qwen template is concatenative; if it weren't, the mask would be wrong
        assert full.startswith(through) and through.startswith(before), "chat template is not prefix-stable"
        spans.append((len(before), len(through)))

    enc = tokenizer(full, return_offsets_mapping=True, truncation=True, max_length=max_len)
    labels = [
        tok if any(a <= start < b for a, b in spans) else -100
        for tok, (start, _) in zip(enc["input_ids"], enc["offset_mapping"])
    ]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    n = max(len(x["input_ids"]) for x in batch)

    def pad(key, value):
        return torch.tensor([x[key] + [value] * (n - len(x[key])) for x in batch])

    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="finetune/adapters/qwen-support-lora")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=2048)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    train = [tokenize(e, tokenizer, args.max_len) for e in load_jsonl(Path(args.data) / "sft_train.jsonl")]
    val_path = Path(args.data) / "sft_val.jsonl"
    val = [tokenize(e, tokenizer, args.max_len) for e in load_jsonl(val_path)] if val_path.exists() else []
    target = sum(sum(t != -100 for t in x["labels"]) for x in train)
    total = sum(len(x["labels"]) for x in train)
    print(f"{len(train)} train examples, {len(val)} validation; loss on {target}/{total} tokens ({100 * target / total:.0f}%)")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto")
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.print_trainable_parameters()

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            lr_scheduler_type="cosine",
            warmup_steps=2,
            bf16=torch.cuda.is_available(),
            logging_steps=2,
            eval_strategy="epoch" if val else "no",
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
        ),
        train_dataset=train,
        eval_dataset=val or None,
        data_collator=lambda batch: collate(batch, tokenizer.pad_token_id),
    )
    trainer.train()

    out = Path(args.out)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (out / "training_history.json").write_text(json.dumps({"base": args.base, "args": vars(args), "log": history}, indent=2), encoding="utf-8")
    print(f"adapter saved to {out.resolve()}")


if __name__ == "__main__":
    main()
