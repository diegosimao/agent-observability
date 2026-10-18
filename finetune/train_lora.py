"""Fine-tuning LoRA de um modelo pequeno com os traces curados da produção.

    uv run --extra finetune python finetune/train_lora.py
    uv run --extra finetune python finetune/train_lora.py --base Qwen/Qwen2.5-1.5B-Instruct --epocas 4

LoRA (Hu et al., 2021, arXiv:2106.09685) treina só matrizes de baixo posto
nas projeções de atenção/MLP — ~1% dos parâmetros, cabe numa GPU de notebook.

A loss é calculada SOMENTE nos turnos do assistente (tool calls e resposta
final). Prompt de sistema, pergunta e resultado das ferramentas são contexto:
treinar neles ensinaria o modelo a "inventar" retornos de API.
"""

import argparse
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def carregar(caminho: Path) -> list[dict]:
    return [json.loads(l) for l in caminho.read_text(encoding="utf-8").splitlines() if l.strip()]


def tokenizar(exemplo: dict, tokenizer, max_len: int) -> dict:
    msgs, tools = exemplo["messages"], exemplo["tools"]
    completo = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False)

    # trechos [início, fim) do texto que pertencem a turnos do assistente
    trechos = []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        antes = tokenizer.apply_chat_template(msgs[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        ate = tokenizer.apply_chat_template(msgs[: i + 1], tools=tools, tokenize=False)
        # o template do Qwen é concatenativo; se não for, a máscara estaria errada
        assert completo.startswith(ate) and ate.startswith(antes), "chat template não é prefixo-estável"
        trechos.append((len(antes), len(ate)))

    enc = tokenizer(completo, return_offsets_mapping=True, truncation=True, max_length=max_len)
    labels = [
        tok if any(a <= ini < b for a, b in trechos) else -100
        for tok, (ini, _) in zip(enc["input_ids"], enc["offset_mapping"])
    ]
    return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": labels}


def colar(lote: list[dict], pad_id: int) -> dict:
    n = max(len(x["input_ids"]) for x in lote)
    def pad(chave, valor):
        return torch.tensor([x[chave] + [valor] * (n - len(x[chave])) for x in lote])
    return {"input_ids": pad("input_ids", pad_id), "attention_mask": pad("attention_mask", 0), "labels": pad("labels", -100)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dados", default="data")
    p.add_argument("--saida", default="finetune/adapters/qwen-suporte-lora")
    p.add_argument("--epocas", type=float, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--max-len", type=int, default=2048)
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    treino = [tokenizar(e, tokenizer, args.max_len) for e in carregar(Path(args.dados) / "sft_train.jsonl")]
    val_path = Path(args.dados) / "sft_val.jsonl"
    val = [tokenizar(e, tokenizer, args.max_len) for e in carregar(val_path)] if val_path.exists() else []
    alvo = sum(sum(t != -100 for t in x["labels"]) for x in treino)
    total = sum(len(x["labels"]) for x in treino)
    print(f"{len(treino)} exemplos de treino, {len(val)} de validação; loss em {alvo}/{total} tokens ({100 * alvo / total:.0f}%)")

    modelo = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="auto")
    modelo = get_peft_model(
        modelo,
        LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    modelo.print_trainable_parameters()

    trainer = Trainer(
        model=modelo,
        args=TrainingArguments(
            output_dir=args.saida,
            num_train_epochs=args.epocas,
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
        train_dataset=treino,
        eval_dataset=val or None,
        data_collator=lambda lote: colar(lote, tokenizer.pad_token_id),
    )
    trainer.train()

    saida = Path(args.saida)
    modelo.save_pretrained(saida)
    tokenizer.save_pretrained(saida)
    historico = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (saida / "historico_treino.json").write_text(json.dumps({"base": args.base, "args": vars(args), "log": historico}, indent=2), encoding="utf-8")
    print(f"adaptador salvo em {saida.resolve()}")


if __name__ == "__main__":
    main()
