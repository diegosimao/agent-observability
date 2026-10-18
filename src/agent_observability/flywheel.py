"""Data flywheel: traces de produção com bom sinal → dataset de fine-tuning.

    uv run obs-export-ft --dias 7

Critério de curadoria (cada um vira um filtro auditável, não um "achismo"):
1. trace raiz `atendimento` sem erro e sem falha de ferramenta
2. nenhum 👎 do usuário (`user_score` = 0 descarta)
3. só a variante de prompt vencedora (v2 por padrão) — destila o melhor comportamento
4. perguntas do dataset golden ficam de fora, para a avaliação não medir memória
5. deduplicação por (pergunta, sequência de tools)

Cada linha guarda o `run_id` de origem: dá para voltar do exemplo ao trace.
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.messages import messages_from_dict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith import Client

from .config import PROJECT, PROMPTS
from .evaluate import EXEMPLOS
from .local_llm import para_chat_template
from .tools import TOOLS

def _norm(texto: str) -> str:
    return "".join(c for c in texto.lower() if c.isalnum() or c == " ").strip()


PERGUNTAS_GOLDEN = {_norm(q) for q, _, _ in EXEMPLOS}


def _mensagens(run) -> list:
    brutas = run.outputs["messages"]
    return messages_from_dict([{"type": m["type"], "data": m} for m in brutas])


def exemplo_de_trace(run, prompt_version: str) -> dict | None:
    """Converte um trace em exemplo de chat, ou None se ele não serve para treino."""
    msgs = _mensagens(run)
    chat = para_chat_template(msgs)
    if any(m.get("role") == "tool" and m["content"].startswith("Error") for m in chat):
        return None
    if not chat or chat[0]["role"] != "user" or chat[-1]["role"] != "assistant" or not chat[-1]["content"]:
        return None
    return {
        "messages": [{"role": "system", "content": PROMPTS[prompt_version]}, *chat],
        "tools": [convert_to_openai_tool(t) for t in TOOLS],
        "origem": {"run_id": str(run.id), "prompt_version": prompt_version, "projeto": PROJECT},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Exporta traces curados do LangSmith para SFT")
    p.add_argument("--dias", type=int, default=7)
    p.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    p.add_argument("--saida", default="data")
    p.add_argument("--val", type=float, default=0.15, help="fração para validação")
    args = p.parse_args()

    client = Client()
    desde = datetime.now(timezone.utc) - timedelta(days=args.dias)
    runs = list(
        client.list_runs(
            project_name=PROJECT,
            is_root=True,
            error=False,
            start_time=desde,
            filter=f'and(eq(name, "atendimento"), eq(metadata_key, "prompt_version"), eq(metadata_value, "{args.prompt}"))',
        )
    )
    negativos = {
        str(f.run_id)
        for f in client.list_feedback(run_ids=[r.id for r in runs], feedback_key=["user_score"])
        if f.score == 0
    }

    motivos: Counter = Counter()
    vistos, exemplos = set(), []
    for run in runs:
        pergunta = run.inputs["messages"][0]["content"].strip()
        if str(run.id) in negativos:
            motivos["👎 do usuário"] += 1
            continue
        if _norm(pergunta) in PERGUNTAS_GOLDEN:
            motivos["pergunta do golden (vazamento)"] += 1
            continue
        ex = exemplo_de_trace(run, args.prompt)
        if ex is None:
            motivos["falha de tool / resposta vazia"] += 1
            continue
        chave = (_norm(pergunta), tuple(tc["function"]["name"] for m in ex["messages"] for tc in m.get("tool_calls", [])))
        if chave in vistos:
            motivos["duplicata"] += 1
            continue
        vistos.add(chave)
        exemplos.append(ex)

    random.Random(42).shuffle(exemplos)
    n_val = max(1, int(len(exemplos) * args.val)) if len(exemplos) > 4 else 0
    saida = Path(args.saida)
    saida.mkdir(exist_ok=True)
    for nome, parte in (("sft_train.jsonl", exemplos[n_val:]), ("sft_val.jsonl", exemplos[:n_val])):
        with open(saida / nome, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in parte)

    print(f"{len(runs)} traces lidos de '{PROJECT}' (prompt {args.prompt}, últimos {args.dias} dias)")
    for motivo, n in motivos.most_common():
        print(f"  descartados — {motivo}: {n}")
    print(f"{len(exemplos) - n_val} treino + {n_val} validação → {saida.resolve()}")


if __name__ == "__main__":
    main()
