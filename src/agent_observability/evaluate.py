"""Avaliação offline: roda um dataset "golden" contra os prompts v1 e v2 e
publica dois experimentos no LangSmith, prontos para a tela de comparação.

    uv run obs-evaluate
"""

import argparse
import sys
from collections import defaultdict
from math import comb

from langsmith import Client, evaluate

from .agent import atender
from .config import APP_VERSION, llm_mode, model_name, tracing_enabled

DATASET = "suporte-golden"

EXEMPLOS = [
    ("Qual o status do pedido 1002?", ["buscar_pedido"], ["1002", "transporte"]),
    ("Por que o pedido 1004 foi cancelado?", ["buscar_pedido"], ["1004", "cancelado"]),
    ("Quanto custa o frete para 01310-100?", ["calcular_frete"], ["R$"]),
    ("Como funciona a devolução?", ["consultar_politica"], ["30 dias"]),
    ("Em quanto tempo sai o reembolso?", ["consultar_politica"], ["7 dias"]),
    ("Posso fazer troca de tamanho?", ["consultar_politica"], ["troca"]),
    ("O pedido 1002 está atrasado? E o frete para 30140-071?", ["buscar_pedido", "calcular_frete"], ["1002", "R$"]),
    ("Quero falar com um atendente humano", ["abrir_chamado"], ["CH-"]),
    ("Isso é um absurdo, pedido 1003 travado!", ["buscar_pedido", "abrir_chamado"], ["CH-"]),
    ("Oi, tudo bem?", [], ["pedido"]),
]


def _garantir_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET):
        return
    ds = client.create_dataset(DATASET, description="Perguntas de suporte com ferramentas e trechos esperados")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {"inputs": {"pergunta": q}, "outputs": {"tools_esperadas": t, "deve_conter": c}}
            for q, t, c in EXEMPLOS
        ],
    )


# --- avaliadores ------------------------------------------------------------
def tool_correta(outputs: dict, reference_outputs: dict) -> bool:
    return sorted(set(outputs["tools"])) == sorted(reference_outputs["tools_esperadas"])


def conteudo_esperado(outputs: dict, reference_outputs: dict) -> float:
    resposta = (outputs["resposta"] or "").lower()
    termos = reference_outputs["deve_conter"]
    return sum(t.lower() in resposta for t in termos) / len(termos)


def concisao(outputs: dict) -> bool:
    return outputs["resposta"] is not None and len(outputs["resposta"]) <= 280


def sem_erro(outputs: dict) -> bool:
    return outputs["erro"] is None


def _aprovado(o: dict, r: dict) -> bool:
    return o["erro"] is None and tool_correta(o, r) and conteudo_esperado(o, r) == 1


def taxa_aprovacao(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    ok = sum(_aprovado(o, r) for o, r in zip(outputs, reference_outputs))
    return {"key": "taxa_aprovacao", "score": ok / len(outputs)}


def pass_hat_k(sucessos_por_tarefa: list[tuple[int, int]], k: int) -> float:
    """pass^k do τ-bench (Yao et al., 2024, arXiv:2406.12045): probabilidade de
    acertar a MESMA tarefa em k tentativas independentes, todas elas. Estimador
    não-viesado com n tentativas e c acertos por tarefa: C(c, k) / C(n, k).
    pass@k pergunta "acerta alguma vez?"; pass^k pergunta "dá para confiar?"."""
    return sum(comb(c, k) / comb(n, k) for c, n in sucessos_por_tarefa) / len(sucessos_por_tarefa)


def confiabilidade(runs, examples) -> dict:
    """Agrupa as repetições por exemplo e publica pass^1..pass^n."""
    por_exemplo: dict = defaultdict(list)
    for run, ex in zip(runs, examples):
        por_exemplo[ex.id].append(_aprovado(run.outputs or {"erro": "sem saída", "tools": [], "resposta": None}, ex.outputs))
    tarefas = [(sum(v), len(v)) for v in por_exemplo.values()]
    n = min(t[1] for t in tarefas)
    return {"results": [{"key": f"pass^{k}", "score": pass_hat_k(tarefas, k)} for k in range(1, n + 1)]}


def _alvo(prompt_version: str):
    def alvo(inputs: dict) -> dict:
        r = atender(inputs["pergunta"], prompt_version=prompt_version, canal="eval", categoria="eval")
        return {"resposta": r.resposta, "tools": r.tools, "erro": r.erro}

    return alvo


def main() -> None:
    p = argparse.ArgumentParser(description="Publica experimentos do dataset golden no LangSmith")
    p.add_argument("--prompts", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    p.add_argument("--k", type=int, default=3, help="repetições por exemplo, para o pass^k")
    p.add_argument("--concorrencia", type=int, default=4)
    args = p.parse_args()

    if not tracing_enabled():
        sys.exit("Defina LANGSMITH_TRACING=true e LANGSMITH_API_KEY no .env para publicar experimentos.")
    client = Client()
    _garantir_dataset(client)
    for versao in args.prompts:
        evaluate(
            _alvo(versao),
            data=DATASET,
            evaluators=[tool_correta, conteudo_esperado, concisao, sem_erro],
            summary_evaluators=[taxa_aprovacao, confiabilidade],
            experiment_prefix=f"suporte-{model_name()}-{versao}",
            metadata={"prompt_version": versao, "app_version": APP_VERSION, "llm_mode": llm_mode(), "modelo": model_name()},
            num_repetitions=args.k,
            max_concurrency=args.concorrencia,
            client=client,
        )
    print(f"Pronto. LangSmith → Datasets & Experiments → {DATASET} → selecione os dois experimentos → Compare")


if __name__ == "__main__":
    main()
