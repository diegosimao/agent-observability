"""Gerador de tráfego de "produção" para popular os dashboards do LangSmith.

Simula clientes reais: perguntas variadas, dois prompts em A/B, planos e canais
diferentes, feedback 👍/👎 do usuário e, opcionalmente, um incidente no meio
(latência x4, transportadora caindo) para mostrar como ele aparece nos gráficos.

    uv run obs-simulate                       # 80 conversas, rápido
    uv run obs-simulate --n 300 --duracao-min 40 --incidente
"""

import argparse
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx

from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from rich.console import Console
from rich.table import Table

from . import chaos
from .agent import Resultado, atender
from .config import PROJECT, llm_mode, project_id, tracing_enabled

# (categoria, peso, perguntas). {p} e {cep} são sorteados a cada conversa, para
# o tráfego (e o dataset de fine-tuning que sai dele) não repetir as mesmas frases.
CENARIOS = [
    ("pedido", 35, [
        "Cadê meu pedido {p}?",
        "Qual o status do pedido #{p}?",
        "Meu pedido {p} ainda não chegou, o que houve?",
        "Pode ver o pedido {p} pra mim?",
        "Quero rastrear o pedido {p}",
        "pedido {p} status",
        "Comprei semana passada, número {p}. Já saiu?",
        "O que aconteceu com a minha compra {p}?",
    ]),
    ("frete", 20, [
        "Quanto custa o frete para {cep}?",
        "Qual o prazo de entrega para o CEP {cep}?",
        "Frete para {cep}, por favor",
        "Vocês entregam no {cep}? Quanto tempo leva?",
        "Moro no CEP {cep}, quanto fica a entrega?",
        "prazo e valor de envio pro {cep}",
    ]),
    ("politica", 20, [
        "Como funciona a devolução?",
        "Posso fazer troca de tamanho?",
        "Qual a garantia dos produtos?",
        "Em quanto tempo sai o reembolso?",
        "Quero devolver um produto, como faço?",
        "Qual o prazo para pedir troca?",
        "O produto tem garantia de quanto tempo?",
        "Quando cai o reembolso no cartão?",
    ]),
    ("multi", 10, [
        "O pedido {p} está atrasado? E qual seria o frete para {cep}?",
        "Recebi o pedido {p} com defeito, como funciona a devolução?",
        "Quero trocar o pedido {p}, qual a política de troca?",
        "Vejam o pedido {p} e me digam o frete para {cep}",
    ]),
    ("escalonamento", 10, [
        "Isso é um absurdo, quero falar com um humano agora!",
        "Pedido {p} travado há dias, quero um atendente",
        "Péssimo atendimento, quero registrar uma reclamação",
        "Me passa para um atendente humano, por favor",
        "Quero fazer uma reclamação formal sobre o pedido {p}",
    ]),
    ("fora_escopo", 5, [
        "Oi, tudo bem?",
        "Vocês vendem ingresso de show?",
        "Qual o horário da loja física?",
    ]),
]
# 1001-1004 existem; os demais caem no "pedido não encontrado" do ERP
PEDIDOS = [("1001", 25), ("1002", 25), ("1003", 20), ("1004", 15), ("5521", 8), ("7789", 7)]
CEPS = ["01310-100", "30140-071", "80010-000", "69005-070", "40020-000", "90010-150", "60060-120", "88010-400"]
PLANOS = [("free", 60), ("plus", 30), ("enterprise", 10)]
CANAIS = [("web", 55), ("whatsapp", 35), ("app", 10)]
COMENTARIOS_RUINS = ["Não resolveu", "Demorou demais", "Resposta genérica", "Deu erro"]
COMENTARIOS_BONS = ["Rápido e claro", "Resolveu!", "Ótimo"]


def _escolher(opcoes):
    return random.choices([o[0] for o in opcoes], weights=[o[1] for o in opcoes])[0]


API_URL: str | None = None  # definido por --api-url


def _via_api(pergunta: str) -> tuple[str, Resultado]:
    """Mesma conversa, mas como um cliente HTTP real: a API escolhe a variante
    do A/B pela sessão, e o run_id volta no corpo (ou no 503)."""
    corpo = {"mensagem": pergunta, "sessao": f"cliente-{random.randint(1, 60)}", "plano": _escolher(PLANOS), "canal": _escolher(CANAIS)}
    inicio = time.perf_counter()
    resp = httpx.post(f"{API_URL}/v1/chat", json=corpo, timeout=120)
    latencia = time.perf_counter() - inicio
    dados = resp.json()
    if resp.status_code != 200:
        detalhe = dados["detail"]
        return "?", Resultado(uuid.UUID(detalhe["run_id"]), None, detalhe["erro"], latencia)
    r = Resultado(uuid.UUID(dados["run_id"]), dados["resposta"], None, latencia, dados["tokens"], dados["tools"], dados["tool_erros"])
    return dados["prompt_version"], r


def _uma_conversa(_: int) -> tuple[str, Resultado]:
    cat_idx = random.choices(range(len(CENARIOS)), weights=[c[1] for c in CENARIOS])[0]
    categoria, _, perguntas = CENARIOS[cat_idx]
    pergunta = random.choice(perguntas).format(p=_escolher(PEDIDOS), cep=random.choice(CEPS))
    if API_URL:
        return _via_api(pergunta)
    prompt_version = random.choice(["v1", "v2"])
    r = atender(
        pergunta,
        prompt_version=prompt_version,
        plano=_escolher(PLANOS),
        canal=_escolher(CANAIS),
        categoria=categoria,
        extra_tags=["simulacao"],
    )
    return prompt_version, r


def _feedback_usuario(prompt_version: str, r: Resultado) -> tuple[float | None, str]:
    """Probabilidade de 👍 que reage às mesmas coisas que irritam um cliente real."""
    if random.random() > 0.45:  # a maioria dos usuários não avalia
        return None, ""
    if r.erro:
        p = 0.05
    elif r.tool_erros:
        p = 0.25
    elif not r.tools:
        p = 0.45
    else:
        p = 0.82 if prompt_version == "v2" else 0.66
    if r.latencia_s > 6:
        p -= 0.3
    gostou = random.random() < max(p, 0.02)
    return (1.0 if gostou else 0.0), random.choice(COMENTARIOS_BONS if gostou else COMENTARIOS_RUINS)


def _enviar_feedback(client: Client, execucoes: list[tuple[str, Resultado]]) -> int:
    enviados = 0
    for prompt_version, r in execucoes:
        score, comentario = _feedback_usuario(prompt_version, r)
        if score is not None:
            if API_URL:  # o caminho que um front-end usaria
                httpx.post(f"{API_URL}/v1/feedback", json={"run_id": str(r.run_id), "score": int(score), "comentario": comentario}, timeout=30).raise_for_status()
            else:
                client.create_feedback(r.run_id, key="user_score", score=score, comment=comentario, session_id=project_id())
            enviados += 1
        if not r.erro:
            # sinal automático: resolvido sem escalar para humano e sem falha de tool
            resolvido = "abrir_chamado" not in r.tools and r.tool_erros == 0 and bool(r.tools)
            client.create_feedback(r.run_id, key="resolvido_sem_humano", score=float(resolvido), session_id=project_id())
    return enviados


def _caos(modo: str) -> None:
    getattr(chaos, modo)()  # estado local, também usado como "já está em incidente?"
    if API_URL:  # o caos precisa acontecer no processo da API, não aqui
        httpx.post(f"{API_URL}/_chaos/{modo}", timeout=10).raise_for_status()


def _resumo(execucoes: list[tuple[str, Resultado]], console: Console) -> None:
    tabela = Table(title="Resumo local (o mesmo aparece no LangSmith)")
    for col in ("prompt", "conversas", "erro %", "p50 s", "p95 s", "tokens méd.", "falhas de tool"):
        tabela.add_column(col, justify="right")
    for versao in ("v1", "v2", "?"):  # "?" = 503 vindo da API, sem variante conhecida
        rs = [r for v, r in execucoes if v == versao]
        if not rs:
            continue
        lat = sorted(r.latencia_s for r in rs)
        ok = [r for r in rs if not r.erro]
        tabela.add_row(
            versao,
            str(len(rs)),
            f"{100 * (len(rs) - len(ok)) / len(rs):.1f}",
            f"{statistics.median(lat):.2f}",
            f"{lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.2f}",
            f"{statistics.mean(r.tokens for r in ok):.0f}" if ok else "-",
            str(sum(r.tool_erros for r in rs)),
        )
    console.print(tabela)


def main() -> None:
    p = argparse.ArgumentParser(description="Gera tráfego simulado para o agente")
    p.add_argument("--n", type=int, default=80, help="número de conversas")
    p.add_argument("--concorrencia", type=int, default=4)
    p.add_argument("--duracao-min", type=float, default=0, help="espalha as conversas por N minutos (gráficos de série temporal ficam mais bonitos)")
    p.add_argument("--incidente", action="store_true", help="degrada o sistema entre 40%% e 65%% da simulação")
    p.add_argument("--api-url", help="gera o tráfego via HTTP contra a API (ex.: http://127.0.0.1:8000)")
    args = p.parse_args()

    global API_URL
    API_URL = args.api_url.rstrip("/") if args.api_url else None

    console = Console()
    console.print(f"[bold]llm[/bold]={llm_mode()}  [bold]tracing[/bold]={'on → projeto ' + PROJECT if tracing_enabled() else 'off (só resumo local)'}")

    # Fases rodam em sequência (cada uma espera a anterior terminar) para que o
    # estado de caos valha para todas as conversas daquela fase.
    a, b = int(args.n * 0.40), int(args.n * 0.65)
    fases = [("normal", a), ("incidente", b - a), ("normal", args.n - b)] if args.incidente else [("normal", args.n)]
    intervalo = args.duracao_min * 60 / args.n if args.duracao_min else 0
    execucoes: list[tuple[str, Resultado]] = []
    with ThreadPoolExecutor(args.concorrencia) as pool, console.status("simulando...") as status:
        for nome, qtd in fases:
            if nome == "incidente":
                _caos("incidente")
                console.print("[red]>>> incidente iniciado: provedor lento, transportadora instável[/red]")
            elif chaos.state.latency_multiplier != 1.0:
                _caos("normal")
                console.print("[green]>>> incidente resolvido[/green]")
            futuros = []
            for i in range(qtd):
                futuros.append(pool.submit(_uma_conversa, i))
                if intervalo:
                    time.sleep(random.expovariate(1 / intervalo))
            for f in futuros:
                execucoes.append(f.result())
                status.update(f"simulando... {len(execucoes)}/{args.n}")

    _resumo(execucoes, console)
    if tracing_enabled():
        wait_for_all_tracers()
        enviados = _enviar_feedback(Client(), execucoes)
        console.print(f"{enviados} avaliações de usuário enviadas. Abra https://smith.langchain.com → Projects → {PROJECT}")


if __name__ == "__main__":
    main()
