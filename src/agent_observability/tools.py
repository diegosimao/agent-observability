"""Ferramentas do agente. Os "sistemas externos" são simulados, com latência e
falhas injetadas para que os dashboards tenham o que mostrar."""

import random
import time

from langchain_core.tools import ToolException, tool

from . import chaos

_PEDIDOS = {
    "1001": {"status": "entregue", "data": "2026-10-02", "itens": 2, "total": 349.90},
    "1002": {"status": "em transporte", "previsao": "2026-10-21", "itens": 1, "total": 89.90},
    "1003": {"status": "aguardando pagamento", "itens": 3, "total": 1299.00},
    "1004": {"status": "cancelado", "motivo": "estoque indisponível", "total": 59.90},
}

_POLITICAS = {
    "devolucao": "Devolução gratuita em até 30 dias após o recebimento, produto sem uso e na embalagem original.",
    "troca": "Troca por tamanho ou cor em até 30 dias; o primeiro envio da troca é grátis.",
    "garantia": "Garantia legal de 90 dias mais garantia do fabricante quando informada na página do produto.",
    "reembolso": "Reembolso em até 7 dias úteis após o recebimento da devolução, no mesmo meio de pagamento.",
}


def _latencia(media_s: float) -> None:
    time.sleep(random.lognormvariate(0, 0.5) * media_s * chaos.state.latency_multiplier)


@tool
def buscar_pedido(pedido_id: str) -> dict:
    """Consulta status, valores e previsão de entrega de um pedido pelo número."""
    _latencia(0.15)
    pedido = _PEDIDOS.get(pedido_id.strip().lstrip("#"))
    if pedido is None:
        raise ToolException(f"Pedido {pedido_id} não encontrado no ERP.")
    return {"pedido_id": pedido_id, **pedido}


@tool
def calcular_frete(cep: str) -> dict:
    """Calcula prazo e valor do frete para um CEP brasileiro."""
    _latencia(0.4)
    # A "API da transportadora" é o ponto frágil do sistema.
    if random.random() < chaos.state.tool_error_rate:
        raise ToolException("Timeout na API da transportadora (504).")
    digitos = "".join(c for c in cep if c.isdigit())
    regiao = int(digitos[0]) if digitos else 0
    return {"cep": cep, "prazo_dias": 2 + regiao, "valor": round(12.9 + regiao * 3.5, 2)}


@tool
def consultar_politica(tema: str) -> str:
    """Retorna a política da loja sobre devolução, troca, garantia ou reembolso."""
    _latencia(0.05)
    chave = tema.lower().replace("ç", "c").replace("ã", "a")
    for nome, texto in _POLITICAS.items():
        if nome[:5] in chave:
            return texto
    return "Não há política cadastrada para esse tema."


@tool
def abrir_chamado(resumo: str, prioridade: str = "normal") -> dict:
    """Abre um chamado para atendimento humano. Prioridade: baixa, normal ou alta."""
    _latencia(0.2)
    return {"chamado": f"CH-{random.randint(10000, 99999)}", "prioridade": prioridade, "sla_horas": 4 if prioridade == "alta" else 24}


TOOLS = [buscar_pedido, calcular_frete, consultar_politica, abrir_chamado]


def erro_para_o_modelo(exc: Exception, request) -> str | None:
    """Usado pelo ToolErrorMiddleware: falhas esperadas (ToolException) voltam ao
    modelo como ToolMessage de erro, para ele se recuperar. A exceção sai da tool
    antes disso, então o run da tool fica marcado como erro no LangSmith — o que
    não acontece com `handle_tool_error=True`, que registra a falha como sucesso.
    Qualquer outra exceção propaga e derruba o trace."""
    if isinstance(exc, ToolException):
        return f"Error: {exc}"
    return None
