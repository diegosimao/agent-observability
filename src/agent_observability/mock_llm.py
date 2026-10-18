"""LLM simulado que se comporta como um chat model real para o LangSmith.

Faz tool calling por palavras-chave, preenche `usage_metadata` (tokens) e se
identifica como `gemini-2.5-flash` via `_get_ls_params`, então o LangSmith
calcula custo, latência e tokens exatamente como faria com o modelo real.
Assim a demo inteira roda sem gastar um centavo de API de LLM.
"""

import ast
import json
import random
import re
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import chaos
from .config import GEMINI_MODEL

_TEMAS = ("devol", "troca", "garantia", "reembols")


def _tokens(texto: str) -> int:
    return max(1, len(texto) // 4)


def _parse(conteudo: str) -> Any:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(conteudo)
        except (ValueError, SyntaxError):
            continue
    return conteudo


class FakeSupportModel(BaseChatModel):
    model: str = GEMINI_MODEL
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-support-model"

    def _get_ls_params(self, stop=None, **kwargs):
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "google_genai"
        params["ls_model_name"] = self.model
        return params

    def bind_tools(self, tools, **kwargs):
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": names})

    # ------------------------------------------------------------------ #
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if random.random() < chaos.state.llm_error_rate:
            time.sleep(random.uniform(0.5, 2.0) * chaos.state.latency_multiplier)
            raise RuntimeError("503 UNAVAILABLE: modelo sobrecarregado (simulado)")

        conciso = "Regras:" in str(messages[0].content)
        ultima = messages[-1]

        if isinstance(ultima, HumanMessage):
            msg = self._planejar(str(ultima.content), conciso)
        else:
            resultados = [m for m in messages if isinstance(m, ToolMessage)]
            msg = AIMessage(content=self._responder(resultados, conciso))

        entrada = sum(_tokens(str(m.content)) for m in messages) + 180 * len(self.tool_names)
        saida = _tokens(str(msg.content)) + 25 * len(msg.tool_calls)
        msg.usage_metadata = {"input_tokens": entrada, "output_tokens": saida, "total_tokens": entrada + saida}
        msg.response_metadata = {"model_name": self.model, "finish_reason": "STOP"}

        # tempo ~ primeiro token + geração proporcional à saída
        time.sleep((0.25 + saida * 0.004) * random.lognormvariate(0, 0.35) * chaos.state.latency_multiplier)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _planejar(self, texto: str, conciso: bool) -> AIMessage:
        t = texto.lower()
        calls = []
        if pedido := re.search(r"\b(\d{4})\b", t):
            calls.append(("buscar_pedido", {"pedido_id": pedido.group(1)}))
        if cep := re.search(r"\b(\d{5}-?\d{3})\b", t):
            calls.append(("calcular_frete", {"cep": cep.group(1)}))
        for tema in _TEMAS:
            if tema in t:
                calls.append(("consultar_politica", {"tema": tema}))
                break
        if any(p in t for p in ("humano", "atendente", "absurdo", "reclama", "péssimo")):
            prioridade = "alta" if conciso or "absurdo" in t else "normal"
            calls.append(("abrir_chamado", {"resumo": texto[:120], "prioridade": prioridade}))

        calls = [(n, a) for n, a in calls if n in self.tool_names]
        if not calls:
            resposta = "Posso ajudar com pedidos, frete, trocas e devoluções. Qual o número do seu pedido?"
            if not conciso:
                resposta = "Olá! Obrigado por entrar em contato com a Vértice. " + resposta + " Fico à disposição para o que precisar."
            return AIMessage(content=resposta)
        return AIMessage(
            content="",
            tool_calls=[{"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"} for n, a in calls],
        )

    def _responder(self, resultados: list[ToolMessage], conciso: bool) -> str:
        partes = []
        for r in resultados:
            dado = _parse(str(r.content))
            if "não encontrado" in str(r.content):
                partes.append("Não encontrei esse número de pedido; pode conferir no e-mail de confirmação?")
            elif r.status == "error" or "Error" in str(r.content):
                partes.append("Não consegui consultar essa informação agora, desculpe — tente novamente em alguns minutos.")
            elif r.name == "buscar_pedido":
                partes.append(f"O pedido {dado['pedido_id']} está {dado['status']} (total R$ {dado['total']:.2f}).")
            elif r.name == "calcular_frete":
                partes.append(f"O frete para {dado['cep']} custa R$ {dado['valor']:.2f} com prazo de {dado['prazo_dias']} dias úteis.")
            elif r.name == "consultar_politica":
                partes.append(str(dado))
            elif r.name == "abrir_chamado":
                partes.append(f"Abri o chamado {dado['chamado']}; um atendente responde em até {dado['sla_horas']}h.")
        texto = " ".join(partes)
        if not conciso:
            texto = (
                "Olá! Agradeço muito por entrar em contato com a Vértice. Verifiquei as informações nos nossos sistemas. "
                + texto
                + " Caso tenha qualquer outra dúvida, sobre este ou outros pedidos, estou sempre à disposição para ajudar!"
            )
        return texto
