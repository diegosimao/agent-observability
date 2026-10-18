"""Configuração central: modo do LLM, versões de prompt e metadados de deploy."""

import os
from functools import cache

from dotenv import load_dotenv

load_dotenv()

APP_VERSION = os.getenv("APP_VERSION", "1.3.0")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# id do Hugging Face (modelo base) ou pasta com o adaptador LoRA treinado
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT = os.getenv("LANGSMITH_PROJECT", "agente-suporte-prod")


def llm_mode() -> str:
    mode = os.getenv("AGENT_MODE", "auto").lower()
    if mode == "auto":
        return "gemini" if os.getenv("GOOGLE_API_KEY") else "mock"
    return mode


@cache
def project_id():
    """id do projeto no LangSmith; o SDK passou a exigir em create_feedback."""
    from langsmith import Client

    return Client().read_project(project_name=PROJECT).id


def model_name() -> str:
    match llm_mode():
        case "local":
            return LOCAL_MODEL.replace("\\", "/").rstrip("/").split("/")[-1]
        case "mock":
            return "mock"
        case _:
            return GEMINI_MODEL


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


# Duas versões de system prompt rodando em paralelo (A/B) — cada trace grava
# qual versão atendeu, e o dashboard compara latência, custo e satisfação.
PROMPTS = {
    "v1": (
        "Você é o assistente de atendimento da loja online Vértice. "
        "Use as ferramentas disponíveis para responder sobre pedidos, frete e políticas. "
        "Seja educado e completo."
    ),
    "v2": (
        "Você é o assistente de atendimento da loja online Vértice.\n"
        "Regras:\n"
        "1. Sempre consulte uma ferramenta antes de afirmar algo sobre pedido, frete ou política.\n"
        "2. Responda em no máximo 3 frases, em português.\n"
        "3. Se o cliente pedir um humano ou estiver insatisfeito, abra um chamado.\n"
        "4. Nunca invente número de pedido, prazo ou valor."
    ),
}
