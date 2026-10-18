"""Agente de suporte (LangChain `create_agent`) e a chamada instrumentada.

Toda a observabilidade vem de três coisas passadas no `config` de cada invoke:
- `run_id`  : id conhecido de antemão, para anexar feedback do usuário depois
- `tags`    : rótulos baixos em cardinalidade, bons para filtro rápido
- `metadata`: dimensões para agrupar gráficos (versão do prompt, plano do
              cliente, canal, release do app...)
"""

import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cache

from langchain.agents import create_agent
from langchain.agents.middleware import ToolErrorMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from .config import APP_VERSION, GEMINI_MODEL, LOCAL_MODEL, PROMPTS, llm_mode, model_name
from .tools import TOOLS, erro_para_o_modelo


def _model():
    match llm_mode():
        case "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)
        case "local":
            from .local_llm import LocalChatModel

            return LocalChatModel(model_path=LOCAL_MODEL)
        case _:
            from .mock_llm import FakeSupportModel

            return FakeSupportModel()


@cache
def build_agent(prompt_version: str):
    return create_agent(
        _model(),
        TOOLS,
        system_prompt=PROMPTS[prompt_version],
        middleware=[ToolErrorMiddleware(on_error=erro_para_o_modelo)],
        name="agente-suporte",
    )


@dataclass
class Resultado:
    run_id: uuid.UUID
    resposta: str | None
    erro: str | None
    latencia_s: float
    tokens: int = 0
    tools: list[str] = field(default_factory=list)
    tool_erros: int = 0


def _config(run_id: uuid.UUID, prompt_version: str, plano: str, canal: str, categoria: str, extra_tags, sessao: str | None) -> dict:
    metadata = {
        "prompt_version": prompt_version,
        "plano_cliente": plano,
        "canal": canal,
        "categoria": categoria,
        "app_version": APP_VERSION,
        "llm_mode": llm_mode(),
        "modelo": model_name(),
    }
    if sessao:
        # `session_id` é a chave que o LangSmith usa para agrupar traces em Threads
        metadata["session_id"] = sessao
    return {
        "run_id": run_id,
        "run_name": "atendimento",
        "tags": ["suporte", f"prompt:{prompt_version}", f"canal:{canal}", *(extra_tags or [])],
        "metadata": metadata,
    }


def atender(
    pergunta: str,
    *,
    prompt_version: str = "v2",
    plano: str = "free",
    canal: str = "web",
    categoria: str = "geral",
    extra_tags: list[str] | None = None,
    sessao: str | None = None,
) -> Resultado:
    run_id = uuid.uuid4()
    config = _config(run_id, prompt_version, plano, canal, categoria, extra_tags, sessao)
    inicio = time.perf_counter()
    try:
        estado = build_agent(prompt_version).invoke({"messages": [{"role": "user", "content": pergunta}]}, config=config)
    except Exception as exc:  # o trace no LangSmith já registra o erro; aqui só contabilizamos
        return Resultado(run_id, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - inicio)

    msgs = estado["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    return Resultado(
        run_id=run_id,
        resposta=msgs[-1].text,  # Gemini 3 devolve blocos (texto + assinatura de raciocínio)
        erro=None,
        latencia_s=time.perf_counter() - inicio,
        tokens=sum((m.usage_metadata or {}).get("total_tokens", 0) for m in msgs if isinstance(m, AIMessage)),
        tools=[m.name for m in tool_msgs],
        tool_erros=sum(1 for m in tool_msgs if m.status == "error"),
    )


def atender_stream(
    pergunta: str,
    *,
    prompt_version: str = "v2",
    plano: str = "free",
    canal: str = "web",
    sessao: str | None = None,
) -> Iterator[dict]:
    """Mesmo trace de `atender`, mas emite cada passo do agente assim que ele
    acontece: chamada de ferramenta, resultado e resposta final."""
    run_id = uuid.uuid4()
    config = _config(run_id, prompt_version, plano, canal, "geral", ["stream"], sessao)
    yield {"evento": "inicio", "run_id": str(run_id), "prompt_version": prompt_version}
    try:
        for passo in build_agent(prompt_version).stream(
            {"messages": [{"role": "user", "content": pergunta}]}, config=config, stream_mode="updates"
        ):
            for atualizacao in passo.values():
                for m in (atualizacao or {}).get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            yield {"evento": "tool_call", "tool": tc["name"], "args": tc["args"]}
                    elif isinstance(m, ToolMessage):
                        yield {"evento": "tool_result", "tool": m.name, "ok": m.status != "error"}
                    elif isinstance(m, AIMessage):
                        yield {"evento": "resposta", "texto": m.text}
    except Exception as exc:
        yield {"evento": "erro", "erro": f"{type(exc).__name__}: {exc}"}
    yield {"evento": "fim", "run_id": str(run_id)}
