"""API HTTP do agente — o "produção" de verdade que o simulador e um front-end chamam.

    uv run obs-api            # http://localhost:8000/docs

- POST /v1/chat          resposta completa + run_id
- POST /v1/chat/stream   Server-Sent Events com cada passo do agente
- POST /v1/feedback      👍/👎 do usuário, anexado ao trace no LangSmith
- GET  /health           modo do LLM, tracing e versão
"""

import hashlib
import json
import os
import time
import uuid
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langsmith import Client
from pydantic import BaseModel, Field

from . import chaos
from .agent import atender, atender_stream
from .config import APP_VERSION, PROJECT, llm_mode, model_name, project_id, tracing_enabled

app = FastAPI(title="Agente de suporte Vértice", version=APP_VERSION)


class ChatIn(BaseModel):
    mensagem: str = Field(min_length=1, max_length=2000)
    sessao: str | None = Field(None, description="id estável do cliente; fixa a variante do A/B")
    prompt_version: Literal["v1", "v2"] | None = Field(None, description="força uma variante (senão, A/B por sessão)")
    plano: Literal["free", "plus", "enterprise"] = "free"
    canal: Literal["web", "whatsapp", "app", "api"] = "api"


class ChatOut(BaseModel):
    run_id: uuid.UUID
    resposta: str
    prompt_version: str
    tools: list[str]
    tool_erros: int
    tokens: int
    latencia_ms: int


class FeedbackIn(BaseModel):
    run_id: uuid.UUID
    score: Literal[0, 1]
    comentario: str | None = Field(None, max_length=500)


def variante(sessao: str | None) -> str:
    """A/B determinístico: a mesma sessão sempre cai na mesma variante, então um
    cliente não vê o comportamento do agente mudar no meio da conversa."""
    if not sessao:
        return "v2"
    return "v1" if hashlib.sha256(sessao.encode()).digest()[0] % 2 == 0 else "v2"


@app.middleware("http")
async def tempo_de_resposta(request: Request, call_next):
    inicio = time.perf_counter()
    resposta = await call_next(request)
    resposta.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - inicio) * 1000:.0f}"
    return resposta


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_mode": llm_mode(), "modelo": model_name(), "tracing": tracing_enabled(), "versao": APP_VERSION}


@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn) -> ChatOut:
    versao = body.prompt_version or variante(body.sessao)
    # o agente é síncrono (tools bloqueantes); roda fora do event loop
    r = await run_in_threadpool(
        atender, body.mensagem, prompt_version=versao, plano=body.plano, canal=body.canal, sessao=body.sessao, extra_tags=["api"]
    )
    if r.erro:
        # o run_id vai junto para o suporte achar o trace exato no LangSmith
        raise HTTPException(503, detail={"erro": "agente indisponível", "run_id": str(r.run_id)})
    return ChatOut(
        run_id=r.run_id,
        resposta=r.resposta or "",
        prompt_version=versao,
        tools=r.tools,
        tool_erros=r.tool_erros,
        tokens=r.tokens,
        latencia_ms=int(r.latencia_s * 1000),
    )


@app.post("/v1/chat/stream")
def chat_stream(body: ChatIn) -> StreamingResponse:
    versao = body.prompt_version or variante(body.sessao)
    eventos = atender_stream(body.mensagem, prompt_version=versao, plano=body.plano, canal=body.canal, sessao=body.sessao)
    sse = (f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in eventos)
    return StreamingResponse(sse, media_type="text/event-stream")


@app.post("/v1/feedback", status_code=202)
def feedback(body: FeedbackIn) -> dict:
    if not tracing_enabled():
        raise HTTPException(503, detail="LangSmith não configurado")
    Client().create_feedback(body.run_id, key="user_score", score=body.score, comment=body.comentario, session_id=project_id())
    return {"aceito": True, "projeto": PROJECT}


@app.post("/_chaos/{modo}", include_in_schema=False)
def caos(modo: Literal["incidente", "normal"]) -> dict:
    """Liga o incidente simulado neste processo. Só existe com CHAOS_ENDPOINT=true."""
    if os.getenv("CHAOS_ENDPOINT", "").lower() != "true":
        raise HTTPException(404)
    getattr(chaos, modo)()
    return {"modo": modo}


def main() -> None:
    import uvicorn

    uvicorn.run("agent_observability.api:app", host="127.0.0.1", port=8000)
