"""HTTP API for the agent: the actual "production" that the simulator and a front end call.

    uv run obs-api            # http://localhost:8000/docs

- POST /v1/chat          full answer + run_id
- POST /v1/chat/stream   Server-Sent Events with each agent step
- POST /v1/feedback      user 👍/👎, attached to the trace in LangSmith
- GET  /health           LLM mode, tracing and version
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
from .agent import handle_message, handle_message_stream
from .config import APP_VERSION, PROJECT, llm_mode, model_name, project_id, tracing_enabled

app = FastAPI(title="Vertice support agent", version=APP_VERSION)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session: str | None = Field(None, description="stable customer id; pins the A/B variant")
    prompt_version: Literal["v1", "v2"] | None = Field(None, description="force a variant (otherwise A/B by session)")
    plan: Literal["free", "plus", "enterprise"] = "free"
    channel: Literal["web", "whatsapp", "app", "api"] = "api"


class ChatOut(BaseModel):
    run_id: uuid.UUID
    answer: str
    prompt_version: str
    tools: list[str]
    tool_errors: int
    tokens: int
    latency_ms: int


class FeedbackIn(BaseModel):
    run_id: uuid.UUID
    score: Literal[0, 1]
    comment: str | None = Field(None, max_length=500)


def variant(session: str | None) -> str:
    """Deterministic A/B: the same session always lands on the same variant, so a
    customer never sees the agent's behavior change mid-conversation."""
    if not session:
        return "v2"
    return "v1" if hashlib.sha256(session.encode()).digest()[0] % 2 == 0 else "v2"


@app.middleware("http")
async def server_timing(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["Server-Timing"] = f"app;dur={(time.perf_counter() - start) * 1000:.0f}"
    return response


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "llm_mode": llm_mode(), "model": model_name(), "tracing": tracing_enabled(), "version": APP_VERSION}


@app.post("/v1/chat", response_model=ChatOut)
async def chat(body: ChatIn) -> ChatOut:
    version = body.prompt_version or variant(body.session)
    # the agent is synchronous (blocking tools); run it off the event loop
    r = await run_in_threadpool(
        handle_message, body.message, prompt_version=version, plan=body.plan, channel=body.channel, session=body.session, extra_tags=["api"]
    )
    if r.error:
        # the run_id goes along so support can find the exact trace in LangSmith
        raise HTTPException(503, detail={"error": "agent unavailable", "run_id": str(r.run_id)})
    return ChatOut(
        run_id=r.run_id,
        answer=r.answer or "",
        prompt_version=version,
        tools=r.tools,
        tool_errors=r.tool_errors,
        tokens=r.tokens,
        latency_ms=int(r.latency_s * 1000),
    )


@app.post("/v1/chat/stream")
def chat_stream(body: ChatIn) -> StreamingResponse:
    version = body.prompt_version or variant(body.session)
    events = handle_message_stream(body.message, prompt_version=version, plan=body.plan, channel=body.channel, session=body.session)
    sse = (f"data: {json.dumps(e, ensure_ascii=False)}\n\n" for e in events)
    return StreamingResponse(sse, media_type="text/event-stream")


@app.post("/v1/feedback", status_code=202)
def feedback(body: FeedbackIn) -> dict:
    if not tracing_enabled():
        raise HTTPException(503, detail="LangSmith is not configured")
    Client().create_feedback(body.run_id, key="user_score", score=body.score, comment=body.comment, session_id=project_id())
    return {"accepted": True, "project": PROJECT}


@app.post("/_chaos/{mode}", include_in_schema=False)
def set_chaos(mode: Literal["incident", "normal"]) -> dict:
    """Turns the simulated incident on in this process. Only exists with CHAOS_ENDPOINT=true."""
    if os.getenv("CHAOS_ENDPOINT", "").lower() != "true":
        raise HTTPException(404)
    getattr(chaos, mode)()
    return {"mode": mode}


def main() -> None:
    import uvicorn

    uvicorn.run("agent_observability.api:app", host="127.0.0.1", port=8000)
