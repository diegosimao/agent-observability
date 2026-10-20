"""Support agent (LangChain `create_agent`) and the instrumented call.

All the observability comes from three things passed in the `config` of each invoke:
- `run_id`  : an id known up front, so user feedback can be attached later
- `tags`    : low-cardinality labels, good for quick filtering
- `metadata`: dimensions to group charts by (prompt version, customer plan,
              channel, app release...)
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
from .tools import TOOLS, tool_error_to_model

# Name of the root trace of every conversation; dashboards and the flywheel filter on it.
RUN_NAME = "support-chat"


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
        middleware=[ToolErrorMiddleware(on_error=tool_error_to_model)],
        name="support-agent",
    )


@dataclass
class AgentResult:
    run_id: uuid.UUID
    answer: str | None
    error: str | None
    latency_s: float
    tokens: int = 0
    tools: list[str] = field(default_factory=list)
    tool_errors: int = 0


def _config(run_id: uuid.UUID, prompt_version: str, plan: str, channel: str, category: str, extra_tags, session: str | None) -> dict:
    metadata = {
        "prompt_version": prompt_version,
        "customer_plan": plan,
        "channel": channel,
        "category": category,
        "app_version": APP_VERSION,
        "llm_mode": llm_mode(),
        "model": model_name(),
    }
    if session:
        # `session_id` is the key LangSmith uses to group traces into Threads
        metadata["session_id"] = session
    return {
        "run_id": run_id,
        "run_name": RUN_NAME,
        "tags": ["support", f"prompt:{prompt_version}", f"channel:{channel}", *(extra_tags or [])],
        "metadata": metadata,
    }


def handle_message(
    message: str,
    *,
    prompt_version: str = "v2",
    plan: str = "free",
    channel: str = "web",
    category: str = "general",
    extra_tags: list[str] | None = None,
    session: str | None = None,
) -> AgentResult:
    run_id = uuid.uuid4()
    config = _config(run_id, prompt_version, plan, channel, category, extra_tags, session)
    start = time.perf_counter()
    try:
        state = build_agent(prompt_version).invoke({"messages": [{"role": "user", "content": message}]}, config=config)
    except Exception as exc:  # the LangSmith trace already records the error; here we just count it
        return AgentResult(run_id, None, f"{type(exc).__name__}: {exc}", time.perf_counter() - start)

    msgs = state["messages"]
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    return AgentResult(
        run_id=run_id,
        answer=msgs[-1].text,  # Gemini 3 returns content blocks (text + reasoning signature)
        error=None,
        latency_s=time.perf_counter() - start,
        tokens=sum((m.usage_metadata or {}).get("total_tokens", 0) for m in msgs if isinstance(m, AIMessage)),
        tools=[m.name for m in tool_msgs],
        tool_errors=sum(1 for m in tool_msgs if m.status == "error"),
    )


def handle_message_stream(
    message: str,
    *,
    prompt_version: str = "v2",
    plan: str = "free",
    channel: str = "web",
    session: str | None = None,
) -> Iterator[dict]:
    """Same trace as `handle_message`, but emits each agent step as it happens:
    tool call, tool result and final answer."""
    run_id = uuid.uuid4()
    config = _config(run_id, prompt_version, plan, channel, "general", ["stream"], session)
    yield {"event": "start", "run_id": str(run_id), "prompt_version": prompt_version}
    try:
        for step in build_agent(prompt_version).stream(
            {"messages": [{"role": "user", "content": message}]}, config=config, stream_mode="updates"
        ):
            for update in step.values():
                for m in (update or {}).get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        for tc in m.tool_calls:
                            yield {"event": "tool_call", "tool": tc["name"], "args": tc["args"]}
                    elif isinstance(m, ToolMessage):
                        yield {"event": "tool_result", "tool": m.name, "ok": m.status != "error"}
                    elif isinstance(m, AIMessage):
                        yield {"event": "answer", "text": m.text}
    except Exception as exc:
        yield {"event": "error", "error": f"{type(exc).__name__}: {exc}"}
    yield {"event": "end", "run_id": str(run_id)}
