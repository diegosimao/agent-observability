import os

# Before any package import: tests never talk to an LLM or to LangSmith.
os.environ["AGENT_MODE"] = "mock"
os.environ["LANGSMITH_TRACING"] = "false"

import pytest  # noqa: E402

from agent_observability import chaos  # noqa: E402


@pytest.fixture(autouse=True)
def no_chaos():
    """Deterministic tests: no random 503s, no artificial latency."""
    chaos.state.latency_multiplier = 0.0
    chaos.state.tool_error_rate = 0.0
    chaos.state.llm_error_rate = 0.0
    yield
    chaos.normal()
