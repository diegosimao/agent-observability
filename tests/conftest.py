import os

# Antes de qualquer import do pacote: testes nunca falam com LLM nem LangSmith.
os.environ["AGENT_MODE"] = "mock"
os.environ["LANGSMITH_TRACING"] = "false"

import pytest  # noqa: E402

from agent_observability import chaos  # noqa: E402


@pytest.fixture(autouse=True)
def sem_caos():
    """Testes determinísticos: sem 503 aleatório, sem latência artificial."""
    chaos.state.latency_multiplier = 0.0
    chaos.state.tool_error_rate = 0.0
    chaos.state.llm_error_rate = 0.0
    yield
    chaos.normal()
