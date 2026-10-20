from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler

from agent_observability import chaos
from agent_observability.agent import build_agent, handle_message
from agent_observability.evaluate import pass_hat_k
from agent_observability.flywheel import example_from_trace
from agent_observability.local_llm import parse_tool_calls


class _ToolCounter(BaseCallbackHandler):
    def __init__(self):
        self.errors, self.successes = [], []

    def on_tool_error(self, error, **kwargs):
        self.errors.append(str(error))

    def on_tool_end(self, output, **kwargs):
        self.successes.append(output)


def test_tool_failure_is_traced_as_error_and_conversation_recovers():
    # Regression: with handle_tool_error=True the failure was recorded as a success
    # and the "errors per tool" chart stayed at zero forever.
    cb = _ToolCounter()
    state = build_agent("v2").invoke({"messages": [{"role": "user", "content": "order 9999"}]}, config={"callbacks": [cb]})
    assert cb.errors == ["Order 9999 not found in the ERP."]
    assert cb.successes == []
    assert "couldn't find" in state["messages"][-1].text


def test_agent_uses_the_right_tools_and_counts_tokens():
    r = handle_message("What's the status of order 1002 and shipping to 30140-071?")
    assert r.error is None
    assert r.tools == ["get_order", "calculate_shipping"]
    assert r.tokens > 0 and r.tool_errors == 0


def test_llm_error_becomes_result_with_run_id():
    chaos.state.llm_error_rate = 1.0
    r = handle_message("Hi")
    assert r.answer is None and "503" in r.error and r.run_id


def test_pass_hat_k_is_stricter_than_pass_1():
    tasks = [(3, 3), (0, 3), (2, 3)]  # (successes, attempts)
    assert pass_hat_k(tasks, 1) == (1 + 0 + 2 / 3) / 3
    assert pass_hat_k(tasks, 3) == 1 / 3  # only the task that always succeeded counts
    assert pass_hat_k(tasks, 2) < pass_hat_k(tasks, 1)


def test_flywheel_builds_example_and_drops_tool_failures():
    def run(tool_content):
        return SimpleNamespace(
            id="00000000-0000-0000-0000-000000000001",
            outputs={"messages": [
                {"type": "human", "content": "order 1002"},
                {"type": "ai", "content": "", "tool_calls": [{"name": "get_order", "args": {"order_id": "1002"}, "id": "c1", "type": "tool_call"}]},
                {"type": "tool", "content": tool_content, "name": "get_order", "tool_call_id": "c1"},
                {"type": "ai", "content": "It is in transit."},
            ]},
        )

    ex = example_from_trace(run("{'status': 'in transit'}"), "v2")
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert ex["messages"][2]["tool_calls"][0]["function"] == {"name": "get_order", "arguments": {"order_id": "1002"}}
    assert ex["source"]["run_id"].endswith("1")
    assert example_from_trace(run("Error: Carrier API timeout (504)."), "v2") is None


def test_qwen_tool_call_parser_skips_broken_json():
    text = '<tool_call>\n{"name": "calculate_shipping", "arguments": {"zip_code": "01310-100"}}\n</tool_call>\n<tool_call>{broken</tool_call>'
    _, calls = parse_tool_calls(text)
    assert [(c["name"], c["args"]) for c in calls] == [("calculate_shipping", {"zip_code": "01310-100"})]
