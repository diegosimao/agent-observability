import json

from fastapi.testclient import TestClient

from agent_observability import chaos
from agent_observability.api import app, variant

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["llm_mode"] == "mock"


def test_chat_returns_run_id_and_tools():
    r = client.post("/v1/chat", json={"message": "How do returns work?", "prompt_version": "v2"})
    body = r.json()
    assert r.status_code == 200
    assert body["tools"] == ["get_policy"] and body["run_id"]
    assert "Server-Timing" in r.headers


def test_chat_with_llm_down_returns_503_with_run_id():
    chaos.state.llm_error_rate = 1.0
    r = client.post("/v1/chat", json={"message": "Hi"})
    assert r.status_code == 503 and r.json()["detail"]["run_id"]


def test_ab_is_sticky_per_session():
    assert all(variant("customer-7") == variant("customer-7") for _ in range(5))
    assert {variant(f"customer-{i}") for i in range(40)} == {"v1", "v2"}


def test_stream_emits_steps_in_order():
    with client.stream("POST", "/v1/chat/stream", json={"message": "order 1002", "prompt_version": "v2"}) as r:
        events = [json.loads(line[6:])["event"] for line in r.iter_lines() if line.startswith("data: ")]
    assert events == ["start", "tool_call", "tool_result", "answer", "end"]


def test_feedback_without_langsmith_is_503_and_score_is_validated():
    assert client.post("/v1/feedback", json={"run_id": "00000000-0000-0000-0000-000000000001", "score": 1}).status_code == 503
    assert client.post("/v1/feedback", json={"run_id": "x", "score": 5}).status_code == 422


def test_chaos_endpoint_does_not_exist_by_default():
    assert client.post("/_chaos/incident").status_code == 404
