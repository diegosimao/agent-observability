import json

from fastapi.testclient import TestClient

from agent_observability import chaos
from agent_observability.api import app, variante

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["llm_mode"] == "mock"


def test_chat_devolve_run_id_e_tools():
    r = client.post("/v1/chat", json={"mensagem": "Como funciona a devolução?", "prompt_version": "v2"})
    corpo = r.json()
    assert r.status_code == 200
    assert corpo["tools"] == ["consultar_politica"] and corpo["run_id"]
    assert "Server-Timing" in r.headers


def test_chat_com_llm_fora_devolve_503_com_run_id():
    chaos.state.llm_error_rate = 1.0
    r = client.post("/v1/chat", json={"mensagem": "Oi"})
    assert r.status_code == 503 and r.json()["detail"]["run_id"]


def test_ab_e_fixo_por_sessao():
    assert all(variante("cliente-7") == variante("cliente-7") for _ in range(5))
    assert {variante(f"cliente-{i}") for i in range(40)} == {"v1", "v2"}


def test_stream_emite_passos_na_ordem():
    with client.stream("POST", "/v1/chat/stream", json={"mensagem": "pedido 1002", "prompt_version": "v2"}) as r:
        eventos = [json.loads(l[6:])["evento"] for l in r.iter_lines() if l.startswith("data: ")]
    assert eventos == ["inicio", "tool_call", "tool_result", "resposta", "fim"]


def test_feedback_sem_langsmith_e_503_e_valida_score():
    assert client.post("/v1/feedback", json={"run_id": "00000000-0000-0000-0000-000000000001", "score": 1}).status_code == 503
    assert client.post("/v1/feedback", json={"run_id": "x", "score": 5}).status_code == 422


def test_endpoint_de_caos_nao_existe_por_padrao():
    assert client.post("/_chaos/incidente").status_code == 404
