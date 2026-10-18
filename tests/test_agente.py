from types import SimpleNamespace

from langchain_core.callbacks import BaseCallbackHandler

from agent_observability import chaos
from agent_observability.agent import atender, build_agent
from agent_observability.evaluate import pass_hat_k
from agent_observability.flywheel import exemplo_de_trace
from agent_observability.local_llm import extrair_tool_calls


class _ContaTools(BaseCallbackHandler):
    def __init__(self):
        self.erros, self.sucessos = [], []

    def on_tool_error(self, error, **kwargs):
        self.erros.append(str(error))

    def on_tool_end(self, output, **kwargs):
        self.sucessos.append(output)


def test_falha_de_tool_aparece_como_erro_no_trace_e_conversa_se_recupera():
    # Regressão: com handle_tool_error=True a falha era registrada como sucesso
    # e o gráfico "erros por ferramenta" ficava sempre zerado.
    cb = _ContaTools()
    estado = build_agent("v2").invoke({"messages": [{"role": "user", "content": "pedido 9999"}]}, config={"callbacks": [cb]})
    assert cb.erros == ["Pedido 9999 não encontrado no ERP."]
    assert cb.sucessos == []
    assert "Não encontrei" in estado["messages"][-1].text


def test_agente_usa_as_tools_certas_e_conta_tokens():
    r = atender("Qual o status do pedido 1002 e o frete para 30140-071?")
    assert r.erro is None
    assert r.tools == ["buscar_pedido", "calcular_frete"]
    assert r.tokens > 0 and r.tool_erros == 0


def test_erro_do_llm_vira_resultado_com_run_id():
    chaos.state.llm_error_rate = 1.0
    r = atender("Oi")
    assert r.resposta is None and "503" in r.erro and r.run_id


def test_pass_hat_k_e_mais_rigoroso_que_pass_1():
    tarefas = [(3, 3), (0, 3), (2, 3)]  # (acertos, tentativas)
    assert pass_hat_k(tarefas, 1) == (1 + 0 + 2 / 3) / 3
    assert pass_hat_k(tarefas, 3) == 1 / 3  # só a tarefa que acertou sempre conta
    assert pass_hat_k(tarefas, 2) < pass_hat_k(tarefas, 1)


def test_flywheel_monta_exemplo_e_descarta_falha_de_tool():
    def run(conteudo_tool):
        return SimpleNamespace(
            id="00000000-0000-0000-0000-000000000001",
            outputs={"messages": [
                {"type": "human", "content": "pedido 1002"},
                {"type": "ai", "content": "", "tool_calls": [{"name": "buscar_pedido", "args": {"pedido_id": "1002"}, "id": "c1", "type": "tool_call"}]},
                {"type": "tool", "content": conteudo_tool, "name": "buscar_pedido", "tool_call_id": "c1"},
                {"type": "ai", "content": "Está em transporte."},
            ]},
        )

    ex = exemplo_de_trace(run("{'status': 'em transporte'}"), "v2")
    assert [m["role"] for m in ex["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert ex["messages"][2]["tool_calls"][0]["function"] == {"name": "buscar_pedido", "arguments": {"pedido_id": "1002"}}
    assert ex["origem"]["run_id"].endswith("1")
    assert exemplo_de_trace(run("Error: Timeout na API da transportadora (504)."), "v2") is None


def test_parser_de_tool_call_do_qwen_ignora_json_quebrado():
    texto = '<tool_call>\n{"name": "calcular_frete", "arguments": {"cep": "01310-100"}}\n</tool_call>\n<tool_call>{quebrado</tool_call>'
    resto, calls = extrair_tool_calls(texto)
    assert [(c["name"], c["args"]) for c in calls] == [("calcular_frete", {"cep": "01310-100"})]
