"""Chat model local (Hugging Face transformers) com tool calling no formato Qwen.

Carrega um modelo base ou um adaptador LoRA treinado por `finetune/train_lora.py`
e se pluga no mesmo `create_agent` — o LangSmith vê tokens, latência e tools
exatamente como vê o Gemini, então base × LoRA × Gemini viram experimentos
comparáveis lado a lado.
"""

import json
import re
import threading
import uuid
from functools import cache
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

_TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_lock = threading.Lock()  # uma GPU, uma geração por vez


@cache
def _carregar(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adaptador = Path(model_path) / "adapter_config.json"
    if adaptador.exists():
        from peft import AutoPeftModelForCausalLM

        modelo = AutoPeftModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    else:
        modelo = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return modelo.eval(), tokenizer


def para_chat_template(messages: list[BaseMessage]) -> list[dict]:
    """Mensagens LangChain → formato de chat do Hugging Face (o mesmo do treino)."""
    saida = []
    for m in messages:
        if isinstance(m, SystemMessage):
            saida.append({"role": "system", "content": m.text})
        elif isinstance(m, HumanMessage):
            saida.append({"role": "user", "content": m.text})
        elif isinstance(m, AIMessage):
            item = {"role": "assistant", "content": m.text}
            if m.tool_calls:
                item["tool_calls"] = [{"type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}} for tc in m.tool_calls]
            saida.append(item)
        elif isinstance(m, ToolMessage):
            saida.append({"role": "tool", "name": m.name, "content": m.text})
    return saida


def extrair_tool_calls(texto: str) -> tuple[str, list[dict]]:
    calls = []
    for bloco in _TOOL_CALL.findall(texto):
        try:
            dado = json.loads(bloco)
            calls.append({"name": dado["name"], "args": dado.get("arguments", {}), "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"})
        except (json.JSONDecodeError, KeyError):
            continue  # tool call malformada: o modelo pequeno às vezes erra o JSON
    return _TOOL_CALL.sub("", texto).strip(), calls


class LocalChatModel(BaseChatModel):
    model_path: str
    max_new_tokens: int = 256
    tools_schema: list[dict] = []

    @property
    def _llm_type(self) -> str:
        return "local-transformers"

    def _get_ls_params(self, stop=None, **kwargs):
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "huggingface"
        params["ls_model_name"] = Path(self.model_path).name
        return params

    def bind_tools(self, tools, **kwargs):
        return self.model_copy(update={"tools_schema": [convert_to_openai_tool(t) for t in tools]})

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        import torch

        modelo, tokenizer = _carregar(self.model_path)
        ids = tokenizer.apply_chat_template(
            para_chat_template(messages),
            tools=self.tools_schema or None,
            add_generation_prompt=True,
            return_dict=True,  # transformers 5 sempre devolve dict; explícito para funcionar no 4.x também
            return_tensors="pt",
        )["input_ids"].to(modelo.device)
        with _lock, torch.inference_mode():
            gerado = modelo.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        novos = gerado[0, ids.shape[1]:]
        texto, calls = extrair_tool_calls(tokenizer.decode(novos, skip_special_tokens=True))
        msg = AIMessage(content=texto, tool_calls=calls)
        msg.usage_metadata = {"input_tokens": ids.shape[1], "output_tokens": len(novos), "total_tokens": ids.shape[1] + len(novos)}
        msg.response_metadata = {"model_name": Path(self.model_path).name}
        return ChatResult(generations=[ChatGeneration(message=msg)])
