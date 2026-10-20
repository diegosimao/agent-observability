"""Local chat model (Hugging Face transformers) with Qwen-style tool calling.

Loads a base model or a LoRA adapter trained by `finetune/train_lora.py` and
plugs into the same `create_agent`. LangSmith sees tokens, latency and tools
exactly as it does for Gemini, so base vs. LoRA vs. Gemini become experiments
you can compare side by side.
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
_lock = threading.Lock()  # one GPU, one generation at a time


@cache
def _load(model_path: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter = Path(model_path) / "adapter_config.json"
    if adapter.exists():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model.eval(), tokenizer


def to_chat_template(messages: list[BaseMessage]) -> list[dict]:
    """LangChain messages → Hugging Face chat format (the same one used in training)."""
    out = []
    for m in messages:
        if isinstance(m, SystemMessage):
            out.append({"role": "system", "content": m.text})
        elif isinstance(m, HumanMessage):
            out.append({"role": "user", "content": m.text})
        elif isinstance(m, AIMessage):
            item = {"role": "assistant", "content": m.text}
            if m.tool_calls:
                item["tool_calls"] = [{"type": "function", "function": {"name": tc["name"], "arguments": tc["args"]}} for tc in m.tool_calls]
            out.append(item)
        elif isinstance(m, ToolMessage):
            out.append({"role": "tool", "name": m.name, "content": m.text})
    return out


def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    calls = []
    for block in _TOOL_CALL.findall(text):
        try:
            data = json.loads(block)
            calls.append({"name": data["name"], "args": data.get("arguments", {}), "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"})
        except (json.JSONDecodeError, KeyError):
            continue  # malformed tool call: small models sometimes get the JSON wrong
    return _TOOL_CALL.sub("", text).strip(), calls


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

        model, tokenizer = _load(self.model_path)
        ids = tokenizer.apply_chat_template(
            to_chat_template(messages),
            tools=self.tools_schema or None,
            add_generation_prompt=True,
            return_dict=True,  # transformers 5 always returns a dict; explicit so 4.x works too
            return_tensors="pt",
        )["input_ids"].to(model.device)
        with _lock, torch.inference_mode():
            generated = model.generate(ids, max_new_tokens=self.max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        new_tokens = generated[0, ids.shape[1]:]
        text, calls = parse_tool_calls(tokenizer.decode(new_tokens, skip_special_tokens=True))
        msg = AIMessage(content=text, tool_calls=calls)
        msg.usage_metadata = {"input_tokens": ids.shape[1], "output_tokens": len(new_tokens), "total_tokens": ids.shape[1] + len(new_tokens)}
        msg.response_metadata = {"model_name": Path(self.model_path).name}
        return ChatResult(generations=[ChatGeneration(message=msg)])
