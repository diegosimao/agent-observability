"""Simulated LLM that behaves like a real chat model as far as LangSmith can tell.

It does tool calling by keyword, fills in `usage_metadata` (tokens) and reports
itself as the configured Gemini model via `_get_ls_params`, so LangSmith computes
cost, latency and tokens exactly as it would for the real model. The whole demo
runs without spending a cent on LLM APIs.
"""

import ast
import json
import random
import re
import time
import uuid
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from . import chaos
from .config import GEMINI_MODEL

# (keyword in the question, topic passed to get_policy)
_POLICY_TOPICS = (("return", "returns"), ("exchang", "exchange"), ("swap", "exchange"), ("warrant", "warranty"), ("refund", "refund"))
_ESCALATION_WORDS = ("human", "agent", "ridiculous", "complain", "terrible", "manager")


def _tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _parse(content: str) -> Any:
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(content)
        except (ValueError, SyntaxError):
            continue
    return content


class FakeSupportModel(BaseChatModel):
    model: str = GEMINI_MODEL
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-support-model"

    def _get_ls_params(self, stop=None, **kwargs):
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "google_genai"
        params["ls_model_name"] = self.model
        return params

    def bind_tools(self, tools, **kwargs):
        names = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": names})

    # ------------------------------------------------------------------ #
    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if random.random() < chaos.state.llm_error_rate:
            time.sleep(random.uniform(0.5, 2.0) * chaos.state.latency_multiplier)
            raise RuntimeError("503 UNAVAILABLE: model overloaded (simulated)")

        concise = "Rules:" in str(messages[0].content)
        last = messages[-1]

        if isinstance(last, HumanMessage):
            msg = self._plan(str(last.content), concise)
        else:
            results = [m for m in messages if isinstance(m, ToolMessage)]
            msg = AIMessage(content=self._answer(results, concise))

        input_tokens = sum(_tokens(str(m.content)) for m in messages) + 180 * len(self.tool_names)
        output_tokens = _tokens(str(msg.content)) + 25 * len(msg.tool_calls)
        msg.usage_metadata = {"input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens}
        msg.response_metadata = {"model_name": self.model, "finish_reason": "STOP"}

        # time ~ first token + generation proportional to output length
        time.sleep((0.25 + output_tokens * 0.004) * random.lognormvariate(0, 0.35) * chaos.state.latency_multiplier)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _plan(self, text: str, concise: bool) -> AIMessage:
        t = text.lower()
        calls = []
        if order := re.search(r"\b(\d{4})\b", t):
            calls.append(("get_order", {"order_id": order.group(1)}))
        if zip_code := re.search(r"\b(\d{5}-?\d{3})\b", t):
            calls.append(("calculate_shipping", {"zip_code": zip_code.group(1)}))
        for keyword, topic in _POLICY_TOPICS:
            if keyword in t:
                calls.append(("get_policy", {"topic": topic}))
                break
        if any(w in t for w in _ESCALATION_WORDS):
            priority = "high" if concise or "ridiculous" in t else "normal"
            calls.append(("create_ticket", {"summary": text[:120], "priority": priority}))

        calls = [(n, a) for n, a in calls if n in self.tool_names]
        if not calls:
            answer = "I can help with orders, shipping, exchanges and returns. What is your order number?"
            if not concise:
                answer = "Hello! Thank you for contacting Vertice. " + answer + " I'm happy to help with anything you need."
            return AIMessage(content=answer)
        return AIMessage(
            content="",
            tool_calls=[{"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"} for n, a in calls],
        )

    def _answer(self, results: list[ToolMessage], concise: bool) -> str:
        parts = []
        for r in results:
            data = _parse(str(r.content))
            if "not found" in str(r.content):
                parts.append("I couldn't find that order number; could you check your confirmation email?")
            elif r.status == "error" or "Error" in str(r.content):
                parts.append("Sorry, I couldn't look that up right now. Please try again in a few minutes.")
            elif r.name == "get_order":
                parts.append(f"Order {data['order_id']} is {data['status']} (total R$ {data['total']:.2f}).")
            elif r.name == "calculate_shipping":
                parts.append(f"Shipping to {data['zip_code']} costs R$ {data['cost']:.2f} and takes {data['delivery_days']} business days.")
            elif r.name == "get_policy":
                parts.append(str(data))
            elif r.name == "create_ticket":
                parts.append(f"I opened ticket {data['ticket']}; a human agent will reply within {data['sla_hours']}h.")
        text = " ".join(parts)
        if not concise:
            text = (
                "Hello! Thank you so much for contacting Vertice. I checked the information in our systems. "
                + text
                + " If you have any other questions, about this or any other order, I'm always happy to help!"
            )
        return text
