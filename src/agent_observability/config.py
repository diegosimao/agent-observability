"""Central configuration: LLM mode, prompt versions and deploy metadata."""

import os
from functools import cache

from dotenv import load_dotenv

load_dotenv()

APP_VERSION = os.getenv("APP_VERSION", "1.3.0")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
# Hugging Face model id (base model) or a folder containing a trained LoRA adapter
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
PROJECT = os.getenv("LANGSMITH_PROJECT", "support-agent-prod")


def llm_mode() -> str:
    mode = os.getenv("AGENT_MODE", "auto").lower()
    if mode == "auto":
        return "gemini" if os.getenv("GOOGLE_API_KEY") else "mock"
    return mode


@cache
def project_id():
    """LangSmith project id; the SDK now requires it in create_feedback."""
    from langsmith import Client

    return Client().read_project(project_name=PROJECT).id


def model_name() -> str:
    match llm_mode():
        case "local":
            return LOCAL_MODEL.replace("\\", "/").rstrip("/").split("/")[-1]
        case "mock":
            return "mock"
        case _:
            return GEMINI_MODEL


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(
        os.getenv("LANGSMITH_API_KEY")
    )


# Two system prompt versions running side by side (A/B). Every trace records
# which version answered, and the dashboard compares latency, cost and satisfaction.
PROMPTS = {
    "v1": (
        "You are the customer support assistant for the Vertice online store. "
        "Use the available tools to answer questions about orders, shipping and policies. "
        "Be polite and thorough."
    ),
    "v2": (
        "You are the customer support assistant for the Vertice online store.\n"
        "Rules:\n"
        "1. Always check a tool before stating anything about an order, shipping or a policy.\n"
        "2. Answer in at most 3 sentences.\n"
        "3. If the customer asks for a human or is unhappy, open a ticket.\n"
        "4. Never make up an order number, delivery time or price."
    ),
}
