"""Data flywheel: production traces with a good signal → fine-tuning dataset.

    uv run obs-export-ft --days 7

Curation criteria (each one is an auditable filter, not a judgment call):
1. root `support-chat` trace with no error and no tool failure
2. no user 👎 (`user_score` = 0 is discarded)
3. only the winning prompt variant (v2 by default), to distill the best behavior
4. golden-dataset questions are excluded, so evaluation doesn't measure memorization
5. deduplication by (question, tool sequence)

Every line keeps the source `run_id`: you can go from an example back to its trace.
"""

import argparse
import json
import random
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langchain_core.messages import messages_from_dict
from langchain_core.utils.function_calling import convert_to_openai_tool
from langsmith import Client

from .agent import RUN_NAME
from .config import PROJECT, PROMPTS
from .evaluate import EXAMPLES
from .local_llm import to_chat_template
from .tools import TOOLS


def _norm(text: str) -> str:
    return "".join(c for c in text.lower() if c.isalnum() or c == " ").strip()


GOLDEN_QUESTIONS = {_norm(q) for q, _, _ in EXAMPLES}


def _messages(run) -> list:
    raw = run.outputs["messages"]
    return messages_from_dict([{"type": m["type"], "data": m} for m in raw])


def example_from_trace(run, prompt_version: str) -> dict | None:
    """Turns a trace into a chat example, or None if it's not fit for training."""
    chat = to_chat_template(_messages(run))
    if any(m.get("role") == "tool" and m["content"].startswith("Error") for m in chat):
        return None
    if not chat or chat[0]["role"] != "user" or chat[-1]["role"] != "assistant" or not chat[-1]["content"]:
        return None
    return {
        "messages": [{"role": "system", "content": PROMPTS[prompt_version]}, *chat],
        "tools": [convert_to_openai_tool(t) for t in TOOLS],
        "source": {"run_id": str(run.id), "prompt_version": prompt_version, "project": PROJECT},
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Export curated LangSmith traces for SFT")
    p.add_argument("--days", type=int, default=7)
    p.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    p.add_argument("--out", default="data")
    p.add_argument("--val", type=float, default=0.15, help="fraction held out for validation")
    args = p.parse_args()

    client = Client()
    since = datetime.now(timezone.utc) - timedelta(days=args.days)
    runs = list(
        client.list_runs(
            project_name=PROJECT,
            is_root=True,
            error=False,
            start_time=since,
            filter=f'and(eq(name, "{RUN_NAME}"), eq(metadata_key, "prompt_version"), eq(metadata_value, "{args.prompt}"))',
        )
    )
    negative = {
        str(f.run_id)
        for f in client.list_feedback(run_ids=[r.id for r in runs], feedback_key=["user_score"])
        if f.score == 0
    }

    reasons: Counter = Counter()
    seen, examples = set(), []
    for run in runs:
        question = run.inputs["messages"][0]["content"].strip()
        if str(run.id) in negative:
            reasons["user 👎"] += 1
            continue
        if _norm(question) in GOLDEN_QUESTIONS:
            reasons["golden question (leakage)"] += 1
            continue
        ex = example_from_trace(run, args.prompt)
        if ex is None:
            reasons["tool failure / empty answer"] += 1
            continue
        key = (_norm(question), tuple(tc["function"]["name"] for m in ex["messages"] for tc in m.get("tool_calls", [])))
        if key in seen:
            reasons["duplicate"] += 1
            continue
        seen.add(key)
        examples.append(ex)

    random.Random(42).shuffle(examples)
    n_val = max(1, int(len(examples) * args.val)) if len(examples) > 4 else 0
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    for name, part in (("sft_train.jsonl", examples[n_val:]), ("sft_val.jsonl", examples[:n_val])):
        with open(out / name, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in part)

    print(f"{len(runs)} traces read from '{PROJECT}' (prompt {args.prompt}, last {args.days} days)")
    for reason, n in reasons.most_common():
        print(f"  discarded — {reason}: {n}")
    print(f"{len(examples) - n_val} train + {n_val} validation → {out.resolve()}")


if __name__ == "__main__":
    main()
