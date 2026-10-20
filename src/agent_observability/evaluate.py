"""Offline evaluation: runs a "golden" dataset against the v1 and v2 prompts and
publishes one experiment per prompt to LangSmith, ready for the compare view.

    uv run obs-evaluate
"""

import argparse
import sys
from collections import defaultdict
from math import comb

from langsmith import Client, evaluate

from .agent import handle_message
from .config import APP_VERSION, llm_mode, model_name, tracing_enabled

DATASET = "support-golden"

# (question, expected tools, snippets the answer must contain)
EXAMPLES = [
    ("What's the status of order 1002?", ["get_order"], ["1002", "in transit"]),
    ("Why was order 1004 cancelled?", ["get_order"], ["1004", "cancelled"]),
    ("How much is shipping to 01310-100?", ["calculate_shipping"], ["R$"]),
    ("How do returns work?", ["get_policy"], ["30 days"]),
    ("How long does a refund take?", ["get_policy"], ["7 business days"]),
    ("Can I exchange for a different size?", ["get_policy"], ["exchange"]),
    ("Is order 1002 late? And shipping to 30140-071?", ["get_order", "calculate_shipping"], ["1002", "R$"]),
    ("I want to talk to a human agent", ["create_ticket"], ["TK-"]),
    ("This is ridiculous, order 1003 is stuck!", ["get_order", "create_ticket"], ["TK-"]),
    ("Hi, how are you?", [], ["order"]),
]


def _ensure_dataset(client: Client) -> None:
    if client.has_dataset(dataset_name=DATASET):
        return
    ds = client.create_dataset(DATASET, description="Support questions with expected tools and required answer snippets")
    client.create_examples(
        dataset_id=ds.id,
        examples=[
            {"inputs": {"question": q}, "outputs": {"expected_tools": t, "must_contain": c}}
            for q, t, c in EXAMPLES
        ],
    )


# --- evaluators --------------------------------------------------------------
def correct_tools(outputs: dict, reference_outputs: dict) -> bool:
    return sorted(set(outputs["tools"])) == sorted(reference_outputs["expected_tools"])


def expected_content(outputs: dict, reference_outputs: dict) -> float:
    answer = (outputs["answer"] or "").lower()
    snippets = reference_outputs["must_contain"]
    return sum(s.lower() in answer for s in snippets) / len(snippets)


def concise(outputs: dict) -> bool:
    return outputs["answer"] is not None and len(outputs["answer"]) <= 280


def no_error(outputs: dict) -> bool:
    return outputs["error"] is None


def _passed(o: dict, r: dict) -> bool:
    return o["error"] is None and correct_tools(o, r) and expected_content(o, r) == 1


def pass_rate(outputs: list[dict], reference_outputs: list[dict]) -> dict:
    ok = sum(_passed(o, r) for o, r in zip(outputs, reference_outputs))
    return {"key": "pass_rate", "score": ok / len(outputs)}


def pass_hat_k(successes_per_task: list[tuple[int, int]], k: int) -> float:
    """pass^k from τ-bench (Yao et al., 2024, arXiv:2406.12045): the probability of
    solving the SAME task in all of k independent attempts. Unbiased estimator with
    n attempts and c successes per task: C(c, k) / C(n, k).
    pass@k asks "does it ever get it right?"; pass^k asks "can I rely on it?"."""
    return sum(comb(c, k) / comb(n, k) for c, n in successes_per_task) / len(successes_per_task)


def reliability(runs, examples) -> dict:
    """Groups repetitions by example and publishes pass^1..pass^n."""
    by_example: dict = defaultdict(list)
    for run, ex in zip(runs, examples):
        by_example[ex.id].append(_passed(run.outputs or {"error": "no output", "tools": [], "answer": None}, ex.outputs))
    tasks = [(sum(v), len(v)) for v in by_example.values()]
    n = min(t[1] for t in tasks)
    return {"results": [{"key": f"pass^{k}", "score": pass_hat_k(tasks, k)} for k in range(1, n + 1)]}


def _target(prompt_version: str):
    def support_agent(inputs: dict) -> dict:
        r = handle_message(inputs["question"], prompt_version=prompt_version, channel="eval", category="eval")
        return {"answer": r.answer, "tools": r.tools, "error": r.error}

    return support_agent


def main() -> None:
    p = argparse.ArgumentParser(description="Publish golden-dataset experiments to LangSmith")
    p.add_argument("--prompts", nargs="+", default=["v1", "v2"], choices=["v1", "v2"])
    p.add_argument("--k", type=int, default=3, help="repetitions per example, for pass^k")
    p.add_argument("--concurrency", type=int, default=4)
    args = p.parse_args()

    if not tracing_enabled():
        sys.exit("Set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env to publish experiments.")
    client = Client()
    _ensure_dataset(client)
    for version in args.prompts:
        evaluate(
            _target(version),
            data=DATASET,
            evaluators=[correct_tools, expected_content, concise, no_error],
            summary_evaluators=[pass_rate, reliability],
            experiment_prefix=f"support-{model_name()}-{version}",
            metadata={"prompt_version": version, "app_version": APP_VERSION, "llm_mode": llm_mode(), "model": model_name()},
            num_repetitions=args.k,
            max_concurrency=args.concurrency,
            client=client,
        )
    print(f"Done. LangSmith → Datasets & Experiments → {DATASET} → select the experiments → Compare")


if __name__ == "__main__":
    main()
