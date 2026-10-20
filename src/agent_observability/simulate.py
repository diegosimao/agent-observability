""""Production" traffic generator that populates the LangSmith dashboards.

Simulates real customers: varied questions, two prompts in A/B, different plans
and channels, user 👍/👎 feedback and, optionally, an incident in the middle
(4x latency, failing carrier) to show how it shows up in the charts.

    uv run obs-simulate                       # 80 conversations, fast
    uv run obs-simulate --n 300 --duration-min 40 --incident
"""

import argparse
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
from langchain_core.tracers.langchain import wait_for_all_tracers
from langsmith import Client
from rich.console import Console
from rich.table import Table

from . import chaos
from .agent import AgentResult, handle_message
from .config import PROJECT, llm_mode, project_id, tracing_enabled

# (category, weight, questions). {o} and {zip} are drawn for every conversation, so
# the traffic (and the fine-tuning dataset built from it) doesn't repeat the same phrases.
SCENARIOS = [
    ("order", 35, [
        "Where is my order {o}?",
        "What's the status of order #{o}?",
        "My order {o} still hasn't arrived, what happened?",
        "Can you check order {o} for me?",
        "I want to track order {o}",
        "order {o} status",
        "I bought something last week, number {o}. Has it shipped?",
        "What happened to my purchase {o}?",
    ]),
    ("shipping", 20, [
        "How much is shipping to {zip}?",
        "What's the delivery time for ZIP code {zip}?",
        "Shipping to {zip}, please",
        "Do you deliver to {zip}? How long does it take?",
        "I live at ZIP {zip}, how much is delivery?",
        "delivery time and cost to {zip}",
    ]),
    ("policy", 20, [
        "How do returns work?",
        "Can I exchange for a different size?",
        "What is the warranty on your products?",
        "How long does a refund take?",
        "I want to return a product, how do I do it?",
        "What's the deadline to request an exchange?",
        "How long is the product warranty?",
        "When will the refund show up on my card?",
    ]),
    ("multi", 10, [
        "Is order {o} late? And what would shipping to {zip} cost?",
        "Order {o} arrived damaged, how do returns work?",
        "I want to exchange order {o}, what's the exchange policy?",
        "Check order {o} and tell me the shipping cost to {zip}",
    ]),
    ("escalation", 10, [
        "This is ridiculous, I want to talk to a human now!",
        "Order {o} has been stuck for days, I want an agent",
        "Terrible service, I want to file a complaint",
        "Please transfer me to a human agent",
        "I want to file a formal complaint about order {o}",
    ]),
    ("out_of_scope", 5, [
        "Hi, how are you?",
        "Do you sell concert tickets?",
        "What are the physical store's opening hours?",
    ]),
]
# 1001-1004 exist; the others hit the ERP's "order not found"
ORDERS = [("1001", 25), ("1002", 25), ("1003", 20), ("1004", 15), ("5521", 8), ("7789", 7)]
ZIP_CODES = ["01310-100", "30140-071", "80010-000", "69005-070", "40020-000", "90010-150", "60060-120", "88010-400"]
PLANS = [("free", 60), ("plus", 30), ("enterprise", 10)]
CHANNELS = [("web", 55), ("whatsapp", 35), ("app", 10)]
BAD_COMMENTS = ["Didn't solve it", "Took too long", "Generic answer", "Got an error"]
GOOD_COMMENTS = ["Fast and clear", "Solved it!", "Great"]

API_URL: str | None = None  # set by --api-url


def _pick(options):
    return random.choices([o[0] for o in options], weights=[o[1] for o in options])[0]


def _via_api(question: str) -> tuple[str, AgentResult]:
    """Same conversation, but as a real HTTP client: the API picks the A/B
    variant from the session, and the run_id comes back in the body (or in the 503)."""
    body = {"message": question, "session": f"customer-{random.randint(1, 60)}", "plan": _pick(PLANS), "channel": _pick(CHANNELS)}
    start = time.perf_counter()
    resp = httpx.post(f"{API_URL}/v1/chat", json=body, timeout=120)
    latency = time.perf_counter() - start
    data = resp.json()
    if resp.status_code != 200:
        detail = data["detail"]
        return "?", AgentResult(uuid.UUID(detail["run_id"]), None, detail["error"], latency)
    r = AgentResult(uuid.UUID(data["run_id"]), data["answer"], None, latency, data["tokens"], data["tools"], data["tool_errors"])
    return data["prompt_version"], r


def _one_conversation(_: int) -> tuple[str, AgentResult]:
    idx = random.choices(range(len(SCENARIOS)), weights=[s[1] for s in SCENARIOS])[0]
    category, _, questions = SCENARIOS[idx]
    question = random.choice(questions).format(o=_pick(ORDERS), zip=random.choice(ZIP_CODES))
    if API_URL:
        return _via_api(question)
    prompt_version = random.choice(["v1", "v2"])
    r = handle_message(
        question,
        prompt_version=prompt_version,
        plan=_pick(PLANS),
        channel=_pick(CHANNELS),
        category=category,
        extra_tags=["simulation"],
    )
    return prompt_version, r


def _user_feedback(prompt_version: str, r: AgentResult) -> tuple[float | None, str]:
    """Probability of 👍 that reacts to the same things that annoy a real customer."""
    if random.random() > 0.45:  # most users don't rate
        return None, ""
    if r.error:
        p = 0.05
    elif r.tool_errors:
        p = 0.25
    elif not r.tools:
        p = 0.45
    else:
        p = 0.82 if prompt_version == "v2" else 0.66
    if r.latency_s > 6:
        p -= 0.3
    liked = random.random() < max(p, 0.02)
    return (1.0 if liked else 0.0), random.choice(GOOD_COMMENTS if liked else BAD_COMMENTS)


def _send_feedback(client: Client, results: list[tuple[str, AgentResult]]) -> int:
    sent = 0
    for prompt_version, r in results:
        score, comment = _user_feedback(prompt_version, r)
        if score is not None:
            if API_URL:  # the path a front end would use
                httpx.post(f"{API_URL}/v1/feedback", json={"run_id": str(r.run_id), "score": int(score), "comment": comment}, timeout=30).raise_for_status()
            else:
                client.create_feedback(r.run_id, key="user_score", score=score, comment=comment, session_id=project_id())
            sent += 1
        if not r.error:
            # automatic signal: resolved without escalating to a human and without tool failures
            resolved = "create_ticket" not in r.tools and r.tool_errors == 0 and bool(r.tools)
            client.create_feedback(r.run_id, key="resolved_without_human", score=float(resolved), session_id=project_id())
    return sent


def _set_chaos(mode: str) -> None:
    getattr(chaos, mode)()  # local state, also used as "are we in an incident?"
    if API_URL:  # the chaos must happen in the API process, not here
        httpx.post(f"{API_URL}/_chaos/{mode}", timeout=10).raise_for_status()


def _summary(results: list[tuple[str, AgentResult]], console: Console) -> None:
    table = Table(title="Local summary (the same shows up in LangSmith)")
    for col in ("prompt", "conversations", "error %", "p50 s", "p95 s", "avg tokens", "tool failures"):
        table.add_column(col, justify="right")
    for version in ("v1", "v2", "?"):  # "?" = 503 from the API, variant unknown
        rs = [r for v, r in results if v == version]
        if not rs:
            continue
        lat = sorted(r.latency_s for r in rs)
        ok = [r for r in rs if not r.error]
        table.add_row(
            version,
            str(len(rs)),
            f"{100 * (len(rs) - len(ok)) / len(rs):.1f}",
            f"{statistics.median(lat):.2f}",
            f"{lat[min(len(lat) - 1, int(0.95 * len(lat)))]:.2f}",
            f"{statistics.mean(r.tokens for r in ok):.0f}" if ok else "-",
            str(sum(r.tool_errors for r in rs)),
        )
    console.print(table)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate simulated traffic for the agent")
    p.add_argument("--n", type=int, default=80, help="number of conversations")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--duration-min", type=float, default=0, help="spread conversations over N minutes (nicer time-series charts)")
    p.add_argument("--incident", action="store_true", help="degrade the system between 40%% and 65%% of the run")
    p.add_argument("--api-url", help="send traffic over HTTP to the API (e.g. http://127.0.0.1:8000)")
    args = p.parse_args()

    global API_URL
    API_URL = args.api_url.rstrip("/") if args.api_url else None

    console = Console()
    console.print(f"[bold]llm[/bold]={llm_mode()}  [bold]tracing[/bold]={'on → project ' + PROJECT if tracing_enabled() else 'off (local summary only)'}")

    # Phases run in sequence (each waits for the previous one to finish) so the
    # chaos state applies to every conversation in that phase.
    a, b = int(args.n * 0.40), int(args.n * 0.65)
    phases = [("normal", a), ("incident", b - a), ("normal", args.n - b)] if args.incident else [("normal", args.n)]
    interval = args.duration_min * 60 / args.n if args.duration_min else 0
    results: list[tuple[str, AgentResult]] = []
    with ThreadPoolExecutor(args.concurrency) as pool, console.status("simulating...") as status:
        for name, count in phases:
            if name == "incident":
                _set_chaos("incident")
                console.print("[red]>>> incident started: slow provider, flaky carrier[/red]")
            elif chaos.state.latency_multiplier != 1.0:
                _set_chaos("normal")
                console.print("[green]>>> incident resolved[/green]")
            futures = []
            for i in range(count):
                futures.append(pool.submit(_one_conversation, i))
                if interval:
                    time.sleep(random.expovariate(1 / interval))
            for f in futures:
                results.append(f.result())
                status.update(f"simulating... {len(results)}/{args.n}")

    _summary(results, console)
    if tracing_enabled():
        wait_for_all_tracers()
        sent = _send_feedback(Client(), results)
        console.print(f"{sent} user ratings sent. Open https://smith.langchain.com → Projects → {PROJECT}")


if __name__ == "__main__":
    main()
