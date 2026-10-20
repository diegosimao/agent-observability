"""`obs-ask "question"`: a one-off call to the agent, for manual testing."""

import argparse

from langchain_core.tracers.langchain import wait_for_all_tracers
from rich import print

from .agent import handle_message
from .config import llm_mode, tracing_enabled


def ask() -> None:
    p = argparse.ArgumentParser(description="Ask the support agent a question")
    p.add_argument("question")
    p.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    args = p.parse_args()

    r = handle_message(args.question, prompt_version=args.prompt, channel="cli")
    wait_for_all_tracers()
    print(f"[dim]llm={llm_mode()} tracing={'on' if tracing_enabled() else 'off'} run_id={r.run_id}[/dim]")
    print(f"[bold]tools:[/bold] {r.tools}  [bold]tokens:[/bold] {r.tokens}  [bold]latency:[/bold] {r.latency_s:.2f}s")
    print(f"[red]{r.error}[/red]" if r.error else f"[green]{r.answer}[/green]")
