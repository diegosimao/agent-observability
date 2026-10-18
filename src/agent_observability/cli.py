"""`obs-ask "pergunta"` — uma chamada avulsa ao agente, para testar na mão."""

import argparse

from langchain_core.tracers.langchain import wait_for_all_tracers
from rich import print

from .agent import atender
from .config import llm_mode, tracing_enabled


def ask() -> None:
    p = argparse.ArgumentParser(description="Faz uma pergunta ao agente de suporte")
    p.add_argument("pergunta")
    p.add_argument("--prompt", default="v2", choices=["v1", "v2"])
    args = p.parse_args()

    r = atender(args.pergunta, prompt_version=args.prompt, canal="cli")
    wait_for_all_tracers()
    print(f"[dim]llm={llm_mode()} tracing={'on' if tracing_enabled() else 'off'} run_id={r.run_id}[/dim]")
    print(f"[bold]tools:[/bold] {r.tools}  [bold]tokens:[/bold] {r.tokens}  [bold]latência:[/bold] {r.latencia_s:.2f}s")
    print(f"[red]{r.erro}[/red]" if r.erro else f"[green]{r.resposta}[/green]")
