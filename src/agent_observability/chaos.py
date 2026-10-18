"""Controles de "caos" para simular incidentes de produção durante a demo."""

from dataclasses import dataclass


@dataclass
class ChaosState:
    latency_multiplier: float = 1.0
    tool_error_rate: float = 0.06
    llm_error_rate: float = 0.02


state = ChaosState()


def incidente() -> None:
    """Degradação típica: provedor lento, transportadora instável."""
    state.latency_multiplier = 4.0
    state.tool_error_rate = 0.45
    state.llm_error_rate = 0.12


def normal() -> None:
    state.latency_multiplier = 1.0
    state.tool_error_rate = 0.06
    state.llm_error_rate = 0.02
