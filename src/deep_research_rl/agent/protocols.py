"""Protocols for model-backed policy generation."""

from __future__ import annotations

from typing import Protocol

from deep_research_rl.agent.contracts import PolicyOutput
from deep_research_rl.core.models import AgentState


class GenerativePolicy(Protocol):
    """Generate one raw response and its exact token-level sampling record."""

    model_name: str
    model_revision: str
    prompt_format: str

    def generate(self, state: AgentState, *, max_searches: int) -> PolicyOutput: ...
