"""Replaceable interfaces for the research agent's main seams."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from deep_research_rl.core.models import (
    Action,
    AgentState,
    Document,
    Example,
    Observation,
)


class Retriever(Protocol):
    """Return ranked documents for a policy-selected query."""

    def search(self, query: str) -> tuple[Document, ...]: ...


class Policy(Protocol):
    """Produce one raw action string from the current state."""

    def choose_action(self, state: AgentState) -> str: ...


class ContextPolicy(Protocol):
    """Construct the next policy-visible context."""

    def update(
        self,
        context: tuple[Observation, ...],
        observation: Observation,
    ) -> tuple[Observation, ...]: ...


class RewardFunction(Protocol):
    """Score an individual transition."""

    def score(
        self,
        example: Example,
        action: Action,
        next_state: AgentState,
    ) -> float: ...


class CreditAssigner(Protocol):
    """Map step rewards to step-level learning credit."""

    def assign(self, rewards: Sequence[float]) -> tuple[float, ...]: ...


class CostFunction(Protocol):
    """Return a transition cost independently from task reward."""

    def cost(self, action: Action, observation: Observation) -> float: ...
