"""Dependency-light contracts and deterministic local implementations."""

from deep_research_rl.core.actions import ActionParseError, parse_action
from deep_research_rl.core.models import (
    Action,
    AgentState,
    AnswerAction,
    Document,
    EpisodeMetrics,
    Example,
    Observation,
    SearchAction,
    Step,
    Trajectory,
)
from deep_research_rl.core.protocols import (
    ContextPolicy,
    CostFunction,
    CreditAssigner,
    Policy,
    Retriever,
    RewardFunction,
)

__all__ = [
    "Action",
    "ActionParseError",
    "AgentState",
    "AnswerAction",
    "ContextPolicy",
    "CostFunction",
    "CreditAssigner",
    "Document",
    "EpisodeMetrics",
    "Example",
    "Observation",
    "Policy",
    "Retriever",
    "RewardFunction",
    "SearchAction",
    "Step",
    "Trajectory",
    "parse_action",
]
