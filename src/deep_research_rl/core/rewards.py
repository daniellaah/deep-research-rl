"""Outcome-only reward functions."""

from __future__ import annotations

from deep_research_rl.core.metrics import normalized_exact_match
from deep_research_rl.core.models import Action, AgentState, AnswerAction, Example


class TerminalExactMatchReward:
    """Return normalized exact match only for the terminating answer action."""

    def score(
        self,
        example: Example,
        action: Action,
        next_state: AgentState,
    ) -> float:
        if not isinstance(action, AnswerAction):
            return 0.0
        if not next_state.terminated or next_state.answer is None:
            raise ValueError("answer action must produce a terminated state")
        return max(
            normalized_exact_match(next_state.answer, reference) for reference in example.answers
        )
