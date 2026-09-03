"""Context-construction policies."""

from __future__ import annotations

from deep_research_rl.core.models import Observation


class AppendOnlyContextPolicy:
    """Append every environment observation without rewriting prior context."""

    def update(
        self,
        context: tuple[Observation, ...],
        observation: Observation,
    ) -> tuple[Observation, ...]:
        return (*context, observation)
