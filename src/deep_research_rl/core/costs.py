"""Cost functions for environment transitions."""

from __future__ import annotations

from deep_research_rl.core.models import Action, Observation


class ZeroCost:
    """Assign no search or token cost."""

    def cost(self, action: Action, observation: Observation) -> float:
        del action, observation
        return 0.0
