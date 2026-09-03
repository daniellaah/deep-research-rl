"""Dependency-light policies for deterministic validation."""

from __future__ import annotations

from dataclasses import dataclass, field

from deep_research_rl.core.models import AgentState


@dataclass(slots=True)
class ScriptedPolicy:
    """Emit a fixed sequence of raw actions, one per call."""

    actions: tuple[str, ...]
    _cursor: int = field(default=0, init=False)

    def choose_action(self, state: AgentState) -> str:
        del state
        if self._cursor >= len(self.actions):
            raise RuntimeError("scripted policy exhausted before the episode terminated")
        action = self.actions[self._cursor]
        self._cursor += 1
        return action
