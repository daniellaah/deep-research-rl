"""Step-level credit assignment implementations."""

from __future__ import annotations

from collections.abc import Sequence


class TerminalOnlyCreditAssigner:
    """Place the terminal step reward only on the final transition."""

    def assign(self, rewards: Sequence[float]) -> tuple[float, ...]:
        if not rewards:
            return ()
        return (*(0.0 for _ in rewards[:-1]), float(rewards[-1]))
