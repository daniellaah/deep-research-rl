"""Strict parsing for the baseline action language."""

from __future__ import annotations

import re

from deep_research_rl.core.models import Action, AnswerAction, SearchAction

_ACTION_PATTERN = re.compile(r"(SEARCH|ANSWER)\((.*)\)")


class ActionParseError(ValueError):
    """Raised when policy output is not exactly one supported action."""


def parse_action(raw_action: str) -> Action:
    """Parse exactly ``SEARCH(query)`` or ``ANSWER(answer)``.

    Leading/trailing whitespace, empty payloads, newlines, unsupported action names,
    and extra text are rejected instead of being repaired implicitly.
    """

    match = _ACTION_PATTERN.fullmatch(raw_action)
    if match is None:
        raise ActionParseError("expected exactly SEARCH(query) or ANSWER(answer)")

    action_name, payload = match.groups()
    if not payload or payload != payload.strip():
        raise ActionParseError("action payload must be non-empty and trimmed")

    if action_name == "SEARCH":
        return SearchAction(query=payload)
    return AnswerAction(answer=payload)
