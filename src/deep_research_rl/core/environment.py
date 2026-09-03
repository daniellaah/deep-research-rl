"""Deterministic state transitions with hard search-budget enforcement."""

from __future__ import annotations

from dataclasses import replace

from deep_research_rl.core.models import (
    Action,
    AgentState,
    AnswerAction,
    Example,
    Observation,
    SearchAction,
)
from deep_research_rl.core.protocols import ContextPolicy, Retriever


class EpisodeTerminatedError(RuntimeError):
    """Raised when an action is attempted after an answer."""


class ResearchEnvironment:
    """Apply search and answer actions to immutable agent states."""

    def __init__(
        self,
        retriever: Retriever,
        context_policy: ContextPolicy,
        *,
        max_searches: int = 5,
    ) -> None:
        if max_searches < 0:
            raise ValueError("max_searches must not be negative")
        self._retriever = retriever
        self._context_policy = context_policy
        self._max_searches = max_searches

    @property
    def max_searches(self) -> int:
        """Return the hard limit on executed policy-selected searches."""

        return self._max_searches

    def reset(self, example: Example) -> AgentState:
        """Create an empty initial state for an example."""

        return AgentState(
            example_id=example.example_id,
            question=example.question,
            context=(),
            executed_searches=0,
            terminated=False,
        )

    def transition(self, state: AgentState, action: Action) -> tuple[AgentState, Observation]:
        """Execute one accepted action or return explicit budget rejection feedback."""

        if state.terminated:
            raise EpisodeTerminatedError("cannot act after an episode has terminated")

        if isinstance(action, AnswerAction):
            observation = Observation(
                status="answer_recorded",
                message="answer recorded; episode terminated",
            )
            next_context = self._context_policy.update(state.context, observation)
            return (
                replace(
                    state,
                    context=next_context,
                    terminated=True,
                    answer=action.answer,
                ),
                observation,
            )

        if not isinstance(action, SearchAction):  # pragma: no cover - closed typed union guard
            raise TypeError(f"unsupported action type: {type(action).__name__}")

        if state.executed_searches >= self._max_searches:
            observation = Observation(
                status="search_rejected",
                query=action.query,
                message=f"search budget exhausted at {self._max_searches} executed calls",
            )
            next_context = self._context_policy.update(state.context, observation)
            return replace(state, context=next_context), observation

        documents = self._retriever.search(action.query)
        observation = Observation(
            status="search_executed",
            query=action.query,
            documents=documents,
            message=f"retrieved {len(documents)} document(s)",
        )
        next_context = self._context_policy.update(state.context, observation)
        return (
            replace(
                state,
                context=next_context,
                executed_searches=state.executed_searches + 1,
            ),
            observation,
        )

    def record_malformed_action(
        self,
        state: AgentState,
        parse_error: str,
    ) -> tuple[AgentState, Observation]:
        """Append deterministic feedback for a response that is not an executable action."""

        if state.terminated:
            raise EpisodeTerminatedError("cannot record an action attempt after termination")
        if not parse_error:
            raise ValueError("parse_error must not be empty")
        observation = Observation(
            status="malformed_action",
            message=f"malformed action rejected: {parse_error}",
        )
        next_context = self._context_policy.update(state.context, observation)
        return replace(state, context=next_context), observation
