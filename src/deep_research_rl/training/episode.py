"""Pure-Python episode controller shared by Agent-R1 and local contract tests."""

from __future__ import annotations

import json
from dataclasses import dataclass

from deep_research_rl.agent.prompting import build_policy_messages
from deep_research_rl.core.actions import ActionParseError, parse_action
from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.models import Action, AgentState, Example, Observation, SearchAction
from deep_research_rl.core.protocols import Retriever
from deep_research_rl.core.rewards import TerminalExactMatchReward
from deep_research_rl.core.serialization import (
    action_to_record,
    observation_to_record,
    state_to_record,
)

AGENT_FLOW_NAME = "deep_research_hotpotqa"
BASELINE_MAX_SEARCHES = 5
BASELINE_MAX_STEPS = 8


@dataclass(frozen=True, slots=True)
class AgentR1EpisodeStep:
    """One policy response and its exact environment transition."""

    index: int
    raw_response: str
    action: Action | None
    parse_error: str | None
    state_before: AgentState
    observation: Observation
    state_after: AgentState
    reward: float
    episode_complete: bool

    def trace_record(self) -> dict[str, object]:
        """Return JSON-compatible state/action/reward evidence for rollout dumps."""

        return {
            "action": None if self.action is None else action_to_record(self.action),
            "episode_complete": self.episode_complete,
            "index": self.index,
            "observation": observation_to_record(self.observation),
            "parse_error": self.parse_error,
            "raw_response": self.raw_response,
            "reward": self.reward,
            "state_after": state_to_record(self.state_after),
            "state_before": state_to_record(self.state_before),
        }

    def reward_extra_info(self, *, attempted_searches: int) -> dict[str, object]:
        """Expose scalar metrics plus a stable transition record to Agent-R1 logging."""

        return {
            "acc": self.reward if self.state_after.terminated else 0.0,
            "attempted_searches": attempted_searches,
            "episode_complete": int(self.episode_complete),
            "executed_searches": self.state_after.executed_searches,
            "parse_error": self.parse_error or "",
            "terminated": int(self.state_after.terminated),
            "transition_json": json.dumps(
                self.trace_record(),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        }


class AgentR1Episode:
    """Advance the frozen baseline MDP one generated response at a time.

    The constructor only resets state. Retrieval can therefore occur only after a
    policy-produced ``SEARCH(query)`` action. The strict parser accepts one action,
    so a response containing parallel or repeated calls is rejected without executing
    any of them.
    """

    def __init__(self, example: Example, retriever: Retriever) -> None:
        self.example = example
        self._environment = ResearchEnvironment(
            retriever,
            AppendOnlyContextPolicy(),
            max_searches=BASELINE_MAX_SEARCHES,
        )
        self._reward = TerminalExactMatchReward()
        self._state = self._environment.reset(example)
        self._steps: list[AgentR1EpisodeStep] = []
        self._attempted_searches = 0
        self._complete = False

    @property
    def state(self) -> AgentState:
        """Return the current immutable policy state."""

        return self._state

    @property
    def steps(self) -> tuple[AgentR1EpisodeStep, ...]:
        """Return completed transitions in generation order."""

        return tuple(self._steps)

    @property
    def attempted_searches(self) -> int:
        """Count parsed search requests, including budget-rejected requests."""

        return self._attempted_searches

    @property
    def complete(self) -> bool:
        """Return whether the episode answered or reached the step bound."""

        return self._complete

    @property
    def termination_reason(self) -> str | None:
        """Return the explicit completion reason, if any."""

        if not self._complete:
            return None
        return "answered" if self._state.terminated else "max_steps"

    def prompt_messages(self) -> tuple[dict[str, str], ...]:
        """Build the same append-only prompt used by model-backed evaluation."""

        if self._complete:
            raise RuntimeError("cannot build another prompt after episode completion")
        return build_policy_messages(self._state, max_searches=BASELINE_MAX_SEARCHES)

    def apply_response(self, raw_response: str) -> AgentR1EpisodeStep:
        """Parse one response, apply at most one action, and assign immediate reward."""

        if self._complete:
            raise RuntimeError("cannot apply a response after episode completion")
        if not isinstance(raw_response, str):
            raise TypeError("raw_response must be a string")

        state_before = self._state
        action: Action | None
        parse_error: str | None
        try:
            action = parse_action(raw_response)
            parse_error = None
        except ActionParseError as error:
            action = None
            parse_error = str(error)

        if action is None:
            state_after, observation = self._environment.record_malformed_action(
                state_before,
                parse_error or "unknown parse failure",
            )
            reward = 0.0
        else:
            if isinstance(action, SearchAction):
                self._attempted_searches += 1
            state_after, observation = self._environment.transition(state_before, action)
            reward = self._reward.score(self.example, action, state_after)

        step_index = len(self._steps)
        episode_complete = state_after.terminated or step_index + 1 >= BASELINE_MAX_STEPS
        step = AgentR1EpisodeStep(
            index=step_index,
            raw_response=raw_response,
            action=action,
            parse_error=parse_error,
            state_before=state_before,
            observation=observation,
            state_after=state_after,
            reward=reward,
            episode_complete=episode_complete,
        )
        self._steps.append(step)
        self._state = state_after
        self._complete = episode_complete
        return step
