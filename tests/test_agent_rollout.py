from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import pytest

from deep_research_rl.agent.contracts import AgentRollout, PolicyOutput
from deep_research_rl.agent.prompting import PROMPT_FORMAT_VERSION
from deep_research_rl.agent.rollout import DEBUG_RESULT_SCOPE, run_model_rollout
from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.costs import ZeroCost
from deep_research_rl.core.credit import TerminalOnlyCreditAssigner
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.fixtures import synthetic_two_hop_fixture
from deep_research_rl.core.models import AgentState, SearchResult
from deep_research_rl.core.rewards import TerminalExactMatchReward

FAKE_REVISION = "0" * 40


@dataclass(slots=True)
class ScriptedGenerativePolicy:
    responses: tuple[str, ...]
    model_name: str = "test/model"
    model_revision: str = FAKE_REVISION
    prompt_format: str = PROMPT_FORMAT_VERSION
    seen_states: list[AgentState] = field(default_factory=list)
    _cursor: int = field(default=0, init=False)

    def generate(self, state: AgentState, *, max_searches: int) -> PolicyOutput:
        assert max_searches == 5
        self.seen_states.append(state)
        raw_response = self.responses[self._cursor]
        self._cursor += 1
        prompt_ids = (100, self._cursor)
        response_ids = tuple(200 + index for index, _ in enumerate(raw_response, 1))
        return PolicyOutput.from_generation(
            raw_response=raw_response,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_logprobs=(-0.25,) * len(response_ids),
            finish_reason="eos",
        )


class CountingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[SearchResult, ...]:
        self.queries.append(query)
        return ()

    def search_batch(self, queries: Sequence[str]) -> tuple[tuple[SearchResult, ...], ...]:
        return tuple(self.search(query) for query in queries)


def _run(
    policy: ScriptedGenerativePolicy, retriever: CountingRetriever, *, max_steps: int = 8
) -> AgentRollout:
    example, _ = synthetic_two_hop_fixture()
    return run_model_rollout(
        example,
        policy,
        ResearchEnvironment(retriever, AppendOnlyContextPolicy(), max_searches=5),
        TerminalExactMatchReward(),
        TerminalOnlyCreditAssigner(),
        ZeroCost(),
        max_steps=max_steps,
    )


def test_first_decision_is_policy_selected_without_forced_search() -> None:
    retriever = CountingRetriever()
    policy = ScriptedGenerativePolicy(("ANSWER(Lumen City)",))

    rollout = _run(policy, retriever)

    assert retriever.queries == []
    assert policy.seen_states[0].context == ()
    assert policy.seen_states[0].executed_searches == 0
    assert rollout.termination_reason == "answered"
    assert rollout.result_scope == DEBUG_RESULT_SCOPE
    assert rollout.metrics.exact_match == 1.0


def test_malformed_action_is_logged_appended_and_does_not_execute_retrieval() -> None:
    retriever = CountingRetriever()
    policy = ScriptedGenerativePolicy(
        (
            "I should search first.",
            "ANSWER(Lumen City)",
        )
    )

    rollout = _run(policy, retriever)

    malformed = rollout.steps[0]
    assert malformed.action is None
    assert malformed.parse_error == "expected exactly SEARCH(query) or ANSWER(answer)"
    assert malformed.observation.status == "malformed_action"
    assert malformed.reward == 0.0
    assert malformed.cost == 0.0
    assert malformed.state_after.executed_searches == 0
    assert malformed.state_after.context == (malformed.observation,)
    assert policy.seen_states[1].context == (malformed.observation,)
    assert retriever.queries == []
    assert [step.credit for step in rollout.steps] == [0.0, 1.0]


def test_five_search_limit_counts_executions_not_rejected_attempts() -> None:
    retriever = CountingRetriever()
    policy = ScriptedGenerativePolicy(
        (
            "SEARCH(query zero)",
            "SEARCH(query one)",
            "SEARCH(query two)",
            "SEARCH(query three)",
            "SEARCH(query four)",
            "SEARCH(query rejected)",
            "ANSWER(Lumen City)",
        )
    )

    rollout = _run(policy, retriever)

    assert retriever.queries == [
        "query zero",
        "query one",
        "query two",
        "query three",
        "query four",
    ]
    assert rollout.steps[5].observation.status == "search_rejected"
    assert rollout.steps[5].state_after.executed_searches == 5
    assert rollout.final_state.executed_searches == 5


def test_maximum_step_bound_returns_a_reviewable_truncated_rollout() -> None:
    retriever = CountingRetriever()
    policy = ScriptedGenerativePolicy(("malformed", "still malformed"))

    rollout = _run(policy, retriever, max_steps=2)

    assert rollout.termination_reason == "max_steps"
    assert rollout.final_state.terminated is False
    assert len(rollout.steps) == 2
    assert [step.observation.status for step in rollout.steps] == [
        "malformed_action",
        "malformed_action",
    ]
    assert rollout.metrics.exact_match == 0.0
    assert [step.credit for step in rollout.steps] == [0.0, 0.0]


def test_policy_output_matches_agent_r1_unpadded_token_contract() -> None:
    output = PolicyOutput.from_generation(
        raw_response="SEARCH(query)",
        prompt_ids=(10, 11),
        response_ids=(20, 21, 22),
        response_logprobs=(-0.1, -0.2, -0.3),
        finish_reason="eos",
    )

    assert output.input_ids == (10, 11, 20, 21, 22)
    assert output.position_ids == (0, 1, 2, 3, 4)
    assert output.attention_mask == (1, 1, 1, 1, 1)
    assert output.response_mask == (1, 1, 1)
    assert output.response_logprobs == (-0.1, -0.2, -0.3)

    with pytest.raises(ValueError, match="unpadded response_mask"):
        PolicyOutput(
            raw_response=output.raw_response,
            prompt_ids=output.prompt_ids,
            response_ids=output.response_ids,
            input_ids=output.input_ids,
            position_ids=output.position_ids,
            attention_mask=output.attention_mask,
            response_mask=(1, 0, 1),
            response_logprobs=output.response_logprobs,
            finish_reason=output.finish_reason,
        )
