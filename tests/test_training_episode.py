import json
from collections.abc import Sequence

import pytest

from deep_research_rl.core.fixtures import synthetic_two_hop_fixture
from deep_research_rl.core.models import SearchResult
from deep_research_rl.training.episode import BASELINE_MAX_STEPS, AgentR1Episode


class CountingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[SearchResult, ...]:
        self.queries.append(query)
        return ()

    def search_batch(self, queries: Sequence[str]) -> tuple[tuple[SearchResult, ...], ...]:
        return tuple(self.search(query) for query in queries)


def test_answer_can_be_the_first_policy_action_without_forced_search() -> None:
    example, _ = synthetic_two_hop_fixture()
    retriever = CountingRetriever()
    episode = AgentR1Episode(example, retriever)

    step = episode.apply_response("ANSWER(Lumen City)")

    assert retriever.queries == []
    assert step.reward == 1.0
    assert step.episode_complete is True
    assert episode.termination_reason == "answered"


def test_parallel_actions_are_rejected_without_any_retrieval() -> None:
    example, _ = synthetic_two_hop_fixture()
    retriever = CountingRetriever()
    episode = AgentR1Episode(example, retriever)

    step = episode.apply_response("SEARCH(first)\nSEARCH(second)")

    assert step.action is None
    assert step.parse_error is not None
    assert step.observation.status == "malformed_action"
    assert step.reward == 0.0
    assert retriever.queries == []
    assert episode.state.context == (step.observation,)


def test_sixth_search_is_logged_but_not_executed_and_terminal_reward_stays_sparse() -> None:
    example, _ = synthetic_two_hop_fixture()
    retriever = CountingRetriever()
    episode = AgentR1Episode(example, retriever)

    steps = [episode.apply_response(f"SEARCH(query {index})") for index in range(6)]
    answer = episode.apply_response("ANSWER(the lumen city)")

    assert retriever.queries == [f"query {index}" for index in range(5)]
    assert episode.attempted_searches == 6
    assert steps[-1].observation.status == "search_rejected"
    assert steps[-1].state_after.executed_searches == 5
    assert [step.reward for step in steps] == [0.0] * 6
    assert answer.reward == 1.0

    metadata = steps[-1].reward_extra_info(attempted_searches=episode.attempted_searches)
    transition = json.loads(str(metadata["transition_json"]))
    assert metadata["executed_searches"] == 5
    assert transition["observation"]["status"] == "search_rejected"


def test_max_step_bound_completes_without_fabricating_terminal_reward() -> None:
    example, _ = synthetic_two_hop_fixture()
    episode = AgentR1Episode(example, CountingRetriever())

    for _ in range(BASELINE_MAX_STEPS):
        final_step = episode.apply_response("not an action")

    assert episode.complete is True
    assert episode.termination_reason == "max_steps"
    assert final_step.reward == 0.0
    assert final_step.state_after.terminated is False
    with pytest.raises(RuntimeError, match="after episode completion"):
        episode.apply_response("ANSWER(Lumen City)")
