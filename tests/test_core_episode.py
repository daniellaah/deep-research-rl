from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.costs import ZeroCost
from deep_research_rl.core.credit import TerminalOnlyCreditAssigner
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.fixtures import synthetic_two_hop_fixture
from deep_research_rl.core.models import Document
from deep_research_rl.core.policies import ScriptedPolicy
from deep_research_rl.core.retrieval import BM25Retriever
from deep_research_rl.core.rewards import TerminalExactMatchReward
from deep_research_rl.core.rollout import run_episode


def test_two_hop_episode_terminates_with_append_only_context_and_terminal_credit() -> None:
    example, documents = synthetic_two_hop_fixture()
    trajectory = run_episode(
        example,
        ScriptedPolicy(
            (
                "SEARCH(Brindle Process)",
                "SEARCH(Mira Voss)",
                "ANSWER(The Lumen City)",
            )
        ),
        ResearchEnvironment(
            BM25Retriever(documents, top_k=1),
            AppendOnlyContextPolicy(),
            max_searches=5,
        ),
        TerminalExactMatchReward(),
        TerminalOnlyCreditAssigner(),
        ZeroCost(),
    )

    assert trajectory.final_state.terminated is True
    assert trajectory.final_state.answer == "The Lumen City"
    assert trajectory.final_state.executed_searches == 2
    assert [step.observation.documents[0].document_id for step in trajectory.steps[:2]] == [
        "brindle-process",
        "mira-voss",
    ]
    assert [step.reward for step in trajectory.steps] == [0.0, 0.0, 1.0]
    assert [step.credit for step in trajectory.steps] == [0.0, 0.0, 1.0]
    assert [step.cost for step in trajectory.steps] == [0.0, 0.0, 0.0]
    assert trajectory.metrics.exact_match == 1.0
    assert trajectory.metrics.token_f1 == 1.0

    for step in trajectory.steps:
        assert step.state_after.context[:-1] == step.state_before.context
        assert step.state_after.context[-1] == step.observation


class CountingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[Document, ...]:
        self.queries.append(query)
        return ()


def test_five_search_budget_rejects_without_executing_or_counting_attempt() -> None:
    example, _ = synthetic_two_hop_fixture()
    retriever = CountingRetriever()
    trajectory = run_episode(
        example,
        ScriptedPolicy(
            (
                "SEARCH(query zero)",
                "SEARCH(query one)",
                "SEARCH(query two)",
                "SEARCH(query three)",
                "SEARCH(query four)",
                "SEARCH(rejected query)",
                "ANSWER(Lumen City)",
            )
        ),
        ResearchEnvironment(retriever, AppendOnlyContextPolicy(), max_searches=5),
        TerminalExactMatchReward(),
        TerminalOnlyCreditAssigner(),
        ZeroCost(),
    )

    rejected_step = trajectory.steps[5]
    assert retriever.queries == [
        "query zero",
        "query one",
        "query two",
        "query three",
        "query four",
    ]
    assert rejected_step.observation.status == "search_rejected"
    assert rejected_step.state_before.executed_searches == 5
    assert rejected_step.state_after.executed_searches == 5
    assert trajectory.final_state.executed_searches == 5
    assert [step.reward for step in trajectory.steps[:-1]] == [0.0] * 6
    assert [step.credit for step in trajectory.steps[:-1]] == [0.0] * 6
    assert trajectory.steps[-1].reward == 1.0
    assert trajectory.steps[-1].credit == 1.0
