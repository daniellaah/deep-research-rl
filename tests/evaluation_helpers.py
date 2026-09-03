from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from deep_research_rl.agent.contracts import AgentRollout, PolicyOutput
from deep_research_rl.agent.rollout import run_model_rollout
from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.costs import ZeroCost
from deep_research_rl.core.credit import TerminalOnlyCreditAssigner
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.models import AgentState, Example, SearchResult
from deep_research_rl.core.rewards import TerminalExactMatchReward

TEST_REVISION = "1" * 40


@dataclass(slots=True)
class FixedPolicy:
    responses: tuple[str, ...]
    prompt_lengths: tuple[int, ...]
    response_lengths: tuple[int, ...]
    model_name: str = "test/qwen"
    model_revision: str = TEST_REVISION
    prompt_format: str = "qwen3_strict_search_answer_v1"
    seen_max_searches: list[int] = field(default_factory=list)
    _cursor: int = field(default=0, init=False)

    def generate(self, state: AgentState, *, max_searches: int) -> PolicyOutput:
        self.seen_max_searches.append(max_searches)
        index = self._cursor
        self._cursor += 1
        prompt_length = self.prompt_lengths[index]
        response_length = self.response_lengths[index]
        return PolicyOutput.from_generation(
            raw_response=self.responses[index],
            prompt_ids=tuple(range(100, 100 + prompt_length)),
            response_ids=tuple(range(200, 200 + response_length)),
            response_logprobs=(-0.5,) * response_length,
            finish_reason="eos",
        )


class FixedRetriever:
    def __init__(self, results: tuple[SearchResult, ...]) -> None:
        self.results = results
        self.queries: list[str] = []

    def search(self, query: str) -> tuple[SearchResult, ...]:
        self.queries.append(query)
        return self.results

    def search_batch(self, queries: Sequence[str]) -> tuple[tuple[SearchResult, ...], ...]:
        return tuple(self.search(query) for query in queries)


def example(
    example_id: str,
    *,
    answers: tuple[str, ...] = ("wrong alias", "blue whale"),
    supporting_document_ids: tuple[str, ...] = ("doc-a", "doc-b"),
) -> Example:
    return Example(
        example_id=example_id,
        question="Which animal is described?",
        answers=answers,
        supporting_document_ids=supporting_document_ids,
        source="hotpotqa_distractor:validation:test-revision",
        synthetic=False,
        benchmark_eligible=True,
    )


def search_result(document_id: str, rank: int, title: str) -> SearchResult:
    return SearchResult(
        document_id=document_id,
        title=title,
        text=f"Evidence from {title}.",
        score=1.0 / rank,
        rank=rank,
    )


def rollout(
    evaluation_example: Example,
    policy: FixedPolicy,
    retriever: FixedRetriever,
    *,
    max_searches: int,
    max_steps: int,
) -> AgentRollout:
    return run_model_rollout(
        evaluation_example,
        policy,
        ResearchEnvironment(
            retriever,
            AppendOnlyContextPolicy(),
            max_searches=max_searches,
        ),
        TerminalExactMatchReward(),
        TerminalOnlyCreditAssigner(),
        ZeroCost(),
        max_steps=max_steps,
    )
