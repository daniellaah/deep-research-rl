from __future__ import annotations

from dataclasses import dataclass

from evaluation_helpers import FixedPolicy, FixedRetriever, example

from deep_research_rl.agent.contracts import PolicyOutput
from deep_research_rl.core.models import AgentState
from deep_research_rl.evaluation.contracts import EvaluationFailure, EvaluationRecord
from deep_research_rl.evaluation.runner import (
    run_no_search_evaluation,
    run_prompted_agent_evaluation,
    run_rl_agent_evaluation,
)


def test_three_policy_entry_points_preserve_their_only_intended_differences() -> None:
    evaluation_example = example("entry-point")
    no_search_policy = FixedPolicy(
        responses=("ANSWER(blue whale)",),
        prompt_lengths=(2,),
        response_lengths=(1,),
    )
    no_search = run_no_search_evaluation(
        (evaluation_example,),
        policy=no_search_policy,
        run_id="no-search",
        result_scope="debug_validation_not_benchmark",
        count_tool_tokens=lambda _: 0,
    )
    prompted_policy = FixedPolicy(
        responses=("ANSWER(blue whale)",),
        prompt_lengths=(2,),
        response_lengths=(1,),
    )
    prompted = run_prompted_agent_evaluation(
        (evaluation_example,),
        policy=prompted_policy,
        retriever=FixedRetriever(()),
        run_id="prompted",
        result_scope="debug_validation_not_benchmark",
        count_tool_tokens=lambda _: 0,
    )
    rl_policy = FixedPolicy(
        responses=("ANSWER(blue whale)",),
        prompt_lengths=(2,),
        response_lengths=(1,),
    )
    trained = run_rl_agent_evaluation(
        (evaluation_example,),
        policy=rl_policy,
        retriever=FixedRetriever(()),
        run_id="trained",
        result_scope="debug_validation_not_benchmark",
        count_tool_tokens=lambda _: 0,
    )

    assert isinstance(no_search[0], EvaluationRecord)
    assert isinstance(prompted[0], EvaluationRecord)
    assert isinstance(trained[0], EvaluationRecord)
    assert no_search[0].policy_condition == "no_search"
    assert prompted[0].policy_condition == "prompted_agent"
    assert trained[0].policy_condition == "rl_agent"
    assert no_search_policy.seen_max_searches == [0]
    assert prompted_policy.seen_max_searches == [5]
    assert rl_policy.seen_max_searches == [5]


@dataclass(slots=True)
class FailingPolicy:
    model_name: str = "test/qwen"
    model_revision: str = "2" * 40
    prompt_format: str = "qwen3_strict_search_answer_v1"

    def generate(self, state: AgentState, *, max_searches: int) -> PolicyOutput:
        raise RuntimeError("model worker unavailable")


def test_infrastructure_exception_becomes_invalidating_failure_record() -> None:
    items = run_prompted_agent_evaluation(
        (example("failure"),),
        policy=FailingPolicy(),
        retriever=FixedRetriever(()),
        run_id="failed-run",
        result_scope="debug_validation_not_benchmark",
        count_tool_tokens=lambda _: 0,
    )

    assert len(items) == 1
    assert isinstance(items[0], EvaluationFailure)
    assert items[0].error_type == "RuntimeError"
    assert items[0].message == "model worker unavailable"
