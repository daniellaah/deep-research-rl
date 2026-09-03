"""Policy-condition entry points for the frozen baseline evaluation protocol."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from deep_research_rl.agent.protocols import GenerativePolicy
from deep_research_rl.agent.rollout import run_model_rollout
from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.costs import ZeroCost
from deep_research_rl.core.credit import TerminalOnlyCreditAssigner
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.models import Example, SearchResult
from deep_research_rl.core.protocols import Retriever
from deep_research_rl.core.rewards import TerminalExactMatchReward
from deep_research_rl.evaluation.contracts import (
    EvaluationFailure,
    EvaluationItem,
    EvaluationResultScope,
    PolicyCondition,
)
from deep_research_rl.evaluation.metrics import evaluate_rollout

BASELINE_MAX_SEARCHES = 5
BASELINE_MAX_STEPS = 8
BASELINE_RETRIEVAL_TOP_K = 3
BASELINE_MAX_PROMPT_TOKENS = 8192
BASELINE_MAX_NEW_TOKENS = 96
BASELINE_EVALUATION_SEED = 0
FULL_VALIDATION_EXAMPLES = 7405
FULL_VALIDATION_EXAMPLES_SHA256 = "3ed88155b5c524d0d75c9f7a99b955b934bd0c3cfd8f5a9f62533c126e2a2944"
FULL_CORPUS_SHA256 = "fc515d9f4686b3e10179bb1743b0444f369650915a71e1a18299411974f6bc69"


class EvaluationRunError(RuntimeError):
    """Raised after an infrastructure exception has been captured as a failure record."""


class NoRetrieval:
    """A retriever boundary that fails loudly if the no-search condition reaches it."""

    def search(self, query: str) -> tuple[SearchResult, ...]:
        raise EvaluationRunError(f"no-search policy attempted an executable retrieval: {query}")

    def search_batch(self, queries: Sequence[str]) -> tuple[tuple[SearchResult, ...], ...]:
        if queries:
            raise EvaluationRunError("no-search policy attempted batch retrieval")
        return ()


def _evaluate_policy(
    examples: Sequence[Example],
    *,
    policy: GenerativePolicy,
    retriever: Retriever,
    run_id: str,
    policy_condition: PolicyCondition,
    result_scope: EvaluationResultScope,
    max_searches: int,
    max_steps: int,
    count_tool_tokens: Callable[[str], int],
) -> tuple[EvaluationItem, ...]:
    if not examples:
        raise ValueError("evaluation requires at least one example")
    environment = ResearchEnvironment(
        retriever,
        AppendOnlyContextPolicy(),
        max_searches=max_searches,
    )
    items: list[EvaluationItem] = []
    for example in examples:
        try:
            rollout = run_model_rollout(
                example,
                policy,
                environment,
                TerminalExactMatchReward(),
                TerminalOnlyCreditAssigner(),
                ZeroCost(),
                max_steps=max_steps,
                result_scope=result_scope,
            )
            items.append(
                evaluate_rollout(
                    rollout,
                    run_id=run_id,
                    policy_condition=policy_condition,
                    count_tool_tokens=count_tool_tokens,
                )
            )
        except Exception as error:
            items.append(
                EvaluationFailure(
                    run_id=run_id,
                    policy_condition=policy_condition,
                    result_scope=result_scope,
                    example_id=example.example_id,
                    error_type=type(error).__name__,
                    message=str(error) or "infrastructure exception without a message",
                )
            )
            break
    return tuple(items)


def run_no_search_evaluation(
    examples: Sequence[Example],
    *,
    policy: GenerativePolicy,
    run_id: str,
    result_scope: EvaluationResultScope,
    count_tool_tokens: Callable[[str], int],
) -> tuple[EvaluationItem, ...]:
    """Evaluate exactly one ANSWER-only generation with retrieval disabled."""

    return _evaluate_policy(
        examples,
        policy=policy,
        retriever=NoRetrieval(),
        run_id=run_id,
        policy_condition="no_search",
        result_scope=result_scope,
        max_searches=0,
        max_steps=1,
        count_tool_tokens=count_tool_tokens,
    )


def run_prompted_agent_evaluation(
    examples: Sequence[Example],
    *,
    policy: GenerativePolicy,
    retriever: Retriever,
    run_id: str,
    result_scope: EvaluationResultScope,
    count_tool_tokens: Callable[[str], int],
) -> tuple[EvaluationItem, ...]:
    """Evaluate the frozen prompt/environment with the pinned base checkpoint."""

    return _evaluate_policy(
        examples,
        policy=policy,
        retriever=retriever,
        run_id=run_id,
        policy_condition="prompted_agent",
        result_scope=result_scope,
        max_searches=BASELINE_MAX_SEARCHES,
        max_steps=BASELINE_MAX_STEPS,
        count_tool_tokens=count_tool_tokens,
    )


def run_rl_agent_evaluation(
    examples: Sequence[Example],
    *,
    policy: GenerativePolicy,
    retriever: Retriever,
    run_id: str,
    result_scope: EvaluationResultScope,
    count_tool_tokens: Callable[[str], int],
) -> tuple[EvaluationItem, ...]:
    """Evaluate a trained policy while preserving prompted-agent controls."""

    return _evaluate_policy(
        examples,
        policy=policy,
        retriever=retriever,
        run_id=run_id,
        policy_condition="rl_agent",
        result_scope=result_scope,
        max_searches=BASELINE_MAX_SEARCHES,
        max_steps=BASELINE_MAX_STEPS,
        count_tool_tokens=count_tool_tokens,
    )
