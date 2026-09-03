"""Typed contracts for per-example evaluation and invalid infrastructure outcomes."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

from deep_research_rl.agent.contracts import AgentRollout
from deep_research_rl.core.models import SearchAction

PolicyCondition = Literal["no_search", "prompted_agent", "rl_agent"]
EvaluationResultScope = Literal["debug_validation_not_benchmark", "baseline_validation"]
BASELINE_EVALUATION_PROTOCOL = "baseline_evaluation_v1"


@dataclass(frozen=True, slots=True)
class EvaluationExampleMetrics:
    """All recomputable metrics for one complete policy outcome."""

    exact_match: float
    token_f1: float
    success: bool
    completed: bool
    attempted_searches: int
    executed_searches: int
    rejected_searches: int
    malformed_actions: int
    step_count: int
    prompt_tokens_processed: int
    response_tokens_generated: int
    total_model_tokens: int
    tool_tokens_appended: int
    supporting_labels_available: bool
    supporting_documents: int
    supporting_document_hits: int
    supporting_document_recall: float | None
    complete_support_set: bool | None

    def __post_init__(self) -> None:
        for name, value in (("exact_match", self.exact_match), ("token_f1", self.token_f1)):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if self.success != (self.exact_match == 1.0):
            raise ValueError("success must mean exact_match == 1")
        integer_values = (
            self.attempted_searches,
            self.executed_searches,
            self.rejected_searches,
            self.malformed_actions,
            self.step_count,
            self.prompt_tokens_processed,
            self.response_tokens_generated,
            self.total_model_tokens,
            self.tool_tokens_appended,
            self.supporting_documents,
            self.supporting_document_hits,
        )
        if any(value < 0 for value in integer_values):
            raise ValueError("metric counts must not be negative")
        if self.attempted_searches != self.executed_searches + self.rejected_searches:
            raise ValueError("attempted searches must equal executed plus rejected searches")
        if self.total_model_tokens != (
            self.prompt_tokens_processed + self.response_tokens_generated
        ):
            raise ValueError("total_model_tokens must equal prompt plus response tokens")
        if self.supporting_document_hits > self.supporting_documents:
            raise ValueError("supporting-document hits cannot exceed the label count")
        if self.supporting_labels_available:
            if self.supporting_documents == 0:
                raise ValueError("labeled examples must contain supporting documents")
            expected_recall = self.supporting_document_hits / self.supporting_documents
            if self.supporting_document_recall != expected_recall:
                raise ValueError("supporting-document recall does not match counts")
            if self.complete_support_set != (
                self.supporting_document_hits == self.supporting_documents
            ):
                raise ValueError("complete-support-set flag does not match counts")
        elif (
            self.supporting_documents != 0
            or self.supporting_document_hits != 0
            or self.supporting_document_recall is not None
            or self.complete_support_set is not None
        ):
            raise ValueError("unlabeled examples must use zero counts and null evidence metrics")


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """A valid policy outcome whose embedded rollout is the metric source of truth."""

    run_id: str
    policy_condition: PolicyCondition
    result_scope: EvaluationResultScope
    rollout: AgentRollout
    metrics: EvaluationExampleMetrics
    retrieved_document_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.run_id or self.run_id != self.run_id.strip():
            raise ValueError("run_id must be non-empty and trimmed")
        if self.policy_condition not in {"no_search", "prompted_agent", "rl_agent"}:
            raise ValueError(f"unsupported policy condition: {self.policy_condition}")
        if self.result_scope not in {
            "debug_validation_not_benchmark",
            "baseline_validation",
        }:
            raise ValueError(f"unsupported result scope: {self.result_scope}")
        if self.rollout.result_scope != self.result_scope:
            raise ValueError("rollout and evaluation result scopes must match")
        if not self.rollout.steps:
            raise ValueError("evaluation trajectory must contain at least one policy step")
        if self.policy_condition == "no_search":
            if (
                len(self.rollout.steps) != 1
                or self.rollout.metrics.executed_searches != 0
                or self.metrics.tool_tokens_appended != 0
            ):
                raise ValueError("no-search evaluation requires one step and zero retrieval calls")
        elif len(self.rollout.steps) > 8 or self.rollout.metrics.executed_searches > 5:
            raise ValueError("search-agent evaluation exceeds the frozen step or search limit")
        if len(self.retrieved_document_ids) != len(set(self.retrieved_document_ids)):
            raise ValueError("retrieved_document_ids must be a duplicate-free union")
        if self.metrics.completed != (self.rollout.termination_reason == "answered"):
            raise ValueError("completion must mean termination by a valid ANSWER")
        if self.metrics.executed_searches != self.rollout.metrics.executed_searches:
            raise ValueError("evaluated and rollout executed-search counts must match")
        if self.metrics.step_count != len(self.rollout.steps):
            raise ValueError("evaluated step count must match the embedded rollout")
        if self.metrics.prompt_tokens_processed != self.rollout.prompt_tokens:
            raise ValueError("evaluated prompt-token count must match the embedded rollout")
        if self.metrics.response_tokens_generated != self.rollout.response_tokens:
            raise ValueError("evaluated response-token count must match the embedded rollout")
        if self.metrics.exact_match != self.rollout.metrics.exact_match:
            raise ValueError("evaluated exact match must match the embedded rollout")
        if self.metrics.token_f1 != self.rollout.metrics.token_f1:
            raise ValueError("evaluated token F1 must match the embedded rollout")
        attempted = sum(isinstance(step.action, SearchAction) for step in self.rollout.steps)
        rejected = sum(step.observation.status == "search_rejected" for step in self.rollout.steps)
        malformed = sum(
            step.observation.status == "malformed_action" for step in self.rollout.steps
        )
        if (
            self.metrics.attempted_searches != attempted
            or self.metrics.rejected_searches != rejected
            or self.metrics.malformed_actions != malformed
        ):
            raise ValueError("action metrics must recompute from the embedded rollout")
        retrieved = tuple(
            dict.fromkeys(
                document.document_id
                for step in self.rollout.steps
                if step.observation.status == "search_executed"
                for document in step.observation.documents
            )
        )
        if self.retrieved_document_ids != retrieved:
            raise ValueError("retrieved-document union must recompute from the embedded rollout")
        supporting = set(self.rollout.example.supporting_document_ids)
        hits = len(supporting.intersection(retrieved))
        if supporting:
            evidence_matches = (
                self.metrics.supporting_labels_available
                and self.metrics.supporting_documents == len(supporting)
                and self.metrics.supporting_document_hits == hits
            )
        else:
            evidence_matches = not self.metrics.supporting_labels_available
        if not evidence_matches:
            raise ValueError("evidence metrics must recompute from the embedded rollout")

    @property
    def example_id(self) -> str:
        """Return the stable dataset identifier for coverage checks."""

        return self.rollout.example.example_id


@dataclass(frozen=True, slots=True)
class EvaluationFailure:
    """An infrastructure exception that invalidates, rather than scores, a run."""

    run_id: str
    policy_condition: PolicyCondition
    result_scope: EvaluationResultScope
    example_id: str
    error_type: str
    message: str

    def __post_init__(self) -> None:
        values = (self.run_id, self.example_id, self.error_type, self.message)
        if any(not value or value != value.strip() for value in values):
            raise ValueError("failure fields must be non-empty and trimmed")
        if self.policy_condition not in {"no_search", "prompted_agent", "rl_agent"}:
            raise ValueError(f"unsupported policy condition: {self.policy_condition}")
        if self.result_scope not in {
            "debug_validation_not_benchmark",
            "baseline_validation",
        }:
            raise ValueError(f"unsupported result scope: {self.result_scope}")


EvaluationItem: TypeAlias = EvaluationRecord | EvaluationFailure
