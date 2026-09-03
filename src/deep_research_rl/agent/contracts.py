"""Dependency-light contracts for model-backed, step-native agent rollouts."""

from __future__ import annotations

import math
import string
from dataclasses import dataclass
from typing import Literal

from deep_research_rl.core.models import (
    Action,
    AgentState,
    EpisodeMetrics,
    Example,
    Observation,
)

FinishReason = Literal["eos", "length", "stop"]
TerminationReason = Literal["answered", "max_steps"]


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Immutable model identity recorded with every rollout."""

    name: str
    revision: str

    def __post_init__(self) -> None:
        if not self.name or self.name != self.name.strip():
            raise ValueError("model name must be non-empty and trimmed")
        if len(self.revision) != 40 or any(
            character not in string.hexdigits for character in self.revision
        ):
            raise ValueError("model revision must be a 40-character hexadecimal commit")


@dataclass(frozen=True, slots=True)
class PolicyOutput:
    """One model response plus the unpadded token contract consumed by Agent-R1/verl."""

    raw_response: str
    prompt_ids: tuple[int, ...]
    response_ids: tuple[int, ...]
    input_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    response_mask: tuple[int, ...]
    response_logprobs: tuple[float, ...] | None
    finish_reason: FinishReason

    def __post_init__(self) -> None:
        for field_name, values in (
            ("prompt_ids", self.prompt_ids),
            ("response_ids", self.response_ids),
            ("input_ids", self.input_ids),
            ("position_ids", self.position_ids),
        ):
            if any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in values
            ):
                raise ValueError(f"{field_name} must contain non-negative integer token IDs")
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        if self.input_ids != (*self.prompt_ids, *self.response_ids):
            raise ValueError("input_ids must equal prompt_ids followed by response_ids")
        if self.position_ids != tuple(range(len(self.input_ids))):
            raise ValueError("position_ids must be contiguous and zero-based")
        if len(self.attention_mask) != len(self.input_ids) or any(
            value != 1 for value in self.attention_mask
        ):
            raise ValueError("unpadded attention_mask must contain one per input token")
        if len(self.response_mask) != len(self.response_ids) or any(
            value != 1 for value in self.response_mask
        ):
            raise ValueError("unpadded response_mask must contain one per generated token")
        if self.response_logprobs is not None:
            if len(self.response_logprobs) != len(self.response_ids):
                raise ValueError("response_logprobs must align one-to-one with response_ids")
            if any(not math.isfinite(value) for value in self.response_logprobs):
                raise ValueError("response_logprobs must be finite")

    @classmethod
    def from_generation(
        cls,
        *,
        raw_response: str,
        prompt_ids: tuple[int, ...],
        response_ids: tuple[int, ...],
        response_logprobs: tuple[float, ...] | None,
        finish_reason: FinishReason,
    ) -> PolicyOutput:
        """Build the canonical unpadded masks and positions for generated tokens."""

        input_ids = (*prompt_ids, *response_ids)
        return cls(
            raw_response=raw_response,
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            input_ids=input_ids,
            position_ids=tuple(range(len(input_ids))),
            attention_mask=(1,) * len(input_ids),
            response_mask=(1,) * len(response_ids),
            response_logprobs=response_logprobs,
            finish_reason=finish_reason,
        )


@dataclass(frozen=True, slots=True)
class AgentRolloutStep:
    """A model response and its deterministic environment handling."""

    index: int
    policy_output: PolicyOutput
    action: Action | None
    parse_error: str | None
    state_before: AgentState
    observation: Observation
    state_after: AgentState
    reward: float
    cost: float
    credit: float

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("step index must not be negative")
        if (self.action is None) == (self.parse_error is None):
            raise ValueError("a step must contain either a parsed action or a parse error")
        if self.parse_error is not None:
            if not self.parse_error:
                raise ValueError("parse_error must not be empty")
            if self.observation.status != "malformed_action":
                raise ValueError("a parse error must produce a malformed_action observation")
        elif self.observation.status == "malformed_action":
            raise ValueError("a parsed action cannot produce a malformed_action observation")
        if not all(math.isfinite(value) for value in (self.reward, self.cost, self.credit)):
            raise ValueError("reward, cost and credit must be finite")
        if self.state_before.example_id != self.state_after.example_id:
            raise ValueError("step states must belong to the same example")


@dataclass(frozen=True, slots=True)
class AgentRollout:
    """Complete model-backed trajectory with explicit termination and token provenance."""

    model: ModelIdentity
    prompt_format: str
    result_scope: str
    example: Example
    initial_state: AgentState
    steps: tuple[AgentRolloutStep, ...]
    final_state: AgentState
    termination_reason: TerminationReason
    metrics: EpisodeMetrics
    prompt_tokens: int
    response_tokens: int

    def __post_init__(self) -> None:
        if not self.prompt_format or not self.result_scope:
            raise ValueError("prompt_format and result_scope must not be empty")
        if self.initial_state.example_id != self.example.example_id:
            raise ValueError("initial state must belong to the rollout example")
        if self.final_state.example_id != self.example.example_id:
            raise ValueError("final state must belong to the rollout example")
        if any(step.index != index for index, step in enumerate(self.steps)):
            raise ValueError("step indices must be contiguous and zero-based")
        if self.steps:
            if self.steps[0].state_before != self.initial_state:
                raise ValueError("first step must start at initial_state")
            if self.steps[-1].state_after != self.final_state:
                raise ValueError("last step must reach final_state")
            if any(
                current.state_after != following.state_before
                for current, following in zip(self.steps, self.steps[1:], strict=False)
            ):
                raise ValueError("adjacent rollout states must form a continuous chain")
        if (self.termination_reason == "answered") != self.final_state.terminated:
            raise ValueError("answered termination must agree with the final state")
        if self.prompt_tokens != sum(len(step.policy_output.prompt_ids) for step in self.steps):
            raise ValueError("prompt_tokens must equal the recorded prompt token total")
        if self.response_tokens != sum(len(step.policy_output.response_ids) for step in self.steps):
            raise ValueError("response_tokens must equal the recorded response token total")
