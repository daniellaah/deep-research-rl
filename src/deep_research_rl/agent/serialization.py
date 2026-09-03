"""Stable JSONL serialization for model-backed agent rollouts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from deep_research_rl.agent.contracts import (
    AgentRollout,
    AgentRolloutStep,
    FinishReason,
    ModelIdentity,
    PolicyOutput,
    TerminationReason,
)
from deep_research_rl.core.serialization import (
    TrajectoryFormatError,
    action_from_record,
    action_to_record,
    example_from_record,
    example_to_record,
    metrics_from_record,
    metrics_to_record,
    observation_from_record,
    observation_to_record,
    state_from_record,
    state_to_record,
)

AGENT_ROLLOUT_SCHEMA_VERSION = 1


def _policy_output_to_record(output: PolicyOutput) -> dict[str, object]:
    return {
        "attention_mask": list(output.attention_mask),
        "finish_reason": output.finish_reason,
        "input_ids": list(output.input_ids),
        "position_ids": list(output.position_ids),
        "prompt_ids": list(output.prompt_ids),
        "raw_response": output.raw_response,
        "response_ids": list(output.response_ids),
        "response_logprobs": (
            None if output.response_logprobs is None else list(output.response_logprobs)
        ),
        "response_mask": list(output.response_mask),
    }


def _step_to_record(step: AgentRolloutStep) -> dict[str, object]:
    return {
        "action": None if step.action is None else action_to_record(step.action),
        "cost": step.cost,
        "credit": step.credit,
        "index": step.index,
        "observation": observation_to_record(step.observation),
        "parse_error": step.parse_error,
        "policy_output": _policy_output_to_record(step.policy_output),
        "reward": step.reward,
        "state_after": state_to_record(step.state_after),
        "state_before": state_to_record(step.state_before),
    }


def agent_rollout_to_dict(rollout: AgentRollout) -> dict[str, object]:
    """Convert one rollout to the stable JSON-compatible artifact shape."""

    return {
        "benchmark_eligible": rollout.example.benchmark_eligible,
        "example": example_to_record(rollout.example),
        "final_state": state_to_record(rollout.final_state),
        "initial_state": state_to_record(rollout.initial_state),
        "metrics": {
            **metrics_to_record(rollout.metrics),
            "prompt_tokens": rollout.prompt_tokens,
            "response_tokens": rollout.response_tokens,
        },
        "model": {"name": rollout.model.name, "revision": rollout.model.revision},
        "prompt_format": rollout.prompt_format,
        "record_type": "agent_rollout",
        "result_scope": rollout.result_scope,
        "schema_version": AGENT_ROLLOUT_SCHEMA_VERSION,
        "steps": [_step_to_record(step) for step in rollout.steps],
        "synthetic": rollout.example.synthetic,
        "termination_reason": rollout.termination_reason,
    }


def agent_rollout_as_json(rollout: AgentRollout) -> str:
    """Return one stable JSONL-ready line."""

    return json.dumps(agent_rollout_to_dict(rollout), ensure_ascii=False, sort_keys=True)


def write_agent_rollout_jsonl(
    path: str | Path,
    rollouts: tuple[AgentRollout, ...],
) -> Path:
    """Write complete rollouts atomically enough for bounded local validation."""

    if not rollouts:
        raise ValueError("at least one rollout is required")
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    contents = "".join(f"{agent_rollout_as_json(rollout)}\n" for rollout in rollouts)
    output_path.write_text(contents, encoding="utf-8")
    return output_path


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TrajectoryFormatError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise TrajectoryFormatError(f"{field} must be an array")
    return list(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TrajectoryFormatError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrajectoryFormatError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrajectoryFormatError(f"{field} must be a number")
    return float(value)


def _integer_tuple(value: object, field: str) -> tuple[int, ...]:
    return tuple(_integer(item, f"{field}[]") for item in _list(value, field))


def _policy_output_from_record(value: object) -> PolicyOutput:
    record = _mapping(value, "policy_output")
    logprobs_value = record.get("response_logprobs")
    if logprobs_value is None:
        logprobs = None
    else:
        logprobs = tuple(
            _number(item, "policy_output.response_logprobs[]")
            for item in _list(logprobs_value, "policy_output.response_logprobs")
        )
    finish_reason = _string(record.get("finish_reason"), "policy_output.finish_reason")
    if finish_reason not in {"eos", "length", "stop"}:
        raise TrajectoryFormatError(f"unsupported finish_reason: {finish_reason}")
    return PolicyOutput(
        raw_response=_string(record.get("raw_response"), "policy_output.raw_response"),
        prompt_ids=_integer_tuple(record.get("prompt_ids"), "policy_output.prompt_ids"),
        response_ids=_integer_tuple(record.get("response_ids"), "policy_output.response_ids"),
        input_ids=_integer_tuple(record.get("input_ids"), "policy_output.input_ids"),
        position_ids=_integer_tuple(record.get("position_ids"), "policy_output.position_ids"),
        attention_mask=_integer_tuple(record.get("attention_mask"), "policy_output.attention_mask"),
        response_mask=_integer_tuple(record.get("response_mask"), "policy_output.response_mask"),
        response_logprobs=logprobs,
        finish_reason=cast(FinishReason, finish_reason),
    )


def _step_from_record(value: object) -> AgentRolloutStep:
    record = _mapping(value, "step")
    action_value = record.get("action")
    parse_error_value = record.get("parse_error")
    if parse_error_value is not None and not isinstance(parse_error_value, str):
        raise TrajectoryFormatError("step.parse_error must be a string or null")
    return AgentRolloutStep(
        index=_integer(record.get("index"), "step.index"),
        policy_output=_policy_output_from_record(record.get("policy_output")),
        action=None if action_value is None else action_from_record(action_value),
        parse_error=parse_error_value,
        state_before=state_from_record(record.get("state_before")),
        observation=observation_from_record(record.get("observation")),
        state_after=state_from_record(record.get("state_after")),
        reward=_number(record.get("reward"), "step.reward"),
        cost=_number(record.get("cost"), "step.cost"),
        credit=_number(record.get("credit"), "step.credit"),
    )


def agent_rollout_from_dict(value: object) -> AgentRollout:
    """Validate and reconstruct one decoded model-rollout record."""

    record = _mapping(value, "agent rollout")
    if _integer(record.get("schema_version"), "schema_version") != AGENT_ROLLOUT_SCHEMA_VERSION:
        raise TrajectoryFormatError("unsupported agent rollout schema_version")
    if _string(record.get("record_type"), "record_type") != "agent_rollout":
        raise TrajectoryFormatError("record_type must be agent_rollout")
    example = example_from_record(record.get("example"))
    if record.get("synthetic") != example.synthetic:
        raise TrajectoryFormatError("top-level synthetic marker must match example")
    if record.get("benchmark_eligible") != example.benchmark_eligible:
        raise TrajectoryFormatError("top-level benchmark_eligible marker must match example")

    model_record = _mapping(record.get("model"), "model")
    metrics_record = _mapping(record.get("metrics"), "metrics")
    termination_reason = _string(record.get("termination_reason"), "termination_reason")
    if termination_reason not in {"answered", "max_steps"}:
        raise TrajectoryFormatError(f"unsupported termination_reason: {termination_reason}")
    return AgentRollout(
        model=ModelIdentity(
            name=_string(model_record.get("name"), "model.name"),
            revision=_string(model_record.get("revision"), "model.revision"),
        ),
        prompt_format=_string(record.get("prompt_format"), "prompt_format"),
        result_scope=_string(record.get("result_scope"), "result_scope"),
        example=example,
        initial_state=state_from_record(record.get("initial_state")),
        steps=tuple(_step_from_record(item) for item in _list(record.get("steps"), "steps")),
        final_state=state_from_record(record.get("final_state")),
        termination_reason=cast(TerminationReason, termination_reason),
        metrics=metrics_from_record(metrics_record),
        prompt_tokens=_integer(metrics_record.get("prompt_tokens"), "metrics.prompt_tokens"),
        response_tokens=_integer(metrics_record.get("response_tokens"), "metrics.response_tokens"),
    )


def agent_rollout_from_json(line: str) -> AgentRollout:
    """Decode one model-rollout JSON line."""

    try:
        value: object = json.loads(line)
    except json.JSONDecodeError as error:
        raise TrajectoryFormatError(f"invalid agent rollout JSON: {error}") from error
    return agent_rollout_from_dict(value)


def read_agent_rollout_jsonl(path: str | Path) -> tuple[AgentRollout, ...]:
    """Read and validate every non-empty model-rollout line."""

    input_path = Path(path)
    rollouts: list[AgentRollout] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rollouts.append(agent_rollout_from_json(line))
        except TrajectoryFormatError as error:
            raise TrajectoryFormatError(f"{input_path}:{line_number}: {error}") from error
    return tuple(rollouts)
