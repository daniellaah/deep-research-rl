"""Stable JSONL serialization for complete trajectories."""

from __future__ import annotations

import json
from pathlib import Path

from deep_research_rl.core.models import (
    Action,
    AgentState,
    AnswerAction,
    EpisodeMetrics,
    Example,
    Observation,
    SearchAction,
    SearchResult,
    Step,
    Trajectory,
)

SCHEMA_VERSION = 1


class TrajectoryFormatError(ValueError):
    """Raised when serialized trajectory data does not match the schema."""


def _document_to_dict(document: SearchResult) -> dict[str, object]:
    return {
        "document_id": document.document_id,
        "rank": document.rank,
        "score": document.score,
        "title": document.title,
        "text": document.text,
    }


def _example_to_dict(example: Example) -> dict[str, object]:
    return {
        "example_id": example.example_id,
        "question": example.question,
        "answers": list(example.answers),
        "supporting_document_ids": list(example.supporting_document_ids),
        "source": example.source,
        "synthetic": example.synthetic,
        "benchmark_eligible": example.benchmark_eligible,
    }


def _action_to_dict(action: Action) -> dict[str, object]:
    if isinstance(action, SearchAction):
        return {"type": "search", "query": action.query}
    return {"type": "answer", "answer": action.answer}


def _observation_to_dict(observation: Observation) -> dict[str, object]:
    return {
        "status": observation.status,
        "message": observation.message,
        "query": observation.query,
        "documents": [_document_to_dict(document) for document in observation.documents],
    }


def _state_to_dict(state: AgentState) -> dict[str, object]:
    return {
        "example_id": state.example_id,
        "question": state.question,
        "context": [_observation_to_dict(observation) for observation in state.context],
        "executed_searches": state.executed_searches,
        "terminated": state.terminated,
        "answer": state.answer,
    }


def _step_to_dict(step: Step) -> dict[str, object]:
    return {
        "index": step.index,
        "raw_action": step.raw_action,
        "action": _action_to_dict(step.action),
        "state_before": _state_to_dict(step.state_before),
        "observation": _observation_to_dict(step.observation),
        "state_after": _state_to_dict(step.state_after),
        "reward": step.reward,
        "cost": step.cost,
        "credit": step.credit,
    }


def _metrics_to_dict(metrics: EpisodeMetrics) -> dict[str, object]:
    return {
        "exact_match": metrics.exact_match,
        "token_f1": metrics.token_f1,
        "terminated": metrics.terminated,
        "executed_searches": metrics.executed_searches,
        "step_count": metrics.step_count,
    }


def trajectory_to_dict(trajectory: Trajectory) -> dict[str, object]:
    """Convert a trajectory to the versioned JSON-compatible record shape."""

    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "trajectory",
        "synthetic": trajectory.example.synthetic,
        "benchmark_eligible": trajectory.example.benchmark_eligible,
        "example": _example_to_dict(trajectory.example),
        "initial_state": _state_to_dict(trajectory.initial_state),
        "steps": [_step_to_dict(step) for step in trajectory.steps],
        "final_state": _state_to_dict(trajectory.final_state),
        "metrics": _metrics_to_dict(trajectory.metrics),
    }


def trajectory_as_json(trajectory: Trajectory) -> str:
    """Return one stable JSONL-ready line."""

    return json.dumps(trajectory_to_dict(trajectory), ensure_ascii=False, sort_keys=True)


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TrajectoryFormatError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _sequence(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise TrajectoryFormatError(f"{field} must be an array")
    return list(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TrajectoryFormatError(f"{field} must be a string")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TrajectoryFormatError(f"{field} must be a boolean")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TrajectoryFormatError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TrajectoryFormatError(f"{field} must be a number")
    return float(value)


def _document_from_dict(value: object, *, default_rank: int) -> SearchResult:
    record = _mapping(value, "document")
    raw_rank = record.get("rank", default_rank)
    raw_score = record.get("score", 0.0)
    return SearchResult(
        document_id=_string(record.get("document_id"), "document.document_id"),
        rank=_integer(raw_rank, "document.rank"),
        score=_number(raw_score, "document.score"),
        title=_string(record.get("title"), "document.title"),
        text=_string(record.get("text"), "document.text"),
    )


def _example_from_dict(value: object) -> Example:
    record = _mapping(value, "example")
    answers = tuple(
        _string(answer, "example.answers[]")
        for answer in _sequence(record.get("answers"), "example.answers")
    )
    supporting_ids = tuple(
        _string(document_id, "example.supporting_document_ids[]")
        for document_id in _sequence(
            record.get("supporting_document_ids"),
            "example.supporting_document_ids",
        )
    )
    return Example(
        example_id=_string(record.get("example_id"), "example.example_id"),
        question=_string(record.get("question"), "example.question"),
        answers=answers,
        supporting_document_ids=supporting_ids,
        source=_string(record.get("source"), "example.source"),
        synthetic=_boolean(record.get("synthetic"), "example.synthetic"),
        benchmark_eligible=_boolean(
            record.get("benchmark_eligible"),
            "example.benchmark_eligible",
        ),
    )


def _action_from_dict(value: object) -> Action:
    record = _mapping(value, "action")
    action_type = _string(record.get("type"), "action.type")
    if action_type == "search":
        return SearchAction(query=_string(record.get("query"), "action.query"))
    if action_type == "answer":
        return AnswerAction(answer=_string(record.get("answer"), "action.answer"))
    raise TrajectoryFormatError(f"unsupported action.type: {action_type}")


def _observation_from_dict(value: object) -> Observation:
    record = _mapping(value, "observation")
    status = _string(record.get("status"), "observation.status")
    if status not in {
        "search_executed",
        "search_rejected",
        "answer_recorded",
        "malformed_action",
    }:
        raise TrajectoryFormatError(f"unsupported observation.status: {status}")
    documents = tuple(
        _document_from_dict(document, default_rank=rank)
        for rank, document in enumerate(
            _sequence(record.get("documents"), "observation.documents"),
            1,
        )
    )
    if status == "search_executed":
        return Observation(
            status="search_executed",
            message=_string(record.get("message"), "observation.message"),
            query=_optional_string(record.get("query"), "observation.query"),
            documents=documents,
        )
    if status == "search_rejected":
        return Observation(
            status="search_rejected",
            message=_string(record.get("message"), "observation.message"),
            query=_optional_string(record.get("query"), "observation.query"),
            documents=documents,
        )
    if status == "answer_recorded":
        return Observation(
            status="answer_recorded",
            message=_string(record.get("message"), "observation.message"),
            query=_optional_string(record.get("query"), "observation.query"),
            documents=documents,
        )
    return Observation(
        status="malformed_action",
        message=_string(record.get("message"), "observation.message"),
        query=_optional_string(record.get("query"), "observation.query"),
        documents=documents,
    )


def example_to_record(example: Example) -> dict[str, object]:
    """Return the stable nested record for an example."""

    return _example_to_dict(example)


def action_to_record(action: Action) -> dict[str, object]:
    """Return the stable nested record for one executable action."""

    return _action_to_dict(action)


def observation_to_record(observation: Observation) -> dict[str, object]:
    """Return the stable nested record for an environment observation."""

    return _observation_to_dict(observation)


def state_to_record(state: AgentState) -> dict[str, object]:
    """Return the stable nested record for an agent state."""

    return _state_to_dict(state)


def metrics_to_record(metrics: EpisodeMetrics) -> dict[str, object]:
    """Return the stable nested record for episode metrics."""

    return _metrics_to_dict(metrics)


def _state_from_dict(value: object) -> AgentState:
    record = _mapping(value, "state")
    context = tuple(
        _observation_from_dict(observation)
        for observation in _sequence(record.get("context"), "state.context")
    )
    return AgentState(
        example_id=_string(record.get("example_id"), "state.example_id"),
        question=_string(record.get("question"), "state.question"),
        context=context,
        executed_searches=_integer(record.get("executed_searches"), "state.executed_searches"),
        terminated=_boolean(record.get("terminated"), "state.terminated"),
        answer=_optional_string(record.get("answer"), "state.answer"),
    )


def _step_from_dict(value: object) -> Step:
    record = _mapping(value, "step")
    return Step(
        index=_integer(record.get("index"), "step.index"),
        raw_action=_string(record.get("raw_action"), "step.raw_action"),
        action=_action_from_dict(record.get("action")),
        state_before=_state_from_dict(record.get("state_before")),
        observation=_observation_from_dict(record.get("observation")),
        state_after=_state_from_dict(record.get("state_after")),
        reward=_number(record.get("reward"), "step.reward"),
        cost=_number(record.get("cost"), "step.cost"),
        credit=_number(record.get("credit"), "step.credit"),
    )


def _metrics_from_dict(value: object) -> EpisodeMetrics:
    record = _mapping(value, "metrics")
    return EpisodeMetrics(
        exact_match=_number(record.get("exact_match"), "metrics.exact_match"),
        token_f1=_number(record.get("token_f1"), "metrics.token_f1"),
        terminated=_boolean(record.get("terminated"), "metrics.terminated"),
        executed_searches=_integer(
            record.get("executed_searches"),
            "metrics.executed_searches",
        ),
        step_count=_integer(record.get("step_count"), "metrics.step_count"),
    )


def example_from_record(value: object) -> Example:
    """Validate and reconstruct an example nested record."""

    return _example_from_dict(value)


def action_from_record(value: object) -> Action:
    """Validate and reconstruct an executable action nested record."""

    return _action_from_dict(value)


def observation_from_record(value: object) -> Observation:
    """Validate and reconstruct an observation nested record."""

    return _observation_from_dict(value)


def state_from_record(value: object) -> AgentState:
    """Validate and reconstruct an agent-state nested record."""

    return _state_from_dict(value)


def metrics_from_record(value: object) -> EpisodeMetrics:
    """Validate and reconstruct an episode-metrics nested record."""

    return _metrics_from_dict(value)


def trajectory_from_dict(value: object) -> Trajectory:
    """Validate and reconstruct a trajectory from a decoded JSON object."""

    record = _mapping(value, "trajectory record")
    schema_version = _integer(record.get("schema_version"), "schema_version")
    if schema_version != SCHEMA_VERSION:
        raise TrajectoryFormatError(f"unsupported schema_version: {schema_version}")
    if _string(record.get("record_type"), "record_type") != "trajectory":
        raise TrajectoryFormatError("record_type must be trajectory")

    example = _example_from_dict(record.get("example"))
    if _boolean(record.get("synthetic"), "synthetic") != example.synthetic:
        raise TrajectoryFormatError("top-level synthetic marker must match example")
    if (
        _boolean(record.get("benchmark_eligible"), "benchmark_eligible")
        != example.benchmark_eligible
    ):
        raise TrajectoryFormatError("top-level benchmark_eligible marker must match example")

    steps = tuple(_step_from_dict(step) for step in _sequence(record.get("steps"), "steps"))
    return Trajectory(
        example=example,
        initial_state=_state_from_dict(record.get("initial_state")),
        steps=steps,
        final_state=_state_from_dict(record.get("final_state")),
        metrics=_metrics_from_dict(record.get("metrics")),
    )


def trajectory_from_json(line: str) -> Trajectory:
    """Decode one JSONL line into a trajectory."""

    try:
        value: object = json.loads(line)
    except json.JSONDecodeError as error:
        raise TrajectoryFormatError(f"invalid trajectory JSON: {error}") from error
    return trajectory_from_dict(value)


def write_trajectory_jsonl(path: str | Path, trajectory: Trajectory) -> Path:
    """Write one trajectory as a UTF-8 JSONL file, creating parent directories."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{trajectory_as_json(trajectory)}\n", encoding="utf-8")
    return output_path


def read_trajectory_jsonl(path: str | Path) -> tuple[Trajectory, ...]:
    """Read every non-empty trajectory line from a UTF-8 JSONL file."""

    input_path = Path(path)
    trajectories: list[Trajectory] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            trajectories.append(trajectory_from_json(line))
        except TrajectoryFormatError as error:
            raise TrajectoryFormatError(f"{input_path}:{line_number}: {error}") from error
    return tuple(trajectories)
