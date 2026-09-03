import json
from pathlib import Path

import pytest

from deep_research_rl.core.serialization import (
    TrajectoryFormatError,
    read_trajectory_jsonl,
    trajectory_as_json,
    trajectory_from_json,
)
from deep_research_rl.core.smoke import run_synthetic_smoke


def test_trajectory_round_trips_through_json_and_jsonl(tmp_path: Path) -> None:
    output_path = tmp_path / "trajectory.jsonl"
    trajectory = run_synthetic_smoke(output_path)

    serialized = trajectory_as_json(trajectory)
    record: object = json.loads(serialized)
    assert isinstance(record, dict)
    assert record["synthetic"] is True
    assert record["benchmark_eligible"] is False
    assert trajectory_from_json(serialized) == trajectory
    assert read_trajectory_jsonl(output_path) == (trajectory,)


def test_rejects_inconsistent_top_level_provenance_marker(tmp_path: Path) -> None:
    trajectory = run_synthetic_smoke(tmp_path / "trajectory.jsonl")
    record: object = json.loads(trajectory_as_json(trajectory))
    assert isinstance(record, dict)
    record["synthetic"] = False

    with pytest.raises(TrajectoryFormatError, match="synthetic marker"):
        trajectory_from_json(json.dumps(record))


def test_schema_v1_reader_accepts_pre_score_search_documents(tmp_path: Path) -> None:
    trajectory = run_synthetic_smoke(tmp_path / "trajectory.jsonl")
    record: object = json.loads(trajectory_as_json(trajectory))
    assert isinstance(record, dict)
    steps = record["steps"]
    assert isinstance(steps, list)
    for step in steps:
        assert isinstance(step, dict)
        observation = step["observation"]
        assert isinstance(observation, dict)
        documents = observation["documents"]
        assert isinstance(documents, list)
        for document in documents:
            assert isinstance(document, dict)
            document.pop("rank")
            document.pop("score")

    restored = trajectory_from_json(json.dumps(record))

    assert [result.rank for result in restored.steps[0].observation.documents] == [1]
    assert [result.score for result in restored.steps[0].observation.documents] == [0.0]
