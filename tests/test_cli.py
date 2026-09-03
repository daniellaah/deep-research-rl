from pathlib import Path

import pytest

from deep_research_rl.cli import main

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BASELINE_CONFIG = REPOSITORY_ROOT / "configs" / "baseline.toml"


def test_top_level_help(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit, match="0"):
        main(["--help"])

    assert "reproducible multi-step retrieval research" in capsys.readouterr().out


def test_validate_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "validate", str(BASELINE_CONFIG)]) == 0

    output = capsys.readouterr().out
    assert "valid defaults configuration" in output


def test_show_baseline(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["config", "show", str(BASELINE_CONFIG)]) == 0

    output = capsys.readouterr().out
    assert '"config_kind": "defaults"' in output
    assert '"max_policy_searches": 5' in output


def test_synthetic_smoke_writes_reviewable_trajectory(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_path = tmp_path / "smoke" / "trajectory.jsonl"

    assert main(["smoke", "--output", str(output_path)]) == 0

    output = capsys.readouterr().out
    assert "synthetic non-benchmark smoke passed" in output
    assert "exact_match=1.0" in output
    assert output_path.is_file()
    assert '"synthetic": true' in output_path.read_text(encoding="utf-8")
    assert '"benchmark_eligible": false' in output_path.read_text(encoding="utf-8")
