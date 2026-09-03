from pathlib import Path

import pytest

from deep_research_rl.cli import build_parser, main

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
    assert '"checkpoint": "Qwen/Qwen3-4B-Instruct-2507"' in output


def test_agent_rollout_parser_has_frozen_baseline_defaults(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "agent",
            "rollout",
            "--examples",
            str(tmp_path / "examples.jsonl"),
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--index-dir",
            str(tmp_path / "index"),
            "--output",
            str(tmp_path / "rollout.jsonl"),
        ]
    )

    assert args.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert args.model_revision == "cdbee75f17c01a7cc42f958dc650907174af0554"
    assert args.max_searches == 5
    assert args.max_steps == 8
    assert args.do_sample is False


def test_agent_rollout_rejects_budget_above_frozen_limit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "agent",
                "rollout",
                "--examples",
                str(tmp_path / "missing-examples.jsonl"),
                "--corpus",
                str(tmp_path / "missing-corpus.jsonl"),
                "--index-dir",
                str(tmp_path / "missing-index"),
                "--output",
                str(tmp_path / "rollout.jsonl"),
                "--max-searches",
                "6",
            ]
        )

    assert "baseline max_searches must be between 0 and 5" in capsys.readouterr().err


def test_evaluation_policy_entry_points_expose_frozen_controls(tmp_path: Path) -> None:
    no_search = build_parser().parse_args(
        [
            "evaluation",
            "no-search",
            "--run-id",
            "debug-no-search",
            "--examples",
            str(tmp_path / "examples.jsonl"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    prompted = build_parser().parse_args(
        [
            "evaluation",
            "prompted-agent",
            "--run-id",
            "debug-prompted",
            "--examples",
            str(tmp_path / "examples.jsonl"),
            "--corpus",
            str(tmp_path / "corpus.jsonl"),
            "--index-dir",
            str(tmp_path / "index"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )

    assert no_search.model_name == prompted.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert no_search.model_revision == prompted.model_revision
    assert no_search.max_examples is None
    assert prompted.final_validation is False


def test_rl_evaluation_requires_an_explicit_trained_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="2"):
        build_parser().parse_args(
            [
                "evaluation",
                "rl-agent",
                "--run-id",
                "debug-rl",
                "--examples",
                str(tmp_path / "examples.jsonl"),
                "--corpus",
                str(tmp_path / "corpus.jsonl"),
                "--index-dir",
                str(tmp_path / "index"),
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )


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
