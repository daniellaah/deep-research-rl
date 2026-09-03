import json
from dataclasses import replace
from itertools import islice
from pathlib import Path

import pytest

from deep_research_rl.cli import main
from deep_research_rl.data.hotpotqa import (
    DataPipelineError,
    build_hotpotqa,
    iter_json_array,
    verify_hotpotqa_build,
)
from deep_research_rl.data.models import hotpotqa_example_from_dict
from deep_research_rl.data.source import load_source_config, sha256_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = REPOSITORY_ROOT / "tests" / "fixtures" / "hotpotqa"
FIXTURE_CONFIG = FIXTURE_ROOT / "source.json"
FIXTURE_RAW = FIXTURE_ROOT / "raw"
OFFICIAL_CONFIG = REPOSITORY_ROOT / "configs" / "data" / "hotpotqa-distractor-v1.1.json"
REFERENCE_MANIFEST_ROOT = REPOSITORY_ROOT / "configs" / "data" / "manifests"


def _jsonl(path: Path) -> list[dict[str, object]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        value: object = json.loads(line)
        assert isinstance(value, dict)
        assert all(isinstance(key, str) for key in value)
        records.append({key: item for key, item in value.items() if isinstance(key, str)})
    return records


def _artifact_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_official_source_descriptor_pins_data_and_agent_adapter() -> None:
    config = load_source_config(OFFICIAL_CONFIG)

    assert config.dataset == "hotpot_qa"
    assert config.variant == "distractor"
    assert config.source_repository_revision == "3635853403a8735609ee997664e1528f4480762a"
    assert config.agent_r1_revision == "b124aa46534cbf2fb8bc8af11405774984c42ac7"
    assert [source_split.records for source_split in config.splits] == [90447, 7405]
    assert [source_split.sha256 for source_split in config.splits] == [
        "26650cf50234ef5fb2e664ed70bbecdfd87815e6bffc257e068efea5cf7cd316",
        "4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea",
    ]


def test_reference_manifests_match_locked_source_and_expected_split_counts() -> None:
    config = load_source_config(OFFICIAL_CONFIG)
    for filename, expected_mode, expected_counts in (
        (
            "hotpotqa-distractor-v1.1-debug-5.json",
            "debug",
            (5, 5),
        ),
        (
            "hotpotqa-distractor-v1.1-full.json",
            "full",
            (90447, 7405),
        ),
    ):
        manifest: object = json.loads(
            (REFERENCE_MANIFEST_ROOT / filename).read_text(encoding="utf-8")
        )
        assert isinstance(manifest, dict)
        assert manifest["build"]["mode"] == expected_mode
        assert (
            manifest["counts"]["train_examples"],
            manifest["counts"]["validation_examples"],
        ) == expected_counts
        assert [source_file["sha256"] for source_file in manifest["source"]["files"]] == [
            source_split.sha256 for source_split in config.splits
        ]
        assert manifest["compatibility"]["agent_r1"]["revision"] == config.agent_r1_revision


def test_full_fixture_build_is_deterministic_and_preserves_all_labels(tmp_path: Path) -> None:
    config = load_source_config(FIXTURE_CONFIG)
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"

    first = build_hotpotqa(config, FIXTURE_RAW, first_root)
    second = build_hotpotqa(config, FIXTURE_RAW, second_root)

    assert first.build_mode == "full"
    assert first.train_examples == second.train_examples == 3
    assert first.validation_examples == second.validation_examples == 3
    assert first.corpus_documents == second.corpus_documents == 11
    assert _artifact_bytes(first_root) == _artifact_bytes(second_root)

    train_records = _jsonl(first_root / "examples" / "train.jsonl")
    validation_records = _jsonl(first_root / "examples" / "validation.jsonl")
    examples = [hotpotqa_example_from_dict(record) for record in train_records + validation_records]
    assert len(examples) == 6
    assert {example.example_id for example in examples[:5]} == {
        "fixture-train-001",
        "fixture-train-002",
        "fixture-train-003",
        "fixture-validation-001",
        "fixture-validation-002",
    }
    assert examples[0].answers == ("Northbridge",)
    assert [(fact.title, fact.sentence_index) for fact in examples[0].supporting_facts] == [
        ("Mira Chen", 0),
        ("Aster Observatory", 0),
    ]
    assert examples[0].supporting_titles == ("Mira Chen", "Aster Observatory")
    assert examples[0].prompt == [
        {"content": "Which city contains the observatory where Mira Chen works?", "role": "user"}
    ]

    core_example = examples[0].to_core_example()
    assert core_example.example_id == examples[0].example_id
    assert core_example.answers == examples[0].answers
    assert core_example.supporting_document_ids == examples[0].supporting_document_ids

    manifest = verify_hotpotqa_build(first_root)
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    assert counts == {
        "context_document_occurrences": 18,
        "corpus_documents": 11,
        "deduplicated_context_occurrences": 7,
        "supporting_fact_reference_issues": 0,
        "train_examples": 3,
        "validation_examples": 3,
    }
    corpus_ids = [
        document_id
        for record in _jsonl(first_root / "corpus.jsonl")
        if isinstance(document_id := record["document_id"], str)
    ]
    assert corpus_ids == sorted(set(corpus_ids))


def test_agent_r1_rows_match_pinned_logical_schema_without_label_leakage(
    tmp_path: Path,
) -> None:
    config = load_source_config(FIXTURE_CONFIG)
    build_hotpotqa(config, FIXTURE_RAW, tmp_path / "build")
    canonical = hotpotqa_example_from_dict(
        _jsonl(tmp_path / "build" / "examples" / "train.jsonl")[0]
    )
    agent_row = _jsonl(tmp_path / "build" / "agent_r1" / "train.jsonl")[0]

    assert set(agent_row) == {"data_source", "prompt", "reward_model", "extra_info"}
    assert agent_row["data_source"] == "hotpotqa_distractor"
    assert agent_row["prompt"] == [{"content": canonical.question, "role": "user"}]
    assert agent_row["reward_model"] == {
        "ground_truth": canonical.answers[0],
        "style": "rule",
    }
    assert isinstance(agent_row["extra_info"], dict)
    assert agent_row["extra_info"]["supporting_facts"] == [
        fact.to_dict() for fact in canonical.supporting_facts
    ]
    assert canonical.answers[0] not in json.dumps(agent_row["prompt"])


def test_cli_builds_and_verifies_bounded_debug_subset(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output_dir = tmp_path / "debug"

    assert (
        main(
            [
                "data",
                "download",
                "--source-config",
                str(FIXTURE_CONFIG),
                "--raw-dir",
                str(FIXTURE_RAW),
            ]
        )
        == 0
    )
    assert "verified 2 raw source files" in capsys.readouterr().out
    assert (
        main(
            [
                "data",
                "prepare",
                "--source-config",
                str(FIXTURE_CONFIG),
                "--raw-dir",
                str(FIXTURE_RAW),
                "--output-dir",
                str(output_dir),
                "--max-train",
                "2",
                "--max-validation",
                "1",
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert "debug HotpotQA build verified: train=2, validation=1" in output

    manifest = verify_hotpotqa_build(output_dir)
    assert manifest["build"] == {
        "limits": {"train": 2, "validation": 1},
        "mode": "debug",
    }
    assert main(["data", "verify", "--output-dir", str(output_dir)]) == 0
    assert "HotpotQA build verified: train=2, validation=1" in capsys.readouterr().out


def test_train_validation_overlap_is_rejected(tmp_path: Path) -> None:
    config = load_source_config(FIXTURE_CONFIG)
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    train_path = raw_dir / "train.json"
    validation_path = raw_dir / "validation.json"
    train_path.write_bytes((FIXTURE_RAW / "train.json").read_bytes())
    validation_value: object = json.loads(
        (FIXTURE_RAW / "validation.json").read_text(encoding="utf-8")
    )
    assert isinstance(validation_value, list)
    assert isinstance(validation_value[0], dict)
    validation_value[0]["_id"] = "fixture-train-001"
    validation_path.write_text(
        json.dumps(validation_value, ensure_ascii=False),
        encoding="utf-8",
    )
    changed_validation = replace(
        config.splits[1],
        bytes=validation_path.stat().st_size,
        sha256=sha256_file(validation_path),
    )
    changed_config = replace(config, splits=(config.splits[0], changed_validation))

    with pytest.raises(DataPipelineError, match="train/validation example id overlap"):
        build_hotpotqa(changed_config, raw_dir, tmp_path / "output")


def test_manifest_verification_detects_artifact_tampering(tmp_path: Path) -> None:
    config = load_source_config(FIXTURE_CONFIG)
    output_dir = tmp_path / "build"
    build_hotpotqa(config, FIXTURE_RAW, output_dir)
    corpus_path = output_dir / "corpus.jsonl"
    corpus_path.write_text(corpus_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")

    with pytest.raises(DataPipelineError, match="output size mismatch"):
        verify_hotpotqa_build(output_dir)


def test_manifest_verification_detects_false_counts(tmp_path: Path) -> None:
    config = load_source_config(FIXTURE_CONFIG)
    output_dir = tmp_path / "build"
    build_hotpotqa(config, FIXTURE_RAW, output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest: object = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert isinstance(manifest, dict)
    assert isinstance(manifest["counts"], dict)
    manifest["counts"]["corpus_documents"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(DataPipelineError, match="manifest counts"):
        verify_hotpotqa_build(output_dir)


def test_streaming_reader_can_stop_before_invalid_suffix(tmp_path: Path) -> None:
    source_path = tmp_path / "bounded.json"
    source_path.write_text('[{"_id": "first"}, INVALID]', encoding="utf-8")

    prefix = list(islice(iter_json_array(source_path, chunk_size=4), 1))

    assert prefix == [{"_id": "first"}]
