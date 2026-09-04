import importlib
import json
from pathlib import Path

import pytest

from deep_research_rl.training.episode import AGENT_FLOW_NAME
from deep_research_rl.training.runtime import (
    AGENT_R1_REVISION,
    CONTAINER_IMAGE,
    MODEL_REVISION,
    VERL_REVISION,
    VERL_VERSION,
    TrainingPreflightError,
    TrainingRuntimeError,
    TrainingRuntimePaths,
    build_training_launch_plan,
    load_training_runtime,
    run_training_launch,
    run_training_preflight,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = REPOSITORY_ROOT / "configs" / "training" / "agentr1-verl-grpo.toml"


def _paths(tmp_path: Path) -> TrainingRuntimePaths:
    return TrainingRuntimePaths(
        agent_r1_root=tmp_path / "Agent-R1",
        verl_root=tmp_path / "verl",
        model_path=tmp_path / "models" / MODEL_REVISION,
        training_data_dir=tmp_path / "training-data",
        corpus_path=tmp_path / "corpus.jsonl",
        index_dir=tmp_path / "index",
        flow_config_path=REPOSITORY_ROOT / "configs" / "training" / "agentr1-flow.yaml",
    )


def test_training_config_and_container_lock_exact_upstream_revisions() -> None:
    config = load_training_runtime(TRAINING_CONFIG)

    assert config.section("upstream")["agent_r1"]["revision"] == AGENT_R1_REVISION
    assert config.section("upstream")["verl"] == {
        "repository": "https://github.com/verl-project/verl.git",
        "revision": VERL_REVISION,
        "version": VERL_VERSION,
    }
    assert config.section("container")["image"] == CONTAINER_IMAGE
    assert config.section("environment")["agent_flow"] == AGENT_FLOW_NAME
    assert config.section("environment")["retrieval_backend"] == "faiss_bge"
    dockerfile = (REPOSITORY_ROOT / "docker" / "training.Dockerfile").read_text(encoding="utf-8")
    assert f"FROM {CONTAINER_IMAGE}" in dockerfile
    assert AGENT_R1_REVISION in dockerfile
    assert VERL_REVISION in dockerfile


def test_training_config_rejects_baseline_drift(tmp_path: Path) -> None:
    changed = TRAINING_CONFIG.read_text(encoding="utf-8").replace(
        "max_searches = 5", "max_searches = 4"
    )
    config_path = tmp_path / "drift.toml"
    config_path.write_text(changed, encoding="utf-8")

    with pytest.raises(TrainingRuntimeError, match=r"environment\.max_searches"):
        load_training_runtime(config_path)


def test_sanity_launch_plan_freezes_update_reward_rollout_and_evaluation_controls(
    tmp_path: Path,
) -> None:
    plan = build_training_launch_plan(
        load_training_runtime(TRAINING_CONFIG),
        _paths(tmp_path),
        phase="sanity",
        run_dir=tmp_path / "run",
        n_gpus=1,
        min_gpu_memory_gib=40,
    )
    arguments = set(plan.command[3:])

    assert plan.start_step == 0
    assert plan.target_step == 1
    assert "algorithm.adv_estimator=grpo" in arguments
    assert "algorithm.use_kl_in_reward=false" in arguments
    assert "actor_rollout_ref.actor.use_kl_loss=false" in arguments
    assert "actor_rollout_ref.rollout.mode=async" in arguments
    assert "actor_rollout_ref.rollout.calculate_log_probs=true" in arguments
    assert "actor_rollout_ref.rollout.n=2" in arguments
    assert f"actor_rollout_ref.rollout.agent.default_agent_flow={AGENT_FLOW_NAME}" in arguments
    assert "trainer.val_before_train=true" in arguments
    assert "trainer.save_freq=1" in arguments
    assert "trainer.test_freq=1" in arguments
    assert "trainer.total_training_steps=1" in arguments
    assert "trainer.resume_mode=disable" in arguments
    assert plan.environment["DEEP_RESEARCH_RL_RETRIEVAL_DEVICE"] == "cpu"
    assert not any("force_first_search" in argument for argument in arguments)
    assert not any("max_parallel_calls" in argument for argument in arguments)


def test_resume_plan_requires_the_exact_first_update_checkpoint(tmp_path: Path) -> None:
    config = load_training_runtime(TRAINING_CONFIG)
    paths = _paths(tmp_path)
    checkpoint = tmp_path / "run" / "checkpoints" / "global_step_1"

    plan = build_training_launch_plan(
        config,
        paths,
        phase="resume",
        run_dir=tmp_path / "run",
        n_gpus=1,
        min_gpu_memory_gib=40,
        resume_from=checkpoint,
    )

    assert (plan.start_step, plan.target_step) == (1, 2)
    assert "trainer.resume_mode=resume_path" in plan.command
    assert f"trainer.resume_from_path={checkpoint.resolve()}" in plan.command
    assert "trainer.total_training_steps=2" in plan.command
    with pytest.raises(TrainingRuntimeError, match="global_step_1"):
        build_training_launch_plan(
            config,
            paths,
            phase="resume",
            run_dir=tmp_path / "run",
            n_gpus=1,
            min_gpu_memory_gib=40,
            resume_from=tmp_path / "global_step_2",
        )


def test_dry_run_writes_resolved_config_and_manifest_without_claiming_execution(
    tmp_path: Path,
) -> None:
    config = load_training_runtime(TRAINING_CONFIG)
    plan = build_training_launch_plan(
        config,
        _paths(tmp_path),
        phase="sanity",
        run_dir=tmp_path / "run",
        n_gpus=1,
        min_gpu_memory_gib=40,
    )

    manifest_path = run_training_launch(
        config,
        plan,
        repository_root=REPOSITORY_ROOT,
        dry_run=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    resolved = json.loads((plan.phase_dir / "resolved-config.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "dry_run"
    assert manifest["target_step"] == 1
    assert resolved["preflight"] == {"status": "not_run_for_dry_run"}
    assert resolved["launch"]["command"] == list(plan.command)


def test_failed_preflight_report_remains_json_serializable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEP_RESEARCH_RL_CONTAINER_IMAGE", raising=False)
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError(f"missing test dependency: {name}")),
    )
    monkeypatch.setattr(
        "deep_research_rl.training.runtime._gpu_inventory",
        lambda: (_ for _ in ()).throw(TrainingRuntimeError("no test GPU")),
    )

    with pytest.raises(TrainingPreflightError) as caught:
        run_training_preflight(
            load_training_runtime(TRAINING_CONFIG),
            _paths(tmp_path),
            n_gpus=1,
            min_gpu_memory_gib=1,
        )

    encoded = json.dumps(caught.value.report)
    assert '"status": "failed"' in encoded
    checks = caught.value.report["checks"]
    assert isinstance(checks, list)
    assert any(check["status"] == "failed" for check in checks if isinstance(check, dict))
