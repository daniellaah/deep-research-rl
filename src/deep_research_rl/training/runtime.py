"""Pinned runtime validation, launch planning, and run artifacts for GPU training."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import re
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from deep_research_rl.config import ConfigError, load_config
from deep_research_rl.evaluation.artifacts import capture_code_state
from deep_research_rl.retrieval import verify_index
from deep_research_rl.retrieval.index import (
    manifest_backend,
    sha256_file,
    stable_json_bytes,
    write_bytes_atomic,
)
from deep_research_rl.training.data import verify_training_data
from deep_research_rl.training.episode import AGENT_FLOW_NAME

AGENT_R1_REPOSITORY = "https://github.com/AgentR1/Agent-R1.git"
AGENT_R1_REVISION = "b124aa46534cbf2fb8bc8af11405774984c42ac7"
VERL_REPOSITORY = "https://github.com/verl-project/verl.git"
VERL_VERSION = "0.7.0"
VERL_REVISION = "f9c855f7cf04d603c9546bc01776c74806a879c1"
VERL_SOURCE_VERSION = "0.7.0.dev"
CONTAINER_IMAGE = (
    "docker.io/verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d"
)
MODEL_CHECKPOINT = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
TRAINING_CONFIG_SCHEMA_VERSION = 1

LaunchPhase = Literal["sanity", "resume"]


class TrainingRuntimeError(ValueError):
    """Raised when a training runtime, plan, or completed artifact is invalid."""


class TrainingPreflightError(TrainingRuntimeError):
    """Raised when one or more mandatory CUDA runtime checks fail."""

    def __init__(self, report: dict[str, object]) -> None:
        self.report = report
        checks_value = report.get("checks", [])
        checks = checks_value if isinstance(checks_value, list) else []
        failed = [
            str(check.get("name"))
            for check in checks
            if isinstance(check, dict) and check.get("status") == "failed"
        ]
        super().__init__(f"training preflight failed: {', '.join(failed)}")


@dataclass(frozen=True, slots=True)
class TrainingRuntimeConfig:
    """Validated immutable defaults for the baseline training adapter."""

    path: Path
    values: dict[str, Any]

    def section(self, name: str) -> dict[str, Any]:
        value = self.values.get(name)
        if not isinstance(value, dict):
            raise TrainingRuntimeError(f"training config section {name!r} must be an object")
        return value


@dataclass(frozen=True, slots=True)
class TrainingRuntimePaths:
    """External paths mounted into the pinned training container."""

    agent_r1_root: Path
    verl_root: Path
    model_path: Path
    training_data_dir: Path
    corpus_path: Path
    index_dir: Path
    flow_config_path: Path


@dataclass(frozen=True, slots=True)
class TrainingLaunchPlan:
    """Exact environment and argument vector for one sanity or resume phase."""

    phase: LaunchPhase
    command: tuple[str, ...]
    environment: dict[str, str]
    paths: TrainingRuntimePaths
    run_dir: Path
    phase_dir: Path
    checkpoint_dir: Path
    rollout_dir: Path
    validation_dir: Path
    start_step: int
    target_step: int
    seed: int
    n_gpus: int
    min_gpu_memory_gib: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible resolved launch plan."""

        return {
            "checkpoint_dir": str(self.checkpoint_dir),
            "command": list(self.command),
            "environment": dict(sorted(self.environment.items())),
            "min_gpu_memory_gib": self.min_gpu_memory_gib,
            "n_gpus": self.n_gpus,
            "phase": self.phase,
            "phase_dir": str(self.phase_dir),
            "rollout_dir": str(self.rollout_dir),
            "run_dir": str(self.run_dir),
            "seed": self.seed,
            "start_step": self.start_step,
            "target_step": self.target_step,
            "validation_dir": str(self.validation_dir),
        }


def _nested(config: dict[str, Any], first: str, second: str) -> dict[str, Any]:
    outer = config.get(first)
    if not isinstance(outer, dict):
        raise TrainingRuntimeError(f"training config section {first!r} must be an object")
    inner = outer.get(second)
    if not isinstance(inner, dict):
        raise TrainingRuntimeError(f"training config section {first}.{second} must be an object")
    return inner


def load_training_runtime(path: str | Path) -> TrainingRuntimeConfig:
    """Load the training defaults and reject any baseline or revision drift."""

    try:
        values = load_config(path)
    except ConfigError as error:
        raise TrainingRuntimeError(str(error)) from error
    if values.get("schema_version") != TRAINING_CONFIG_SCHEMA_VERSION:
        raise TrainingRuntimeError("unsupported training config schema_version")
    expected_values: tuple[tuple[str, object, object], ...] = (
        (
            "upstream.agent_r1.repository",
            _nested(values, "upstream", "agent_r1").get("repository"),
            AGENT_R1_REPOSITORY,
        ),
        (
            "upstream.agent_r1.revision",
            _nested(values, "upstream", "agent_r1").get("revision"),
            AGENT_R1_REVISION,
        ),
        (
            "upstream.verl.repository",
            _nested(values, "upstream", "verl").get("repository"),
            VERL_REPOSITORY,
        ),
        ("upstream.verl.version", _nested(values, "upstream", "verl").get("version"), VERL_VERSION),
        (
            "upstream.verl.revision",
            _nested(values, "upstream", "verl").get("revision"),
            VERL_REVISION,
        ),
        ("container.image", values.get("container", {}).get("image"), CONTAINER_IMAGE),
        ("container.platform", values.get("container", {}).get("platform"), "linux/amd64"),
        ("container.cuda_minimum", values.get("container", {}).get("cuda_minimum"), "12.8"),
        ("model.checkpoint", values.get("model", {}).get("checkpoint"), MODEL_CHECKPOINT),
        ("model.revision", values.get("model", {}).get("revision"), MODEL_REVISION),
        (
            "environment.agent_flow",
            values.get("environment", {}).get("agent_flow"),
            AGENT_FLOW_NAME,
        ),
        (
            "environment.action_format",
            values.get("environment", {}).get("action_format"),
            "strict_search_answer_v1",
        ),
        (
            "environment.context_policy",
            values.get("environment", {}).get("context_policy"),
            "append_only",
        ),
        ("environment.max_searches", values.get("environment", {}).get("max_searches"), 5),
        ("environment.max_steps", values.get("environment", {}).get("max_steps"), 8),
        (
            "environment.max_prompt_tokens",
            values.get("environment", {}).get("max_prompt_tokens"),
            8192,
        ),
        (
            "environment.max_response_tokens",
            values.get("environment", {}).get("max_response_tokens"),
            96,
        ),
        (
            "environment.retrieval_backend",
            values.get("environment", {}).get("retrieval_backend"),
            "faiss_bge",
        ),
        ("environment.retrieval_top_k", values.get("environment", {}).get("retrieval_top_k"), 3),
        ("reward.answer", values.get("reward", {}).get("answer"), "normalized_exact_match"),
        ("reward.intermediate", values.get("reward", {}).get("intermediate"), 0.0),
        ("reward.search_cost", values.get("reward", {}).get("search_cost"), 0.0),
        ("reward.token_cost", values.get("reward", {}).get("token_cost"), 0.0),
        ("reward.credit", values.get("reward", {}).get("credit"), "terminal_only"),
        ("grpo.algorithm", values.get("grpo", {}).get("algorithm"), "grpo"),
        ("grpo.rollouts_per_prompt", values.get("grpo", {}).get("rollouts_per_prompt"), 2),
        ("grpo.kl_in_reward", values.get("grpo", {}).get("kl_in_reward"), False),
        ("grpo.kl_in_loss", values.get("grpo", {}).get("kl_in_loss"), False),
        ("sanity.target_step", values.get("sanity", {}).get("target_step"), 1),
        ("sanity.resume_target_step", values.get("sanity", {}).get("resume_target_step"), 2),
        (
            "evaluation.validation_before_train",
            values.get("evaluation", {}).get("validation_before_train"),
            True,
        ),
        (
            "evaluation.validation_after_each_step",
            values.get("evaluation", {}).get("validation_after_each_step"),
            True,
        ),
        ("evaluation.decoding", values.get("evaluation", {}).get("decoding"), "greedy"),
        ("checkpoint.save_every_steps", values.get("checkpoint", {}).get("save_every_steps"), 1),
        (
            "checkpoint.resume_mode",
            values.get("checkpoint", {}).get("resume_mode"),
            "explicit_path",
        ),
    )
    for field, actual, expected in expected_values:
        if actual != expected:
            raise TrainingRuntimeError(
                f"{field} must remain {expected!r} for the baseline; received {actual!r}"
            )
    return TrainingRuntimeConfig(path=Path(path).resolve(), values=values)


def _checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    if match is None:
        raise TrainingRuntimeError("resume checkpoint path must end with global_step_<N>")
    return int(match.group(1))


def build_training_launch_plan(
    config: TrainingRuntimeConfig,
    paths: TrainingRuntimePaths,
    *,
    phase: LaunchPhase,
    run_dir: str | Path,
    n_gpus: int,
    min_gpu_memory_gib: int,
    seed: int = 0,
    resume_from: str | Path | None = None,
) -> TrainingLaunchPlan:
    """Build the exact Agent-R1 Hydra command for a one-step update or resume."""

    if n_gpus < 1:
        raise TrainingRuntimeError("n_gpus must be at least 1")
    if min_gpu_memory_gib < 1:
        raise TrainingRuntimeError("min_gpu_memory_gib must be at least 1")
    if seed < 0:
        raise TrainingRuntimeError("seed must not be negative")
    sanity = config.section("sanity")
    grpo = config.section("grpo")
    environment_config = config.section("environment")
    root = Path(run_dir).resolve()
    phase_dir = root / phase
    checkpoint_dir = root / "checkpoints"
    rollout_dir = phase_dir / "rollouts"
    validation_dir = phase_dir / "validation"

    if phase == "sanity":
        if resume_from is not None:
            raise TrainingRuntimeError("sanity phase cannot accept resume_from")
        start_step = 0
        target_step = int(sanity["target_step"])
        resume_mode = "disable"
        resume_override = "trainer.resume_from_path=null"
    else:
        if resume_from is None:
            raise TrainingRuntimeError("resume phase requires resume_from")
        resume_path = Path(resume_from).resolve()
        start_step = _checkpoint_step(resume_path)
        if start_step != int(sanity["target_step"]):
            expected_name = f"global_step_{sanity['target_step']}"
            raise TrainingRuntimeError(
                f"resume sanity expects {expected_name}, received {resume_path.name}"
            )
        target_step = int(sanity["resume_target_step"])
        resume_mode = "resume_path"
        resume_override = f"trainer.resume_from_path={resume_path}"

    train_path = paths.training_data_dir.resolve() / "train.parquet"
    validation_path = paths.training_data_dir.resolve() / "validation.parquet"
    command = (
        sys.executable,
        "-m",
        "agent_r1.trainer.main_agent_ppo",
        "algorithm.adv_estimator=grpo",
        f"algorithm.norm_adv_by_std_in_grpo={str(bool(grpo['normalize_advantage_by_std'])).lower()}",
        "algorithm.use_kl_in_reward=false",
        "algorithm.gamma=1.0",
        f"data.train_files={train_path}",
        f"data.val_files={validation_path}",
        f"data.train_batch_size={sanity['train_batch_size']}",
        f"data.val_batch_size={sanity['validation_batch_size']}",
        f"data.train_max_samples={sanity['train_examples']}",
        f"data.val_max_samples={sanity['validation_examples']}",
        f"data.max_prompt_length={environment_config['max_prompt_tokens']}",
        f"data.max_response_length={environment_config['max_response_tokens']}",
        "data.filter_overlong_prompts=false",
        "data.truncation=error",
        "data.return_raw_chat=true",
        "data.shuffle=false",
        "data.validation_shuffle=false",
        "data.dataloader_num_workers=0",
        f"data.seed={seed}",
        f"actor_rollout_ref.model.path={paths.model_path.resolve()}",
        "actor_rollout_ref.model.use_remove_padding=true",
        "actor_rollout_ref.model.enable_gradient_checkpointing=true",
        f"actor_rollout_ref.actor.optim.lr={grpo['actor_learning_rate']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={sanity['ppo_mini_batch_size']}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={sanity['micro_batch_size_per_gpu']}",
        "actor_rollout_ref.actor.ppo_epochs=1",
        "actor_rollout_ref.actor.use_kl_loss=false",
        f"actor_rollout_ref.actor.policy_loss.loss_mode={grpo['policy_loss']}",
        f"actor_rollout_ref.actor.loss_agg_mode={grpo['loss_aggregation']}",
        "actor_rollout_ref.actor.fsdp_config.param_offload=true",
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=true",
        "actor_rollout_ref.rollout.mode=async",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.calculate_log_probs=true",
        f"actor_rollout_ref.rollout.n={grpo['rollouts_per_prompt']}",
        "actor_rollout_ref.rollout.temperature=1.0",
        "actor_rollout_ref.rollout.top_p=1.0",
        "actor_rollout_ref.rollout.do_sample=true",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={sanity['rollout_gpu_memory_utilization']}",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={sanity['micro_batch_size_per_gpu']}",
        "actor_rollout_ref.rollout.val_kwargs.temperature=0.0",
        "actor_rollout_ref.rollout.val_kwargs.top_p=1.0",
        "actor_rollout_ref.rollout.val_kwargs.do_sample=false",
        "actor_rollout_ref.rollout.val_kwargs.n=1",
        f"actor_rollout_ref.rollout.agent.agent_flow_config_path={paths.flow_config_path.resolve()}",
        f"actor_rollout_ref.rollout.agent.num_workers={sanity['agent_workers']}",
        f"actor_rollout_ref.rollout.agent.default_agent_flow={AGENT_FLOW_NAME}",
        "actor_rollout_ref.ref.fsdp_config.param_offload=true",
        "critic.enable=false",
        "reward_model.enable=false",
        "custom_reward_function.path=null",
        "trainer.logger=[console]",
        "trainer.project_name=DeepResearchRL",
        "trainer.experiment_name=agentr1_grpo_sanity",
        f"trainer.n_gpus_per_node={n_gpus}",
        "trainer.nnodes=1",
        "trainer.val_before_train=true",
        "trainer.val_only=false",
        "trainer.save_freq=1",
        "trainer.test_freq=1",
        "trainer.max_actor_ckpt_to_keep=2",
        "trainer.total_epochs=1",
        f"trainer.total_training_steps={target_step}",
        f"trainer.default_local_dir={checkpoint_dir}",
        f"trainer.rollout_data_dir={rollout_dir}",
        f"trainer.validation_data_dir={validation_dir}",
        f"trainer.resume_mode={resume_mode}",
        resume_override,
    )
    environment = {
        "DEEP_RESEARCH_RL_CORPUS_PATH": str(paths.corpus_path.resolve()),
        "DEEP_RESEARCH_RL_INDEX_DIR": str(paths.index_dir.resolve()),
        "DEEP_RESEARCH_RL_RETRIEVAL_DEVICE": "cpu",
        "HYDRA_FULL_ERROR": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_V1": "1",
    }
    return TrainingLaunchPlan(
        phase=phase,
        command=command,
        environment=environment,
        paths=paths,
        run_dir=root,
        phase_dir=phase_dir,
        checkpoint_dir=checkpoint_dir,
        rollout_dir=rollout_dir,
        validation_dir=validation_dir,
        start_step=start_step,
        target_step=target_step,
        seed=seed,
        n_gpus=n_gpus,
        min_gpu_memory_gib=min_gpu_memory_gib,
    )


def _git_state(path: Path) -> tuple[str, bool]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise TrainingRuntimeError(f"could not inspect Git checkout {path}: {error}") from error
    return revision, bool(status)


def _cuda_version_tuple(value: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", value)
    if match is None:
        raise TrainingRuntimeError(f"invalid CUDA version: {value}")
    return int(match.group(1)), int(match.group(2))


def _gpu_inventory() -> list[dict[str, object]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise TrainingRuntimeError(f"nvidia-smi is unavailable or failed: {error}") from error
    inventory = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise TrainingRuntimeError(f"unexpected nvidia-smi output: {line}")
        inventory.append(
            {
                "driver_version": fields[3],
                "index": int(fields[0]),
                "memory_mib": int(fields[2]),
                "name": fields[1],
            }
        )
    return inventory


def run_training_preflight(
    config: TrainingRuntimeConfig,
    paths: TrainingRuntimePaths,
    *,
    n_gpus: int,
    min_gpu_memory_gib: int,
) -> dict[str, object]:
    """Verify the complete pinned NVIDIA runtime and every external training input."""

    checks: list[dict[str, object]] = []

    def check(name: str, function: Callable[[], object]) -> object | None:
        try:
            detail = function()
        except Exception as error:
            checks.append({"detail": str(error), "name": name, "status": "failed"})
            return None
        checks.append({"detail": detail, "name": name, "status": "passed"})
        return detail

    check(
        "platform",
        lambda: (
            {"machine": platform.machine(), "system": platform.system()}
            if platform.system() == "Linux" and platform.machine() == "x86_64"
            else (_ for _ in ()).throw(
                TrainingRuntimeError("training requires the pinned linux/amd64 container")
            )
        ),
    )
    check(
        "container_image",
        lambda: (
            CONTAINER_IMAGE
            if os.environ.get("DEEP_RESEARCH_RL_CONTAINER_IMAGE") == CONTAINER_IMAGE
            else (_ for _ in ()).throw(
                TrainingRuntimeError("runtime does not declare the pinned container digest")
            )
        ),
    )

    def check_upstream(path: Path, expected: str) -> dict[str, object]:
        revision, dirty = _git_state(path)
        if revision != expected:
            raise TrainingRuntimeError(f"expected revision {expected}, found {revision}")
        if dirty:
            raise TrainingRuntimeError("upstream checkout has uncommitted changes")
        return {"dirty": False, "path": str(path.resolve()), "revision": revision}

    check("agent_r1_revision", lambda: check_upstream(paths.agent_r1_root, AGENT_R1_REVISION))
    check("verl_revision", lambda: check_upstream(paths.verl_root, VERL_REVISION))

    def check_verl_source() -> dict[str, object]:
        spec = importlib.util.find_spec("verl")
        if spec is None or spec.origin is None:
            raise TrainingRuntimeError("verl cannot be imported")
        origin = Path(spec.origin).resolve()
        if not origin.is_relative_to(paths.verl_root.resolve()):
            raise TrainingRuntimeError(f"verl imports from unexpected path: {origin}")
        version_path = paths.verl_root / "verl" / "version" / "version"
        try:
            source_version = version_path.read_text(encoding="utf-8").strip()
        except OSError as error:
            raise TrainingRuntimeError(f"could not read verl source version: {error}") from error
        if source_version != VERL_SOURCE_VERSION:
            raise TrainingRuntimeError(
                f"verl source version is {source_version!r}, expected {VERL_SOURCE_VERSION!r}"
            )
        return {"import_path": str(origin), "source_version": source_version}

    check("verl_source", check_verl_source)

    def check_agent_import() -> str:
        spec = importlib.util.find_spec("agent_r1")
        if spec is None:
            raise TrainingRuntimeError("agent_r1 cannot be imported")
        locations = [Path(item).resolve() for item in spec.submodule_search_locations or ()]
        if spec.origin is not None:
            locations.append(Path(spec.origin).resolve())
        if not locations:
            raise TrainingRuntimeError("agent_r1 import has no source location")
        expected_root = paths.agent_r1_root.resolve()
        unexpected = [path for path in locations if not path.is_relative_to(expected_root)]
        if unexpected:
            raise TrainingRuntimeError(f"agent_r1 imports from unexpected path: {unexpected[0]}")
        return ",".join(str(path) for path in locations)

    check("agent_r1_import", check_agent_import)
    check(
        "model_revision",
        lambda: (
            str(paths.model_path.resolve())
            if paths.model_path.is_dir() and paths.model_path.resolve().name == MODEL_REVISION
            else (_ for _ in ()).throw(
                TrainingRuntimeError(
                    "model_path must be a local Hugging Face snapshot directory whose name "
                    f"is the pinned revision {MODEL_REVISION}"
                )
            )
        ),
    )

    def check_training_data() -> dict[str, object]:
        build = verify_training_data(paths.training_data_dir)
        return {
            "manifest": str(build.manifest_path),
            "train_rows": build.train_rows,
            "validation_rows": build.validation_rows,
        }

    training_data = check("training_data", check_training_data)

    def check_retrieval_index() -> dict[str, object]:
        backend = manifest_backend(paths.index_dir)
        if backend != "faiss_bge":
            raise TrainingRuntimeError(
                f"baseline GPU training requires a faiss_bge index, received {backend}"
            )
        return verify_index(paths.index_dir, paths.corpus_path)

    check("retrieval_index", check_retrieval_index)
    check(
        "agent_flow_config",
        lambda: (
            sha256_file(paths.flow_config_path)
            if paths.flow_config_path.is_file()
            else (_ for _ in ()).throw(TrainingRuntimeError("AgentFlow config file is missing"))
        ),
    )

    inventory = check("nvidia_smi", _gpu_inventory)

    def check_torch_cuda() -> dict[str, object]:
        torch = importlib.import_module("torch")
        if not torch.cuda.is_available():
            raise TrainingRuntimeError("torch.cuda.is_available() is false")
        cuda_version = str(torch.version.cuda or "")
        minimum = str(config.section("container")["cuda_minimum"])
        if _cuda_version_tuple(cuda_version) < _cuda_version_tuple(minimum):
            raise TrainingRuntimeError(
                f"torch CUDA runtime {cuda_version} is below required {minimum}"
            )
        if not torch.cuda.is_bf16_supported():
            raise TrainingRuntimeError("selected CUDA devices do not support bfloat16")
        return {
            "available": True,
            "bf16_supported": True,
            "device_count": torch.cuda.device_count(),
            "runtime_version": cuda_version,
            "torch_version": str(torch.__version__),
        }

    torch_cuda = check("torch_cuda", check_torch_cuda)

    def check_capacity() -> dict[str, object]:
        if not isinstance(inventory, list):
            raise TrainingRuntimeError("GPU inventory is unavailable")
        if len(inventory) < n_gpus:
            raise TrainingRuntimeError(f"requested {n_gpus} GPUs but found {len(inventory)}")
        minimum_mib = min_gpu_memory_gib * 1024
        selected = inventory[:n_gpus]
        undersized = [gpu for gpu in selected if int(gpu["memory_mib"]) < minimum_mib]
        if undersized:
            raise TrainingRuntimeError(
                f"selected GPU memory is below requested minimum {min_gpu_memory_gib} GiB"
            )
        return {"requested": n_gpus, "selected": selected}

    capacity = check("gpu_capacity", check_capacity)
    report: dict[str, object] = {
        "agent_r1_revision": AGENT_R1_REVISION,
        "checks": checks,
        "container_image": CONTAINER_IMAGE,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "gpu_capacity": capacity,
        "record_type": "training_preflight",
        "schema_version": 1,
        "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed",
        "torch_cuda": torch_cuda,
        "training_data": training_data if isinstance(training_data, dict) else None,
        "verl_revision": VERL_REVISION,
        "verl_version": VERL_VERSION,
    }
    if report["status"] != "passed":
        raise TrainingPreflightError(report)
    return report


def _write_json(path: Path, value: object) -> Path:
    write_bytes_atomic(path, stable_json_bytes(value))
    return path


def _artifact(path: Path, *, relative_to: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
    }


def _manifest_artifacts(manifest: dict[str, object]) -> dict[str, object]:
    value = manifest.get("artifacts")
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TrainingRuntimeError("training run manifest artifacts must be an object")
    return {str(key): item for key, item in value.items()}


def _verify_trajectory_dump(path: Path) -> int:
    records = 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise TrainingRuntimeError(f"could not read trajectory dump {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            record: object = json.loads(line)
        except json.JSONDecodeError as error:
            raise TrainingRuntimeError(
                f"invalid trajectory JSON at {path}:{line_number}"
            ) from error
        if not isinstance(record, dict) or not isinstance(record.get("steps"), list):
            raise TrainingRuntimeError(
                f"trajectory dump record is incomplete at {path}:{line_number}"
            )
        steps = record["steps"]
        if not steps or any(
            not isinstance(step, dict) or not isinstance(step.get("transition_json"), str)
            for step in steps
        ):
            raise TrainingRuntimeError(
                f"trajectory dump lacks per-step transition evidence at {path}:{line_number}"
            )
        records += 1
    if records == 0:
        raise TrainingRuntimeError(f"trajectory dump is empty: {path}")
    return records


def _completed_phase_evidence(
    plan: TrainingLaunchPlan,
    log_path: Path,
) -> tuple[Path, dict[str, object], dict[str, int]]:
    checkpoint = plan.checkpoint_dir / f"global_step_{plan.target_step}"
    tracker = plan.checkpoint_dir / "latest_checkpointed_iteration.txt"
    if not checkpoint.is_dir():
        raise TrainingRuntimeError(f"expected checkpoint was not created: {checkpoint}")
    try:
        tracked_step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise TrainingRuntimeError(f"invalid latest checkpoint tracker: {tracker}") from error
    if tracked_step != plan.target_step:
        raise TrainingRuntimeError(
            f"checkpoint tracker reports step {tracked_step}, expected {plan.target_step}"
        )
    before_validation = plan.validation_dir / f"{plan.start_step}.jsonl"
    after_validation = plan.validation_dir / f"{plan.target_step}.jsonl"
    rollout = plan.rollout_dir / f"{plan.target_step}.jsonl"
    trajectory_records = {
        "after_validation": _verify_trajectory_dump(after_validation),
        "before_validation": _verify_trajectory_dump(before_validation),
        "rollout": _verify_trajectory_dump(rollout),
    }
    artifacts: dict[str, object] = {
        "after_validation": _artifact(after_validation, relative_to=plan.phase_dir),
        "before_validation": _artifact(before_validation, relative_to=plan.phase_dir),
        "checkpoint_tracker": _artifact(tracker, relative_to=plan.run_dir),
        "rollout_trajectories": _artifact(rollout, relative_to=plan.phase_dir),
        "training_log": _artifact(log_path, relative_to=plan.phase_dir),
    }
    return checkpoint, artifacts, trajectory_records


def run_training_launch(
    config: TrainingRuntimeConfig,
    plan: TrainingLaunchPlan,
    *,
    repository_root: str | Path,
    dry_run: bool = False,
) -> Path:
    """Run a preflighted phase, stream its log, and verify checkpoint/trajectory evidence."""

    plan.phase_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = plan.phase_dir / "resolved-config.json"
    manifest_path = plan.phase_dir / "manifest.json"
    log_path = plan.phase_dir / "training.log"
    if resolved_path.exists() or manifest_path.exists() or log_path.exists():
        raise TrainingRuntimeError(
            f"phase artifacts already exist; choose a new run directory: {plan.phase_dir}"
        )

    if dry_run:
        preflight: dict[str, object] = {"status": "not_run_for_dry_run"}
    else:
        preflight = run_training_preflight(
            config,
            plan.paths,
            n_gpus=plan.n_gpus,
            min_gpu_memory_gib=plan.min_gpu_memory_gib,
        )
    resolved = {
        "code": capture_code_state(repository_root),
        "launch": plan.to_dict(),
        "model": {"checkpoint": MODEL_CHECKPOINT, "revision": MODEL_REVISION},
        "preflight": preflight,
        "record_type": "resolved_training_config",
        "runtime": config.values,
        "schema_version": 1,
    }
    _write_json(resolved_path, resolved)
    manifest: dict[str, object] = {
        "artifacts": {"resolved_config": _artifact(resolved_path, relative_to=plan.phase_dir)},
        "created_at_utc": datetime.now(UTC).isoformat(),
        "phase": plan.phase,
        "record_type": "training_run_manifest",
        "schema_version": 1,
        "start_step": plan.start_step,
        "status": "dry_run" if dry_run else "running",
        "target_step": plan.target_step,
    }
    _write_json(manifest_path, manifest)
    if dry_run:
        return manifest_path

    environment = os.environ.copy()
    environment.update(plan.environment)
    return_code = -1
    try:
        with log_path.open("w", encoding="utf-8", newline="\n") as log_file:
            process = subprocess.Popen(
                plan.command,
                cwd=plan.paths.agent_r1_root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:  # pragma: no cover - guaranteed by stdout=PIPE
                raise TrainingRuntimeError("could not capture trainer output")
            for line in process.stdout:
                print(line, end="", flush=True)
                log_file.write(line)
                log_file.flush()
            return_code = process.wait()
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["return_code"] = return_code
        _write_json(manifest_path, manifest)
        raise
    if return_code != 0:
        manifest["status"] = "failed"
        manifest["return_code"] = return_code
        manifest["artifacts"] = {
            **_manifest_artifacts(manifest),
            "training_log": _artifact(log_path, relative_to=plan.phase_dir),
        }
        _write_json(manifest_path, manifest)
        raise TrainingRuntimeError(f"Agent-R1 trainer exited with status {return_code}")

    try:
        checkpoint, completed_artifacts, trajectory_records = _completed_phase_evidence(
            plan,
            log_path,
        )
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = str(error)
        manifest["return_code"] = return_code
        manifest["artifacts"] = {
            **_manifest_artifacts(manifest),
            "training_log": _artifact(log_path, relative_to=plan.phase_dir),
        }
        _write_json(manifest_path, manifest)
        raise
    artifacts = _manifest_artifacts(manifest)
    artifacts.update(completed_artifacts)
    manifest.update(
        {
            "artifacts": artifacts,
            "checkpoint_path": str(checkpoint),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "return_code": return_code,
            "status": "completed",
            "trajectory_records": trajectory_records,
        }
    )
    _write_json(manifest_path, manifest)
    return manifest_path
