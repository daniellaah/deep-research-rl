"""Command-line interface for dependency-light project operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, cast

from deep_research_rl import __version__
from deep_research_rl.agent.qwen import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_REVISION,
    NoSearchQwenPolicyAdapter,
    QwenDependencyError,
    QwenGenerationSettings,
    QwenInferenceError,
    QwenPolicyAdapter,
)
from deep_research_rl.agent.rollout import DEBUG_RESULT_SCOPE, run_model_rollout
from deep_research_rl.agent.serialization import write_agent_rollout_jsonl
from deep_research_rl.config import ConfigError, load_config
from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.costs import ZeroCost
from deep_research_rl.core.credit import TerminalOnlyCreditAssigner
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.models import SearchResult
from deep_research_rl.core.rewards import TerminalExactMatchReward
from deep_research_rl.core.smoke import run_synthetic_smoke
from deep_research_rl.data.hotpotqa import (
    DataPipelineError,
    build_hotpotqa,
    verify_hotpotqa_build,
)
from deep_research_rl.data.source import (
    SourceConfigError,
    download_source_files,
    load_source_config,
)
from deep_research_rl.evaluation.artifacts import (
    build_resolved_config,
    capture_code_state,
    write_evaluation_artifacts,
)
from deep_research_rl.evaluation.contracts import EvaluationResultScope
from deep_research_rl.evaluation.metrics import EvaluationIntegrityError, aggregate_evaluation
from deep_research_rl.evaluation.reporting import (
    load_aggregate_json,
    write_aggregate_csv,
    write_comparison_csv,
)
from deep_research_rl.evaluation.runner import (
    BASELINE_EVALUATION_SEED,
    BASELINE_MAX_NEW_TOKENS,
    BASELINE_MAX_PROMPT_TOKENS,
    BASELINE_RETRIEVAL_TOP_K,
    FULL_CORPUS_SHA256,
    FULL_VALIDATION_EXAMPLES,
    FULL_VALIDATION_EXAMPLES_SHA256,
    run_no_search_evaluation,
    run_prompted_agent_evaluation,
    run_rl_agent_evaluation,
)
from deep_research_rl.evaluation.serialization import (
    EvaluationFormatError,
    read_evaluation_jsonl,
    write_json_artifact,
)
from deep_research_rl.retrieval import load_retriever, verify_index
from deep_research_rl.retrieval.bm25 import build_bm25_index
from deep_research_rl.retrieval.corpus import load_corpus
from deep_research_rl.retrieval.diagnostics import (
    build_recall_report,
    load_diagnostic_examples,
    write_recall_report,
)
from deep_research_rl.retrieval.errors import RetrievalDependencyError, RetrievalError
from deep_research_rl.retrieval.faiss_bge import (
    DEFAULT_BGE_MODEL,
    DEFAULT_BGE_MODEL_REVISION,
    DEFAULT_QUERY_INSTRUCTION,
    build_faiss_bge_index,
)
from deep_research_rl.retrieval.index import (
    INDEX_MANIFEST_FILENAME,
    load_manifest,
    manifest_backend,
    sha256_file,
)
from deep_research_rl.training.data import (
    TrainingDataError,
    prepare_training_data,
    verify_training_data,
)
from deep_research_rl.training.runtime import (
    TrainingPreflightError,
    TrainingRuntimeError,
    TrainingRuntimePaths,
    build_training_launch_plan,
    load_training_runtime,
    run_training_launch,
    run_training_preflight,
)


def _add_evaluation_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    retrieval_enabled: bool,
    trained_policy: bool,
) -> None:
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-examples",
        type=int,
        help="ordered-prefix size for debug runs (defaults to 1)",
    )
    parser.add_argument(
        "--final-validation",
        action="store_true",
        help="require the complete pinned validation split and production retrieval",
    )
    if retrieval_enabled:
        parser.add_argument("--corpus", type=Path, required=True)
        parser.add_argument("--index-dir", type=Path, required=True)
        parser.add_argument("--retrieval-device", default="cpu")
    if trained_policy:
        parser.add_argument("--model-name", required=True)
        parser.add_argument("--model-revision", required=True)
    else:
        parser.add_argument("--model-name", default=DEFAULT_QWEN_MODEL)
        parser.add_argument("--model-revision", default=DEFAULT_QWEN_REVISION)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")


def _add_training_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--agent-r1-root", type=Path, required=True)
    parser.add_argument("--verl-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--training-data-dir", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--flow-config", type=Path, required=True)
    parser.add_argument("--n-gpus", type=int, required=True)
    parser.add_argument("--min-gpu-memory-gib", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command parser."""

    parser = argparse.ArgumentParser(
        prog="deep-research-rl",
        description="Tools for reproducible multi-step retrieval research.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    commands = parser.add_subparsers(dest="command", title="commands")
    config_parser = commands.add_parser("config", help="load and inspect TOML configurations")
    config_commands = config_parser.add_subparsers(dest="config_command", title="config commands")

    validate_parser = config_commands.add_parser("validate", help="validate a configuration")
    validate_parser.add_argument("path", type=Path, help="path to a TOML configuration")

    show_parser = config_commands.add_parser("show", help="print a configuration as stable JSON")
    show_parser.add_argument("path", type=Path, help="path to a TOML configuration")

    smoke_parser = commands.add_parser(
        "smoke",
        help="run the deterministic synthetic non-benchmark CPU episode",
    )
    smoke_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="destination for the generated JSONL trajectory",
    )

    agent_parser = commands.add_parser(
        "agent",
        help="run bounded model-backed agent validation",
    )
    agent_commands = agent_parser.add_subparsers(dest="agent_command", title="agent commands")
    rollout_parser = agent_commands.add_parser(
        "rollout",
        help="run strict SEARCH/ANSWER rollouts over a bounded example prefix",
    )
    rollout_parser.add_argument("--examples", type=Path, required=True)
    rollout_parser.add_argument("--corpus", type=Path, required=True)
    rollout_parser.add_argument("--index-dir", type=Path, required=True)
    rollout_parser.add_argument("--output", type=Path, required=True)
    rollout_parser.add_argument("--max-examples", type=int, default=1)
    rollout_parser.add_argument("--top-k", type=int, default=3)
    rollout_parser.add_argument("--max-searches", type=int, default=5)
    rollout_parser.add_argument("--max-steps", type=int, default=8)
    rollout_parser.add_argument("--max-prompt-tokens", type=int, default=8192)
    rollout_parser.add_argument("--max-new-tokens", type=int, default=96)
    rollout_parser.add_argument("--model-name", default=DEFAULT_QWEN_MODEL)
    rollout_parser.add_argument("--model-revision", default=DEFAULT_QWEN_REVISION)
    rollout_parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    rollout_parser.add_argument(
        "--dtype",
        choices=("auto", "float16", "float32", "bfloat16"),
        default="auto",
    )
    rollout_parser.add_argument("--retrieval-device", default="cpu")
    rollout_parser.add_argument("--seed", type=int, default=0)
    rollout_parser.add_argument("--do-sample", action="store_true")
    rollout_parser.add_argument("--temperature", type=float, default=0.7)
    rollout_parser.add_argument("--top-p", type=float, default=0.8)
    rollout_parser.add_argument("--top-k-sampling", type=int, default=20)
    rollout_parser.add_argument("--local-files-only", action="store_true")

    evaluation_parser = commands.add_parser(
        "evaluation",
        help="run and aggregate the frozen baseline evaluation protocol",
    )
    evaluation_commands = evaluation_parser.add_subparsers(
        dest="evaluation_command",
        title="evaluation commands",
    )
    no_search_parser = evaluation_commands.add_parser(
        "no-search",
        help="evaluate one ANSWER-only generation without retrieval",
    )
    _add_evaluation_run_arguments(
        no_search_parser,
        retrieval_enabled=False,
        trained_policy=False,
    )
    prompted_parser = evaluation_commands.add_parser(
        "prompted-agent",
        help="evaluate the pinned base checkpoint with the search agent",
    )
    _add_evaluation_run_arguments(
        prompted_parser,
        retrieval_enabled=True,
        trained_policy=False,
    )
    rl_parser = evaluation_commands.add_parser(
        "rl-agent",
        help="evaluate a trained policy under the same search-agent controls",
    )
    _add_evaluation_run_arguments(
        rl_parser,
        retrieval_enabled=True,
        trained_policy=True,
    )
    aggregate_parser = evaluation_commands.add_parser(
        "aggregate",
        help="recompute aggregate JSON and CSV from per-example source records",
    )
    aggregate_parser.add_argument("--per-example", type=Path, required=True)
    aggregate_parser.add_argument("--examples", type=Path, required=True)
    aggregate_parser.add_argument("--max-examples", type=int)
    aggregate_parser.add_argument("--output-json", type=Path, required=True)
    aggregate_parser.add_argument("--output-csv", type=Path, required=True)
    compare_parser = evaluation_commands.add_parser(
        "compare",
        help="write a compatible policy-condition comparison table",
    )
    compare_parser.add_argument("--aggregates", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    data_parser = commands.add_parser(
        "data",
        help="download, convert, and verify versioned datasets",
    )
    data_commands = data_parser.add_subparsers(dest="data_command", title="data commands")

    download_parser = data_commands.add_parser(
        "download",
        help="download and verify pinned raw source files",
    )
    download_parser.add_argument(
        "--source-config",
        type=Path,
        required=True,
        help="committed JSON source descriptor",
    )
    download_parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="directory for ignored raw source files",
    )

    prepare_parser = data_commands.add_parser(
        "prepare",
        help="convert verified HotpotQA JSON into deterministic artifacts",
    )
    prepare_parser.add_argument(
        "--source-config",
        type=Path,
        required=True,
        help="committed JSON source descriptor",
    )
    prepare_parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="directory containing verified raw source files",
    )
    prepare_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for ignored converted artifacts and manifest",
    )
    prepare_parser.add_argument(
        "--max-train",
        type=int,
        help="positive deterministic train prefix size for a debug build",
    )
    prepare_parser.add_argument(
        "--max-validation",
        type=int,
        help="positive deterministic validation prefix size for a debug build",
    )

    verify_parser = data_commands.add_parser(
        "verify",
        help="verify a converted build against its manifest",
    )
    verify_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory containing converted artifacts and manifest",
    )

    training_parser = commands.add_parser(
        "training",
        help="prepare, preflight, and launch the pinned Agent-R1/verl GRPO path",
    )
    training_commands = training_parser.add_subparsers(
        dest="training_command",
        title="training commands",
    )
    training_prepare = training_commands.add_parser(
        "prepare-data",
        help="materialize verified Agent-R1 Parquet train/validation inputs",
    )
    training_prepare.add_argument("--processed-dir", type=Path, required=True)
    training_prepare.add_argument("--output-dir", type=Path, required=True)
    training_prepare.add_argument("--max-train", type=int)
    training_prepare.add_argument("--max-validation", type=int)
    training_verify = training_commands.add_parser(
        "verify-data",
        help="verify Agent-R1 Parquet hashes, rows, columns, and flow routing",
    )
    training_verify.add_argument("--training-data-dir", type=Path, required=True)
    training_preflight = training_commands.add_parser(
        "preflight",
        help="verify the pinned container, CUDA, upstream checkouts, and inputs",
    )
    _add_training_runtime_arguments(training_preflight)
    training_preflight.add_argument("--output", type=Path, required=True)
    training_launch = training_commands.add_parser(
        "launch",
        help="launch and verify a one-update sanity or explicit resume phase",
    )
    _add_training_runtime_arguments(training_launch)
    training_launch.add_argument("--phase", choices=("sanity", "resume"), required=True)
    training_launch.add_argument("--run-dir", type=Path, required=True)
    training_launch.add_argument("--resume-from", type=Path)
    training_launch.add_argument("--seed", type=int, default=0)
    training_launch.add_argument("--dry-run", action="store_true")

    retrieval_parser = commands.add_parser(
        "retrieval",
        help="build, verify, query, and diagnose retrieval indexes",
    )
    retrieval_commands = retrieval_parser.add_subparsers(
        dest="retrieval_command",
        title="retrieval commands",
    )

    retrieval_build = retrieval_commands.add_parser(
        "build",
        help="build an integrity-checked BM25 or FAISS/BGE index",
    )
    retrieval_build.add_argument("--backend", choices=("bm25", "faiss_bge"), required=True)
    retrieval_build.add_argument("--corpus", type=Path, required=True)
    retrieval_build.add_argument("--index-dir", type=Path, required=True)
    retrieval_build.add_argument("--k1", type=float, default=1.5)
    retrieval_build.add_argument("--b", type=float, default=0.75)
    retrieval_build.add_argument("--model-name", default=DEFAULT_BGE_MODEL)
    retrieval_build.add_argument("--model-revision", default=DEFAULT_BGE_MODEL_REVISION)
    retrieval_build.add_argument("--query-instruction", default=DEFAULT_QUERY_INSTRUCTION)
    retrieval_build.add_argument("--device", default="cpu")
    retrieval_build.add_argument("--batch-size", type=int, default=128)

    retrieval_verify = retrieval_commands.add_parser(
        "verify",
        help="verify an index against its artifacts and exact corpus",
    )
    retrieval_verify.add_argument("--corpus", type=Path, required=True)
    retrieval_verify.add_argument("--index-dir", type=Path, required=True)

    retrieval_search = retrieval_commands.add_parser(
        "search",
        help="query a verified index and print traceable scored results",
    )
    retrieval_search.add_argument("--corpus", type=Path, required=True)
    retrieval_search.add_argument("--index-dir", type=Path, required=True)
    retrieval_search.add_argument("--query", required=True)
    retrieval_search.add_argument("--top-k", type=int, default=5)
    retrieval_search.add_argument("--device", default="cpu")

    retrieval_diagnose = retrieval_commands.add_parser(
        "diagnose",
        help="measure supporting-document recall on a fixed ordered example prefix",
    )
    retrieval_diagnose.add_argument("--corpus", type=Path, required=True)
    retrieval_diagnose.add_argument("--examples", type=Path, required=True)
    retrieval_diagnose.add_argument("--index-dir", type=Path, required=True)
    retrieval_diagnose.add_argument("--output", type=Path, required=True)
    retrieval_diagnose.add_argument("--limit", type=int, default=5)
    retrieval_diagnose.add_argument("--ks", type=int, nargs="+", default=(1, 5, 10))
    retrieval_diagnose.add_argument("--device", default="cpu")

    return parser


def _run_data_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run one dependency-light dataset command."""

    if args.data_command is None:
        parser.parse_args(["data", "--help"])
        return 0
    try:
        if args.data_command == "download":
            source_config = load_source_config(args.source_config)
            paths = download_source_files(source_config, args.raw_dir)
            print(f"verified {len(paths)} raw source files in {args.raw_dir}")
            return 0
        if args.data_command == "prepare":
            source_config = load_source_config(args.source_config)
            result = build_hotpotqa(
                source_config,
                args.raw_dir,
                args.output_dir,
                max_train=args.max_train,
                max_validation=args.max_validation,
            )
            print(
                f"{result.build_mode} HotpotQA build verified: "
                f"train={result.train_examples}, "
                f"validation={result.validation_examples}, "
                f"corpus={result.corpus_documents}, "
                f"manifest={result.manifest_path}"
            )
            return 0
        manifest = verify_hotpotqa_build(args.output_dir)
        counts = manifest["counts"]
        if not isinstance(counts, dict):
            raise DataPipelineError("manifest counts must be an object")
        print(
            f"HotpotQA build verified: train={counts['train_examples']}, "
            f"validation={counts['validation_examples']}, "
            f"corpus={counts['corpus_documents']}"
        )
        return 0
    except (DataPipelineError, OSError, SourceConfigError) as error:
        parser.error(str(error))
    return 2


def _result_record(result: SearchResult) -> dict[str, object]:
    return {
        "document_id": result.document_id,
        "rank": result.rank,
        "score": result.score,
        "text": result.text,
        "title": result.title,
    }


def _training_paths(args: argparse.Namespace) -> TrainingRuntimePaths:
    return TrainingRuntimePaths(
        agent_r1_root=args.agent_r1_root,
        verl_root=args.verl_root,
        model_path=args.model_path,
        training_data_dir=args.training_data_dir,
        corpus_path=args.corpus,
        index_dir=args.index_dir,
        flow_config_path=args.flow_config,
    )


def _run_training_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Prepare data or run the pinned CUDA preflight/launch workflow."""

    if args.training_command is None:
        parser.parse_args(["training", "--help"])
        return 0
    try:
        if args.training_command == "prepare-data":
            result = prepare_training_data(
                args.processed_dir,
                args.output_dir,
                max_train=args.max_train,
                max_validation=args.max_validation,
            )
            print(
                "Agent-R1 training data prepared and verified: "
                f"train={result.train_rows}, validation={result.validation_rows}, "
                f"manifest={result.manifest_path}"
            )
            return 0
        if args.training_command == "verify-data":
            result = verify_training_data(args.training_data_dir)
            print(
                "Agent-R1 training data verified: "
                f"train={result.train_rows}, validation={result.validation_rows}"
            )
            return 0

        config = load_training_runtime(args.config)
        paths = _training_paths(args)
        if args.training_command == "preflight":
            try:
                report = run_training_preflight(
                    config,
                    paths,
                    n_gpus=args.n_gpus,
                    min_gpu_memory_gib=args.min_gpu_memory_gib,
                )
            except TrainingPreflightError as error:
                write_json_artifact(args.output, error.report)
                raise
            write_json_artifact(args.output, report)
            print(f"pinned training preflight passed: report={args.output}")
            return 0

        phase = cast(Literal["sanity", "resume"], args.phase)
        plan = build_training_launch_plan(
            config,
            paths,
            phase=phase,
            run_dir=args.run_dir,
            n_gpus=args.n_gpus,
            min_gpu_memory_gib=args.min_gpu_memory_gib,
            seed=args.seed,
            resume_from=args.resume_from,
        )
        repository_root = Path(__file__).resolve().parents[2]
        manifest_path = run_training_launch(
            config,
            plan,
            repository_root=repository_root,
            dry_run=args.dry_run,
        )
        state = "dry run written" if args.dry_run else "completed and verified"
        print(f"Agent-R1 {phase} phase {state}: manifest={manifest_path}")
        return 0
    except (
        OSError,
        RetrievalDependencyError,
        RetrievalError,
        TrainingDataError,
        TrainingRuntimeError,
    ) as error:
        parser.error(str(error))
    return 2


def _run_agent_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run a bounded real-model debug rollout without changing baseline semantics."""

    if args.agent_command is None:
        parser.parse_args(["agent", "--help"])
        return 0
    try:
        if not 0 <= args.max_searches <= 5:
            raise ValueError("baseline max_searches must be between 0 and 5")
        if args.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        examples = load_diagnostic_examples(args.examples, limit=args.max_examples).examples
        retriever = load_retriever(
            args.index_dir,
            args.corpus,
            top_k=args.top_k,
            device=args.retrieval_device,
        )
        settings = QwenGenerationSettings(
            max_prompt_tokens=args.max_prompt_tokens,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k_sampling,
            seed=args.seed,
        )
        policy = QwenPolicyAdapter.from_pretrained(
            model_name=args.model_name,
            model_revision=args.model_revision,
            device=args.device,
            dtype=args.dtype,
            settings=settings,
            local_files_only=args.local_files_only,
        )
        environment = ResearchEnvironment(
            retriever,
            AppendOnlyContextPolicy(),
            max_searches=args.max_searches,
        )
        rollouts = tuple(
            run_model_rollout(
                example,
                policy,
                environment,
                TerminalExactMatchReward(),
                TerminalOnlyCreditAssigner(),
                ZeroCost(),
                max_steps=args.max_steps,
                result_scope=DEBUG_RESULT_SCOPE,
            )
            for example in examples
        )
        write_agent_rollout_jsonl(args.output, rollouts)
        answered = sum(rollout.termination_reason == "answered" for rollout in rollouts)
        searches = sum(rollout.final_state.executed_searches for rollout in rollouts)
        print(
            "bounded real-model debug rollout completed: "
            f"examples={len(rollouts)}, answered={answered}, executed_searches={searches}, "
            f"device={policy.device}, result_scope={DEBUG_RESULT_SCOPE}, output={args.output}"
        )
        return 0
    except (
        OSError,
        QwenDependencyError,
        QwenInferenceError,
        RetrievalDependencyError,
        RetrievalError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 2


def _evaluation_scope(args: argparse.Namespace, examples_sha256: str, source_records: int) -> str:
    if not args.final_validation:
        return DEBUG_RESULT_SCOPE
    if args.max_examples is not None:
        raise ValueError("final validation cannot use --max-examples")
    if source_records != FULL_VALIDATION_EXAMPLES:
        raise ValueError(
            f"final validation requires {FULL_VALIDATION_EXAMPLES} source examples, "
            f"found {source_records}"
        )
    if examples_sha256 != FULL_VALIDATION_EXAMPLES_SHA256:
        raise ValueError("final validation examples do not match the pinned canonical artifact")
    return "baseline_validation"


def _run_evaluation_policy(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    command: Sequence[str],
) -> int:
    condition = str(args.evaluation_command).replace("-", "_")
    if condition in {"no_search", "prompted_agent"} and (
        args.model_name != DEFAULT_QWEN_MODEL or args.model_revision != DEFAULT_QWEN_REVISION
    ):
        raise ValueError("no-search and prompted-agent require the pinned baseline checkpoint")
    limit = None if args.final_validation else (args.max_examples or 1)
    loaded_examples = load_diagnostic_examples(args.examples, limit=limit)
    scope_value = _evaluation_scope(
        args,
        loaded_examples.source_sha256,
        loaded_examples.source_records,
    )
    scope = cast(EvaluationResultScope, scope_value)

    retrieval_metadata: dict[str, object]
    policy_type: type[QwenPolicyAdapter]
    retriever = None
    if condition == "no_search":
        retrieval_metadata = {"enabled": False, "top_k": None}
        policy_type = NoSearchQwenPolicyAdapter
    else:
        index_manifest = load_manifest(args.index_dir)
        backend = manifest_backend(args.index_dir)
        corpus_metadata = index_manifest.get("corpus")
        if not isinstance(corpus_metadata, dict):
            raise ValueError("retrieval manifest corpus must be an object")
        if args.final_validation and (
            backend != "faiss_bge" or corpus_metadata.get("sha256") != FULL_CORPUS_SHA256
        ):
            raise ValueError(
                "final search-agent validation requires the pinned full-corpus FAISS/BGE index"
            )
        retriever = load_retriever(
            args.index_dir,
            args.corpus,
            top_k=BASELINE_RETRIEVAL_TOP_K,
            device=args.retrieval_device,
        )
        manifest_path = args.index_dir / INDEX_MANIFEST_FILENAME
        retrieval_metadata = {
            "backend": backend,
            "corpus": corpus_metadata,
            "corpus_path": str(args.corpus.resolve()),
            "device": args.retrieval_device,
            "enabled": True,
            "index_dir": str(args.index_dir.resolve()),
            "index_manifest_sha256": sha256_file(manifest_path),
            "top_k": BASELINE_RETRIEVAL_TOP_K,
        }
        policy_type = QwenPolicyAdapter

    settings = QwenGenerationSettings(
        max_prompt_tokens=BASELINE_MAX_PROMPT_TOKENS,
        max_new_tokens=BASELINE_MAX_NEW_TOKENS,
        do_sample=False,
        seed=BASELINE_EVALUATION_SEED,
    )
    policy = policy_type.from_pretrained(
        model_name=args.model_name,
        model_revision=args.model_revision,
        device=args.device,
        dtype=args.dtype,
        settings=settings,
        local_files_only=args.local_files_only,
    )
    if condition == "no_search":
        items = run_no_search_evaluation(
            loaded_examples.examples,
            policy=policy,
            run_id=args.run_id,
            result_scope=scope,
            count_tool_tokens=policy.count_text_tokens,
        )
    elif condition == "prompted_agent":
        if retriever is None:  # pragma: no cover - guarded by the condition branch
            raise RuntimeError("prompted-agent retriever was not initialized")
        items = run_prompted_agent_evaluation(
            loaded_examples.examples,
            policy=policy,
            retriever=retriever,
            run_id=args.run_id,
            result_scope=scope,
            count_tool_tokens=policy.count_text_tokens,
        )
    else:
        if retriever is None:  # pragma: no cover - guarded by the condition branch
            raise RuntimeError("rl-agent retriever was not initialized")
        items = run_rl_agent_evaluation(
            loaded_examples.examples,
            policy=policy,
            retriever=retriever,
            run_id=args.run_id,
            result_scope=scope,
            count_tool_tokens=policy.count_text_tokens,
        )

    repository_root = Path(__file__).resolve().parents[2]
    resolved_config = build_resolved_config(
        run_id=args.run_id,
        policy_condition=condition,
        result_scope=scope,
        examples_path=args.examples,
        examples_sha256=loaded_examples.source_sha256,
        dataset_source=loaded_examples.examples[0].source,
        source_records=loaded_examples.source_records,
        requested_example_ids=[example.example_id for example in loaded_examples.examples],
        model_name=policy.model_name,
        model_revision=policy.model_revision,
        prompt_format=policy.prompt_format,
        device=policy.device,
        dtype=args.dtype,
        retrieval=retrieval_metadata,
        code_state=capture_code_state(repository_root),
        command=command,
    )
    artifacts = write_evaluation_artifacts(
        args.output_dir,
        items=items,
        expected_example_ids=[example.example_id for example in loaded_examples.examples],
        resolved_config=resolved_config,
    )
    if artifacts.status != "completed" or artifacts.aggregate is None:
        parser.error(f"evaluation run is invalid; inspect {artifacts.manifest_path}")
    print(
        f"{condition} evaluation completed: examples={len(items)}, "
        f"result_scope={scope}, benchmark_eligible="
        f"{artifacts.aggregate['benchmark_eligible']}, output={artifacts.output_dir}"
    )
    return 0


def _run_evaluation_command(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    command: Sequence[str],
) -> int:
    """Run a policy condition or recompute/export already-recorded results."""

    if args.evaluation_command is None:
        parser.parse_args(["evaluation", "--help"])
        return 0
    try:
        if args.evaluation_command in {"no-search", "prompted-agent", "rl-agent"}:
            return _run_evaluation_policy(parser, args, command)
        if args.evaluation_command == "aggregate":
            loaded_examples = load_diagnostic_examples(args.examples, limit=args.max_examples)
            items = read_evaluation_jsonl(args.per_example)
            aggregate = aggregate_evaluation(
                items,
                expected_example_ids=[example.example_id for example in loaded_examples.examples],
            )
            write_json_artifact(args.output_json, aggregate)
            write_aggregate_csv(args.output_csv, aggregate)
            print(
                "aggregate exactly recomputed from per-example records: "
                f"examples={len(items)}, json={args.output_json}, csv={args.output_csv}"
            )
            return 0
        aggregates = [load_aggregate_json(path) for path in args.aggregates]
        write_comparison_csv(args.output, aggregates)
        print(f"comparison table written: conditions={len(aggregates)}, output={args.output}")
        return 0
    except (
        EvaluationFormatError,
        EvaluationIntegrityError,
        OSError,
        QwenDependencyError,
        QwenInferenceError,
        RetrievalDependencyError,
        RetrievalError,
        ValueError,
    ) as error:
        parser.error(str(error))
    return 2


def _run_retrieval_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Run one index lifecycle or retrieval diagnostic command."""

    if args.retrieval_command is None:
        parser.parse_args(["retrieval", "--help"])
        return 0
    try:
        if args.retrieval_command == "build":
            if args.backend == "bm25":
                manifest_path = build_bm25_index(
                    args.corpus,
                    args.index_dir,
                    k1=args.k1,
                    b=args.b,
                )
            else:
                manifest_path = build_faiss_bge_index(
                    args.corpus,
                    args.index_dir,
                    model_name=args.model_name,
                    model_revision=args.model_revision,
                    query_instruction=args.query_instruction,
                    device=args.device,
                    batch_size=args.batch_size,
                )
            verify_index(args.index_dir, args.corpus)
            print(f"{args.backend} index built and verified by manifest: {manifest_path}")
            return 0
        if args.retrieval_command == "verify":
            manifest = verify_index(args.index_dir, args.corpus)
            corpus = manifest["corpus"]
            if not isinstance(corpus, dict):
                raise RetrievalError("manifest corpus metadata must be an object")
            print(
                f"{manifest['backend']} index verified: "
                f"documents={corpus['documents']}, corpus_sha256={corpus['sha256']}"
            )
            return 0
        if args.retrieval_command == "search":
            retriever = load_retriever(
                args.index_dir,
                args.corpus,
                top_k=args.top_k,
                device=args.device,
            )
            results = [_result_record(result) for result in retriever.search(args.query)]
            print(json.dumps({"results": results}, ensure_ascii=False, sort_keys=True))
            return 0

        ks = tuple(args.ks)
        examples = load_diagnostic_examples(args.examples, limit=args.limit)
        corpus = load_corpus(args.corpus)
        index_manifest = load_manifest(args.index_dir)
        backend = manifest_backend(args.index_dir)
        retriever = load_retriever(
            args.index_dir,
            args.corpus,
            top_k=max(ks),
            device=args.device,
        )
        manifest_path = args.index_dir / INDEX_MANIFEST_FILENAME
        report = build_recall_report(
            retriever,
            examples,
            backend=backend,
            ks=ks,
            corpus_metadata=corpus.fingerprint.to_dict(),
            index_manifest=index_manifest,
            index_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        output_path = write_recall_report(args.output, report)
        print(
            f"{backend} retrieval-only diagnostic verified: "
            f"queries={len(examples.examples)}, output={output_path}"
        )
        return 0
    except (OSError, RetrievalDependencyError, RetrievalError, ValueError) as error:
        parser.error(str(error))
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface and return a process exit code."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        try:
            trajectory = run_synthetic_smoke(args.output)
        except (OSError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            "synthetic non-benchmark smoke passed: "
            f"exact_match={trajectory.metrics.exact_match:.1f}, "
            f"executed_searches={trajectory.metrics.executed_searches}, "
            f"output={args.output}"
        )
        return 0

    if args.command == "data":
        return _run_data_command(parser, args)

    if args.command == "agent":
        return _run_agent_command(parser, args)

    if args.command == "evaluation":
        invoked = tuple(argv) if argv is not None else tuple(sys.argv[1:])
        return _run_evaluation_command(parser, args, ("deep-research-rl", *invoked))

    if args.command == "training":
        return _run_training_command(parser, args)

    if args.command == "retrieval":
        return _run_retrieval_command(parser, args)

    if args.command != "config":
        parser.print_help()
        return 0

    if args.config_command is None:
        parser.parse_args(["config", "--help"])
        return 0

    try:
        config = load_config(args.path)
    except ConfigError as error:
        parser.error(str(error))

    if args.config_command == "show":
        print(json.dumps(config, indent=2, sort_keys=True, ensure_ascii=False))
    else:
        print(f"valid {config['config_kind']} configuration: {args.path}")
    return 0
