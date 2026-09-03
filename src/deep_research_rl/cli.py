"""Command-line interface for dependency-light project operations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from deep_research_rl import __version__
from deep_research_rl.config import ConfigError, config_as_json, load_config
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
        print(config_as_json(config))
    else:
        print(f"valid {config['config_kind']} configuration: {args.path}")
    return 0
