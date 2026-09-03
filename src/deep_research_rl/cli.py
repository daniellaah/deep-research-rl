"""Command-line interface for dependency-light project operations."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from deep_research_rl import __version__
from deep_research_rl.config import ConfigError, config_as_json, load_config
from deep_research_rl.core.smoke import run_synthetic_smoke


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

    return parser


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
