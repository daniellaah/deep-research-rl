"""Command-line interface for dependency-light project operations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from deep_research_rl import __version__
from deep_research_rl.config import ConfigError, load_config
from deep_research_rl.core.models import SearchResult
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
