"""Supporting-document recall diagnostics for fixed HotpotQA subsets."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from deep_research_rl.core.models import Example, SearchResult
from deep_research_rl.core.protocols import Retriever
from deep_research_rl.data.models import DataRecordError, hotpotqa_example_from_dict
from deep_research_rl.retrieval.errors import RetrievalError
from deep_research_rl.retrieval.index import stable_json_bytes, write_bytes_atomic

DIAGNOSTIC_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class DiagnosticExamples:
    """An exact ordered example subset plus source-file provenance."""

    examples: tuple[Example, ...]
    source_bytes: int
    source_records: int
    source_sha256: str


def load_diagnostic_examples(
    path: str | Path,
    *,
    limit: int | None = None,
) -> DiagnosticExamples:
    """Load an ordered canonical HotpotQA prefix and retain its full-file fingerprint."""

    if limit is not None and limit < 1:
        raise ValueError("diagnostic limit must be at least 1")
    source_path = Path(path)
    try:
        raw = source_path.read_bytes()
    except OSError as error:
        raise RetrievalError(
            f"could not read diagnostic examples {source_path}: {error}"
        ) from error
    examples: list[Example] = []
    source_records = 0
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        source_records += 1
        if limit is not None and len(examples) >= limit:
            continue
        try:
            value: object = json.loads(line)
            examples.append(hotpotqa_example_from_dict(value).to_core_example())
        except (UnicodeDecodeError, json.JSONDecodeError, DataRecordError) as error:
            raise RetrievalError(
                f"invalid canonical diagnostic example on line {line_number}: {error}"
            ) from error
    if not examples:
        raise RetrievalError("diagnostic subset must contain at least one example")
    return DiagnosticExamples(
        examples=tuple(examples),
        source_bytes=len(raw),
        source_records=source_records,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _validate_results(results: Sequence[SearchResult]) -> None:
    ranks = [result.rank for result in results]
    if ranks != list(range(1, len(results) + 1)):
        raise RetrievalError("retriever returned non-contiguous ranks")
    identifiers = [result.document_id for result in results]
    if len(identifiers) != len(set(identifiers)):
        raise RetrievalError("retriever returned duplicate document IDs")


def build_recall_report(
    retriever: Retriever,
    examples: DiagnosticExamples,
    *,
    backend: str,
    ks: Sequence[int],
    corpus_metadata: dict[str, object],
    index_manifest: dict[str, object],
    index_manifest_sha256: str,
) -> dict[str, object]:
    """Compute macro/micro supporting-document recall without agent-performance claims."""

    normalized_ks = tuple(sorted(set(ks)))
    if not normalized_ks or any(k < 1 for k in normalized_ks):
        raise ValueError("ks must contain positive integers")
    if index_manifest.get("backend") != backend:
        raise RetrievalError("diagnostic backend differs from the index manifest")
    query_results = retriever.search_batch(tuple(example.question for example in examples.examples))
    if len(query_results) != len(examples.examples):
        raise RetrievalError("retriever batch output cardinality differs from query count")

    macro_totals = {k: 0.0 for k in normalized_ks}
    micro_hits = {k: 0 for k in normalized_ks}
    complete_hits = {k: 0 for k in normalized_ks}
    total_supporting = 0
    per_example: list[dict[str, object]] = []
    for example, results in zip(examples.examples, query_results, strict=True):
        _validate_results(results)
        supporting = set(example.supporting_document_ids)
        if not supporting:
            raise RetrievalError(
                f"diagnostic example {example.example_id} has no supporting document labels"
            )
        total_supporting += len(supporting)
        retrieved_ids = [result.document_id for result in results]
        recall_values: dict[str, float] = {}
        hits_values: dict[str, int] = {}
        for k in normalized_ks:
            hits = len(supporting.intersection(retrieved_ids[:k]))
            recall = hits / len(supporting)
            macro_totals[k] += recall
            micro_hits[k] += hits
            complete_hits[k] += hits == len(supporting)
            recall_values[str(k)] = recall
            hits_values[str(k)] = hits
        per_example.append(
            {
                "example_id": example.example_id,
                "hits_at_k": hits_values,
                "recall_at_k": recall_values,
                "retrieved_document_ids": retrieved_ids,
                "supporting_document_ids": list(example.supporting_document_ids),
            }
        )

    query_count = len(examples.examples)
    return {
        "backend": backend,
        "corpus": corpus_metadata,
        "index_manifest": index_manifest,
        "index_manifest_sha256": index_manifest_sha256,
        "metrics": {
            "complete_support_set_rate_at_k": {
                str(k): complete_hits[k] / query_count for k in normalized_ks
            },
            "macro_supporting_document_recall_at_k": {
                str(k): macro_totals[k] / query_count for k in normalized_ks
            },
            "micro_supporting_document_recall_at_k": {
                str(k): micro_hits[k] / total_supporting for k in normalized_ks
            },
            "queries": query_count,
            "supporting_documents": total_supporting,
        },
        "per_example": per_example,
        "quality_scope": "retrieval_only_not_agent_performance",
        "record_type": "retrieval_recall_diagnostic",
        "schema_version": DIAGNOSTIC_SCHEMA_VERSION,
        "subset": {
            "example_ids": [example.example_id for example in examples.examples],
            "selection": "ordered_prefix",
            "source_bytes": examples.source_bytes,
            "source_records": examples.source_records,
            "source_sha256": examples.source_sha256,
        },
    }


def write_recall_report(path: str | Path, report: dict[str, object]) -> Path:
    """Write one deterministic diagnostic report."""

    output_path = Path(path)
    write_bytes_atomic(output_path, stable_json_bytes(report))
    return output_path
