import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from deep_research_rl.cli import main
from deep_research_rl.core.models import Document, Example, SearchResult
from deep_research_rl.data.models import HotpotQAExample, SupportingFact
from deep_research_rl.retrieval.bm25 import (
    BM25Retriever,
    build_bm25_index,
    load_bm25_index,
)
from deep_research_rl.retrieval.diagnostics import DiagnosticExamples, build_recall_report
from deep_research_rl.retrieval.errors import IndexIntegrityError
from deep_research_rl.retrieval.index import load_manifest


def _write_corpus(path: Path, documents: Sequence[Document]) -> Path:
    records = [
        {
            "document_id": document.document_id,
            "record_type": "hotpotqa_corpus_document",
            "schema_version": 1,
            "sentences": [document.text],
            "text": document.text,
            "title": document.title,
        }
        for document in documents
    ]
    path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def retrieval_documents() -> tuple[Document, ...]:
    return (
        Document("doc-alpha", "Alpha Scientist", "Alpha works at Beta Observatory."),
        Document("doc-beta", "Beta Observatory", "The observatory is in Gamma City."),
        Document("doc-delta", "Delta Lake", "Delta Lake is outside Epsilon Village."),
    )


def test_bm25_single_and_batch_return_uniform_scored_schema(
    retrieval_documents: tuple[Document, ...],
) -> None:
    retriever = BM25Retriever(retrieval_documents, top_k=2)

    single = retriever.search("Alpha Scientist")
    batch = retriever.search_batch(("Alpha Scientist", "Gamma City"))

    assert batch[0] == single
    assert single[0] == SearchResult(
        document_id="doc-alpha",
        title="Alpha Scientist",
        text="Alpha works at Beta Observatory.",
        score=single[0].score,
        rank=1,
    )
    assert single[0].score > 0
    assert [result.rank for result in batch[1]] == list(range(1, len(batch[1]) + 1))


def test_bm25_index_round_trip_and_result_traceability(
    tmp_path: Path,
    retrieval_documents: tuple[Document, ...],
) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.jsonl", retrieval_documents)
    index_dir = tmp_path / "index"

    manifest_path = build_bm25_index(corpus_path, index_dir)
    retriever = load_bm25_index(index_dir, corpus_path, top_k=3)
    result = retriever.search("Gamma City")[0]

    assert manifest_path == index_dir / "manifest.json"
    assert (result.document_id, result.title, result.text) == (
        retrieval_documents[1].document_id,
        retrieval_documents[1].title,
        retrieval_documents[1].text,
    )
    manifest = load_manifest(index_dir)
    assert manifest["backend"] == "bm25"
    corpus_metadata = manifest["corpus"]
    assert isinstance(corpus_metadata, dict)
    assert corpus_metadata["documents"] == 3


def test_index_rejects_same_cardinality_corpus_hash_mismatch(
    tmp_path: Path,
    retrieval_documents: tuple[Document, ...],
) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.jsonl", retrieval_documents)
    index_dir = tmp_path / "index"
    build_bm25_index(corpus_path, index_dir)
    changed = list(retrieval_documents)
    changed[0] = Document("doc-alpha", "Alpha Scientist", "Changed text.")
    _write_corpus(corpus_path, changed)

    with pytest.raises(IndexIntegrityError, match="index/corpus mismatch"):
        load_bm25_index(index_dir, corpus_path)


def test_index_rejects_corpus_cardinality_mismatch(
    tmp_path: Path,
    retrieval_documents: tuple[Document, ...],
) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.jsonl", retrieval_documents)
    index_dir = tmp_path / "index"
    build_bm25_index(corpus_path, index_dir)
    _write_corpus(
        corpus_path,
        (*retrieval_documents, Document("doc-extra", "Extra", "Extra text.")),
    )

    with pytest.raises(IndexIntegrityError, match="index/corpus mismatch"):
        load_bm25_index(index_dir, corpus_path)


def test_index_rejects_corrupted_artifact(
    tmp_path: Path,
    retrieval_documents: tuple[Document, ...],
) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.jsonl", retrieval_documents)
    index_dir = tmp_path / "index"
    build_bm25_index(corpus_path, index_dir)
    index_path = index_dir / "bm25.json"
    index_path.write_bytes(index_path.read_bytes() + b"corruption")

    with pytest.raises(IndexIntegrityError, match="artifact size mismatch"):
        load_bm25_index(index_dir, corpus_path)


def test_index_rejects_wrong_backend_artifact_declaration(
    tmp_path: Path,
    retrieval_documents: tuple[Document, ...],
) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.jsonl", retrieval_documents)
    index_dir = tmp_path / "index"
    build_bm25_index(corpus_path, index_dir)
    manifest_path = index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"] = {"other": manifest["artifacts"]["bm25"]}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(IndexIntegrityError, match="exactly the bm25 artifact"):
        load_bm25_index(index_dir, corpus_path)


def test_supporting_document_recall_report_is_deterministic_and_retrieval_only(
    retrieval_documents: tuple[Document, ...],
) -> None:
    example = Example(
        example_id="example-1",
        question="Where does Alpha Scientist work and which city is it in?",
        answers=("Gamma City",),
        supporting_document_ids=("doc-alpha", "doc-beta"),
        source="fixture",
        synthetic=True,
        benchmark_eligible=False,
    )
    diagnostic_examples = DiagnosticExamples(
        examples=(example,),
        source_bytes=123,
        source_records=1,
        source_sha256="a" * 64,
    )
    report = build_recall_report(
        BM25Retriever(retrieval_documents, top_k=3),
        diagnostic_examples,
        backend="bm25",
        ks=(1, 3),
        corpus_metadata={"documents": 3},
        index_manifest={"backend": "bm25"},
        index_manifest_sha256="b" * 64,
    )

    assert report["quality_scope"] == "retrieval_only_not_agent_performance"
    metrics = report["metrics"]
    per_example = report["per_example"]
    assert isinstance(metrics, dict)
    assert isinstance(metrics["macro_supporting_document_recall_at_k"], dict)
    assert metrics["macro_supporting_document_recall_at_k"]["3"] == 1.0
    assert isinstance(per_example, list)
    assert isinstance(per_example[0], dict)
    assert per_example[0]["supporting_document_ids"] == ["doc-alpha", "doc-beta"]


def test_retrieval_cli_build_verify_search_and_diagnose(
    tmp_path: Path,
    retrieval_documents: tuple[Document, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    corpus_path = _write_corpus(tmp_path / "corpus.jsonl", retrieval_documents)
    index_dir = tmp_path / "index"
    examples_path = tmp_path / "validation.jsonl"
    example = HotpotQAExample(
        example_id="fixture-validation-1",
        question="Where does Alpha Scientist work and which city is it in?",
        answers=("Gamma City",),
        split="validation",
        level="easy",
        question_type="bridge",
        supporting_facts=(
            SupportingFact("Alpha Scientist", 0, "doc-alpha"),
            SupportingFact("Beta Observatory", 0, "doc-beta"),
        ),
        supporting_titles=("Alpha Scientist", "Beta Observatory"),
        supporting_document_ids=("doc-alpha", "doc-beta"),
        context_document_ids=("doc-alpha", "doc-beta", "doc-delta"),
        source_revision="fixture-v1",
    )
    examples_path.write_text(f"{json.dumps(example.to_dict(), sort_keys=True)}\n", encoding="utf-8")

    assert (
        main(
            [
                "retrieval",
                "build",
                "--backend",
                "bm25",
                "--corpus",
                str(corpus_path),
                "--index-dir",
                str(index_dir),
            ]
        )
        == 0
    )
    assert "bm25 index built" in capsys.readouterr().out
    assert (
        main(
            [
                "retrieval",
                "verify",
                "--corpus",
                str(corpus_path),
                "--index-dir",
                str(index_dir),
            ]
        )
        == 0
    )
    assert "documents=3" in capsys.readouterr().out
    assert (
        main(
            [
                "retrieval",
                "search",
                "--corpus",
                str(corpus_path),
                "--index-dir",
                str(index_dir),
                "--query",
                "Gamma City",
                "--top-k",
                "1",
            ]
        )
        == 0
    )
    search_payload = json.loads(capsys.readouterr().out)
    assert set(search_payload["results"][0]) == {"document_id", "rank", "score", "text", "title"}
    report_path = tmp_path / "report.json"
    assert (
        main(
            [
                "retrieval",
                "diagnose",
                "--corpus",
                str(corpus_path),
                "--examples",
                str(examples_path),
                "--index-dir",
                str(index_dir),
                "--output",
                str(report_path),
                "--ks",
                "1",
                "3",
            ]
        )
        == 0
    )
    assert "retrieval-only diagnostic verified" in capsys.readouterr().out
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["subset"]["example_ids"] == ["fixture-validation-1"]
