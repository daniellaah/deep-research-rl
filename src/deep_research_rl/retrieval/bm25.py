"""Deterministic Okapi BM25 with a portable, integrity-checked local index."""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path

from deep_research_rl.core.models import Document, SearchResult
from deep_research_rl.retrieval.corpus import load_corpus
from deep_research_rl.retrieval.errors import IndexIntegrityError
from deep_research_rl.retrieval.index import (
    INDEX_MANIFEST_SCHEMA_VERSION,
    file_metadata,
    manifest_mapping,
    require_array,
    require_integer,
    require_number,
    stable_json_bytes,
    verify_common_manifest,
    write_bytes_atomic,
    write_manifest,
)

BM25_INDEX_FILENAME = "bm25.json"
BM25_FORMAT_VERSION = 1
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> tuple[str, ...]:
    """Apply the frozen dependency-light BM25 tokenization rule."""

    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def _build_statistics(
    documents: Sequence[Document],
) -> tuple[tuple[int, ...], dict[str, tuple[tuple[int, int], ...]]]:
    lengths: list[int] = []
    mutable_postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for document_index, document in enumerate(documents):
        frequencies = Counter(tokenize(f"{document.title} {document.text}"))
        lengths.append(sum(frequencies.values()))
        for term, frequency in sorted(frequencies.items()):
            mutable_postings[term].append((document_index, frequency))
    return (
        tuple(lengths),
        {term: tuple(values) for term, values in sorted(mutable_postings.items())},
    )


class BM25Retriever:
    """Rank a fixed corpus with standard Okapi BM25 scoring."""

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        top_k: int = 1,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        lengths, postings = _build_statistics(documents)
        self._initialize(tuple(documents), lengths, postings, top_k=top_k, k1=k1, b=b)

    @classmethod
    def _from_statistics(
        cls,
        documents: tuple[Document, ...],
        lengths: tuple[int, ...],
        postings: dict[str, tuple[tuple[int, int], ...]],
        *,
        top_k: int,
        k1: float,
        b: float,
    ) -> BM25Retriever:
        instance = cls.__new__(cls)
        instance._initialize(documents, lengths, postings, top_k=top_k, k1=k1, b=b)
        return instance

    def _initialize(
        self,
        documents: tuple[Document, ...],
        lengths: tuple[int, ...],
        postings: dict[str, tuple[tuple[int, int], ...]],
        *,
        top_k: int,
        k1: float,
        b: float,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")
        if len(documents) != len(lengths):
            raise ValueError("documents and BM25 length statistics must have equal cardinality")
        self._documents = documents
        self._document_lengths = lengths
        self._postings = postings
        self._top_k = top_k
        self._k1 = k1
        self._b = b
        self._average_document_length = sum(lengths) / len(lengths) if lengths else 0.0

    def search(self, query: str) -> tuple[SearchResult, ...]:
        """Return positive-score results with stable score/ID tie breaking."""

        query_terms = set(tokenize(query))
        if not query_terms or not self._documents:
            return ()

        corpus_size = len(self._documents)
        scores: dict[int, float] = defaultdict(float)
        for term in query_terms:
            postings = self._postings.get(term, ())
            if not postings:
                continue
            document_frequency = len(postings)
            inverse_document_frequency = math.log(
                1 + (corpus_size - document_frequency + 0.5) / (document_frequency + 0.5)
            )
            for document_index, term_frequency in postings:
                document_length = self._document_lengths[document_index]
                length_ratio = (
                    document_length / self._average_document_length
                    if self._average_document_length
                    else 0.0
                )
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                scores[document_index] += inverse_document_frequency * (
                    term_frequency * (self._k1 + 1) / denominator
                )

        ranked = sorted(
            (
                (score, self._documents[document_index])
                for document_index, score in scores.items()
                if score > 0
            ),
            key=lambda item: (-item[0], item[1].document_id),
        )[: self._top_k]
        return tuple(
            SearchResult(
                document_id=document.document_id,
                title=document.title,
                text=document.text,
                score=score,
                rank=rank,
            )
            for rank, (score, document) in enumerate(ranked, 1)
        )

    def search_batch(self, queries: Sequence[str]) -> tuple[tuple[SearchResult, ...], ...]:
        """Apply exactly the single-query path to every query in order."""

        return tuple(self.search(query) for query in queries)


def _index_payload(
    lengths: tuple[int, ...],
    postings: dict[str, tuple[tuple[int, int], ...]],
) -> dict[str, object]:
    return {
        "document_lengths": list(lengths),
        "format_version": BM25_FORMAT_VERSION,
        "postings": {
            term: [list(posting) for posting in term_postings]
            for term, term_postings in postings.items()
        },
    }


def build_bm25_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> Path:
    """Build a deterministic portable BM25 index and integrity manifest."""

    if k1 <= 0:
        raise ValueError("k1 must be positive")
    if not 0 <= b <= 1:
        raise ValueError("b must be between 0 and 1")
    corpus = load_corpus(corpus_path)
    lengths, postings = _build_statistics(corpus.documents)
    directory = Path(index_dir)
    index_path = directory / BM25_INDEX_FILENAME
    write_bytes_atomic(index_path, stable_json_bytes(_index_payload(lengths, postings)))
    manifest: dict[str, object] = {
        "artifacts": {"bm25": file_metadata(index_path, relative_to=directory)},
        "backend": "bm25",
        "corpus": corpus.fingerprint.to_dict(),
        "parameters": {"b": b, "k1": k1, "tokenizer": "unicode_word_lower_v1"},
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
        "statistics": {
            "documents": len(lengths),
            "terms": len(postings),
            "total_tokens": sum(lengths),
        },
    }
    return write_manifest(directory, manifest)


def _load_payload(
    path: Path,
    *,
    document_count: int,
) -> tuple[tuple[int, ...], dict[str, tuple[tuple[int, int], ...]]]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise IndexIntegrityError(f"could not read BM25 index {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise IndexIntegrityError(f"invalid BM25 index JSON {path}: {error}") from error
    if not isinstance(value, dict):
        raise IndexIntegrityError("BM25 index must be an object")
    payload = {key: item for key, item in value.items() if isinstance(key, str)}
    version = require_integer(payload.get("format_version"), "format_version")
    if version != BM25_FORMAT_VERSION:
        raise IndexIntegrityError(f"unsupported BM25 format_version: {version}")

    lengths = tuple(
        require_integer(item, "document_lengths[]")
        for item in require_array(payload.get("document_lengths"), "document_lengths")
    )
    if len(lengths) != document_count or any(length < 0 for length in lengths):
        raise IndexIntegrityError("BM25 document lengths do not match corpus cardinality")

    raw_postings = payload.get("postings")
    if not isinstance(raw_postings, dict) or any(
        not isinstance(term, str) for term in raw_postings
    ):
        raise IndexIntegrityError("postings must be an object keyed by terms")
    postings: dict[str, tuple[tuple[int, int], ...]] = {}
    for raw_term, raw_values in raw_postings.items():
        if not isinstance(raw_term, str) or not raw_term:
            raise IndexIntegrityError("postings terms must be non-empty strings")
        parsed: list[tuple[int, int]] = []
        for raw_posting in require_array(raw_values, f"postings.{raw_term}"):
            pair = require_array(raw_posting, f"postings.{raw_term}[]")
            if len(pair) != 2:
                raise IndexIntegrityError(f"postings.{raw_term} entries must have two integers")
            document_index = require_integer(pair[0], f"postings.{raw_term}[].document")
            frequency = require_integer(pair[1], f"postings.{raw_term}[].frequency")
            if not 0 <= document_index < document_count or frequency < 1:
                raise IndexIntegrityError(f"invalid posting for term {raw_term}")
            parsed.append((document_index, frequency))
        if [item[0] for item in parsed] != sorted({item[0] for item in parsed}):
            raise IndexIntegrityError(f"postings for term {raw_term} are duplicated or unordered")
        postings[raw_term] = tuple(parsed)
    return lengths, postings


def load_bm25_index(
    index_dir: str | Path,
    corpus_path: str | Path,
    *,
    top_k: int = 5,
) -> BM25Retriever:
    """Load BM25 only after verifying every manifest and corpus invariant."""

    corpus = load_corpus(corpus_path)
    directory = Path(index_dir)
    manifest = verify_common_manifest(directory, corpus, expected_backend="bm25")
    artifacts = manifest_mapping(manifest, "artifacts")
    bm25_artifact = artifacts.get("bm25")
    if set(artifacts) != {"bm25"} or not isinstance(bm25_artifact, dict):
        raise IndexIntegrityError("BM25 manifest must declare exactly the bm25 artifact")
    if bm25_artifact.get("path") != BM25_INDEX_FILENAME:
        raise IndexIntegrityError("BM25 artifact path must be bm25.json")
    parameters = manifest_mapping(manifest, "parameters")
    k1 = require_number(parameters.get("k1"), "parameters.k1")
    b = require_number(parameters.get("b"), "parameters.b")
    if parameters.get("tokenizer") != "unicode_word_lower_v1":
        raise IndexIntegrityError("unsupported BM25 tokenizer")
    lengths, postings = _load_payload(
        directory / BM25_INDEX_FILENAME,
        document_count=len(corpus.documents),
    )
    statistics = manifest_mapping(manifest, "statistics")
    expected_statistics = {
        "documents": len(lengths),
        "terms": len(postings),
        "total_tokens": sum(lengths),
    }
    if statistics != expected_statistics:
        raise IndexIntegrityError("BM25 statistics do not match the saved index")
    return BM25Retriever._from_statistics(
        corpus.documents,
        lengths,
        postings,
        top_k=top_k,
        k1=k1,
        b=b,
    )


def verify_bm25_index(index_dir: str | Path, corpus_path: str | Path) -> dict[str, object]:
    """Verify BM25 artifacts and return their manifest without executing a query."""

    load_bm25_index(index_dir, corpus_path, top_k=1)
    corpus = load_corpus(corpus_path)
    return verify_common_manifest(index_dir, corpus, expected_backend="bm25")
