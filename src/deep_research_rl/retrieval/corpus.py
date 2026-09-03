"""Strict loading and fingerprinting for traceable retrieval corpora."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from deep_research_rl.core.models import Document
from deep_research_rl.retrieval.errors import RetrievalError


@dataclass(frozen=True, slots=True)
class CorpusFingerprint:
    """Content and identity digest for one ordered JSONL corpus."""

    bytes: int
    documents: int
    sha256: str
    document_ids_sha256: str

    def to_dict(self) -> dict[str, object]:
        """Return the stable index-manifest representation."""

        return {
            "bytes": self.bytes,
            "document_ids_sha256": self.document_ids_sha256,
            "documents": self.documents,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class CorpusSnapshot:
    """An immutable ordered corpus plus lookup and integrity metadata."""

    path: Path
    documents: tuple[Document, ...]
    fingerprint: CorpusFingerprint

    def resolve(self, document_id: str) -> Document:
        """Return the exact corpus record for a stable document ID."""

        for document in self.documents:
            if document.document_id == document_id:
                return document
        raise KeyError(document_id)


def _required_string(record: object, field: str, *, line_number: int) -> str:
    if not isinstance(record, dict):
        raise RetrievalError(f"corpus line {line_number} must be a JSON object")
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise RetrievalError(f"corpus line {line_number} field {field} must be non-empty text")
    return value


def load_corpus(path: str | Path) -> CorpusSnapshot:
    """Load an ordered JSONL corpus and compute byte/cardinality/ID hashes."""

    corpus_path = Path(path)
    documents: list[Document] = []
    seen_ids: set[str] = set()
    id_hasher = hashlib.sha256()
    corpus_hasher = hashlib.sha256()
    byte_count = 0
    try:
        with corpus_path.open("rb") as corpus_file:
            for line_number, raw_line in enumerate(corpus_file, 1):
                corpus_hasher.update(raw_line)
                byte_count += len(raw_line)
                if not raw_line.strip():
                    continue
                try:
                    value: object = json.loads(raw_line)
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise RetrievalError(
                        f"invalid JSON on corpus line {line_number}: {error}"
                    ) from error
                document_id = _required_string(value, "document_id", line_number=line_number)
                title = _required_string(value, "title", line_number=line_number)
                text = _required_string(value, "text", line_number=line_number)
                if document_id in seen_ids:
                    raise RetrievalError(
                        f"duplicate corpus document_id on line {line_number}: {document_id}"
                    )
                seen_ids.add(document_id)
                id_hasher.update(document_id.encode("utf-8"))
                id_hasher.update(b"\n")
                documents.append(Document(document_id=document_id, title=title, text=text))
    except OSError as error:
        raise RetrievalError(f"could not read corpus {corpus_path}: {error}") from error

    if not documents:
        raise RetrievalError(f"corpus must contain at least one document: {corpus_path}")
    return CorpusSnapshot(
        path=corpus_path,
        documents=tuple(documents),
        fingerprint=CorpusFingerprint(
            bytes=byte_count,
            documents=len(documents),
            sha256=corpus_hasher.hexdigest(),
            document_ids_sha256=id_hasher.hexdigest(),
        ),
    )
