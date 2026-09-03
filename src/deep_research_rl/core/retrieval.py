"""A deterministic in-memory BM25 retriever for local validation."""

from __future__ import annotations

import math
import re
from collections import Counter

from deep_research_rl.core.models import Document

_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever:
    """Rank a fixed corpus with standard Okapi BM25 scoring."""

    def __init__(
        self,
        documents: tuple[Document, ...],
        *,
        top_k: int = 1,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self._documents = documents
        self._top_k = top_k
        self._k1 = k1
        self._b = b
        tokenized_documents = [
            _tokenize(f"{document.title} {document.text}") for document in documents
        ]
        self._term_frequencies = tuple(Counter(tokens) for tokens in tokenized_documents)
        self._document_lengths = tuple(len(tokens) for tokens in tokenized_documents)
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

        document_frequencies: Counter[str] = Counter()
        for frequencies in self._term_frequencies:
            document_frequencies.update(frequencies.keys())
        corpus_size = len(documents)
        self._inverse_document_frequencies = {
            term: math.log(1 + (corpus_size - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequencies.items()
        }

    def search(self, query: str) -> tuple[Document, ...]:
        """Return up to ``top_k`` positive-score documents in stable rank order."""

        query_terms = set(_tokenize(query))
        if not query_terms or not self._documents:
            return ()

        scores: list[tuple[float, Document]] = []
        for document, frequencies, document_length in zip(
            self._documents,
            self._term_frequencies,
            self._document_lengths,
            strict=True,
        ):
            score = 0.0
            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                length_ratio = (
                    document_length / self._average_document_length
                    if self._average_document_length
                    else 0.0
                )
                denominator = term_frequency + self._k1 * (1 - self._b + self._b * length_ratio)
                score += self._inverse_document_frequencies[term] * (
                    term_frequency * (self._k1 + 1) / denominator
                )
            if score > 0:
                scores.append((score, document))

        ranked = sorted(scores, key=lambda item: (-item[0], item[1].document_id))
        return tuple(document for _, document in ranked[: self._top_k])
