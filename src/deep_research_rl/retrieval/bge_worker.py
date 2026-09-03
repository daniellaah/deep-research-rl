"""Isolated BGE encoder entry point for runtimes with incompatible OpenMP libraries."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from deep_research_rl.retrieval.errors import RetrievalError
from deep_research_rl.retrieval.faiss_bge import BGEEncoder, _host_array, _optional_module


def _string(record: dict[str, object], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise RetrievalError(f"{field} must be a non-empty string")
    return value


def _load_request(path: Path) -> dict[str, object]:
    value: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise RetrievalError("BGE worker request must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def main(argv: Sequence[str] | None = None) -> int:
    """Encode one corpus or query batch without importing FAISS."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    request = _load_request(args.request)
    raw_texts = request.get("texts")
    if not isinstance(raw_texts, list) or any(not isinstance(text, str) for text in raw_texts):
        raise RetrievalError("texts must be an array of strings")
    texts = [text for text in raw_texts if isinstance(text, str)]
    encoder = BGEEncoder(
        model_name=_string(request, "model_name"),
        model_revision=_string(request, "model_revision"),
        query_instruction=_string(request, "query_instruction"),
        device=_string(request, "device"),
        process_isolation=False,
    )
    mode = _string(request, "mode")
    if mode == "corpus":
        batch_size = request.get("batch_size")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool):
            raise RetrievalError("batch_size must be an integer for corpus encoding")
        encoded = encoder.encode_corpus(texts, batch_size=batch_size)
    elif mode == "queries":
        encoded = encoder.encode_queries(texts)
    else:
        raise RetrievalError(f"unsupported BGE worker mode: {mode}")
    numpy = _optional_module("numpy")
    vectors = _host_array(encoded, numpy)
    with args.output.open("wb") as output_file:
        numpy.save(output_file, vectors)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
