"""Agent-R1-compatible FAISS/BGE indexing behind optional runtime dependencies."""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from deep_research_rl.core.models import SearchResult
from deep_research_rl.retrieval.corpus import CorpusSnapshot, load_corpus
from deep_research_rl.retrieval.errors import IndexIntegrityError, RetrievalDependencyError
from deep_research_rl.retrieval.index import (
    INDEX_MANIFEST_SCHEMA_VERSION,
    file_metadata,
    manifest_mapping,
    require_integer,
    require_string,
    verify_common_manifest,
    write_manifest,
)

AGENT_R1_REVISION = "b124aa46534cbf2fb8bc8af11405774984c42ac7"
DEFAULT_BGE_MODEL = "BAAI/bge-large-en-v1.5"
DEFAULT_BGE_MODEL_REVISION = "d4aa6901d3a41ba39fb536a557fa166f842b0e09"
DEFAULT_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
FAISS_INDEX_FILENAME = "index.bin"
EMBEDDINGS_FILENAME = "hpqa_corpus.npy"
FAISS_FORMAT_VERSION = 1


class VectorEncoder(Protocol):
    """Minimal corpus/query encoder seam used by the FAISS adapter."""

    model_name: str
    model_revision: str
    query_instruction: str

    def encode_corpus(self, texts: Sequence[str], *, batch_size: int) -> object: ...

    def encode_queries(self, queries: Sequence[str]) -> object: ...


def _optional_module(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as error:
        raise RetrievalDependencyError(
            f"optional retrieval dependency {name!r} is unavailable; "
            "install the pinned retrieval extra with: pip install -e '.[retrieval]'"
        ) from error


class BGEEncoder:
    """Revision-pinned FlagEmbedding wrapper matching the Agent-R1 recipe semantics."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BGE_MODEL,
        model_revision: str = DEFAULT_BGE_MODEL_REVISION,
        query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
        device: str = "cpu",
        process_isolation: bool | None = None,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must not be empty")
        if not model_revision or model_revision == "UNRESOLVED":
            raise ValueError("model_revision must be an immutable resolved revision")
        if not query_instruction:
            raise ValueError("query_instruction must not be empty")
        if not device:
            raise ValueError("device must not be empty")
        self.model_name = model_name
        self.model_revision = model_revision
        self.query_instruction = query_instruction
        self.device = device
        self.process_isolation = (
            sys.platform == "darwin" if process_isolation is None else process_isolation
        )
        self._model: Any | None = None

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        model_path = Path(self.model_name).expanduser()
        if model_path.exists():
            resolved_model = str(model_path.resolve())
        else:
            hub = _optional_module("huggingface_hub")
            resolved_model = hub.snapshot_download(
                repo_id=self.model_name,
                revision=self.model_revision,
            )
        flag_embedding = _optional_module("FlagEmbedding")
        self._model = flag_embedding.FlagModel(
            resolved_model,
            query_instruction_for_retrieval=self.query_instruction,
            devices=self.device,
        )
        return self._model

    def encode_corpus(self, texts: Sequence[str], *, batch_size: int) -> object:
        """Encode title-prefixed passages through FlagEmbedding's corpus path."""

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.process_isolation:
            return self._encode_isolated("corpus", texts, batch_size=batch_size)
        model = self._load_model()
        try:
            return model.encode_corpus(list(texts), batch_size=batch_size)
        except TypeError:
            return model.encode_corpus(list(texts))

    def prepare(self) -> None:
        """Load the pinned model before FAISS on runtimes with competing OpenMP libraries."""

        if not self.process_isolation:
            self._load_model()

    def encode_queries(self, queries: Sequence[str]) -> object:
        """Encode queries with the frozen BGE retrieval instruction."""

        if self.process_isolation:
            return self._encode_isolated("queries", queries, batch_size=None)
        return self._load_model().encode_queries(list(queries))

    def _encode_isolated(
        self,
        mode: str,
        texts: Sequence[str],
        *,
        batch_size: int | None,
    ) -> object:
        numpy = _optional_module("numpy")
        request = {
            "batch_size": batch_size,
            "device": self.device,
            "mode": mode,
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "query_instruction": self.query_instruction,
            "texts": list(texts),
        }
        with tempfile.TemporaryDirectory(prefix="deepresearch-rl-bge-") as temporary_dir:
            directory = Path(temporary_dir)
            request_path = directory / "request.json"
            output_path = directory / "embeddings.npy"
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "deep_research_rl.retrieval.bge_worker",
                "--request",
                str(request_path),
                "--output",
                str(output_path),
            ]
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
            if completed.returncode != 0:
                detail = completed.stderr.strip() or completed.stdout.strip() or "unknown failure"
                raise RetrievalDependencyError(f"isolated BGE encoder failed: {detail}")
            return numpy.load(output_path)


def _host_array(value: object, numpy: Any) -> Any:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "float"):
        candidate = candidate.float()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    array = numpy.asarray(candidate, dtype=numpy.float32)
    if not array.flags.c_contiguous:
        array = numpy.ascontiguousarray(array)
    return array


def _validate_vectors(
    vectors: Any,
    numpy: Any,
    *,
    expected_rows: int,
    expected_dimension: int | None = None,
) -> Any:
    if vectors.ndim != 2:
        raise IndexIntegrityError("encoded vectors must be a two-dimensional matrix")
    if int(vectors.shape[0]) != expected_rows:
        raise IndexIntegrityError(
            f"encoded vector cardinality mismatch: expected {expected_rows}, "
            f"received {int(vectors.shape[0])}"
        )
    dimension = int(vectors.shape[1])
    if dimension < 1:
        raise IndexIntegrityError("encoded vector dimension must be positive")
    if expected_dimension is not None and dimension != expected_dimension:
        raise IndexIntegrityError(
            f"query vector dimension mismatch: expected {expected_dimension}, received {dimension}"
        )
    if not bool(numpy.isfinite(vectors).all()):
        raise IndexIntegrityError("encoded vectors must contain only finite values")
    return vectors


def _corpus_texts(corpus: CorpusSnapshot) -> list[str]:
    return [f"{document.title} {document.text}".strip() for document in corpus.documents]


def _write_numpy_atomic(numpy: Any, path: Path, vectors: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as output_file:
            numpy.save(output_file, vectors)
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _write_faiss_atomic(faiss: Any, path: Path, index: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        faiss.write_index(index, str(temporary))
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def build_faiss_bge_index(
    corpus_path: str | Path,
    index_dir: str | Path,
    *,
    encoder: VectorEncoder | None = None,
    model_name: str = DEFAULT_BGE_MODEL,
    model_revision: str = DEFAULT_BGE_MODEL_REVISION,
    query_instruction: str = DEFAULT_QUERY_INSTRUCTION,
    device: str = "cpu",
    batch_size: int = 128,
) -> Path:
    """Encode a corpus and build Agent-R1's exact Flat inner-product FAISS layout."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    corpus = load_corpus(corpus_path)
    effective_encoder = encoder or BGEEncoder(
        model_name=model_name,
        model_revision=model_revision,
        query_instruction=query_instruction,
        device=device,
    )
    if not effective_encoder.model_revision or effective_encoder.model_revision == "UNRESOLVED":
        raise ValueError("encoder model_revision must be an immutable resolved revision")

    numpy = _optional_module("numpy")
    vectors = _host_array(
        effective_encoder.encode_corpus(_corpus_texts(corpus), batch_size=batch_size),
        numpy,
    )
    vectors = _validate_vectors(vectors, numpy, expected_rows=len(corpus.documents))
    dimension = int(vectors.shape[1])
    faiss = _optional_module("faiss")
    index = faiss.index_factory(dimension, "Flat", faiss.METRIC_INNER_PRODUCT)
    index.add(vectors)
    if int(index.ntotal) != len(corpus.documents):
        raise IndexIntegrityError("FAISS index cardinality differs after add")

    directory = Path(index_dir)
    embeddings_path = directory / EMBEDDINGS_FILENAME
    index_path = directory / FAISS_INDEX_FILENAME
    _write_numpy_atomic(numpy, embeddings_path, vectors)
    _write_faiss_atomic(faiss, index_path, index)
    manifest: dict[str, object] = {
        "artifacts": {
            "embeddings": file_metadata(embeddings_path, relative_to=directory),
            "faiss_index": file_metadata(index_path, relative_to=directory),
        },
        "backend": "faiss_bge",
        "compatibility": {
            "agent_r1": {
                "corpus_text_format": "{title} {text}",
                "embedding_filename": EMBEDDINGS_FILENAME,
                "index_filename": FAISS_INDEX_FILENAME,
                "revision": AGENT_R1_REVISION,
            }
        },
        "corpus": corpus.fingerprint.to_dict(),
        "encoder": {
            "family": "bge",
            "model_name": effective_encoder.model_name,
            "model_revision": effective_encoder.model_revision,
            "query_instruction": effective_encoder.query_instruction,
        },
        "index": {
            "dimension": dimension,
            "factory": "Flat",
            "format_version": FAISS_FORMAT_VERSION,
            "metric": "inner_product",
            "vectors": len(corpus.documents),
        },
        "schema_version": INDEX_MANIFEST_SCHEMA_VERSION,
    }
    return write_manifest(directory, manifest)


def _verify_faiss_objects(
    directory: Path,
    manifest: dict[str, object],
    *,
    corpus_count: int,
) -> tuple[Any, int]:
    numpy = _optional_module("numpy")
    faiss = _optional_module("faiss")
    artifacts = manifest_mapping(manifest, "artifacts")
    if set(artifacts) != {"embeddings", "faiss_index"}:
        raise IndexIntegrityError(
            "FAISS manifest must declare exactly embeddings and faiss_index artifacts"
        )
    expected_artifact_paths = {
        "embeddings": EMBEDDINGS_FILENAME,
        "faiss_index": FAISS_INDEX_FILENAME,
    }
    for name, expected_path in expected_artifact_paths.items():
        metadata = artifacts.get(name)
        if not isinstance(metadata, dict) or metadata.get("path") != expected_path:
            raise IndexIntegrityError(f"FAISS {name} artifact path must be {expected_path}")
    index_metadata = manifest_mapping(manifest, "index")
    if index_metadata.get("factory") != "Flat" or index_metadata.get("metric") != "inner_product":
        raise IndexIntegrityError("FAISS index recipe must be Flat inner_product")
    version = require_integer(index_metadata.get("format_version"), "index.format_version")
    if version != FAISS_FORMAT_VERSION:
        raise IndexIntegrityError(f"unsupported FAISS format_version: {version}")
    dimension = require_integer(index_metadata.get("dimension"), "index.dimension")
    vector_count = require_integer(index_metadata.get("vectors"), "index.vectors")
    if vector_count != corpus_count:
        raise IndexIntegrityError("FAISS manifest vector count does not match corpus")

    vectors = numpy.load(directory / EMBEDDINGS_FILENAME, mmap_mode="r")
    _validate_vectors(
        vectors,
        numpy,
        expected_rows=corpus_count,
        expected_dimension=dimension,
    )
    if str(vectors.dtype) != "float32":
        raise IndexIntegrityError("saved FAISS embeddings must use float32")
    index = faiss.read_index(str(directory / FAISS_INDEX_FILENAME))
    if int(index.ntotal) != corpus_count:
        raise IndexIntegrityError("FAISS index.ntotal does not match corpus cardinality")
    if int(index.d) != dimension:
        raise IndexIntegrityError("FAISS index dimension does not match manifest")
    if not bool(index.is_trained):
        raise IndexIntegrityError("FAISS index must be trained")
    if int(index.metric_type) != int(faiss.METRIC_INNER_PRODUCT):
        raise IndexIntegrityError("FAISS index metric must be inner product")
    if type(index).__name__ not in {"IndexFlat", "IndexFlatIP"}:
        raise IndexIntegrityError("FAISS index implementation must be exact Flat search")
    return index, dimension


def _verify_compatibility(manifest: dict[str, object]) -> None:
    compatibility = manifest_mapping(manifest, "compatibility")
    agent_r1_value = compatibility.get("agent_r1")
    if not isinstance(agent_r1_value, dict):
        raise IndexIntegrityError("compatibility.agent_r1 must be an object")
    agent_r1 = {key: value for key, value in agent_r1_value.items() if isinstance(key, str)}
    expected = {
        "corpus_text_format": "{title} {text}",
        "embedding_filename": EMBEDDINGS_FILENAME,
        "index_filename": FAISS_INDEX_FILENAME,
        "revision": AGENT_R1_REVISION,
    }
    if agent_r1 != expected:
        raise IndexIntegrityError("Agent-R1 compatibility metadata differs from the pinned recipe")


def verify_faiss_bge_index(
    index_dir: str | Path,
    corpus_path: str | Path,
) -> dict[str, object]:
    """Verify corpus hashes, artifact hashes, vector shape and FAISS internals."""

    corpus = load_corpus(corpus_path)
    directory = Path(index_dir)
    manifest = verify_common_manifest(directory, corpus, expected_backend="faiss_bge")
    _verify_compatibility(manifest)
    _verify_faiss_objects(directory, manifest, corpus_count=len(corpus.documents))
    return manifest


class FaissBGERetriever:
    """Return uniform, traceable results from a verified Flat-IP FAISS index."""

    def __init__(
        self,
        corpus: CorpusSnapshot,
        index: Any,
        encoder: VectorEncoder,
        *,
        dimension: int,
        top_k: int = 5,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self._corpus = corpus
        self._index = index
        self._encoder = encoder
        self._dimension = dimension
        self._top_k = top_k

    def search(self, query: str) -> tuple[SearchResult, ...]:
        """Use the batch path for guaranteed single/batch behavioral consistency."""

        return self.search_batch((query,))[0]

    def search_batch(self, queries: Sequence[str]) -> tuple[tuple[SearchResult, ...], ...]:
        """Encode and search a query batch while preserving its input order."""

        if not queries:
            return ()
        if any(not query or query != query.strip() for query in queries):
            raise ValueError("queries must be non-empty and trimmed")
        numpy = _optional_module("numpy")
        query_vectors = _host_array(self._encoder.encode_queries(queries), numpy)
        query_vectors = _validate_vectors(
            query_vectors,
            numpy,
            expected_rows=len(queries),
            expected_dimension=self._dimension,
        )
        scores, identifiers = self._index.search(
            query_vectors,
            min(self._top_k, len(self._corpus.documents)),
        )
        batches: list[tuple[SearchResult, ...]] = []
        for row_index in range(len(queries)):
            hits = []
            for raw_score, raw_identifier in zip(
                scores[row_index], identifiers[row_index], strict=True
            ):
                identifier = int(raw_identifier)
                score = float(raw_score)
                if identifier < 0:
                    continue
                if identifier >= len(self._corpus.documents):
                    raise IndexIntegrityError("FAISS returned an out-of-range corpus row")
                if not math.isfinite(score):
                    raise IndexIntegrityError("FAISS returned a non-finite score")
                hits.append((score, self._corpus.documents[identifier]))
            hits.sort(key=lambda item: (-item[0], item[1].document_id))
            batches.append(
                tuple(
                    SearchResult(
                        document_id=document.document_id,
                        title=document.title,
                        text=document.text,
                        score=score,
                        rank=rank,
                    )
                    for rank, (score, document) in enumerate(hits, 1)
                )
            )
        return tuple(batches)


def load_faiss_bge_index(
    index_dir: str | Path,
    corpus_path: str | Path,
    *,
    top_k: int = 5,
    encoder: VectorEncoder | None = None,
    device: str = "cpu",
) -> FaissBGERetriever:
    """Load a dense retriever only after checking corpus, artifacts and encoder identity."""

    corpus = load_corpus(corpus_path)
    directory = Path(index_dir)
    manifest = verify_common_manifest(directory, corpus, expected_backend="faiss_bge")
    _verify_compatibility(manifest)
    encoder_metadata = manifest_mapping(manifest, "encoder")
    if encoder_metadata.get("family") != "bge":
        raise IndexIntegrityError("unsupported dense encoder family")
    model_name = require_string(encoder_metadata.get("model_name"), "encoder.model_name")
    model_revision = require_string(
        encoder_metadata.get("model_revision"), "encoder.model_revision"
    )
    query_instruction = require_string(
        encoder_metadata.get("query_instruction"), "encoder.query_instruction"
    )
    effective_encoder = encoder or BGEEncoder(
        model_name=model_name,
        model_revision=model_revision,
        query_instruction=query_instruction,
        device=device,
    )
    actual_identity = (
        effective_encoder.model_name,
        effective_encoder.model_revision,
        effective_encoder.query_instruction,
    )
    expected_identity = (model_name, model_revision, query_instruction)
    if actual_identity != expected_identity:
        raise IndexIntegrityError("runtime encoder identity does not match the saved index")
    if isinstance(effective_encoder, BGEEncoder):
        effective_encoder.prepare()
    index, dimension = _verify_faiss_objects(
        directory,
        manifest,
        corpus_count=len(corpus.documents),
    )
    return FaissBGERetriever(
        corpus,
        index,
        effective_encoder,
        dimension=dimension,
        top_k=top_k,
    )


def agent_r1_result_content(results: Sequence[SearchResult]) -> str:
    """Render the pinned recipe's legacy ``{\"results\": [...]}`` tool payload."""

    passages = [f"{result.title} {result.text}".strip() for result in results]
    return json.dumps({"results": passages}, ensure_ascii=False)


class AgentR1FaissBGEToolAdapter:
    """Expose a verified retriever through Agent-R1's execute/batch_execute shape."""

    def __init__(self, retriever: FaissBGERetriever) -> None:
        self._retriever = retriever

    def execute(self, args: dict[str, object]) -> dict[str, object]:
        """Execute one legacy tool request without hiding failures as valid results."""

        try:
            query = args.get("query")
            if not isinstance(query, str):
                raise ValueError("query must be a string")
            return {
                "content": agent_r1_result_content(self._retriever.search(query)),
                "success": True,
            }
        except (IndexIntegrityError, RetrievalDependencyError, ValueError) as error:
            return {"content": str(error), "success": False}

    def batch_execute(self, args_list: Sequence[dict[str, object]]) -> list[dict[str, object]]:
        """Execute a legacy request batch with one encoder/index call."""

        try:
            queries = []
            for args in args_list:
                query = args.get("query")
                if not isinstance(query, str):
                    raise ValueError("query must be a string")
                queries.append(query)
            return [
                {"content": agent_r1_result_content(results), "success": True}
                for results in self._retriever.search_batch(queries)
            ]
        except (IndexIntegrityError, RetrievalDependencyError, ValueError) as error:
            return [{"content": str(error), "success": False} for _ in args_list]
