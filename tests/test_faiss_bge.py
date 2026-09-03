import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from deep_research_rl.core.models import Document
from deep_research_rl.retrieval.errors import IndexIntegrityError
from deep_research_rl.retrieval.faiss_bge import (
    AGENT_R1_REVISION,
    DEFAULT_QUERY_INSTRUCTION,
    AgentR1FaissBGEToolAdapter,
    BGEEncoder,
    build_faiss_bge_index,
    load_faiss_bge_index,
)
from deep_research_rl.retrieval.index import load_manifest

pytest.importorskip("faiss")
pytest.importorskip("numpy")


class FixtureBGEEncoder:
    model_name = "fixture/bge"
    model_revision = "1" * 40
    query_instruction = DEFAULT_QUERY_INSTRUCTION

    @staticmethod
    def _vectors(texts: Sequence[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            lowered = text.lower()
            vectors.append(
                [
                    float("alpha" in lowered),
                    float("gamma" in lowered),
                    float("delta" in lowered),
                ]
            )
        return vectors

    def encode_corpus(self, texts: Sequence[str], *, batch_size: int) -> object:
        assert batch_size > 0
        return self._vectors(texts)

    def encode_queries(self, queries: Sequence[str]) -> object:
        return self._vectors(queries)


class WrongFixtureBGEEncoder(FixtureBGEEncoder):
    model_revision = "2" * 40


def _write_corpus(path: Path) -> tuple[Path, tuple[Document, ...]]:
    documents = (
        Document("doc-alpha", "Alpha", "Alpha passage."),
        Document("doc-gamma", "Gamma", "Gamma passage."),
        Document("doc-delta", "Delta", "Delta passage."),
    )
    records = [
        {"document_id": item.document_id, "text": item.text, "title": item.title}
        for item in documents
    ]
    path.write_text(
        "".join(f"{json.dumps(record, sort_keys=True)}\n" for record in records),
        encoding="utf-8",
    )
    return path, documents


@pytest.mark.integration
def test_faiss_bge_small_index_batch_consistency_traceability_and_agent_r1_adapter(
    tmp_path: Path,
) -> None:
    corpus_path, documents = _write_corpus(tmp_path / "corpus.jsonl")
    index_dir = tmp_path / "index"
    encoder = FixtureBGEEncoder()

    build_faiss_bge_index(corpus_path, index_dir, encoder=encoder, batch_size=2)
    retriever = load_faiss_bge_index(
        index_dir,
        corpus_path,
        top_k=1,
        encoder=encoder,
    )
    single = retriever.search("gamma")
    batch = retriever.search_batch(("gamma", "delta"))

    assert batch[0] == single
    assert (single[0].document_id, single[0].title, single[0].text) == (
        documents[1].document_id,
        documents[1].title,
        documents[1].text,
    )
    assert single[0].rank == 1
    assert single[0].score == pytest.approx(1.0)
    assert batch[1][0].document_id == "doc-delta"
    manifest = load_manifest(index_dir)
    compatibility = manifest["compatibility"]
    assert isinstance(compatibility, dict)
    agent_r1 = compatibility["agent_r1"]
    assert isinstance(agent_r1, dict)
    assert agent_r1["revision"] == AGENT_R1_REVISION

    adapter = AgentR1FaissBGEToolAdapter(retriever)
    response = adapter.execute({"query": "gamma"})
    assert response["success"] is True
    assert json.loads(str(response["content"])) == {"results": ["Gamma Gamma passage."]}
    assert adapter.batch_execute([{"query": "gamma"}]) == [response]


@pytest.mark.integration
def test_faiss_index_rejects_runtime_encoder_identity_mismatch(tmp_path: Path) -> None:
    corpus_path, _ = _write_corpus(tmp_path / "corpus.jsonl")
    index_dir = tmp_path / "index"
    build_faiss_bge_index(corpus_path, index_dir, encoder=FixtureBGEEncoder())

    with pytest.raises(IndexIntegrityError, match="encoder identity"):
        load_faiss_bge_index(
            index_dir,
            corpus_path,
            encoder=WrongFixtureBGEEncoder(),
        )


def test_bge_encoder_requires_resolved_revision() -> None:
    with pytest.raises(ValueError, match="immutable resolved revision"):
        BGEEncoder(model_revision="UNRESOLVED")
