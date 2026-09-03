"""Clearly labeled synthetic data for deterministic local checks."""

from __future__ import annotations

from deep_research_rl.core.models import Document, Example


def synthetic_two_hop_fixture() -> tuple[Example, tuple[Document, ...]]:
    """Return a fictional two-hop example that is never eligible for benchmarking."""

    example = Example(
        example_id="synthetic-two-hop-001",
        question="Which city is home to the scientist who developed the Brindle Process?",
        answers=("Lumen City",),
        supporting_document_ids=("brindle-process", "mira-voss"),
        source="synthetic_non_benchmark",
        synthetic=True,
        benchmark_eligible=False,
    )
    documents = (
        Document(
            document_id="brindle-process",
            title="Brindle Process",
            text="The fictional Brindle Process was developed by scientist Mira Voss.",
        ),
        Document(
            document_id="mira-voss",
            title="Mira Voss",
            text="Mira Voss is a fictional scientist whose home is Lumen City.",
        ),
        Document(
            document_id="lumen-lake",
            title="Lumen Lake",
            text="Lumen Lake is a fictional lake near Copper Village.",
        ),
    )
    return example, documents
