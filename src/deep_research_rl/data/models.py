"""Stable logical records produced by the HotpotQA conversion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from deep_research_rl.core.models import Example

DATA_SCHEMA_VERSION = 1
DATASET_NAME = "hotpot_qa"
DATASET_VARIANT = "distractor"
DATA_SOURCE_NAME = "hotpotqa_distractor"

Split = Literal["train", "validation"]


class DataRecordError(ValueError):
    """Raised when a logical dataset record violates its schema."""


@dataclass(frozen=True, slots=True)
class SupportingFact:
    """One sentence-level HotpotQA evidence label."""

    title: str
    sentence_index: int
    document_id: str

    def __post_init__(self) -> None:
        if not self.title:
            raise DataRecordError("supporting fact title must not be empty")
        if self.sentence_index < 0:
            raise DataRecordError("supporting fact sentence_index must not be negative")
        if not self.document_id:
            raise DataRecordError("supporting fact document_id must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return the version-independent nested JSON shape."""

        return {
            "document_id": self.document_id,
            "sentence_index": self.sentence_index,
            "title": self.title,
        }


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    """One deduplicated paragraph with sentence boundaries preserved."""

    document_id: str
    title: str
    text: str
    sentences: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.document_id:
            raise DataRecordError("corpus document_id must not be empty")
        if not self.title:
            raise DataRecordError("corpus title must not be empty")
        if not self.text:
            raise DataRecordError("corpus text must not be empty")
        if not self.sentences:
            raise DataRecordError("corpus sentences must not be empty")

    def to_dict(self) -> dict[str, object]:
        """Return the stable corpus JSONL record."""

        return {
            "document_id": self.document_id,
            "record_type": "hotpotqa_corpus_document",
            "schema_version": DATA_SCHEMA_VERSION,
            "sentences": list(self.sentences),
            "text": self.text,
            "title": self.title,
        }


@dataclass(frozen=True, slots=True)
class HotpotQAExample:
    """One prompt plus evaluation labels and corpus references."""

    example_id: str
    question: str
    answers: tuple[str, ...]
    split: Split
    level: str
    question_type: str
    supporting_facts: tuple[SupportingFact, ...]
    supporting_titles: tuple[str, ...]
    supporting_document_ids: tuple[str, ...]
    context_document_ids: tuple[str, ...]
    source_revision: str

    def __post_init__(self) -> None:
        if not self.example_id:
            raise DataRecordError("example_id must not be empty")
        if not self.question or self.question != self.question.strip():
            raise DataRecordError("question must be non-empty and trimmed")
        if not self.answers or any(not answer for answer in self.answers):
            raise DataRecordError("answers must contain non-empty strings")
        if self.split not in {"train", "validation"}:
            raise DataRecordError(f"unsupported split: {self.split}")
        if not self.level:
            raise DataRecordError("level must not be empty")
        if not self.question_type:
            raise DataRecordError("question_type must not be empty")
        if not self.supporting_facts:
            raise DataRecordError("supporting_facts must not be empty")
        if not self.supporting_titles:
            raise DataRecordError("supporting_titles must not be empty")
        if not self.supporting_document_ids:
            raise DataRecordError("supporting_document_ids must not be empty")
        if not self.context_document_ids:
            raise DataRecordError("context_document_ids must not be empty")
        if not self.source_revision:
            raise DataRecordError("source_revision must not be empty")
        fact_titles = tuple(dict.fromkeys(fact.title for fact in self.supporting_facts))
        if fact_titles != self.supporting_titles:
            raise DataRecordError("supporting_titles must match supporting_facts order")
        fact_document_ids = tuple(dict.fromkeys(fact.document_id for fact in self.supporting_facts))
        if fact_document_ids != self.supporting_document_ids:
            raise DataRecordError("supporting_document_ids must match supporting_facts order")
        if not set(self.supporting_document_ids).issubset(self.context_document_ids):
            raise DataRecordError("supporting documents must appear in context_document_ids")

    @property
    def prompt(self) -> list[dict[str, str]]:
        """Return the complete policy-visible initial prompt.

        Evaluation labels intentionally cannot be injected through this property.
        """

        return [{"content": self.question, "role": "user"}]

    def to_dict(self) -> dict[str, object]:
        """Return the stable canonical example JSONL record."""

        return {
            "answers": list(self.answers),
            "context_document_ids": list(self.context_document_ids),
            "dataset": DATASET_NAME,
            "example_id": self.example_id,
            "level": self.level,
            "prompt": self.prompt,
            "question": self.question,
            "question_type": self.question_type,
            "record_type": "hotpotqa_example",
            "schema_version": DATA_SCHEMA_VERSION,
            "source_revision": self.source_revision,
            "split": self.split,
            "supporting_document_ids": list(self.supporting_document_ids),
            "supporting_facts": [fact.to_dict() for fact in self.supporting_facts],
            "supporting_titles": list(self.supporting_titles),
            "variant": DATASET_VARIANT,
        }

    def to_agent_r1_dict(self, row_index: int) -> dict[str, object]:
        """Adapt to the logical columns consumed by Agent-R1's RLHFDataset.

        Extra evaluation metadata stays outside ``prompt`` and is therefore not policy-visible.
        """

        if row_index < 0:
            raise DataRecordError("row_index must not be negative")
        return {
            "data_source": DATA_SOURCE_NAME,
            "extra_info": {
                "answers": list(self.answers),
                "index": row_index,
                "level": self.level,
                "question_id": self.example_id,
                "source_revision": self.source_revision,
                "split": self.split,
                "supporting_document_ids": list(self.supporting_document_ids),
                "supporting_facts": [fact.to_dict() for fact in self.supporting_facts],
                "supporting_titles": list(self.supporting_titles),
                "type": self.question_type,
            },
            "prompt": self.prompt,
            "reward_model": {"ground_truth": self.answers[0], "style": "rule"},
        }

    def to_core_example(self) -> Example:
        """Adapt to the dependency-light local episode contract."""

        return Example(
            example_id=self.example_id,
            question=self.question,
            answers=self.answers,
            supporting_document_ids=self.supporting_document_ids,
            source=f"{DATA_SOURCE_NAME}:{self.split}:{self.source_revision}",
            synthetic=False,
            benchmark_eligible=True,
        )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DataRecordError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _list(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise DataRecordError(f"{field} must be an array")
    return list(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise DataRecordError(f"{field} must be a string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataRecordError(f"{field} must be an integer")
    return value


def _string_tuple(value: object, field: str) -> tuple[str, ...]:
    return tuple(_string(item, f"{field}[]") for item in _list(value, field))


def hotpotqa_example_from_dict(value: object) -> HotpotQAExample:
    """Validate and reconstruct one canonical example record."""

    record = _mapping(value, "example")
    if _integer(record.get("schema_version"), "schema_version") != DATA_SCHEMA_VERSION:
        raise DataRecordError("unsupported example schema_version")
    if _string(record.get("record_type"), "record_type") != "hotpotqa_example":
        raise DataRecordError("record_type must be hotpotqa_example")
    if _string(record.get("dataset"), "dataset") != DATASET_NAME:
        raise DataRecordError(f"dataset must be {DATASET_NAME}")
    if _string(record.get("variant"), "variant") != DATASET_VARIANT:
        raise DataRecordError(f"variant must be {DATASET_VARIANT}")

    question = _string(record.get("question"), "question")
    prompt = _list(record.get("prompt"), "prompt")
    expected_prompt: list[object] = [{"content": question, "role": "user"}]
    if prompt != expected_prompt:
        raise DataRecordError("prompt must contain only the user question")

    split_value = _string(record.get("split"), "split")
    if split_value == "train":
        split: Split = "train"
    elif split_value == "validation":
        split = "validation"
    else:
        raise DataRecordError(f"unsupported split: {split_value}")

    facts = []
    for index, fact_value in enumerate(_list(record.get("supporting_facts"), "supporting_facts")):
        fact = _mapping(fact_value, f"supporting_facts[{index}]")
        facts.append(
            SupportingFact(
                title=_string(fact.get("title"), f"supporting_facts[{index}].title"),
                sentence_index=_integer(
                    fact.get("sentence_index"),
                    f"supporting_facts[{index}].sentence_index",
                ),
                document_id=_string(
                    fact.get("document_id"),
                    f"supporting_facts[{index}].document_id",
                ),
            )
        )

    return HotpotQAExample(
        example_id=_string(record.get("example_id"), "example_id"),
        question=question,
        answers=_string_tuple(record.get("answers"), "answers"),
        split=split,
        level=_string(record.get("level"), "level"),
        question_type=_string(record.get("question_type"), "question_type"),
        supporting_facts=tuple(facts),
        supporting_titles=_string_tuple(record.get("supporting_titles"), "supporting_titles"),
        supporting_document_ids=_string_tuple(
            record.get("supporting_document_ids"),
            "supporting_document_ids",
        ),
        context_document_ids=_string_tuple(
            record.get("context_document_ids"),
            "context_document_ids",
        ),
        source_revision=_string(record.get("source_revision"), "source_revision"),
    )
