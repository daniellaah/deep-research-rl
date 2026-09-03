"""Deterministic HotpotQA distractor conversion and build verification."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from contextlib import suppress
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

from deep_research_rl.data.models import (
    DATA_SCHEMA_VERSION,
    CorpusDocument,
    DataRecordError,
    HotpotQAExample,
    Split,
    SupportingFact,
    hotpotqa_example_from_dict,
)
from deep_research_rl.data.source import (
    DataSourceConfig,
    sha256_file,
    verify_source_file,
)

BUILD_MANIFEST_SCHEMA_VERSION = 1
SPLIT_ORDER: tuple[Split, ...] = ("train", "validation")


class DataPipelineError(ValueError):
    """Raised when conversion or build verification detects invalid data."""


@dataclass(frozen=True, slots=True)
class HotpotQABuildResult:
    """Paths and high-level counts for one completed build."""

    output_dir: Path
    manifest_path: Path
    train_examples: int
    validation_examples: int
    corpus_documents: int
    build_mode: str


def _stable_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise DataPipelineError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise DataPipelineError(f"{field} must be an array")
    return list(value)


def _string(
    value: object,
    field: str,
    *,
    trim: bool = False,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise DataPipelineError(f"{field} must be a string")
    result = value.strip() if trim else value
    if not result and not allow_empty:
        raise DataPipelineError(f"{field} must not be empty")
    return result


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataPipelineError(f"{field} must be an integer")
    return value


def _skip_whitespace(buffer: str, position: int) -> int:
    while position < len(buffer) and buffer[position].isspace():
        position += 1
    return position


def iter_json_array(path: str | Path, *, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    """Stream values from a top-level JSON array.

    This avoids loading the roughly 566 MB training source into memory and lets debug builds stop
    parsing after their requested prefix.
    """

    if chunk_size <= 0:
        raise DataPipelineError("chunk_size must be positive")
    source_path = Path(path)
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    eof = False
    started = False
    needs_separator = False

    try:
        with source_path.open("r", encoding="utf-8") as input_file:
            while True:
                if position > chunk_size:
                    buffer = buffer[position:]
                    position = 0
                if not eof and len(buffer) - position < chunk_size:
                    chunk = input_file.read(chunk_size)
                    if chunk:
                        buffer += chunk
                    else:
                        eof = True

                position = _skip_whitespace(buffer, position)
                if not started:
                    if position >= len(buffer):
                        if eof:
                            raise DataPipelineError(f"empty JSON source: {source_path}")
                        continue
                    if buffer[position] != "[":
                        raise DataPipelineError(
                            f"source must contain a top-level JSON array: {source_path}"
                        )
                    started = True
                    position += 1
                    continue

                position = _skip_whitespace(buffer, position)
                if position >= len(buffer):
                    if eof:
                        raise DataPipelineError(f"unterminated JSON array: {source_path}")
                    continue

                if needs_separator:
                    if buffer[position] == "]":
                        position = _skip_whitespace(buffer, position + 1)
                        if buffer[position:].strip() or input_file.read().strip():
                            raise DataPipelineError(
                                f"unexpected content after JSON array: {source_path}"
                            )
                        return
                    if buffer[position] != ",":
                        raise DataPipelineError(
                            f"expected a comma between JSON records: {source_path}"
                        )
                    position += 1
                    needs_separator = False
                    position = _skip_whitespace(buffer, position)
                    if position >= len(buffer) and not eof:
                        continue
                elif buffer[position] == "]":
                    position = _skip_whitespace(buffer, position + 1)
                    if buffer[position:].strip() or input_file.read().strip():
                        raise DataPipelineError(
                            f"unexpected content after JSON array: {source_path}"
                        )
                    return

                try:
                    value, next_position = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as error:
                    if not eof:
                        chunk = input_file.read(chunk_size)
                        if chunk:
                            buffer += chunk
                            continue
                        eof = True
                        continue
                    raise DataPipelineError(
                        f"invalid JSON record in {source_path}: {error}"
                    ) from error
                yield value
                position = next_position
                needs_separator = True
    except (OSError, UnicodeError) as error:
        raise DataPipelineError(f"could not read JSON source {source_path}: {error}") from error


def _document_id(title: str, sentences: tuple[str, ...]) -> str:
    canonical = _stable_json([title, list(sentences)]).encode()
    return f"hotpotqa:{hashlib.sha256(canonical).hexdigest()}"


def _parse_context(
    raw: dict[str, object],
    *,
    location: str,
) -> tuple[tuple[CorpusDocument, ...], dict[str, CorpusDocument]]:
    documents = []
    documents_by_title: dict[str, CorpusDocument] = {}
    for context_index, context_value in enumerate(
        _array(raw.get("context"), f"{location}.context")
    ):
        context = _array(context_value, f"{location}.context[{context_index}]")
        if len(context) != 2:
            raise DataPipelineError(
                f"{location}.context[{context_index}] must contain title and sentences"
            )
        title = _string(context[0], f"{location}.context[{context_index}].title")
        sentences = tuple(
            _string(
                sentence,
                f"{location}.context[{context_index}].sentences[{index}]",
                allow_empty=True,
            )
            for index, sentence in enumerate(
                _array(context[1], f"{location}.context[{context_index}].sentences")
            )
        )
        if not sentences:
            raise DataPipelineError(
                f"{location}.context[{context_index}].sentences must not be empty"
            )
        text = " ".join(sentences).strip()
        document = CorpusDocument(
            document_id=_document_id(title, sentences),
            title=title,
            text=text,
            sentences=sentences,
        )
        existing = documents_by_title.get(title)
        if existing is not None and existing.document_id != document.document_id:
            raise DataPipelineError(f"{location} contains ambiguous duplicate title {title!r}")
        documents_by_title[title] = document
        documents.append(document)
    if not documents:
        raise DataPipelineError(f"{location}.context must not be empty")
    return tuple(documents), documents_by_title


def _convert_example(
    value: object,
    *,
    split: Split,
    row_index: int,
    source_revision: str,
) -> tuple[HotpotQAExample, tuple[CorpusDocument, ...], int]:
    location = f"{split}[{row_index}]"
    raw = _mapping(value, location)
    example_id = _string(raw.get("_id"), f"{location}._id")
    question = _string(raw.get("question"), f"{location}.question", trim=True)
    answer = _string(raw.get("answer"), f"{location}.answer")
    level = _string(raw.get("level"), f"{location}.level")
    question_type = _string(raw.get("type"), f"{location}.type")
    documents, documents_by_title = _parse_context(raw, location=location)

    facts = []
    reference_issues = 0
    for fact_index, fact_value in enumerate(
        _array(raw.get("supporting_facts"), f"{location}.supporting_facts")
    ):
        fact = _array(fact_value, f"{location}.supporting_facts[{fact_index}]")
        if len(fact) != 2:
            raise DataPipelineError(
                f"{location}.supporting_facts[{fact_index}] must contain title and sentence id"
            )
        title = _string(fact[0], f"{location}.supporting_facts[{fact_index}].title")
        sentence_index = _integer(
            fact[1], f"{location}.supporting_facts[{fact_index}].sentence_index"
        )
        if sentence_index < 0:
            raise DataPipelineError(
                f"{location}.supporting_facts[{fact_index}] has a negative sentence id"
            )
        document = documents_by_title.get(title)
        if document is None:
            raise DataPipelineError(
                f"{location}.supporting_facts[{fact_index}] references missing title {title!r}"
            )
        if sentence_index >= len(document.sentences):
            reference_issues += 1
        facts.append(
            SupportingFact(
                title=title,
                sentence_index=sentence_index,
                document_id=document.document_id,
            )
        )
    if not facts:
        raise DataPipelineError(f"{location}.supporting_facts must not be empty")

    supporting_titles = tuple(dict.fromkeys(fact.title for fact in facts))
    supporting_document_ids = tuple(dict.fromkeys(fact.document_id for fact in facts))
    return (
        HotpotQAExample(
            example_id=example_id,
            question=question,
            answers=(answer,),
            split=split,
            level=level,
            question_type=question_type,
            supporting_facts=tuple(facts),
            supporting_titles=supporting_titles,
            supporting_document_ids=supporting_document_ids,
            context_document_ids=tuple(document.document_id for document in documents),
            source_revision=source_revision,
        ),
        documents,
        reference_issues,
    )


def _write_jsonl(path: Path, records: Iterable[dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.partial")
    count = 0
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in records:
                output_file.write(_stable_json(record))
                output_file.write("\n")
                count += 1
        os.replace(temporary_path, path)
    except OSError as error:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise DataPipelineError(f"could not write {path}: {error}") from error
    return count


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.partial")
    try:
        temporary_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary_path, path)
    except OSError as error:
        with suppress(OSError):
            temporary_path.unlink(missing_ok=True)
        raise DataPipelineError(f"could not write manifest {path}: {error}") from error


def _output_metadata(path: Path, records: int, output_dir: Path) -> dict[str, object]:
    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(output_dir).as_posix(),
        "records": records,
        "sha256": sha256_file(path),
    }


def _validate_limit(value: int | None, field: str) -> None:
    if value is not None and value <= 0:
        raise DataPipelineError(f"{field} must be positive when provided")


def build_hotpotqa(
    config: DataSourceConfig,
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    max_train: int | None = None,
    max_validation: int | None = None,
) -> HotpotQABuildResult:
    """Build canonical, corpus, adapter, and manifest artifacts from pinned raw JSON."""

    _validate_limit(max_train, "max_train")
    _validate_limit(max_validation, "max_validation")
    limits: dict[Split, int | None] = {
        "train": max_train,
        "validation": max_validation,
    }
    source_by_name = {source_split.name: source_split for source_split in config.splits}
    raw_root = Path(raw_dir)
    destination = Path(output_dir)
    examples_by_split: dict[Split, list[HotpotQAExample]] = {
        "train": [],
        "validation": [],
    }
    corpus_by_id: dict[str, CorpusDocument] = {}
    source_files = []
    context_occurrences = 0
    supporting_fact_reference_issues = 0

    for split in SPLIT_ORDER:
        source_split = source_by_name[split]
        source_path = raw_root / source_split.filename
        verify_source_file(source_path, source_split)
        source_files.append(
            {
                "bytes": source_split.bytes,
                "expected_records": source_split.records,
                "filename": source_split.filename,
                "mirrors": list(source_split.mirrors),
                "sha256": source_split.sha256,
                "split": split,
                "url": source_split.url,
            }
        )
        values: Iterable[object] = iter_json_array(source_path)
        if limits[split] is not None:
            values = islice(values, limits[split])
        split_ids: set[str] = set()
        for row_index, value in enumerate(values):
            example, documents, reference_issues = _convert_example(
                value,
                split=split,
                row_index=row_index,
                source_revision=config.source_revision,
            )
            if example.example_id in split_ids:
                raise DataPipelineError(f"duplicate example id {example.example_id!r} in {split}")
            split_ids.add(example.example_id)
            examples_by_split[split].append(example)
            supporting_fact_reference_issues += reference_issues
            for document in documents:
                context_occurrences += 1
                existing = corpus_by_id.get(document.document_id)
                if existing is not None and existing != document:
                    raise DataPipelineError(
                        f"corpus document hash collision: {document.document_id}"
                    )
                corpus_by_id[document.document_id] = document
        if limits[split] is None and len(examples_by_split[split]) != source_split.records:
            raise DataPipelineError(
                f"{split} record count mismatch: expected {source_split.records}, "
                f"got {len(examples_by_split[split])}"
            )

    train_ids = {example.example_id for example in examples_by_split["train"]}
    validation_ids = {example.example_id for example in examples_by_split["validation"]}
    overlap = sorted(train_ids & validation_ids)
    if overlap:
        preview = ", ".join(overlap[:5])
        raise DataPipelineError(f"train/validation example id overlap: {preview}")

    corpus = sorted(corpus_by_id.values(), key=lambda document: document.document_id)
    output_paths: dict[str, tuple[Path, int]] = {}
    for split in SPLIT_ORDER:
        canonical_path = destination / "examples" / f"{split}.jsonl"
        canonical_count = _write_jsonl(
            canonical_path, (example.to_dict() for example in examples_by_split[split])
        )
        output_paths[f"examples_{split}"] = (canonical_path, canonical_count)
        agent_r1_path = destination / "agent_r1" / f"{split}.jsonl"
        agent_r1_count = _write_jsonl(
            agent_r1_path,
            (
                example.to_agent_r1_dict(row_index)
                for row_index, example in enumerate(examples_by_split[split])
            ),
        )
        output_paths[f"agent_r1_{split}"] = (agent_r1_path, agent_r1_count)

    corpus_path = destination / "corpus.jsonl"
    corpus_count = _write_jsonl(corpus_path, (document.to_dict() for document in corpus))
    output_paths["corpus"] = (corpus_path, corpus_count)

    build_mode = "full" if all(limit is None for limit in limits.values()) else "debug"
    manifest: dict[str, object] = {
        "build": {
            "limits": {"train": max_train, "validation": max_validation},
            "mode": build_mode,
        },
        "compatibility": {
            "agent_r1": {
                "logical_columns": [
                    "data_source",
                    "prompt",
                    "reward_model",
                    "extra_info",
                ],
                "repository": config.agent_r1_repository,
                "revision": config.agent_r1_revision,
            },
            "local_core_adapter": "HotpotQAExample.to_core_example",
        },
        "counts": {
            "context_document_occurrences": context_occurrences,
            "corpus_documents": corpus_count,
            "deduplicated_context_occurrences": context_occurrences - corpus_count,
            "supporting_fact_reference_issues": supporting_fact_reference_issues,
            "train_examples": len(examples_by_split["train"]),
            "validation_examples": len(examples_by_split["validation"]),
        },
        "dataset": config.dataset,
        "hash_algorithm": "sha256",
        "integrity": {
            "corpus_document_ids_unique": True,
            "evaluation_labels_excluded_from_prompt": True,
            "train_validation_ids_disjoint": True,
        },
        "license": {"name": config.license_name, "url": config.license_url},
        "outputs": {
            name: _output_metadata(path, record_count, destination)
            for name, (path, record_count) in sorted(output_paths.items())
        },
        "schema_version": BUILD_MANIFEST_SCHEMA_VERSION,
        "source": {
            "files": source_files,
            "repository": config.source_repository,
            "repository_revision": config.source_repository_revision,
            "revision": config.source_revision,
        },
        "variant": config.variant,
    }
    manifest_path = destination / "manifest.json"
    _write_manifest(manifest_path, manifest)
    verify_hotpotqa_build(destination)
    return HotpotQABuildResult(
        output_dir=destination,
        manifest_path=manifest_path,
        train_examples=len(examples_by_split["train"]),
        validation_examples=len(examples_by_split["validation"]),
        corpus_documents=corpus_count,
        build_mode=build_mode,
    )


def _load_json(path: Path, field: str) -> dict[str, object]:
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DataPipelineError(f"could not read {field} {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise DataPipelineError(f"invalid JSON in {field} {path}: {error}") from error
    return _mapping(value, field)


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, object]]]:
    try:
        with path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                try:
                    value: object = json.loads(line)
                except json.JSONDecodeError as error:
                    raise DataPipelineError(
                        f"invalid JSON in {path} line {line_number}: {error}"
                    ) from error
                yield line_number, _mapping(value, f"{path.name} line {line_number}")
    except OSError as error:
        raise DataPipelineError(f"could not read {path}: {error}") from error


def _verify_output_metadata(
    output_dir: Path,
    name: str,
    value: object,
) -> tuple[Path, int]:
    metadata = _mapping(value, f"outputs.{name}")
    relative_path = _string(metadata.get("path"), f"outputs.{name}.path")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise DataPipelineError(f"outputs.{name}.path must stay inside the build directory")
    path = output_dir / relative
    expected_bytes = _integer(metadata.get("bytes"), f"outputs.{name}.bytes")
    expected_records = _integer(metadata.get("records"), f"outputs.{name}.records")
    expected_sha256 = _string(metadata.get("sha256"), f"outputs.{name}.sha256")
    try:
        actual_bytes = path.stat().st_size
    except OSError as error:
        raise DataPipelineError(f"could not inspect output {path}: {error}") from error
    if actual_bytes != expected_bytes:
        raise DataPipelineError(f"output size mismatch for {path}")
    if sha256_file(path) != expected_sha256:
        raise DataPipelineError(f"output sha256 mismatch for {path}")
    actual_records = sum(1 for _ in _iter_jsonl(path))
    if actual_records != expected_records:
        raise DataPipelineError(f"output record count mismatch for {path}")
    return path, expected_records


def _verify_agent_r1_row(
    row: dict[str, object],
    example: HotpotQAExample,
    row_index: int,
) -> None:
    if row != example.to_agent_r1_dict(row_index):
        raise DataPipelineError(
            f"Agent-R1 row does not match canonical example {example.example_id}"
        )
    if row.get("prompt") != [{"content": example.question, "role": "user"}]:
        raise DataPipelineError("Agent-R1 prompt must contain only the user question")


def verify_hotpotqa_build(output_dir: str | Path) -> dict[str, object]:
    """Verify hashes, schemas, ordering, label isolation, and split integrity."""

    destination = Path(output_dir)
    manifest = _load_json(destination / "manifest.json", "build manifest")
    if _integer(manifest.get("schema_version"), "schema_version") != BUILD_MANIFEST_SCHEMA_VERSION:
        raise DataPipelineError("unsupported build manifest schema_version")
    outputs = _mapping(manifest.get("outputs"), "outputs")
    required_outputs = {
        "agent_r1_train",
        "agent_r1_validation",
        "corpus",
        "examples_train",
        "examples_validation",
    }
    if set(outputs) != required_outputs:
        raise DataPipelineError("build manifest outputs do not match the required artifact set")
    verified = {
        name: _verify_output_metadata(destination, name, outputs[name])
        for name in sorted(required_outputs)
    }

    examples_by_split: dict[Split, list[HotpotQAExample]] = {
        "train": [],
        "validation": [],
    }
    for split in SPLIT_ORDER:
        path, _ = verified[f"examples_{split}"]
        for _, record in _iter_jsonl(path):
            try:
                example = hotpotqa_example_from_dict(record)
            except DataRecordError as error:
                raise DataPipelineError(f"invalid canonical example in {path}: {error}") from error
            if example.split != split:
                raise DataPipelineError(f"example split mismatch in {path}")
            examples_by_split[split].append(example)

        agent_path, agent_count = verified[f"agent_r1_{split}"]
        if agent_count != len(examples_by_split[split]):
            raise DataPipelineError(f"Agent-R1 and canonical counts differ for {split}")
        for row_index, ((_, row), example) in enumerate(
            zip(_iter_jsonl(agent_path), examples_by_split[split], strict=True)
        ):
            _verify_agent_r1_row(row, example, row_index)

    train_ids = [example.example_id for example in examples_by_split["train"]]
    validation_ids = [example.example_id for example in examples_by_split["validation"]]
    if len(train_ids) != len(set(train_ids)) or len(validation_ids) != len(set(validation_ids)):
        raise DataPipelineError("duplicate ids found within a converted split")
    if set(train_ids) & set(validation_ids):
        raise DataPipelineError("train and validation ids overlap")

    corpus_path, _ = verified["corpus"]
    corpus_ids = []
    corpus_sentence_counts: dict[str, int] = {}
    for line_number, record in _iter_jsonl(corpus_path):
        if _integer(record.get("schema_version"), "corpus.schema_version") != DATA_SCHEMA_VERSION:
            raise DataPipelineError(f"unsupported corpus schema on line {line_number}")
        if _string(record.get("record_type"), "corpus.record_type") != ("hotpotqa_corpus_document"):
            raise DataPipelineError(f"invalid corpus record type on line {line_number}")
        document_id = _string(record.get("document_id"), "corpus.document_id")
        title = _string(record.get("title"), "corpus.title")
        sentences = tuple(
            _string(sentence, "corpus.sentences[]", allow_empty=True)
            for sentence in _array(record.get("sentences"), "corpus.sentences")
        )
        if not sentences:
            raise DataPipelineError(f"empty corpus sentences on line {line_number}")
        if document_id != _document_id(title, sentences):
            raise DataPipelineError(f"invalid content-derived document id on line {line_number}")
        if _string(record.get("text"), "corpus.text") != " ".join(sentences).strip():
            raise DataPipelineError(f"corpus text mismatch on line {line_number}")
        corpus_ids.append(document_id)
        corpus_sentence_counts[document_id] = len(sentences)
    if corpus_ids != sorted(corpus_ids):
        raise DataPipelineError("corpus records are not deterministically ordered")
    if len(corpus_ids) != len(set(corpus_ids)):
        raise DataPipelineError("corpus document ids are not unique")
    corpus_id_set = set(corpus_ids)
    for example in examples_by_split["train"] + examples_by_split["validation"]:
        if not set(example.context_document_ids).issubset(corpus_id_set):
            raise DataPipelineError(
                f"example {example.example_id} references a missing corpus document"
            )

    all_examples = examples_by_split["train"] + examples_by_split["validation"]
    context_occurrences = sum(len(example.context_document_ids) for example in all_examples)
    reference_issues = sum(
        fact.sentence_index >= corpus_sentence_counts[fact.document_id]
        for example in all_examples
        for fact in example.supporting_facts
    )
    counts = _mapping(manifest.get("counts"), "counts")
    expected_counts = {
        "context_document_occurrences": context_occurrences,
        "corpus_documents": len(corpus_ids),
        "deduplicated_context_occurrences": context_occurrences - len(corpus_ids),
        "supporting_fact_reference_issues": reference_issues,
        "train_examples": len(examples_by_split["train"]),
        "validation_examples": len(examples_by_split["validation"]),
    }
    if counts != expected_counts:
        raise DataPipelineError("manifest counts do not match converted artifacts")
    integrity = _mapping(manifest.get("integrity"), "integrity")
    expected_integrity = {
        "corpus_document_ids_unique": True,
        "evaluation_labels_excluded_from_prompt": True,
        "train_validation_ids_disjoint": True,
    }
    if integrity != expected_integrity:
        raise DataPipelineError("manifest integrity declarations do not match verified checks")

    return manifest
