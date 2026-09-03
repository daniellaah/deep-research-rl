"""Pinned source descriptors, hashing, and downloads for dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from deep_research_rl.data.models import Split

SOURCE_CONFIG_SCHEMA_VERSION = 1


class SourceConfigError(ValueError):
    """Raised when a data source descriptor is invalid."""


@dataclass(frozen=True, slots=True)
class SourceSplit:
    """One immutable raw source artifact."""

    name: Split
    filename: str
    url: str
    sha256: str
    bytes: int
    records: int
    mirrors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.filename or Path(self.filename).name != self.filename:
            raise SourceConfigError("source filename must be one plain filename")
        if not self.url.startswith(("http://", "https://")):
            raise SourceConfigError("source URL must use HTTP or HTTPS")
        if any(not mirror.startswith(("http://", "https://")) for mirror in self.mirrors):
            raise SourceConfigError("source mirror URLs must use HTTP or HTTPS")
        if len(self.sha256) != 64 or any(char not in "0123456789abcdef" for char in self.sha256):
            raise SourceConfigError("source sha256 must contain 64 lowercase hex characters")
        if self.bytes <= 0:
            raise SourceConfigError("source bytes must be positive")
        if self.records <= 0:
            raise SourceConfigError("source records must be positive")


@dataclass(frozen=True, slots=True)
class DataSourceConfig:
    """Complete provenance and compatibility lock for one dataset release."""

    dataset: str
    variant: str
    source_revision: str
    source_repository: str
    source_repository_revision: str
    license_name: str
    license_url: str
    agent_r1_repository: str
    agent_r1_revision: str
    splits: tuple[SourceSplit, ...]

    def __post_init__(self) -> None:
        if self.dataset != "hotpot_qa" or self.variant != "distractor":
            raise SourceConfigError("only hotpot_qa/distractor is supported")
        string_fields = (
            self.source_revision,
            self.source_repository,
            self.source_repository_revision,
            self.license_name,
            self.license_url,
            self.agent_r1_repository,
            self.agent_r1_revision,
        )
        if any(not field for field in string_fields):
            raise SourceConfigError("source provenance fields must not be empty")
        if tuple(split.name for split in self.splits) != ("train", "validation"):
            raise SourceConfigError("source splits must be ordered train, validation")


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise SourceConfigError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def _array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise SourceConfigError(f"{field} must be an array")
    return list(value)


def _string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise SourceConfigError(f"{field} must be a non-empty string")
    return value


def _integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise SourceConfigError(f"{field} must be an integer")
    return value


def load_source_config(path: str | Path) -> DataSourceConfig:
    """Load and validate one committed source lock file."""

    config_path = Path(path)
    try:
        value: object = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise SourceConfigError(f"could not read source config {config_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise SourceConfigError(f"invalid JSON in source config {config_path}: {error}") from error
    record = _mapping(value, "source config")
    if _integer(record.get("schema_version"), "schema_version") != SOURCE_CONFIG_SCHEMA_VERSION:
        raise SourceConfigError("unsupported source config schema_version")

    source = _mapping(record.get("source"), "source")
    license_record = _mapping(record.get("license"), "license")
    compatibility = _mapping(record.get("compatibility"), "compatibility")
    agent_r1 = _mapping(compatibility.get("agent_r1"), "compatibility.agent_r1")
    splits = []
    for index, split_value in enumerate(_array(record.get("splits"), "splits")):
        split_record = _mapping(split_value, f"splits[{index}]")
        split_name = _string(split_record.get("name"), f"splits[{index}].name")
        if split_name == "train":
            name: Split = "train"
        elif split_name == "validation":
            name = "validation"
        else:
            raise SourceConfigError(f"unsupported split: {split_name}")
        splits.append(
            SourceSplit(
                name=name,
                filename=_string(split_record.get("filename"), f"splits[{index}].filename"),
                url=_string(split_record.get("url"), f"splits[{index}].url"),
                sha256=_string(split_record.get("sha256"), f"splits[{index}].sha256"),
                bytes=_integer(split_record.get("bytes"), f"splits[{index}].bytes"),
                records=_integer(split_record.get("records"), f"splits[{index}].records"),
                mirrors=tuple(
                    _string(mirror, f"splits[{index}].mirrors[]")
                    for mirror in _array(
                        split_record.get("mirrors", []),
                        f"splits[{index}].mirrors",
                    )
                ),
            )
        )

    return DataSourceConfig(
        dataset=_string(record.get("dataset"), "dataset"),
        variant=_string(record.get("variant"), "variant"),
        source_revision=_string(record.get("source_revision"), "source_revision"),
        source_repository=_string(source.get("repository"), "source.repository"),
        source_repository_revision=_string(source.get("revision"), "source.revision"),
        license_name=_string(license_record.get("name"), "license.name"),
        license_url=_string(license_record.get("url"), "license.url"),
        agent_r1_repository=_string(agent_r1.get("repository"), "agent_r1.repository"),
        agent_r1_revision=_string(agent_r1.get("revision"), "agent_r1.revision"),
        splits=tuple(splits),
    )


def verify_source_file(path: Path, source_split: SourceSplit) -> None:
    """Check a raw artifact against its expected byte count and digest."""

    try:
        actual_bytes = path.stat().st_size
    except OSError as error:
        raise SourceConfigError(f"could not inspect source file {path}: {error}") from error
    if actual_bytes != source_split.bytes:
        raise SourceConfigError(
            f"source size mismatch for {path}: expected {source_split.bytes}, got {actual_bytes}"
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != source_split.sha256:
        raise SourceConfigError(
            f"source sha256 mismatch for {path}: expected {source_split.sha256}, "
            f"got {actual_sha256}"
        )


def download_source_files(config: DataSourceConfig, raw_dir: str | Path) -> tuple[Path, ...]:
    """Download missing source files and verify every artifact before use.

    Existing files are never replaced silently. A mismatched existing file raises an error.
    """

    destination = Path(raw_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for source_split in config.splits:
        path = destination / source_split.filename
        if not path.exists():
            temporary_path = destination / f".{source_split.filename}.partial"
            errors = []
            for url in (source_split.url, *source_split.mirrors):
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
                try:
                    with (
                        urllib.request.urlopen(url, timeout=60) as response,
                        temporary_path.open("wb") as output_file,
                    ):
                        while chunk := response.read(1024 * 1024):
                            output_file.write(chunk)
                    verify_source_file(temporary_path, source_split)
                    os.replace(temporary_path, path)
                    break
                except (OSError, SourceConfigError, urllib.error.URLError) as error:
                    errors.append(f"{url}: {error}")
            else:
                with suppress(OSError):
                    temporary_path.unlink(missing_ok=True)
                raise SourceConfigError(
                    f"could not download a verified copy to {path}: {'; '.join(errors)}"
                )
        verify_source_file(path, source_split)
        paths.append(path)
    return tuple(paths)
