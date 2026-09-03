"""Shared deterministic index-manifest construction and verification."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from deep_research_rl.retrieval.corpus import CorpusSnapshot
from deep_research_rl.retrieval.errors import IndexIntegrityError

INDEX_MANIFEST_SCHEMA_VERSION = 1
INDEX_MANIFEST_FILENAME = "manifest.json"


def stable_json_bytes(value: object) -> bytes:
    """Encode deterministic UTF-8 JSON with a trailing newline."""

    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode("utf-8")


def write_bytes_atomic(path: Path, content: bytes) -> None:
    """Replace a file only after its complete content reaches a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(content)
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash an arbitrarily large artifact without loading it into memory."""

    hasher = hashlib.sha256()
    with path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def file_metadata(path: Path, *, relative_to: Path) -> dict[str, object]:
    """Return deterministic size/hash metadata for one index artifact."""

    return {
        "bytes": path.stat().st_size,
        "path": path.relative_to(relative_to).as_posix(),
        "sha256": sha256_file(path),
    }


def write_manifest(index_dir: Path, manifest: dict[str, object]) -> Path:
    """Write the canonical manifest after all indexed artifacts exist."""

    manifest_path = index_dir / INDEX_MANIFEST_FILENAME
    write_bytes_atomic(manifest_path, stable_json_bytes(manifest))
    return manifest_path


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise IndexIntegrityError(f"{field} must be an object")
    return {key: item for key, item in value.items() if isinstance(key, str)}


def require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise IndexIntegrityError(f"{field} must be a non-empty string")
    return value


def require_integer(value: object, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise IndexIntegrityError(f"{field} must be an integer")
    return value


def require_number(value: object, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise IndexIntegrityError(f"{field} must be a number")
    return float(value)


def require_array(value: object, field: str) -> list[object]:
    if not isinstance(value, list):
        raise IndexIntegrityError(f"{field} must be an array")
    return list(value)


def load_manifest(index_dir: str | Path) -> dict[str, object]:
    """Read and validate the common index-manifest envelope."""

    directory = Path(index_dir)
    path = directory / INDEX_MANIFEST_FILENAME
    try:
        value: object = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise IndexIntegrityError(f"could not read index manifest {path}: {error}") from error
    except json.JSONDecodeError as error:
        raise IndexIntegrityError(f"invalid index manifest JSON {path}: {error}") from error
    manifest = _mapping(value, "index manifest")
    version = require_integer(manifest.get("schema_version"), "schema_version")
    if version != INDEX_MANIFEST_SCHEMA_VERSION:
        raise IndexIntegrityError(f"unsupported index manifest schema_version: {version}")
    require_string(manifest.get("backend"), "backend")
    return manifest


def manifest_backend(index_dir: str | Path) -> str:
    """Return the validated backend discriminator for a saved index."""

    return require_string(load_manifest(index_dir).get("backend"), "backend")


def verify_common_manifest(
    index_dir: str | Path,
    corpus: CorpusSnapshot,
    *,
    expected_backend: str,
) -> dict[str, object]:
    """Fail on any backend, corpus or artifact mismatch declared by a manifest."""

    directory = Path(index_dir)
    manifest = load_manifest(directory)
    backend = require_string(manifest.get("backend"), "backend")
    if backend != expected_backend:
        raise IndexIntegrityError(
            f"index backend mismatch: expected {expected_backend}, manifest has {backend}"
        )

    corpus_metadata = _mapping(manifest.get("corpus"), "corpus")
    expected_corpus = corpus.fingerprint.to_dict()
    if corpus_metadata != expected_corpus:
        raise IndexIntegrityError(
            "index/corpus mismatch: byte hash, document ID hash, cardinality, or size differs"
        )

    artifacts = _mapping(manifest.get("artifacts"), "artifacts")
    if not artifacts:
        raise IndexIntegrityError("artifacts must not be empty")
    for name, raw_metadata in sorted(artifacts.items()):
        metadata = _mapping(raw_metadata, f"artifacts.{name}")
        relative_path = Path(require_string(metadata.get("path"), f"artifacts.{name}.path"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise IndexIntegrityError(f"artifacts.{name}.path must stay inside the index directory")
        artifact_path = directory / relative_path
        try:
            actual_size = artifact_path.stat().st_size
        except OSError as error:
            raise IndexIntegrityError(
                f"could not read index artifact {artifact_path}: {error}"
            ) from error
        expected_size = require_integer(metadata.get("bytes"), f"artifacts.{name}.bytes")
        if actual_size != expected_size:
            raise IndexIntegrityError(f"index artifact size mismatch: {artifact_path}")
        expected_hash = require_string(metadata.get("sha256"), f"artifacts.{name}.sha256")
        try:
            actual_hash = sha256_file(artifact_path)
        except OSError as error:
            raise IndexIntegrityError(
                f"could not hash index artifact {artifact_path}: {error}"
            ) from error
        if actual_hash != expected_hash:
            raise IndexIntegrityError(f"index artifact sha256 mismatch: {artifact_path}")
    return manifest


def manifest_mapping(manifest: dict[str, object], field: str) -> dict[str, object]:
    """Validate and expose one object-valued backend manifest section."""

    return _mapping(manifest.get(field), field)
