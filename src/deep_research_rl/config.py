"""Dependency-free loading and validation for project TOML configurations."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

SUPPORTED_CONFIG_KINDS = frozenset({"defaults", "resolved"})


class ConfigError(ValueError):
    """Raised when a configuration cannot be loaded or violates the base convention."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration and validate its shared envelope.

    Domain-specific validation belongs to the component that owns the corresponding section.
    This loader only enforces fields shared by every project configuration.
    """

    config_path = Path(path)
    if config_path.suffix.lower() != ".toml":
        raise ConfigError(f"configuration must be a .toml file: {config_path}")

    try:
        with config_path.open("rb") as config_file:
            config = tomllib.load(config_file)
    except OSError as error:
        raise ConfigError(f"could not read configuration {config_path}: {error}") from error
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(f"invalid TOML in {config_path}: {error}") from error

    schema_version = config.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ConfigError("schema_version must be an integer")
    if schema_version < 1:
        raise ConfigError("schema_version must be at least 1")

    config_kind = config.get("config_kind")
    if not isinstance(config_kind, str) or config_kind not in SUPPORTED_CONFIG_KINDS:
        allowed = ", ".join(sorted(SUPPORTED_CONFIG_KINDS))
        raise ConfigError(f"config_kind must be one of: {allowed}")

    return config
