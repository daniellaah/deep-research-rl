"""Dependency-light public package for DeepResearch-RL."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("deepresearch-rl")
except PackageNotFoundError:  # pragma: no cover - only possible outside an installed checkout
    __version__ = "0+unknown"

__all__ = ["__version__"]
