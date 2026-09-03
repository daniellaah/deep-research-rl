"""Retrieval-specific errors with actionable failure messages."""


class RetrievalError(ValueError):
    """Base class for invalid retrieval inputs or artifacts."""


class IndexIntegrityError(RetrievalError):
    """Raised when an index does not match its manifest or corpus."""


class RetrievalDependencyError(RuntimeError):
    """Raised when an optional production retrieval dependency is unavailable."""
