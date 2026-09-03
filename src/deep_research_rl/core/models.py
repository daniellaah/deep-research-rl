"""Immutable data models for dependency-light research trajectories."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

ObservationStatus = Literal["search_executed", "search_rejected", "answer_recorded"]


@dataclass(frozen=True, slots=True)
class Document:
    """A retrievable text document."""

    document_id: str
    title: str
    text: str

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One ranked retrieval hit with its corpus identity and backend score."""

    document_id: str
    title: str
    text: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not self.document_id:
            raise ValueError("document_id must not be empty")
        if not self.title:
            raise ValueError("title must not be empty")
        if not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if self.rank < 1:
            raise ValueError("rank must be at least 1")

    def to_document(self) -> Document:
        """Drop rank metadata while preserving the traceable corpus record."""

        return Document(document_id=self.document_id, title=self.title, text=self.text)


@dataclass(frozen=True, slots=True)
class Example:
    """A question-answer example and its provenance classification."""

    example_id: str
    question: str
    answers: tuple[str, ...]
    supporting_document_ids: tuple[str, ...]
    source: str
    synthetic: bool
    benchmark_eligible: bool

    def __post_init__(self) -> None:
        if not self.example_id:
            raise ValueError("example_id must not be empty")
        if not self.question:
            raise ValueError("question must not be empty")
        if not self.answers or any(not answer for answer in self.answers):
            raise ValueError("answers must contain at least one non-empty answer")
        if not self.source:
            raise ValueError("source must not be empty")
        if self.synthetic and self.benchmark_eligible:
            raise ValueError("synthetic examples cannot be benchmark eligible")


@dataclass(frozen=True, slots=True)
class SearchAction:
    """Request retrieval for one non-empty query."""

    query: str

    def __post_init__(self) -> None:
        if not self.query or self.query != self.query.strip():
            raise ValueError("search query must be non-empty and trimmed")


@dataclass(frozen=True, slots=True)
class AnswerAction:
    """Terminate an episode with one non-empty answer."""

    answer: str

    def __post_init__(self) -> None:
        if not self.answer or self.answer != self.answer.strip():
            raise ValueError("answer must be non-empty and trimmed")


Action = SearchAction | AnswerAction


@dataclass(frozen=True, slots=True)
class Observation:
    """The environment response to an accepted or rejected action."""

    status: ObservationStatus
    message: str
    query: str | None = None
    documents: tuple[SearchResult, ...] = ()

    def __post_init__(self) -> None:
        if not self.message:
            raise ValueError("observation message must not be empty")
        if self.status in {"search_executed", "search_rejected"} and not self.query:
            raise ValueError("search observations require a query")
        if self.status == "search_rejected" and self.documents:
            raise ValueError("rejected searches cannot contain documents")
        if self.status == "answer_recorded" and (self.query is not None or self.documents):
            raise ValueError("answer observations cannot contain a query or documents")


@dataclass(frozen=True, slots=True)
class AgentState:
    """The complete policy-visible state for one episode step."""

    example_id: str
    question: str
    context: tuple[Observation, ...]
    executed_searches: int
    terminated: bool
    answer: str | None = None

    def __post_init__(self) -> None:
        if self.executed_searches < 0:
            raise ValueError("executed_searches must not be negative")
        if self.terminated != (self.answer is not None):
            raise ValueError("terminated state and recorded answer must agree")


@dataclass(frozen=True, slots=True)
class Step:
    """One fully traceable state/action/environment transition."""

    index: int
    raw_action: str
    action: Action
    state_before: AgentState
    observation: Observation
    state_after: AgentState
    reward: float
    cost: float
    credit: float


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    """Dependency-light metrics emitted with a single trajectory."""

    exact_match: float
    token_f1: float
    terminated: bool
    executed_searches: int
    step_count: int


@dataclass(frozen=True, slots=True)
class Trajectory:
    """A complete episode, including provenance, transitions, and metrics."""

    example: Example
    initial_state: AgentState
    steps: tuple[Step, ...]
    final_state: AgentState
    metrics: EpisodeMetrics

    def __post_init__(self) -> None:
        if self.initial_state.example_id != self.example.example_id:
            raise ValueError("initial state must belong to the trajectory example")
        if self.final_state.example_id != self.example.example_id:
            raise ValueError("final state must belong to the trajectory example")
        if self.steps and self.steps[-1].state_after != self.final_state:
            raise ValueError("last step must reach final_state")
        if any(step.index != index for index, step in enumerate(self.steps)):
            raise ValueError("step indices must be contiguous and zero-based")
