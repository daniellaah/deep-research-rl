"""Small, tokenizer-free answer metrics used by the CPU path."""

from __future__ import annotations

import re
import string
from collections import Counter

from deep_research_rl.core.models import AgentState, EpisodeMetrics, Example

_ARTICLES = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)


def normalize_answer(answer: str) -> str:
    """Apply the conventional lowercase/punctuation/article/whitespace normalization."""

    lowered = answer.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = _ARTICLES.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def normalized_exact_match(prediction: str, reference: str) -> float:
    """Return one when normalized strings match and zero otherwise."""

    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Compute bag-of-normalized-token F1 for one reference answer."""

    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()
    if not prediction_tokens or not reference_tokens:
        return float(prediction_tokens == reference_tokens)

    common_count = sum((Counter(prediction_tokens) & Counter(reference_tokens)).values())
    if common_count == 0:
        return 0.0
    precision = common_count / len(prediction_tokens)
    recall = common_count / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)


def build_episode_metrics(
    example: Example,
    final_state: AgentState,
    step_count: int,
) -> EpisodeMetrics:
    """Summarize one finished or truncated episode without external tokenizers."""

    prediction = final_state.answer or ""
    exact_match = max(normalized_exact_match(prediction, answer) for answer in example.answers)
    best_token_f1 = max(token_f1(prediction, answer) for answer in example.answers)
    return EpisodeMetrics(
        exact_match=exact_match,
        token_f1=best_token_f1,
        terminated=final_state.terminated,
        executed_searches=final_state.executed_searches,
        step_count=step_count,
    )
