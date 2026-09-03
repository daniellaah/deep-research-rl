import pytest

from deep_research_rl.core.actions import ActionParseError, parse_action
from deep_research_rl.core.models import AnswerAction, SearchAction


def test_parse_supported_actions_exactly() -> None:
    assert parse_action("SEARCH(Mira Voss)") == SearchAction(query="Mira Voss")
    assert parse_action("ANSWER(Lumen City)") == AnswerAction(answer="Lumen City")


@pytest.mark.parametrize(
    "raw_action",
    [
        " SEARCH(Mira Voss)",
        "SEARCH(Mira Voss) ",
        "SEARCH( Mira Voss)",
        "SEARCH(Mira Voss )",
        "SEARCH()",
        "ANSWER()",
        "LOOKUP(Mira Voss)",
        "SEARCH(Mira Voss) extra",
        "SEARCH(Mira\nVoss)",
    ],
)
def test_reject_malformed_or_repaired_actions(raw_action: str) -> None:
    with pytest.raises(ActionParseError):
        parse_action(raw_action)
