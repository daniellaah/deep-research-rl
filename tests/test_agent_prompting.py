from deep_research_rl.agent.prompting import build_policy_messages
from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.fixtures import synthetic_two_hop_fixture
from deep_research_rl.core.models import SearchAction
from deep_research_rl.retrieval.bm25 import BM25Retriever


def test_prompt_contains_strict_action_contract_and_complete_append_only_history() -> None:
    example, documents = synthetic_two_hop_fixture()
    environment = ResearchEnvironment(
        BM25Retriever(documents, top_k=1),
        AppendOnlyContextPolicy(),
        max_searches=5,
    )
    state = environment.reset(example)
    state, first_observation = environment.transition(state, SearchAction("Brindle Process"))
    state, malformed_observation = environment.record_malformed_action(
        state,
        "expected exactly SEARCH(query) or ANSWER(answer)",
    )

    messages = build_policy_messages(state, max_searches=5)
    system = messages[0]["content"]
    user = messages[1]["content"]

    assert "output exactly one line" in system
    assert "Do not output analysis" in system
    assert f"Observation 1: SEARCH({first_observation.query}) executed." in user
    assert first_observation.documents[0].title in user
    assert first_observation.documents[0].text in user
    assert f"Observation 2: {malformed_observation.message}" in user
    assert user.index("Observation 1") < user.index("Observation 2")
    assert "1 executed, 4 remaining, 5 maximum" in user
