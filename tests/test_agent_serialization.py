from pathlib import Path

from test_agent_rollout import CountingRetriever, ScriptedGenerativePolicy, _run

from deep_research_rl.agent.serialization import (
    agent_rollout_as_json,
    agent_rollout_from_json,
    read_agent_rollout_jsonl,
    write_agent_rollout_jsonl,
)


def test_model_rollout_round_trips_with_tokens_parse_status_and_termination(tmp_path: Path) -> None:
    rollout = _run(
        ScriptedGenerativePolicy(("malformed", "ANSWER(Lumen City)")),
        CountingRetriever(),
    )
    path = tmp_path / "rollout.jsonl"

    serialized = agent_rollout_as_json(rollout)
    write_agent_rollout_jsonl(path, (rollout,))

    assert agent_rollout_from_json(serialized) == rollout
    assert read_agent_rollout_jsonl(path) == (rollout,)
    assert '"result_scope": "debug_validation_not_benchmark"' in serialized
    assert '"response_logprobs":' in serialized
    assert '"termination_reason": "answered"' in serialized
