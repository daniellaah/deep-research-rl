"""Model-backed agent rollout contracts and adapters."""

from deep_research_rl.agent.contracts import (
    AgentRollout,
    AgentRolloutStep,
    ModelIdentity,
    PolicyOutput,
)
from deep_research_rl.agent.protocols import GenerativePolicy
from deep_research_rl.agent.qwen import (
    DEFAULT_QWEN_MODEL,
    DEFAULT_QWEN_REVISION,
    QwenGenerationSettings,
    QwenPolicyAdapter,
)
from deep_research_rl.agent.rollout import DEBUG_RESULT_SCOPE, run_model_rollout
from deep_research_rl.agent.serialization import (
    agent_rollout_as_json,
    agent_rollout_from_json,
    read_agent_rollout_jsonl,
    write_agent_rollout_jsonl,
)

__all__ = [
    "DEBUG_RESULT_SCOPE",
    "DEFAULT_QWEN_MODEL",
    "DEFAULT_QWEN_REVISION",
    "AgentRollout",
    "AgentRolloutStep",
    "GenerativePolicy",
    "ModelIdentity",
    "PolicyOutput",
    "QwenGenerationSettings",
    "QwenPolicyAdapter",
    "agent_rollout_as_json",
    "agent_rollout_from_json",
    "read_agent_rollout_jsonl",
    "run_model_rollout",
    "write_agent_rollout_jsonl",
]
