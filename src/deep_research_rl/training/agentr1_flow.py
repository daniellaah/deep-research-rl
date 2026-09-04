"""Agent-R1 AgentFlow adapter for the frozen DeepResearch-RL baseline."""

from __future__ import annotations

from functools import lru_cache
from typing import Any
from uuid import uuid4

from agent_r1.agent_flow.agent_flow import (  # type: ignore[import-not-found]
    AgentFlowBase,
    AgentFlowOutput,
    AgentFlowStep,
    register,
)
from verl.utils.profiler import simple_timer  # type: ignore[import-not-found]

from deep_research_rl.core.models import Example
from deep_research_rl.core.protocols import Retriever
from deep_research_rl.retrieval import load_retriever
from deep_research_rl.retrieval.index import manifest_backend
from deep_research_rl.training.episode import (
    AGENT_FLOW_NAME,
    BASELINE_MAX_SEARCHES,
    BASELINE_MAX_STEPS,
    AgentR1Episode,
)

BASELINE_PROMPT_TOKENS = 8192
BASELINE_RESPONSE_TOKENS = 96
BASELINE_RETRIEVAL_TOP_K = 3


@lru_cache(maxsize=8)
def _cached_retriever(
    index_dir: str,
    corpus_path: str,
    retrieval_device: str,
) -> Retriever:
    """Load one verified retriever per AgentFlow worker process and runtime identity."""

    backend = manifest_backend(index_dir)
    if backend != "faiss_bge":
        raise ValueError(f"baseline AgentFlow requires a faiss_bge index, received {backend}")
    return load_retriever(
        index_dir,
        corpus_path,
        top_k=BASELINE_RETRIEVAL_TOP_K,
        device=retrieval_device,
    )


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be an object")
    return {str(key): item for key, item in value.items()}


def _strings(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field} must be an array")
    strings = tuple(item for item in value if isinstance(item, str) and item)
    if len(strings) != len(value):
        raise ValueError(f"{field} must contain only non-empty strings")
    return strings


def _example_from_kwargs(kwargs: dict[str, Any]) -> Example:
    raw_prompt = kwargs.get("raw_prompt")
    if not isinstance(raw_prompt, (list, tuple)) or len(raw_prompt) != 1:
        raise ValueError("raw_prompt must contain exactly one user message")
    message = _mapping(raw_prompt[0], "raw_prompt[0]")
    if message.get("role") != "user" or not isinstance(message.get("content"), str):
        raise ValueError("raw_prompt must contain exactly one textual user message")
    question = str(message["content"])

    extra_info = _mapping(kwargs.get("extra_info"), "extra_info")
    answers_value = extra_info.get("answers")
    answers: tuple[str, ...]
    if answers_value is None:
        reward_model = _mapping(kwargs.get("reward_model"), "reward_model")
        answers = (str(reward_model.get("ground_truth", "")),)
    else:
        answers = _strings(answers_value, "extra_info.answers")
    if not answers or any(not answer for answer in answers):
        raise ValueError("training example requires at least one non-empty answer")

    supporting_value = extra_info.get("supporting_document_ids", ())
    supporting_document_ids = (
        ()
        if supporting_value in (None, ())
        else _strings(supporting_value, "extra_info.supporting_document_ids")
    )
    example_id_value = extra_info.get("question_id")
    if not isinstance(example_id_value, str) or not example_id_value:
        raise ValueError("extra_info.question_id must be a non-empty string")
    split = extra_info.get("split", "unknown")
    return Example(
        example_id=example_id_value,
        question=question,
        answers=answers,
        supporting_document_ids=supporting_document_ids,
        source=f"hotpotqa_distractor:{split}",
        synthetic=False,
        benchmark_eligible=True,
    )


@register(AGENT_FLOW_NAME)
class DeepResearchAgentFlow(AgentFlowBase):  # type: ignore[misc]
    """Generate one strict action per step and expose every transition to Agent-R1."""

    def __init__(
        self,
        *args: Any,
        corpus_path: str,
        index_dir: str,
        retrieval_device: str = "cpu",
        max_searches: int = BASELINE_MAX_SEARCHES,
        max_steps: int = BASELINE_MAX_STEPS,
        retrieval_top_k: int = BASELINE_RETRIEVAL_TOP_K,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if max_searches != BASELINE_MAX_SEARCHES:
            raise ValueError(f"baseline AgentFlow requires max_searches={BASELINE_MAX_SEARCHES}")
        if max_steps != BASELINE_MAX_STEPS:
            raise ValueError(f"baseline AgentFlow requires max_steps={BASELINE_MAX_STEPS}")
        if retrieval_top_k != BASELINE_RETRIEVAL_TOP_K:
            raise ValueError(
                f"baseline AgentFlow requires retrieval_top_k={BASELINE_RETRIEVAL_TOP_K}"
            )
        rollout = self.config.actor_rollout_ref.rollout
        if int(rollout.prompt_length) != BASELINE_PROMPT_TOKENS:
            raise ValueError(f"baseline AgentFlow requires prompt_length={BASELINE_PROMPT_TOKENS}")
        if int(rollout.response_length) != BASELINE_RESPONSE_TOKENS:
            raise ValueError(
                f"baseline AgentFlow requires response_length={BASELINE_RESPONSE_TOKENS}"
            )
        if str(rollout.mode) != "async":
            raise ValueError("baseline AgentFlow requires async rollout mode")
        if not corpus_path or not index_dir or not retrieval_device:
            raise ValueError("corpus_path, index_dir, and retrieval_device are required")
        self._retriever = _cached_retriever(index_dir, corpus_path, retrieval_device)

    async def run(self, sampling_params: dict[str, Any], **kwargs: Any) -> AgentFlowOutput:
        """Run the bounded append-only policy loop for one dataset row."""

        example = _example_from_kwargs(kwargs)
        episode = AgentR1Episode(example, self._retriever)
        steps: list[Any] = []
        metrics: dict[str, float] = {}
        response_length = int(self.config.actor_rollout_ref.rollout.response_length)

        while not episode.complete:
            prompt_ids = await self.apply_chat_template(list(episode.prompt_messages()))
            if len(prompt_ids) > BASELINE_PROMPT_TOKENS:
                raise ValueError(
                    f"append-only prompt has {len(prompt_ids)} tokens, exceeding "
                    f"the baseline bound {BASELINE_PROMPT_TOKENS}"
                )
            with simple_timer("generate_sequences", metrics):
                output = await self.server_manager.generate(
                    request_id=uuid4().hex,
                    prompt_ids=prompt_ids,
                    sampling_params=sampling_params,
                )
            response_ids = list(output.token_ids[:response_length])
            if not response_ids:
                raise ValueError("Agent-R1 server returned an empty policy response")
            if output.log_probs is None:
                raise ValueError(
                    "Agent-R1 server did not return response log probabilities; "
                    "set actor_rollout_ref.rollout.calculate_log_probs=true"
                )
            response_logprobs = [float(value) for value in output.log_probs[: len(response_ids)]]
            if len(response_logprobs) != len(response_ids):
                raise ValueError("response token IDs and rollout log probabilities are misaligned")
            raw_response = self.tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            transition = episode.apply_response(raw_response)
            reward_extra_info = transition.reward_extra_info(
                attempted_searches=episode.attempted_searches
            )
            reward_extra_info["termination_reason"] = episode.termination_reason or "in_progress"
            step = AgentFlowStep(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_logprobs=response_logprobs,
                response_mask=[1] * len(response_ids),
                reward_score=transition.reward,
                extra_fields={
                    "example_id": example.example_id,
                    "reward_extra_info": reward_extra_info,
                    "transition": transition.trace_record(),
                },
            )
            steps.append(await self._postprocess(step, **kwargs))

        return AgentFlowOutput(steps=steps, metrics=metrics)
