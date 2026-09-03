"""Bounded model-backed rollout preserving the frozen environment semantics."""

from __future__ import annotations

from dataclasses import replace

from deep_research_rl.agent.contracts import (
    AgentRollout,
    AgentRolloutStep,
    ModelIdentity,
    TerminationReason,
)
from deep_research_rl.agent.protocols import GenerativePolicy
from deep_research_rl.core.actions import ActionParseError, parse_action
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.metrics import build_episode_metrics
from deep_research_rl.core.models import Example
from deep_research_rl.core.protocols import CostFunction, CreditAssigner, RewardFunction

DEBUG_RESULT_SCOPE = "debug_validation_not_benchmark"


def run_model_rollout(
    example: Example,
    policy: GenerativePolicy,
    environment: ResearchEnvironment,
    reward_function: RewardFunction,
    credit_assigner: CreditAssigner,
    cost_function: CostFunction,
    *,
    max_steps: int,
    result_scope: str = DEBUG_RESULT_SCOPE,
) -> AgentRollout:
    """Run until an answer or the independent maximum-step safety bound is reached."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")
    if not result_scope:
        raise ValueError("result_scope must not be empty")

    initial_state = environment.reset(example)
    state = initial_state
    steps: list[AgentRolloutStep] = []
    termination_reason: TerminationReason = "max_steps"

    for index in range(max_steps):
        policy_output = policy.generate(state, max_searches=environment.max_searches)
        parse_error: str | None
        try:
            action = parse_action(policy_output.raw_response)
        except ActionParseError as error:
            parse_error = str(error)
            next_state, observation = environment.record_malformed_action(state, parse_error)
            reward = 0.0
            cost = 0.0
            parsed_action = None
        else:
            parse_error = None
            parsed_action = action
            next_state, observation = environment.transition(state, action)
            reward = reward_function.score(example, action, next_state)
            cost = cost_function.cost(action, observation)

        steps.append(
            AgentRolloutStep(
                index=index,
                policy_output=policy_output,
                action=parsed_action,
                parse_error=parse_error,
                state_before=state,
                observation=observation,
                state_after=next_state,
                reward=reward,
                cost=cost,
                credit=0.0,
            )
        )
        state = next_state
        if state.terminated:
            termination_reason = "answered"
            break

    credits = credit_assigner.assign(tuple(step.reward for step in steps))
    if len(credits) != len(steps):
        raise ValueError("credit assigner must return one value per rollout step")
    credited_steps = tuple(
        replace(step, credit=credit) for step, credit in zip(steps, credits, strict=True)
    )
    metrics = build_episode_metrics(example, state, len(credited_steps))
    return AgentRollout(
        model=ModelIdentity(policy.model_name, policy.model_revision),
        prompt_format=policy.prompt_format,
        result_scope=result_scope,
        example=example,
        initial_state=initial_state,
        steps=credited_steps,
        final_state=state,
        termination_reason=termination_reason,
        metrics=metrics,
        prompt_tokens=sum(len(step.policy_output.prompt_ids) for step in credited_steps),
        response_tokens=sum(len(step.policy_output.response_ids) for step in credited_steps),
    )
