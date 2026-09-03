"""Episode orchestration across replaceable policy, reward, credit, and cost seams."""

from __future__ import annotations

from dataclasses import replace

from deep_research_rl.core.actions import parse_action
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.metrics import build_episode_metrics
from deep_research_rl.core.models import Example, Step, Trajectory
from deep_research_rl.core.protocols import CostFunction, CreditAssigner, Policy, RewardFunction


class RolloutLimitError(RuntimeError):
    """Raised when a policy does not answer within the orchestration safety limit."""


def run_episode(
    example: Example,
    policy: Policy,
    environment: ResearchEnvironment,
    reward_function: RewardFunction,
    credit_assigner: CreditAssigner,
    cost_function: CostFunction,
    *,
    max_steps: int = 64,
) -> Trajectory:
    """Run one episode and attach reward, cost, and assigned credit to every step."""

    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    initial_state = environment.reset(example)
    state = initial_state
    steps: list[Step] = []
    for index in range(max_steps):
        raw_action = policy.choose_action(state)
        action = parse_action(raw_action)
        next_state, observation = environment.transition(state, action)
        reward = reward_function.score(example, action, next_state)
        cost = cost_function.cost(action, observation)
        steps.append(
            Step(
                index=index,
                raw_action=raw_action,
                action=action,
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
            break
    else:
        raise RolloutLimitError(f"policy did not answer within {max_steps} steps")

    credits = credit_assigner.assign(tuple(step.reward for step in steps))
    if len(credits) != len(steps):
        raise ValueError("credit assigner must return one value per step")
    credited_steps = tuple(
        replace(step, credit=credit) for step, credit in zip(steps, credits, strict=True)
    )
    metrics = build_episode_metrics(example, state, len(credited_steps))
    return Trajectory(
        example=example,
        initial_state=initial_state,
        steps=credited_steps,
        final_state=state,
        metrics=metrics,
    )
