"""End-to-end deterministic CPU smoke path."""

from __future__ import annotations

from pathlib import Path

from deep_research_rl.core.context import AppendOnlyContextPolicy
from deep_research_rl.core.costs import ZeroCost
from deep_research_rl.core.credit import TerminalOnlyCreditAssigner
from deep_research_rl.core.environment import ResearchEnvironment
from deep_research_rl.core.fixtures import synthetic_two_hop_fixture
from deep_research_rl.core.models import Trajectory
from deep_research_rl.core.policies import ScriptedPolicy
from deep_research_rl.core.retrieval import BM25Retriever
from deep_research_rl.core.rewards import TerminalExactMatchReward
from deep_research_rl.core.rollout import run_episode
from deep_research_rl.core.serialization import write_trajectory_jsonl


def run_synthetic_smoke(output_path: str | Path) -> Trajectory:
    """Run the fictional two-hop episode and save its complete trajectory."""

    example, documents = synthetic_two_hop_fixture()
    policy = ScriptedPolicy(
        actions=(
            "SEARCH(Brindle Process)",
            "SEARCH(Mira Voss)",
            "ANSWER(Lumen City)",
        )
    )
    environment = ResearchEnvironment(
        BM25Retriever(documents, top_k=1),
        AppendOnlyContextPolicy(),
        max_searches=5,
    )
    trajectory = run_episode(
        example,
        policy,
        environment,
        TerminalExactMatchReward(),
        TerminalOnlyCreditAssigner(),
        ZeroCost(),
    )
    if trajectory.metrics.exact_match != 1.0:
        raise RuntimeError("synthetic smoke trajectory did not reach exact match")
    write_trajectory_jsonl(output_path, trajectory)
    return trajectory
