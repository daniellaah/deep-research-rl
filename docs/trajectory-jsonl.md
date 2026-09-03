# Trajectory JSONL

The CPU path writes one complete trajectory per UTF-8 JSONL line. Records are stable-key JSON with
`schema_version = 1` and `record_type = "trajectory"`. Readers reject unknown schema versions,
invalid field types, unknown action or observation variants, and inconsistent provenance markers.

## Provenance and reporting safety

Every record exposes `synthetic` and `benchmark_eligible` at the top level and repeats them inside
`example`. The two values must match during deserialization. The bundled smoke fixture uses:

```json
{
  "synthetic": true,
  "benchmark_eligible": false,
  "example": {
    "source": "synthetic_non_benchmark",
    "synthetic": true,
    "benchmark_eligible": false
  }
}
```

Scores from such a record are validation signals for the software path, not benchmark results.

## Record shape

| Field | Meaning |
|---|---|
| `schema_version` | Integer version for compatibility checks. |
| `record_type` | Always `"trajectory"`. |
| `synthetic`, `benchmark_eligible` | Prominent reporting-safety markers. |
| `example` | ID, question, accepted answers, supporting document IDs, and source metadata. |
| `initial_state` | Empty-context state before the first policy action. |
| `steps` | Ordered, zero-indexed transitions with raw and parsed actions. |
| `final_state` | State after the final recorded transition. |
| `metrics` | Per-episode exact match, token F1, termination, executed searches, and step count. |

Each step contains:

| Field | Meaning |
|---|---|
| `index` | Contiguous zero-based transition index. |
| `raw_action` | Exact text emitted by the policy. |
| `action` | Parsed `search` with `query`, or `answer` with `answer`. |
| `state_before`, `state_after` | Complete immutable states around the transition. |
| `observation` | `search_executed`, `search_rejected`, or `answer_recorded` response. |
| `reward` | Outcome reward on this transition; intermediate values are zero. |
| `cost` | Independent transition cost; zero in the baseline. |
| `credit` | Learning credit assigned to this step; only the final step receives terminal credit. |

An agent state contains `example_id`, `question`, the ordered `context`, `executed_searches`,
`terminated`, and the optional terminal `answer`. Context is append-only: every next state retains
the entire previous context and appends exactly its new observation. A rejected over-budget search
adds feedback to context but does not invoke retrieval or increment `executed_searches`.
Every document in a successful search observation contains the corpus `document_id`, `title`, and
`text` together with its backend-specific finite `score` and contiguous one-based `rank`.

## Generate and read an example

```bash
deep-research-rl smoke --output artifacts/smoke/synthetic-trajectory.jsonl
```

The output directory is intentionally ignored because trajectories are run artifacts. Python
callers can round-trip records with `trajectory_as_json`, `trajectory_from_json`,
`write_trajectory_jsonl`, and `read_trajectory_jsonl` from
`deep_research_rl.core.serialization`.
