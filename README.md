# DeepResearch-RL

DeepResearch-RL is a research platform for studying sequential retrieval policies under a
limited search budget. The initial baseline intentionally uses conventional retrieval and
outcome-only reinforcement learning so that later policy, reward, and credit-assignment
experiments have a clear comparison point.

The repository provides a dependency-light project foundation, a deterministic CPU vertical slice
across retrieval, state transitions, trajectories, outcome reward, terminal credit, and metrics,
a reproducible HotpotQA distractor data pipeline, integrity-checked BM25 and FAISS/BGE retrieval
backends, and a revision-pinned Qwen3 rollout adapter. Distributed training remains a separate,
independently testable layer.

## Quick start

Python 3.11 or 3.12 is required. The core package has no ML or GPU dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
deep-research-rl --help
deep-research-rl config validate configs/baseline.toml
deep-research-rl smoke --output artifacts/smoke/synthetic-trajectory.jsonl
```

The smoke command uses fictional data and labels both its console output and JSONL record as
synthetic and ineligible for benchmark reporting. Its complete trajectory is useful for reviewing
the state/action/reward/credit path without installing a model or training framework. See
[`docs/trajectory-jsonl.md`](docs/trajectory-jsonl.md) for the output contract.

## Development

Run all checks with the same entry point used by CI:

```bash
make check
```

Individual commands are also available:

```bash
make lint
make format-check
make typecheck
make test
make cli-check
```

## Configuration convention

Committed files under `configs/` contain reviewable defaults. Their top-level `config_kind` is
`"defaults"`, and unresolved provenance fields are explicit. An executable experiment must write
a complete configuration with `config_kind = "resolved"` beneath its run directory, including
the exact data, model, and upstream revisions. Run directories and resolved configurations are
ignored by Git.

Inspect a configuration without importing any training stack:

```bash
deep-research-rl config show configs/baseline.toml
```

## Core contracts

The `deep_research_rl.core` package contains immutable models for examples, documents, actions,
observations, states, steps, metrics, and trajectories. Its replaceable interfaces keep retrieval,
policy, context construction, reward, credit assignment, and cost independent. The local path
provides strict `SEARCH(query)` / `ANSWER(answer)` parsing, append-only context, in-memory BM25,
a scripted policy, normalized terminal exact match, zero cost, and terminal-only credit using only
the Python standard library.

## HotpotQA data

The committed source descriptor pins the official HotpotQA train v1.1 and distractor development
files by URL, byte count, record count, and SHA-256. Dependency-light commands download, convert,
and verify canonical examples, a deduplicated corpus, Agent-R1 logical rows, and a complete build
manifest. Debug builds select bounded deterministic prefixes and require no GPU. See
[`docs/hotpotqa-data.md`](docs/hotpotqa-data.md) for commands and schemas.

## Retrieval indexes

Both retrieval backends return the same ranked result schema: stable document ID, title, text,
score, and one-based rank. Every saved index has a deterministic manifest that binds it to the
exact corpus byte hash, ordered document-ID hash, and cardinality and also hashes each index
artifact. BM25 stays dependency-light; FAISS/BGE uses a separately installable, pinned runtime:

```bash
python -m pip install -e ".[retrieval]"
deep-research-rl retrieval build \
  --backend bm25 \
  --corpus data/processed/hotpotqa-debug/corpus.jsonl \
  --index-dir indexes/hotpotqa-debug/bm25
```

See [`docs/retrieval.md`](docs/retrieval.md) for FAISS/BGE commands, integrity behavior,
Agent-R1 compatibility, and supporting-document Recall@K diagnostics.

## Model-backed rollout

The optional rollout runtime loads `Qwen/Qwen3-4B-Instruct-2507` from an immutable revision and
records exact prompt/response token IDs, masks, and per-response-token generation log
probabilities. It preserves the complete observation history, accepts only exact `SEARCH(query)`
or `ANSWER(answer)` output, never forces a search, and enforces separate five-search and
maximum-step safety bounds.

```bash
python -m pip install -e ".[dev,rollout]"
deep-research-rl agent rollout --help
```

See [`docs/agent-rollout.md`](docs/agent-rollout.md) for transition semantics, malformed-action
handling, the downstream training contract, and a bounded real-model debug command.

## Evaluation and observability

The evaluator exposes separate `no-search`, `prompted-agent`, and `rl-agent` entry points under
`deep-research-rl evaluation`. Every run writes a complete trajectory-backed per-example JSONL,
aggregate JSON and CSV, a resolved configuration, and a hash-bearing run manifest. Debug runs use
ordered prefixes and are explicitly ineligible for benchmark reporting. See
[`docs/evaluation.md`](docs/evaluation.md) for the frozen controls, metric formulas, integrity
rules, artifact schemas, and comparison-table commands.

## Local artifact policy

Datasets, model weights, indexes, checkpoints, run outputs, and generated artifacts are local
inputs or outputs rather than source files. The corresponding top-level directories and common
large-file extensions are ignored. Reproducible runs should record origins, revisions, hashes,
seeds, and output locations in their run metadata.

## Contributor workflow

Project coordination is maintained separately from this distributable repository. Before making
changes, contributors should read the provided `AGENTS.md`, then `STATUS.md`, then the assigned
task specification. Those coordination files are intentionally not copied into this repository.
