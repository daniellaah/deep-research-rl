# Agent-R1/verl GRPO training

This path is intentionally small: one HotpotQA train example, two sampled trajectories, one GRPO
optimizer update, one validation example, and a second update after explicit checkpoint resume. It
is an engineering sanity run, not a benchmark or a useful trained policy.

## Immutable runtime

The committed defaults in `configs/training/agentr1-verl-grpo.toml` lock:

- Agent-R1 revision `b124aa46534cbf2fb8bc8af11405774984c42ac7`;
- verl `0.7.0` revision `f9c855f7cf04d603c9546bc01776c74806a879c1`;
- container
  `docker.io/verlai/verl@sha256:9576682f85ca36f4ef719efccc5a5deb4d0b6f66f06fc14f43fdfed0749fbf5d`;
- Linux `amd64`, CUDA 12.8 or newer, and bfloat16-capable NVIDIA devices;
- `Qwen/Qwen3-4B-Instruct-2507` revision
  `cdbee75f17c01a7cc42f958dc650907174af0554`.

The preflight rejects revision drift, modified upstream checkouts, a model snapshot with the wrong
revision directory, invalid data/index manifests, a different container identity, insufficient GPU
count or memory, and an incompatible CUDA runtime.

Build the runtime on a Linux/amd64 NVIDIA host:

```bash
docker build --platform linux/amd64 \
  -f docker/training.Dockerfile \
  -t deep-research-rl:training .
```

The tag is only a local convenience; the Dockerfile's `FROM` line and the runtime preflight use the
digest, not a floating image tag.

## Training input

First complete and verify the canonical HotpotQA conversion. Then create Agent-R1 Parquet files.
The extra `agent_name` routing column is added only in this derived training layer, so canonical
JSONL remains byte-identical.

```bash
deep-research-rl training prepare-data \
  --processed-dir /data/processed/hotpotqa-distractor \
  --output-dir /data/training/hotpotqa-distractor \
  --max-train 1 \
  --max-validation 1

deep-research-rl training verify-data \
  --training-data-dir /data/training/hotpotqa-distractor
```

The row limits select ordered prefixes and are marked `ordered_prefix_engineering_only` in the
generated manifest. They must never be reported as benchmark results.

## Container and hardware choices

Choose the GPU count, the minimum acceptable memory per selected GPU, and a host-level time or
spend ceiling before starting. The example below uses one GPU and requires at least 40 GiB; these
are operator inputs, not project defaults.

```bash
docker run --rm --gpus all --ipc=host \
  -v /absolute/data:/data \
  -v /absolute/indexes:/indexes:ro \
  -v /absolute/models:/models:ro \
  -v /absolute/runs:/runs \
  deep-research-rl:training \
  bash
```

Inside the container, define paths explicitly. `MODEL_PATH` must end in the pinned 40-character
snapshot revision.

```bash
export MODEL_PATH=/models/Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554
export TRAINING_DATA=/data/training/hotpotqa-distractor
export CORPUS=/data/processed/hotpotqa-distractor/corpus.jsonl
export INDEX=/indexes/hotpotqa-distractor/faiss-bge
export RUN_DIR=/runs/agentr1-grpo-sanity
export N_GPUS=1
export MIN_GPU_MEMORY_GIB=40
```

## Preflight and one-update sanity run

All runtime commands use the same pinned inputs:

```bash
deep-research-rl training preflight \
  --config /workspace/deep-research-rl/configs/training/agentr1-verl-grpo.toml \
  --agent-r1-root /opt/Agent-R1 \
  --verl-root /opt/verl \
  --model-path "$MODEL_PATH" \
  --training-data-dir "$TRAINING_DATA" \
  --corpus "$CORPUS" \
  --index-dir "$INDEX" \
  --flow-config /workspace/deep-research-rl/configs/training/agentr1-flow.yaml \
  --n-gpus "$N_GPUS" \
  --min-gpu-memory-gib "$MIN_GPU_MEMORY_GIB" \
  --output "$RUN_DIR/preflight.json"

/workspace/deep-research-rl/scripts/run_agentr1_grpo.sh \
  --phase sanity \
  --config /workspace/deep-research-rl/configs/training/agentr1-verl-grpo.toml \
  --agent-r1-root /opt/Agent-R1 \
  --verl-root /opt/verl \
  --model-path "$MODEL_PATH" \
  --training-data-dir "$TRAINING_DATA" \
  --corpus "$CORPUS" \
  --index-dir "$INDEX" \
  --flow-config /workspace/deep-research-rl/configs/training/agentr1-flow.yaml \
  --n-gpus "$N_GPUS" \
  --min-gpu-memory-gib "$MIN_GPU_MEMORY_GIB" \
  --run-dir "$RUN_DIR"
```

The launcher fixes asynchronous vLLM rollout, two samples per prompt, GRPO, no critic, no reward
model, no KL reward or loss, one optimizer update, validation at step 0 and step 1, and checkpoint
save frequency 1. It does not force a first search and exposes no parallel-call route. The strict
parser accepts exactly one `SEARCH(query)` or `ANSWER(answer)` per generated response; a sixth
parsed search is recorded as rejected and never reaches the retriever.

## Explicit resume

Resume only from the verified first-update checkpoint:

```bash
/workspace/deep-research-rl/scripts/run_agentr1_grpo.sh \
  --phase resume \
  --resume-from "$RUN_DIR/checkpoints/global_step_1" \
  --config /workspace/deep-research-rl/configs/training/agentr1-verl-grpo.toml \
  --agent-r1-root /opt/Agent-R1 \
  --verl-root /opt/verl \
  --model-path "$MODEL_PATH" \
  --training-data-dir "$TRAINING_DATA" \
  --corpus "$CORPUS" \
  --index-dir "$INDEX" \
  --flow-config /workspace/deep-research-rl/configs/training/agentr1-flow.yaml \
  --n-gpus "$N_GPUS" \
  --min-gpu-memory-gib "$MIN_GPU_MEMORY_GIB" \
  --run-dir "$RUN_DIR"
```

The resume plan requires `global_step_1`, loads it through Agent-R1/verl's explicit resume path,
and targets `global_step_2`. A different checkpoint step is rejected before trainer startup.

## Reward, advantage, and loss path

Each AgentFlow step is one model-generated response:

1. The tokenizer produces `prompt_ids`; the inference server returns `response_ids` and one rollout
   log probability per response token. Missing or misaligned log probabilities fail the rollout.
2. Every generated response token has response mask 1. Retrieval observations are not appended to
   the response token sequence; they enter the next append-only prompt.
3. The strict parser advances the environment by at most one action. Search, malformed, rejected,
   and nonterminal steps receive reward 0. Only a terminating `ANSWER` receives normalized exact
   match, either 0 or 1.
4. Agent-R1 writes each step reward onto that step's last valid response token and groups all steps
   from one episode with a trajectory identifier. The trajectory score is therefore exactly its
   terminal exact-match reward.
5. The pinned Agent-R1 GRPO estimator compares the two trajectory scores for the same source
   prompt, normalizes the outcome advantage, and broadcasts that trajectory advantage to valid
   response tokens. verl recomputes current-policy log probabilities and applies the configured
   vanilla clipped policy loss with sequence-mean/token-mean aggregation.

Thus the only reward/credit signal supplied to GRPO is the terminal outcome; the estimator's
policy-gradient propagation of that outcome to sampled action tokens does not create an
intermediate reward or an evidence-aware step signal.

The per-step `transition_json` travels with reward metadata into both rollout and validation dumps.
It contains state before/after, raw response, parsed action or parse error, observation, immediate
reward, attempted searches, and executed searches. This makes the delayed terminal reward and the
upstream outcome-advantage behavior auditable without inferring actions from decoded text.

## Required evidence

A successful phase is accepted only when the launcher finds all of the following:

- a zero trainer exit code;
- `latest_checkpointed_iteration.txt` equal to the target step;
- the matching `global_step_<N>` checkpoint directory;
- non-empty validation JSONL before and after the update;
- a non-empty rollout JSONL for the update;
- a `transition_json` field on every dumped trajectory step.

Each phase directory contains `resolved-config.json`, `training.log`, and `manifest.json`. The
manifest hashes the resolved config, log, validation dumps, rollout trajectories, and checkpoint
tracker and records the exact model, code state, runtime pins, hardware preflight, seed, start step,
and target step. A dry run writes only its plan and is labeled `dry_run`; it is not execution
evidence.
