# Model-backed agent rollout

The baseline rollout path uses the official non-thinking checkpoint
`Qwen/Qwen3-4B-Instruct-2507` at immutable Hugging Face revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. The checkpoint is Apache-2.0 licensed. Loading
always supplies this exact revision to both the tokenizer and model; the adapter rejects a
different resolved commit when the runtime exposes one.

Install the optional model runtime separately from the dependency-light core:

```bash
python -m pip install -e ".[dev,rollout]"
```

## Prompt and action protocol

The prompt format is versioned as `qwen3_strict_search_answer_v1`. At every step it includes the
question, the exact ordered observation history, and the number of searches used, remaining, and
allowed. Context is append-only: no earlier observation can be summarized, removed, or reordered.
If the complete prompt exceeds the configured token bound, rollout fails explicitly rather than
silently truncating evidence.

The model must emit exactly one of these two forms and no surrounding text:

```text
SEARCH(query)
ANSWER(answer)
```

The parser does not extract an action from prose, remove Markdown fences, trim whitespace, or
repair malformed output. This keeps the behavior seen by evaluation identical to the behavior
that generated the trajectory.

## Environment transitions and safety bounds

| Model output | Environment effect |
|---|---|
| Valid `SEARCH(query)` below budget | Invoke retrieval once, append ranked results, increment executed searches. |
| Valid `SEARCH(query)` at budget | Do not invoke retrieval, append a deterministic budget-rejection observation, leave the counter unchanged. |
| Valid `ANSWER(answer)` | Append the answer observation and terminate. |
| Malformed output | Do not invoke retrieval, append deterministic parser feedback, leave reward, cost, and search count unchanged. |

There is no forced first search. Each valid model-selected search action invokes retrieval exactly
once, with no parallel fan-out inside a trajectory. The search limit and the step limit are
independent:

- `max_searches = 5` limits successful retrieval calls.
- `max_steps = 8` guarantees termination even if the model repeatedly emits malformed or
  over-budget actions.

Multiple trajectories may be evaluated concurrently by a future runner, but concurrency must not
change either per-trajectory rule.

## Token and log-probability contract

Every generated step records the exact tensors needed by a later reinforcement-learning worker:

| Field | Contract |
|---|---|
| `prompt_ids` | Unpadded chat-template token IDs given to generation. |
| `response_ids` | Generated token IDs only. |
| `input_ids` | Exact concatenation of `prompt_ids + response_ids`. |
| `position_ids` | Contiguous zero-based positions over `input_ids`. |
| `attention_mask` | One for every real token in this unpadded record. |
| `response_mask` | One per generated response token; its length equals `response_ids`. Prompt tokens are outside this response-only mask. |
| `response_logprobs` | Optional generation-time log probability for every response token, in the same order and with no prompt entries. |

The JSONL artifact is the unpadded source of truth. A batch collator may add padding later, but it
must pad IDs, masks, positions, and log probabilities consistently and must not re-tokenize the
stored response. Padding or materialized tool-output positions receive zero in the downstream
response mask; generated positions retain the recorded ones. Malformed output still represents a
policy decision: its generated tokens remain selected by `response_mask`, and its log probabilities
remain available. The rollout layer records these values; a later training layer is responsible for
combining current-policy log probabilities, old-policy log probabilities, response masks, and
terminal credit or advantages in its loss.

## Bounded real-model debug run

The command below evaluates one local HotpotQA example against an integrity-checked BM25 index:

```bash
deep-research-rl agent rollout \
  --examples data/processed/hotpotqa-debug-5/examples/validation.jsonl \
  --corpus data/processed/hotpotqa-debug-5/corpus.jsonl \
  --index-dir indexes/hotpotqa-debug-5/bm25 \
  --output artifacts/qwen-debug/trajectory.jsonl \
  --max-examples 1 \
  --max-searches 5 \
  --max-steps 8 \
  --model-name Qwen/Qwen3-4B-Instruct-2507 \
  --model-revision cdbee75f17c01a7cc42f958dc650907174af0554 \
  --seed 0
```

After the pinned snapshot has been downloaded once, add `--local-files-only` to make cache-only
loading a hard no-network operation.

This output is explicitly labeled `debug_validation_not_benchmark`. It verifies model loading,
generation, parsing, transitions, retrieval, and tensor provenance, but it is not a benchmark score
or evidence for a research conclusion. CUDA/verl training validation is a separate requirement.
