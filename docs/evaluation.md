# Evaluation and observability

The baseline evaluator uses one protocol, `baseline_evaluation_v1`, for three policy conditions:

| Condition | Model and allowed behavior |
|---|---|
| `no_search` | The pinned base checkpoint receives one ANSWER-only prompt, gets one generation, and has no retriever. |
| `prompted_agent` | The pinned base checkpoint uses the versioned search-agent prompt and environment. |
| `rl_agent` | A trained checkpoint uses the same prompt, environment, retrieval, and inference controls as `prompted_agent`. |

The full baseline evaluation reads all 7,405 pinned HotpotQA distractor validation examples in
canonical order. A debug run reads a deterministic ordered prefix and always carries
`result_scope = "debug_validation_not_benchmark"`. Passing `--final-validation` removes the prefix
limit and verifies the complete canonical example hash. Search-capable final runs additionally
require the pinned full-corpus FAISS/BGE index. Debug results are engineering evidence, not
benchmark results.

## Frozen controls

- Greedy decoding, seed 0, at most 96 generated tokens per policy step, and at most 8,192 prompt
  tokens without silent truncation.
- Search agents have at most 5 executed policy-selected searches and 8 policy steps. Each executed
  search returns top-3 results. The no-search condition has 0 searches and 1 policy step.
- The answer, prompt, environment, append-only context, terminal reward, zero costs, and
  terminal-only credit contracts are unchanged.
- Exact match and token F1 take the maximum over accepted answer aliases after lowercase,
  punctuation, article, and whitespace normalization.

## Per-example metrics

`success` means normalized exact match equals one. `completed` means the trajectory ended in a
valid `ANSWER`. A max-step outcome without an answer is a complete, valid policy outcome with zero
answer scores; it is not an infrastructure failure.

`attempted_searches` counts every parsed `SEARCH`, including a budget-rejected action.
`executed_searches` counts `search_executed` observations, which correspond one-to-one with actual
retriever calls. Rejected searches, malformed actions, policy steps, and termination reasons remain
separate fields.

Token fields are computed as follows:

```text
prompt_tokens_processed   = sum(len(prompt_ids) for every model call)
response_tokens_generated = sum(len(response_ids) for every model call)
total_model_tokens        = prompt_tokens_processed + response_tokens_generated
```

`tool_tokens_appended` tokenizes each rendered `search_executed` observation exactly once with the
evaluated model tokenizer and without chat-template special tokens. It measures tool-output volume
but is not added again to `total_model_tokens`: the observation is already counted whenever it
appears inside a later prompt.

For a labeled example, let `S` be the supporting-document ID set and `R` the union of document IDs
returned by all executed searches. The per-example evidence recall is `|S ∩ R| / |S|`, and
complete support means `S` is a subset of `R`. Aggregation reports the mean per-example recall
(macro), total hits divided by total supporting documents (micro), and the fraction with complete
support. A labeled no-search example therefore scores zero. An example with no labels is excluded,
and the labeled and excluded denominators are explicit.

## Run commands

A bounded no-search debug run needs only the canonical examples and cached pinned model:

```bash
deep-research-rl evaluation no-search \
  --run-id no-search-debug \
  --examples data/processed/hotpotqa-debug-5/examples/validation.jsonl \
  --output-dir artifacts/evaluation/no-search-debug \
  --max-examples 1 \
  --local-files-only
```

Search-capable conditions also require an integrity-checked corpus and index:

```bash
deep-research-rl evaluation prompted-agent \
  --run-id prompted-debug \
  --examples data/processed/hotpotqa-debug-5/examples/validation.jsonl \
  --corpus data/processed/hotpotqa-debug-5/corpus.jsonl \
  --index-dir indexes/hotpotqa-debug-5/bm25 \
  --output-dir artifacts/evaluation/prompted-debug \
  --max-examples 1 \
  --local-files-only
```

`evaluation rl-agent` has the same retrieval arguments and requires explicit `--model-name` and
`--model-revision` for the trained checkpoint. Final runs use `--final-validation` and omit
`--max-examples`; search-capable final runs reject BM25 and non-pinned corpus/index artifacts.

## Artifacts and integrity

Each output directory contains:

| File | Contract |
|---|---|
| `per-example.jsonl` | Ordered `evaluation_example` records containing all derived metrics and the complete `agent_rollout` trajectory. Infrastructure exceptions use `evaluation_infrastructure_failure`. |
| `aggregate.json` | Nested aggregate recomputed only from valid per-example records. |
| `aggregate.csv` | One flat row using the same stable columns as comparison tables. |
| `resolved-config.json` | Dataset and requested-ID hashes, exact model/prompt/inference/retrieval controls, command, code revision/dirty state, and host runtime. |
| `manifest.json` | Run status plus byte size and SHA-256 for every emitted artifact. |

Aggregation requires every requested example ID exactly once and in requested canonical order.
Duplicate, missing, unexpected, unreadable, or incomplete records fail aggregation. Any
`evaluation_infrastructure_failure` invalidates the run instead of receiving a model score.

The aggregate can be independently regenerated from the source JSONL:

```bash
deep-research-rl evaluation aggregate \
  --per-example artifacts/evaluation/prompted-debug/per-example.jsonl \
  --examples data/processed/hotpotqa-debug-5/examples/validation.jsonl \
  --max-examples 1 \
  --output-json artifacts/evaluation/prompted-debug/recomputed.json \
  --output-csv artifacts/evaluation/prompted-debug/recomputed.csv
```

Comparison tables accept compatible aggregates and sort rows as `no_search`, `prompted_agent`,
then `rl_agent`:

```bash
deep-research-rl evaluation compare \
  --aggregates RUN_A/aggregate.json RUN_B/aggregate.json RUN_C/aggregate.json \
  --output comparison.csv
```

The table identifies each policy condition and model revision and includes correctness,
completion, search, step, malformed-action, model/tool-token, and evidence fields. Inputs must have
the same result scope, dataset source, and requested example-ID hash. A table is descriptive; it
does not itself establish statistical significance across training seeds.
