# HotpotQA data pipeline

The data pipeline converts the official HotpotQA distractor train and development files into
deterministic, versioned JSONL artifacts. It uses only the Python standard library, streams the
top-level source arrays, verifies every raw file before parsing, and does not require a GPU.

## Pinned inputs

[`configs/data/hotpotqa-distractor-v1.1.json`](../configs/data/hotpotqa-distractor-v1.1.json)
locks the following provenance:

- official train v1.1 and distractor development v1 identities, pinned byte-identical download
  locations, byte counts, record counts, and SHA-256 digests;
- the HotpotQA source repository revision used to document the input schema and CC BY-SA 4.0
  license;
- the Agent-R1 revision whose logical `RLHFDataset` row shape is targeted by the adapter output.

Raw and converted files are ignored by Git. Download the two source files once:

```bash
deep-research-rl data download \
  --source-config configs/data/hotpotqa-distractor-v1.1.json \
  --raw-dir data/raw/hotpotqa
```

Existing raw files are checked, never silently replaced. Both byte size and digest must match the
source lock before conversion begins. Each split may list ordered fallback locations, but every
download is accepted only when it matches the same locked size and digest.

## Debug and full builds

A bounded debug build takes deterministic prefixes from both official splits:

```bash
deep-research-rl data prepare \
  --source-config configs/data/hotpotqa-distractor-v1.1.json \
  --raw-dir data/raw/hotpotqa \
  --output-dir data/processed/hotpotqa-debug \
  --max-train 32 \
  --max-validation 32
```

Omit both limits for a full build:

```bash
deep-research-rl data prepare \
  --source-config configs/data/hotpotqa-distractor-v1.1.json \
  --raw-dir data/raw/hotpotqa \
  --output-dir data/processed/hotpotqa-v1.1
```

Each command verifies the completed output. A saved build can also be checked independently:

```bash
deep-research-rl data verify --output-dir data/processed/hotpotqa-v1.1
```

## Output layout

```text
<output-dir>/
  examples/
    train.jsonl
    validation.jsonl
  agent_r1/
    train.jsonl
    validation.jsonl
  corpus.jsonl
  manifest.json
```

The `examples` files are the canonical records. `corpus.jsonl` contains paragraphs deduplicated
by title plus exact sentence sequence and sorted by a content-derived document ID. The
`agent_r1` files expose the pinned logical columns as JSONL; materializing those rows as Parquet
belongs with the training runtime, so this dependency-light stage does not import pandas,
PyArrow, verl, or Agent-R1.

## Canonical example schema

Every canonical record contains:

| Field | Meaning |
|---|---|
| `schema_version`, `record_type` | Logical schema discriminator |
| `dataset`, `variant`, `source_revision` | Dataset provenance |
| `example_id`, `split` | Stable identity and split |
| `question`, `prompt` | Original task text and the policy-visible message list |
| `answers` | Ordered answer aliases; official HotpotQA contributes one alias |
| `level`, `question_type` | Original difficulty and bridge/comparison classification |
| `supporting_facts` | Ordered title, zero-based sentence index, and document ID labels |
| `supporting_titles`, `supporting_document_ids` | First-occurrence-deduplicated evidence labels |
| `context_document_ids` | Ordered references to every distractor context paragraph |

`prompt` is validated as exactly one user message whose content equals `question`. Answers,
supporting facts, evidence titles, document IDs, level, and question type are not copied into that
message. They remain available for reward and evaluation outside policy-visible input.

`HotpotQAExample.to_core_example()` maps a canonical row to the local dependency-light `Example`
contract. `HotpotQAExample.to_agent_r1_dict()` emits these logical Agent-R1 columns:

- `data_source`
- `prompt`
- `reward_model`
- `extra_info`

The primary answer is placed in `reward_model.ground_truth`; complete aliases and evidence labels
are retained under `extra_info`. The adapter output is regenerated from canonical rows and checked
for exact equality during build verification.

## Corpus schema and evidence references

Each corpus row contains `document_id`, `title`, joined `text`, and the original ordered
`sentences`. The document ID is SHA-256 over a canonical JSON representation of title and sentence
sequence. Preserving sentence boundaries keeps supporting-fact indices meaningful, while the
joined text maps directly to the local retrieval `Document` contract.

The converter rejects duplicate example IDs, split overlap, missing supporting titles, ambiguous
same-title contexts, negative sentence indices, hash collisions, and missing corpus references.
An out-of-range nonnegative sentence label is preserved rather than rewritten and counted as a
`supporting_fact_reference_issues` anomaly in the manifest.

## Build manifest

`manifest.json` records source revision, license, pinned source hashes, build mode and limits,
split counts, corpus occurrence/deduplication counts, evidence-reference anomaly counts, adapter
compatibility, relative output paths, record counts, byte sizes, and SHA-256 digests. It contains
no timestamp or absolute path, so repeating a build from the same source lock and limits produces
identical bytes. A reviewable shape is available at
[`configs/data/hotpotqa-build-manifest.template.json`](../configs/data/hotpotqa-build-manifest.template.json).

Reference manifests from verified official-data builds are committed under
[`configs/data/manifests/`](../configs/data/manifests/): a five-example-per-split debug build and a
full 90,447-train / 7,405-validation build. The manifests contain metadata and hashes only; the
raw and converted data remain untracked. The full source contains 23 nonnegative supporting-fact
sentence indices that exceed their referenced paragraph length. Those labels are preserved and
counted rather than silently changed.

The small files under `tests/fixtures/hotpotqa/` are fictional schema fixtures used for CPU tests;
they are not benchmark samples and their outputs must not be reported as benchmark results.
