# Retrieval indexes and diagnostics

The retrieval layer exposes one contract across local BM25 and production FAISS/BGE. A search
returns ordered records with `document_id`, `title`, `text`, finite `score`, and a contiguous
one-based `rank`. The stable ID maps every hit back to the exact corpus JSONL record. Batch search
preserves query order and uses the same ranking path as single-query search.

## Install

BM25 uses the Python standard library. The dense backend is an optional pinned environment:

```bash
python -m pip install -e ".[retrieval]"
```

The default dense model is `BAAI/bge-large-en-v1.5` at revision
`d4aa6901d3a41ba39fb536a557fa166f842b0e09`. The adapter follows Agent-R1 revision
`b124aa46534cbf2fb8bc8af11405774984c42ac7`: passage input is `title + " " + text`, queries use
`Represent this sentence for searching relevant passages: `, and the index is exact FAISS `Flat`
inner-product search. The saved artifact names `hpqa_corpus.npy` and `index.bin` match that recipe.
The optional `AgentR1FaissBGEToolAdapter` renders its legacy `{"results": [...]}` tool payload
without weakening the scored internal contract.

## Build, verify, and search

Build the dependency-light index:

```bash
deep-research-rl retrieval build \
  --backend bm25 \
  --corpus data/processed/hotpotqa-debug-5/corpus.jsonl \
  --index-dir indexes/hotpotqa-debug-5/bm25
```

Build the pinned FAISS/BGE index:

```bash
deep-research-rl retrieval build \
  --backend faiss_bge \
  --corpus data/processed/hotpotqa-debug-5/corpus.jsonl \
  --index-dir indexes/hotpotqa-debug-5/faiss-bge-large \
  --model-name BAAI/bge-large-en-v1.5 \
  --model-revision d4aa6901d3a41ba39fb536a557fa166f842b0e09 \
  --device cpu
```

Every build verifies its output. Verification can be repeated without loading the BGE model:

```bash
deep-research-rl retrieval verify \
  --corpus data/processed/hotpotqa-debug-5/corpus.jsonl \
  --index-dir indexes/hotpotqa-debug-5/faiss-bge-large
```

Inspect the full common result schema and source text:

```bash
deep-research-rl retrieval search \
  --corpus data/processed/hotpotqa-debug-5/corpus.jsonl \
  --index-dir indexes/hotpotqa-debug-5/bm25 \
  --query "Which magazine was started first?" \
  --top-k 5
```

## Integrity contract

Each index directory contains `manifest.json`. It records the corpus byte size, SHA-256,
cardinality, and SHA-256 of the ordered document-ID stream. It also records every index artifact's
relative path, byte size, and SHA-256. Load and verify fail before search if any of these values
differs. BM25 additionally validates document lengths, postings, token counts, and parameters.
FAISS additionally validates embedding shape and dtype, index dimension, training state,
`index.ntotal`, encoder identity, model revision, and the pinned Agent-R1 layout.

On macOS, PyTorch and the PyPI FAISS wheel can load competing OpenMP runtimes. BGE encoding is
therefore isolated in a clean child process on that platform; FAISS remains in the caller. Linux
and GPU deployments use direct in-process encoding. This avoids the unsupported
`KMP_DUPLICATE_LIB_OK` workaround and does not change vectors or ranking semantics.

## Supporting-document recall

The diagnostic command uses canonical HotpotQA questions as queries and keeps evaluation labels
outside retrieval input:

```bash
deep-research-rl retrieval diagnose \
  --corpus data/processed/hotpotqa-debug-5/corpus.jsonl \
  --examples data/processed/hotpotqa-debug-5/examples/validation.jsonl \
  --index-dir indexes/hotpotqa-debug-5/bm25 \
  --output reports/retrieval/hotpotqa-debug-5-bm25.json \
  --limit 5 \
  --ks 1 5 10
```

For each query and K, supporting-document recall is the fraction of labeled supporting document
IDs present in the first K hits. The report includes macro recall, micro recall, complete-support-
set rate, per-example IDs/hits, the complete source-file hash, selected ordered IDs, corpus
fingerprint, a snapshot of the complete index manifest, and its hash.

The committed reports use the first five official validation records and the 100-document corpus
from the reproducible five-train/five-validation debug build:

| Backend | Macro Recall@1 | Macro Recall@5 | Macro Recall@10 |
|---|---:|---:|---:|
| BM25 | 0.40 | 0.70 | 0.90 |
| FAISS/BGE-large | 0.50 | 0.90 | 1.00 |

These are bounded retrieval diagnostics over five questions and a debug corpus, not agent
performance, training results, or full-corpus benchmark estimates. The JSON reports are retained
to verify the plumbing and exact subset, not to support a comparative research conclusion.
