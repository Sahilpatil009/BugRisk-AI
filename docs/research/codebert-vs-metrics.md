# Phase 6 — CodeBERT + Metrics vs. Metrics-Only Baseline

Status: _in progress — numbers below are from the first subset run._

## Question

The PRD (section 17 / roadmap phase 6) asks whether adding **source-code
semantics** to the git/code metrics improves defect prediction:

> Compare: XGBoost metrics only vs. CodeBERT + metrics. Evaluate whether
> CodeBERT improves defect prediction.

## Setup

- **Data**: the 15-project ApacheJIT subset shipped in `ml/data` (106,674
  commits). ApacheJIT carries per-commit metrics but no source text, so the
  extraction stage (`ml/embeddings/extract.py`) clones the public Apache
  GitHub mirrors and records the post-change content of every source file
  (.java/.py/.scala/.kt, capped at 8 files per commit, 20,000 chars per file)
  touched by each dataset commit.
- **Embeddings**: `microsoft/codebert-base` (768-d), mean-pooled over token
  positions, L2-normalized per file; the commit vector is the L2-normalized
  mean of its changed-file vectors. File encodings are cached by content hash
  so re-runs are incremental.
- **Models** (same rows, same protocol as the deployed baseline):
  - Baseline: XGBoost (300 trees, depth 5, lr 0.05) on the 11 git/code
    metrics — the same hyperparameters as `ml/training/train.py`.
  - CodeBERT model: 2-layer MLP (779 → 256 → 1, dropout 0.3, Adam, 1e-3,
    early stopping on validation PR-AUC) on the 11 metrics + 768-d embedding.
- **Evaluation protocol**: per-project temporal 60/20/20 split (no shuffling
  across time), Platt calibration on validation probabilities, recall-aware
  F2 threshold chosen on validation. Metrics: precision, recall, F1, F2,
  ROC-AUC, PR-AUC, confusion matrix.
- **Subset for the first run**: `apache/zookeeper` (839 commits) as an
  end-to-end feasibility pass; the expanded run uses
  zookeeper, spark, hadoop-mapreduce, hadoop-hdfs and kafka (8,916 commits).
  Both models are trained **only on commits whose source resolved in the
  mirrors**, so the comparison is on identical rows.

## Results

_Updated after each run — see `ml/artifacts/codebert/codebert_comparison.json`
for the machine-readable version._

| Metric | XGBoost (metrics only) | CodeBERT + metrics (MLP) | Δ |
| --- | ---: | ---: | ---: |
| Precision | _ | _ | _ |
| Recall | _ | _ | _ |
| F1 | _ | _ | _ |
| F2 (recall-weighted) | _ | _ | _ |
| ROC-AUC | _ | _ | _ |
| PR-AUC | _ | _ | _ |

## Interpretation notes

- PR-AUC and recall are the headline metrics here because the buggy class is
  the minority and, per the PRD, missing a high-risk change is more costly
  than flagging a clean one.
- A null or negative result is a valid, publishable finding: it would mean
  the git/code metrics already capture most of the signal CodeBERT adds at
  this scale, and it sets the ablation baseline for phase 7.
- CodeBERT adds meaningful inference cost (one forward pass per changed file
  vs. an XGBoost predict over 11 numbers), so the metric gap has to justify
  the latency/disk cost before it is promoted into the live `/predict` path.

## How to reproduce

```bash
pip install -e 'backend[codebert]'
python -m ml.embeddings.extract --data ml/data/processed/apachejit.csv \
    --projects apache/zookeeper,apache/spark,apache/hadoop-mapreduce,apache/hadoop-hdfs,apache/kafka \
    --output ml/data/codebert/sources
python -m ml.embeddings.embed --sources ml/data/codebert/sources/sources.parquet \
    --output ml/data/codebert/embeddings
python -m ml.training.train_codebert --data ml/data/processed/apachejit.csv \
    --embeddings ml/data/codebert/embeddings/embeddings.npz \
    --output ml/artifacts/codebert
```

(or `dvc repro` — the `codebert-extract`, `codebert-embed` and `codebert-train`
stages in `dvc.yaml` run the same commands.)
