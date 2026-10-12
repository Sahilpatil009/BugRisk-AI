# Phase 7 — Metrics + CodeBERT + GRU + GNN Ablation

Status: _quick five-project GPU run reported below as preliminary; the full run is pending._

## Question

The PRD (section 20 / roadmap phase 7) asks for a research model that fuses four views of a
change and an ablation that shows what each view adds:

1. Metrics only
2. Metrics + CodeBERT
3. Metrics + CodeBERT + GRU
4. Metrics + CodeBERT + GRU + GNN

The supervised target stays ApacheJIT's commit label, so every variant predicts
**change/commit risk**, not a file-level bug probability. The models are offline research
artifacts and do not change the live `/predict` path.

## Inputs

| View | Source | Representation |
| --- | --- | --- |
| Metrics | ApacheJIT columns (11 features) | log1p, train-median imputation, train standardization |
| CodeBERT | `ml/embeddings` (phase 6) | 768-d mean of the changed files' mean-pooled embeddings |
| GRU history | `ml/advanced/prepare.py` | Metrics of the project's previous 5 commits, left-padded with a mask channel |
| Dependency graph | `ml/advanced/prepare.py` | Up to 8 changed files as nodes; edges are imports between them |

Graph nodes carry five static source features (log lines, characters, imports, branches,
declarations). Import edges are resolved for Java, Kotlin, Scala (including static, wildcard and
brace imports) and Python by matching dotted import names against the changed files' module paths
after stripping `src/main/java`-style roots. GraphSAGE aggregates over edges in both directions.

## Model

`ml/training/advanced_model.py` builds a fusion network whose branches can be switched off:

```text
metrics ────────────────────────────┐
CodeBERT embedding ─────────────────┤
previous 5 commits → GRU(32) ───────┤→ concat → Linear(256) → ReLU → Dropout(0.3) → Linear(1) → sigmoid
changed-file graph → GraphSAGE(32) ─┘
```

Disabled branches are not instantiated, so each ablation variant has only its own parameters.

## Protocol

Identical to the deployed baseline and the phase 6 comparison:

- **Rows**: only commits that have a CodeBERT embedding and a dependency graph, so every variant
  is evaluated on exactly the same commits.
- **Split**: per-project temporal 60/20/20, with no shuffling across time.
- **Training**: Adam (1e-3, weight decay 1e-5), `pos_weight` for class imbalance, early stopping
  on validation PR-AUC (patience 6, up to 40 epochs).
- **Calibration and threshold**: Platt scaling on validation probabilities, then the F2 (recall
  weighted) threshold chosen on validation.
- **Seeds**: 3 seeds per variant; the table reports mean ± standard deviation.
- **Reference**: XGBoost on the metrics, with the deployed hyperparameters, on the same rows.
- **Selection**: the saved `advanced_model.joblib` is the variant and seed with the best
  validation PR-AUC. The test split is never used for any choice.

### Leakage controls

- A history contains only commits with a **strictly earlier** timestamp in the same project.
  Commits that share a timestamp cannot see each other. Histories use metrics only, never labels.
- Graphs use only the post-change source of the files that commit changed.
- Scalers and imputation statistics are fitted on the training split only.
- The laptop/quick commit cap (`--max-commits-per-project`) picks evenly spaced commits across
  each project's history without looking at labels. A newest-only cap was rejected: the newest
  commits have the least mature labels (their bugs have not been found yet). In validation runs
  on Zookeeper it produced a 20% buggy rate and an almost bug-free test split, against 41% for
  evenly spaced sampling.

## Results

_Filled only from a committed `ml/artifacts/advanced/ablation.json` produced by the full run in
[ml/notebooks/phase7_colab.ipynb](../../ml/notebooks/phase7_colab.ipynb)._

| Variant | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost (metrics) | _ | _ | _ | _ | _ | _ |
| MLP (metrics) | _ | _ | _ | _ | _ | _ |
| + CodeBERT | _ | _ | _ | _ | _ | _ |
| + GRU | _ | _ | _ | _ | _ | _ |
| + GNN | _ | _ | _ | _ | _ | _ |

### Preliminary: quick run (≈300 commits per project)

From [ml/artifacts/advanced/ablation_quick.json](../../ml/artifacts/advanced/ablation_quick.json):
Colab T4 GPU, `QUICK = True` (evenly spaced cap of 300 commits per project, 4 files and 8,000
characters per commit, 256 CodeBERT tokens), 3 seeds. 1,321 commits from the five projects with
a 791 / 264 / 266 split. 1,318 commits have at least one prior commit, and 542 dependency graphs
have at least one import edge. Values are test-split mean ± std across seeds.

| Variant | Precision | Recall | F1 | F2 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| XGBoost (metrics) | 0.376 ± 0.011 | 0.958 ± 0.005 | 0.540 ± 0.011 | 0.731 ± 0.006 | 0.785 ± 0.005 | 0.644 ± 0.006 |
| MLP (metrics) | 0.458 ± 0.017 | 0.935 ± 0.027 | 0.614 ± 0.009 | 0.773 ± 0.006 | 0.802 ± 0.021 | 0.661 ± 0.044 |
| + CodeBERT | 0.444 ± 0.021 | 0.950 ± 0.005 | 0.605 ± 0.019 | 0.773 ± 0.012 | 0.794 ± 0.020 | 0.650 ± 0.033 |
| + GRU | 0.439 ± 0.021 | 0.946 ± 0.011 | 0.600 ± 0.018 | 0.768 ± 0.010 | 0.807 ± 0.003 | 0.685 ± 0.009 |
| + GNN | 0.414 ± 0.008 | 0.950 ± 0.005 | 0.577 ± 0.007 | 0.755 ± 0.004 | 0.813 ± 0.007 | 0.682 ± 0.012 |

What this sample does and does not show:

- **CodeBERT adds nothing measurable here**: PR-AUC −0.011 versus MLP (metrics), well inside the
  seed spread (±0.03–0.04).
- **The GRU history is the only promising step**: PR-AUC +0.035 over metrics + CodeBERT, with the
  smallest spread of any MLP variant (±0.009). The gain is still within the MLP (metrics) spread
  of ±0.044, so it is not yet evidence of improvement.
- **The GNN adds nothing**: PR-AUC −0.003 with a small ROC-AUC gain (+0.006). Only 41% of commits
  have any import edge between their changed files, so most graphs carry little structure.
- **All MLP variants beat XGBoost on ranking** (PR-AUC +0.017 to +0.041, ROC-AUC +0.009 to +0.028)
  on this small sample. XGBoost's 300-tree setup was tuned on the full dataset and may overfit
  791 training commits.
- **The operating points over-flag.** The test split is 33% buggy (87 of 266), while the earlier
  training and validation commits are far more often buggy (57% overall). Labels of later commits
  are less mature, so thresholds calibrated on validation land low (0.12–0.42). The result is
  recall ≈ 0.95 at precision ≈ 0.4 for every variant. Compare variants by PR-AUC and ROC-AUC,
  which do not depend on the threshold.
- With 266 test commits, none of these differences is conclusive. The full run decides.

### Pipeline validation (not a result)

The full pipeline (download, preprocess, extract, CPU CodeBERT embedding, prepare, train, MLflow)
ran end to end on 299 Zookeeper commits, and the Colab notebook ran end to end in a simulated
environment. With about 60 test commits, the per-variant scores from those runs have no
statistical meaning and are deliberately not reported.

## Interpretation notes

- PR-AUC and recall are the headline metrics, because the buggy class is the minority and,
  per the PRD, missing a high-risk change costs more than flagging a clean one.
- Compare each step against the previous one (the `deltas` block in `ablation.json`), and the
  whole ladder against XGBoost. A step whose gain is within one standard deviation across seeds
  is not evidence of improvement.
- A null or negative result is a valid finding: the metrics may already capture most of the
  signal at this scale.
- Every extra view adds inference cost (CodeBERT passes, history lookups, graph building), so a
  gain must justify it before any variant is considered for live inference.

## How to reproduce

GPU (recommended): open the notebook in Google Colab with a T4 GPU and run all cells. Set
`QUICK = True` first to check the pipeline, then `QUICK = False` for the full run.

Locally, after the phase 6 extract and embed stages:

```bash
pip install -e 'backend[ml,codebert]'
python -m ml.advanced.prepare --data ml/data/processed/apachejit.csv \
    --sources ml/data/codebert/sources/sources.parquet --output ml/data/advanced
python -m ml.training.train_advanced --data ml/data/processed/apachejit.csv \
    --embeddings ml/data/codebert/embeddings/embeddings.npz --inputs ml/data/advanced \
    --output ml/artifacts/advanced --device auto
```

(or `dvc repro advanced-train`).
