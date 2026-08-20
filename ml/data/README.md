# ApacheJIT data

Place the downloaded ApacheJIT CSV at `ml/data/raw/apachejit.csv`, then run
`dvc repro`. Raw datasets and generated model artifacts are intentionally
excluded from Git. The pipeline maps common ApacheJIT short feature names such
as `la`, `ld`, `nf`, `entropy`, `ndev`, `age`, `nuc`, and `exp` into BugRisk
AI's versioned inference schema.

