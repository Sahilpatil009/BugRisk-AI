# Training Data

## Dataset: ApacheJIT

**Source:** Zenodo record 5907847 (DOI: 10.5281/zenodo.5907847)  
**License:** CC BY 4.0  
**Description:** Per-commit change metrics and bug labels for Apache open-source projects.

This is the **only** dataset used for training the BugRisk AI models. No user repository data is
collected or used for training.

## Permitted sources

| Dataset | Usage |
|---|---|
| ApacheJIT (Zenodo 5907847) | Training, validation, test |
| Explicitly consented public repositories | Training only, with `--allow-user-repos` flag |

## Prohibited sources

- Private repositories
- User repositories connected to the BugRisk AI service (without explicit consent and `--allow-user-repos`)
- Any repository not listed in the permitted sources table above

## Directory layout

```
ml/data/
├── raw/            # Downloaded ApacheJIT CSV (gitignored)
├── processed/      # Prepared CSV after feature normalization (gitignored)
├── codebert/       # CodeBERT source extractions (gitignored)
│   └── sources/
├── file_level/     # Per-file dataset rows (gitignored)
└── README.md       # This file
```

## Provenance verification

Run `dvc status` to verify that each stage's inputs match the recorded checksums in `dvc.lock`.
The `acquire` stage downloads ApacheJIT directly from Zenodo using the stable DOI URL.
