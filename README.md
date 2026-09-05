# BugRisk AI

BugRisk AI is an explainable engineering-risk dashboard for Python repositories. It predicts
the risk of an analyzed **commit/change**, then derives a separate **file-priority score** to
help reviewers and QA engineers decide where to spend attention. A file-priority score is not
a claim that the file contains a bug.

## What is included

- Next.js 16 dashboard with responsive repository, pull-request, file-evidence, history, and model views.
- FastAPI REST API with GitHub OAuth, encrypted server-side tokens, secure sessions, and
  ownership checks.
- PostgreSQL/Supabase-ready schema and RLS policies; SQLite is used for the zero-config demo.
- Database-backed analysis states and a separate worker process.
- Temporary, authenticated Git clones with Python metrics from Git history, AST, and Radon.
- ApacheJIT preprocessing, temporal per-project splits, model comparison, sigmoid calibration,
  SHAP explanations, MLflow logging, and DVC stages.
- Deterministic, evidence-backed recommendations. No LLM is allowed to create risk scores.
- Signed GitHub App webhooks for opened/updated pull requests, changed-file analysis, and reusable PR comments.
- Docker Compose, Render, Vercel, and GitHub Actions configuration.

The CodeBERT source-embedding extension (phase 6) is implemented as an offline research pipeline;
GRU and GNN extensions remain future work.

## Interface preview

| Landing page | Repository dashboard |
| --- | --- |
| ![BugRisk AI landing page](preview-redesign-landing.png) | ![BugRisk AI repository dashboard](preview-redesign-dashboard.png) |

![BugRisk AI mobile landing page](preview-redesign-mobile.png)

## Quick start: demo mode

Requirements: Python 3.11+, Node.js 22+, pnpm 11+, and Git.

```powershell
Copy-Item .env.example .env

Set-Location backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn app.main:app --reload
```

In another terminal:

```powershell
Set-Location frontend
corepack pnpm install
corepack pnpm dev
```

Open `http://localhost:3000/dashboard`. With `DEMO_MODE=true`, the API seeds one completed
repository analysis and does not require credentials. API documentation is available at
`http://localhost:8000/docs`.

## Train the ApacheJIT model

The official ApacheJIT v2 archive is downloaded from Zenodo record `5907847`. The archive is
82.8 MB and contains `apachejit_total.csv`, whose `buggy` field is the commit-level target.

```powershell
pip install -e ".\backend[ml]"
dvc repro
```

The stages download the dataset, normalize its feature columns, split each project in temporal
60/20/20 order, compare Logistic Regression, Random Forest, and XGBoost, calibrate the selected
model, select a recall-aware F2 operating threshold on validation data, and export
`ml/artifacts/bugrisk_model.joblib`. The threshold is used for evaluation reporting only; the
API continues to return calibrated probabilities and the fixed product risk bands. DVC records
dataset and pipeline hashes;
MLflow records the selected model and test metrics when installed.

## GitHub OAuth

Create a GitHub OAuth App with callback:

```text
http://localhost:8000/auth/github/callback
```

Set `GITHUB_CLIENT_ID`, `GITHUB_CLIENT_SECRET`, a random `SESSION_SECRET` of at least 32
characters, and a Fernet `TOKEN_ENCRYPTION_KEY`. Disable demo mode. Tokens are encrypted before
storage and are never sent to the frontend or placed in Git command arguments.

## GitHub Pull Request automation

Create a GitHub App with webhook URL `https://<backend>/webhooks/github`, enable the pull request
event, and grant read access to metadata and contents plus read/write access to pull requests.
Set `GITHUB_APP_ID`, `GITHUB_APP_PRIVATE_KEY`, and `GITHUB_WEBHOOK_SECRET`. Install the app on a
connected repository. The `opened` and `synchronize` events create queued analyses restricted to
the PR's changed Python files. BugRisk AI creates one report comment and updates it on later pushes.

## CodeBERT research extension (phase 6)

To evaluate whether source-code semantics improve defect prediction beyond the git/code metrics:

```powershell
pip install -e ".\backend[codebert]"
python -m ml.embeddings.extract --data ml/data/processed/apachejit.csv `
    --projects apache/zookeeper,apache/spark,apache/hadoop-mapreduce,apache/hadoop-hdfs,apache/kafka `
    --output ml/data/codebert/sources
python -m ml.embeddings.embed --sources ml/data/codebert/sources/sources.parquet `
    --output ml/data/codebert/embeddings
python -m ml.training.train_codebert --data ml/data/processed/apachejit.csv `
    --embeddings ml/data/codebert/embeddings/embeddings.npz --output ml/artifacts/codebert
```

The extract stage clones the public Apache GitHub mirrors (cache in
`~/.cache/bugrisk/repos`) and records each dataset commit's changed source files; the embed stage
produces CodeBERT commit vectors cached by content hash; the train stage compares a
metrics+embedding MLP against metrics-only XGBoost on identical rows with the same split,
calibration, and threshold protocol. Results and methodology live in
[docs/research/codebert-vs-metrics.md](docs/research/codebert-vs-metrics.md). The extension is
offline research tooling and does not change the live `/predict` path.

## Supabase

Create a Supabase project, open its SQL editor, and apply
all SQL files in `backend/supabase/migrations` in numeric order. Set `DATABASE_URL` to the pooled PostgreSQL
connection string used by the API and worker. The migration enables ownership RLS policies on
all application tables; the API also performs explicit ownership checks.

## Docker

After creating `.env`:

```powershell
docker compose up --build
```

Compose starts PostgreSQL, API, worker, and frontend. In Compose, `INLINE_WORKER=false` ensures
only the worker consumes queued analyses.

## Quality checks

```powershell
python -m ruff check backend ml scripts
python -m mypy --config-file backend/pyproject.toml --explicit-package-bases backend/app ml
python -m pytest backend/tests ml/tests --cov=backend/app

Set-Location frontend
corepack pnpm lint
corepack pnpm typecheck
corepack pnpm test
corepack pnpm build
```

See [architecture](docs/architecture.md), [deployment](docs/deployment.md), and
[security](SECURITY.md) for operational details.
