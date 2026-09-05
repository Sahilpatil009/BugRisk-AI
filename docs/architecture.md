# Architecture

```text
Browser -> Next.js dashboard -> FastAPI -> PostgreSQL / Supabase
GitHub PR -> signed webhook ----^             |
       ^                                      +-> pull request state
       +---- updated risk comment <--- GitHub App installation token
                                  |
                                  +-> analysis queue <- Python worker
                                                        |
                                   temporary Git clone + metrics
                                                        |
                                 calibrated model -> SHAP -> file priorities
```

## Risk contract

`change_risk_probability` is the calibrated supervised-model output for a commit. ApacheJIT's
`buggy` label is commit-level, so BugRisk AI does not relabel individual files as known buggy
examples. `file_priority_score` combines 55% change risk with 45% bounded file evidence. It is
used only for ranking review attention.

Risk bands are `LOW` at 0–30%, `MEDIUM` above 30–60%, `HIGH` above 60–80%, and `CRITICAL` above
80–100%.

## Analysis lifecycle

The API inserts `QUEUED` and immediately returns an analysis identifier. A worker claims queued
rows, transitions through `ANALYZING` and `PREDICTING`, then writes files and marks the analysis
`COMPLETED`. Exceptions are stored as sanitized `FAILED` messages. Demo mode can run the same
processor as a FastAPI background task for simple local use.

Repository source is cloned into an OS temporary directory and removed on success or failure.
Private credentials are passed through Git's process environment as an HTTP header, not embedded
in a clone URL.

For pull requests, the webhook handler verifies the raw-body HMAC, deduplicates the GitHub delivery,
fetches the changed-file list with a short-lived installation token, and queues the PR head SHA.
The analyzer fetches GitHub's pull ref and limits results to changed Python paths. A completed run
creates or updates one pull-request comment so synchronize events do not accumulate stale reports.

## Persistence and isolation

SQLite provides a credential-free demo. Production uses `DATABASE_URL` with PostgreSQL/Supabase.
Every API query includes the authenticated user owner condition. The SQL migration additionally
enables RLS based on the transaction-local `app.current_user_id`; a trusted backend service role
may bypass RLS for OAuth bootstrap and worker processing while API ownership checks remain
mandatory.

## Provider-neutral recommendation extension

The MVP recommendation engine is deterministic and based only on SHAP evidence and file metrics.
A future LLM adapter may rewrite those actions into friendlier prose, but it must accept and
return a schema that contains no writable score fields. Invalid or unavailable provider output
falls back to the deterministic actions.

## Phase 6 research extension (CodeBERT)

The deployed model consumes only the 11 git/code metrics. To test whether source-code semantics
add signal, `ml/embeddings` reconstructs the changed source files for historical ApacheJIT commits
from the public Apache GitHub mirrors (one bulk `git log --name-only` plus one bulk
`git cat-file --batch` per mirror) and encodes each changed file with CodeBERT. The commit vector
is the L2-normalized mean of its changed-file vectors, cached by content hash.
`ml/training/train_codebert.py` trains an MLP on metrics + embedding and a metrics-only XGBoost on
the exact same rows, using the identical temporal split, Platt calibration, and F2 threshold
protocol, then writes `codebert_comparison.json`. This stays out of the live `/predict` path until
the comparison justifies the extra inference cost. See
[docs/research/codebert-vs-metrics.md](research/codebert-vs-metrics.md).
