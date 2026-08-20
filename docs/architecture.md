# Architecture

```text
Browser -> Next.js dashboard -> FastAPI -> PostgreSQL / Supabase
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

