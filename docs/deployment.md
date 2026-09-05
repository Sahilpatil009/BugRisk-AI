# Deployment

## Frontend on Vercel

Set the project root to `frontend`, deploy with the included `vercel.json`, and configure
`NEXT_PUBLIC_API_URL` to the public HTTPS backend URL. Rebuild after changing this value because
public Next.js variables are embedded in the browser bundle.

## API with an inline worker on Render

The included `render.yaml` explicitly selects Render's `free` plan for one web service, with analysis jobs handled inside the
API process. Configure
`DATABASE_URL`, `FRONTEND_URL`, `BACKEND_URL`, GitHub OAuth credentials, GitHub App credentials,
`SESSION_SECRET`, and `TOKEN_ENCRYPTION_KEY`.

Set `DEMO_MODE=false` and `INLINE_WORKER=true`. Apply the Supabase SQL migration before serving
traffic. Use a durable object store or image-baked artifact for the trained model; do not depend
on an ephemeral deployment filesystem for training output.

The backend image downloads the `apachejit-xgboost-v2` GitHub Release artifact and verifies its
pinned SHA-256 digest during the image build. Publish that release asset before creating the
Render service. The API and inline worker load `/models/bugrisk_model.joblib` from the same
immutable image. A separate paid worker remains the recommended topology for higher throughput
and stronger process isolation.

## Supabase

Use the direct PostgreSQL connection for migrations and the pooled connection for runtime where
appropriate. Apply every SQL file in `backend/supabase/migrations` in numeric order, grant only the required
table privileges, and verify the RLS policies with two distinct test users before production.
The API remains the only client allowed to hold the database service credential.

## Smoke checks

- `GET /health` returns HTTP 200.
- GitHub callback creates a session without exposing a provider token.
- A small public Python repository reaches `COMPLETED` and its temporary clone is removed.
- A second user receives 404 for the first user's repository, analysis, and file identifiers.
- Frontend dashboard, file evidence, model card, and failed-analysis state render without console
  or network errors.
- A signed `pull_request.opened` webhook creates an analysis containing only changed Python files,
  posts a GitHub report, and a later `synchronize` delivery updates the same comment.
