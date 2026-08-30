# Deployment

## Frontend on Vercel

Set the project root to `frontend`, deploy with the included `vercel.json`, and configure
`NEXT_PUBLIC_API_URL` to the public HTTPS backend URL. Rebuild after changing this value because
public Next.js variables are embedded in the browser bundle.

## API and worker on Render or Railway

The included `render.yaml` defines a web service and worker from the same backend image. Configure
`DATABASE_URL`, `FRONTEND_URL`, `BACKEND_URL`, GitHub OAuth credentials, `SESSION_SECRET`, and
`TOKEN_ENCRYPTION_KEY`. Both services must use the same secrets and model artifact version.

Set `DEMO_MODE=false` and `INLINE_WORKER=false`. Apply the Supabase SQL migration before serving
traffic. Use a durable object store or image-baked artifact for the trained model; do not depend
on an ephemeral deployment filesystem for training output.

The backend image downloads the `apachejit-xgboost-v2` GitHub Release artifact and verifies its
pinned SHA-256 digest during the image build. Publish that release asset before creating the
Render services. Both the API and worker then load `/models/bugrisk_model.joblib` from the same
immutable image.

## Supabase

Use the direct PostgreSQL connection for migrations and the pooled connection for runtime where
appropriate. Apply `backend/supabase/migrations/001_initial_schema.sql`, grant only the required
table privileges, and verify the RLS policies with two distinct test users before production.
The API and worker remain the only clients allowed to hold the database service credential.

## Smoke checks

- `GET /health` returns HTTP 200.
- GitHub callback creates a session without exposing a provider token.
- A small public Python repository reaches `COMPLETED` and its temporary clone is removed.
- A second user receives 404 for the first user's repository, analysis, and file identifiers.
- Frontend dashboard, file evidence, model card, and failed-analysis state render without console
  or network errors.
