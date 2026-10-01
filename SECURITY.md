# Security policy

- Never commit `.env`, OAuth credentials, database URLs, repository source, raw datasets, or model
  artifacts.
- GitHub OAuth uses a cryptographically random state value stored in the signed session cookie.
- GitHub webhooks require the exact `X-Hub-Signature-256` HMAC before payload processing.
- GitHub App installation tokens are short-lived and generated from an encrypted deployment secret.
- Access tokens are encrypted at rest and used only by the API/worker.
- Production must use HTTPS, a new session secret, a dedicated Fernet key, exact CORS origins, and
  `DEMO_MODE=false`.
- Repository and analysis reads are owner-scoped in the API and reinforced by PostgreSQL RLS.
- Private repositories are cloned into temporary directories and removed automatically.
- Treat analyzed source as untrusted input. The analyzer reads files but never imports or executes
  repository code.

Report vulnerabilities privately to the project maintainers; do not open a public issue containing
credentials, private source, or exploitable details.
