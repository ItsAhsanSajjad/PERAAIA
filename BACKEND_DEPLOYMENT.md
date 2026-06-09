# Ask PERA — Backend Deployment Notes

Operational guide for deploying the FastAPI backend of Ask PERA, a
public government-facing RAG assistant. Pair this with `.env.example`.

> Never commit real secrets. `.env`, `.env.bak*`, `env.txt*`, and
> `.admin_secret` are git-ignored — keep it that way.

## 1. Required environment variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI access for embeddings + answer generation. |
| `ADMIN_EMAIL` | Yes (admin) | Admin dashboard login email. |
| `ADMIN_PASSWORD_HASH` | Yes (admin) | PBKDF2-HMAC-SHA256 hash of admin password (never plaintext). |
| `APP_ENV` | Recommended | `development` (default) or `production`. Controls CORS/security strictness. |
| `CORS_ALLOW_ORIGINS` | Prod: Yes | Comma-separated browser origin allow-list. |
| `CORS_ALLOW_CREDENTIALS` | Optional | `1` to send cookies/Authorization on CORS. Incompatible with `*`. |
| `OPENAI_TIMEOUT_SECONDS` | Optional | Per-request OpenAI timeout (default `30`). |
| `OPENAI_MAX_RETRIES` | Optional | SDK retries on transient failure (default `1`). |
| `OPENAI_ANSWER_MAX_TOKENS` | Optional | Max output tokens for the answer (default `1200`). |
| `OPENAI_REFINE_MAX_TOKENS` | Optional | Max output tokens for refinement (default `600`). |
| `LOG_RAW_QUERIES` | Optional | `1` logs raw citizen questions (debug only; default `0`). |
| `LOG_CLIENT_IP` | Optional | `1` logs full client IP (default `0` → salted hash). |
| `AUDIT_HASH_SALT` | Prod: Recommended | Long random salt for query/IP hashes. |
| `ADMIN_JWT_SECRET` | Optional | Fixed admin session-signing secret (else auto-persisted to `.admin_secret`). |

## 2. Generating the admin password hash

```
python scripts/generate_admin_password_hash.py
```

Enter the password when prompted (input is hidden, never logged). Paste
the printed `pbkdf2_sha256$...` value into `ADMIN_PASSWORD_HASH`. The
plaintext is never stored.

> **Rotate the old hardcoded credential.** Any password that was ever
> committed to git history must be treated as compromised — generate a
> fresh hash and set a new password before going live.

## 3. CORS guidance

- **Local / development**: leave `CORS_ALLOW_ORIGINS` empty. Built-in
  `localhost`/`127.0.0.1` origins (ports 3000/3001/3002/5173) are merged
  in automatically. Wildcard `*` works only when credentials are off.
- **Staging**: set `APP_ENV=production` and list the staging origin
  explicitly, e.g. `CORS_ALLOW_ORIGINS=https://staging.ask.pera.gop.pk`.
- **Production**: `APP_ENV=production` and the exact public origin only,
  e.g. `CORS_ALLOW_ORIGINS=https://ask.pera.gop.pk`. Wildcard `*` is
  ignored in production (fails closed); localhost defaults are NOT added.

## 4. Workers / process model (IMPORTANT)

This backend keeps **in-process state**: an in-memory rate-limiter, the
admin login rate-limiter, the audit hash-chain tail, and a `threading.Lock`
guarding blue/green index swaps. These are **not shared across processes.**

- **Run a single worker** (e.g. `uvicorn fastapi_app:app --workers 1`).
- Do **not** scale with `--workers >1` or multiple Gunicorn workers
  until shared/distributed locking + shared rate-limit state are added.
  Multiple workers would each hold their own lock and index pointer,
  risking inconsistent rate limiting and concurrent index swaps.
- Vertical scaling (bigger instance) is the supported scaling path today.

Example single-worker launch:

```
uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --workers 1
```

## 5. Health / readiness

- Use the existing root/health route exposed by the app for liveness.
- Readiness should additionally confirm the FAISS active index pointer
  (`assets/indexes/ACTIVE.json`) resolves and `OPENAI_API_KEY` is set.
- The audit integrity verifier is an ops tool, not a request-path check:
  `python -c "from audit_trail import verify_audit_file; print(verify_audit_file('audit_logs/audit_<date>.jsonl'))"`.

## 6. Privacy / logging defaults

- By default raw questions and full IPs are **not** persisted to the
  audit log — only length + a short salted hash. Set `AUDIT_HASH_SALT`
  to a strong random value in production.
- Keep `LOG_RAW_QUERIES=0` and `LOG_CLIENT_IP=0` in production.
- `audit_logs/` is git-ignored and server-side only.

## 7. Dependencies / Python version

- Python **3.11** (the project venv is `venv311`).
- Runtime deps live in `requirements.txt` (unpinned for flexibility).
- A pinned snapshot of the known-good environment is captured in
  `requirements.lock.txt` (generated via `pip freeze`); install from it
  for reproducible deploys. Regenerate only when intentionally upgrading.
