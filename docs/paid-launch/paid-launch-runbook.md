# Paid-launch runbook

## Local sandbox

1. Copy `deploy/paid-launch/.env.example` to `deploy/paid-launch/.env` and replace every placeholder with test-only values. Do not add the file to Git.
2. Configure a real test OIDC tenant/callback and Stripe test-mode product/price/webhook. Keep `PAID_LIVE_BILLING_ENABLED=false` and `PAID_SALES_ENABLED=false`.
3. Start the isolated stack:

   ```powershell
   docker compose --env-file deploy/paid-launch/.env -f deploy/paid-launch/docker-compose.yml up --build
   ```

The migration container runs before the API and worker. The shell is available only on `127.0.0.1:8081`; Mailpit is on `127.0.0.1:8025`. The root research dependency stack is not installed into these images.

## Tests

The one-command paid-service test requires PostgreSQL:

```powershell
$env:PAID_TEST_DATABASE_URL='postgresql://subscriber:test-only@localhost:5432/subscriber_test'; $env:PYTHONPATH='.'; python -m pytest -q tests/paid_launch/case_policy_and_contracts.py tests/paid_launch/case_postgres_api.py tests/paid_launch/case_isolation_and_scope.py
```

Run the inherited compatibility suite without changing it:

```powershell
python -m pytest -q tests/test_refresh_lock_performance.py tests/test_lock_storage_performance.py tests/test_prediction_evidence.py
python scripts/benchmark_refresh_lock_storage.py --work-dir <isolated-dir> --source-commit <sha>
```

## Staging sequence

1. Provision an isolated PostgreSQL database with encrypted connections, least-privilege API/worker users, continuous backups, and point-in-time recovery.
2. Deploy the pinned API, worker, and shell images with a source revision label. Route the shell/API through TLS. Keep the inherited operator UI behind a separately configured OIDC/MFA gateway.
3. Apply only `services/subscriber/migrations` and verify `subscriber.schema_migration`.
4. Confirm anonymous direct requests to API, JSON, internal probe, old paths, origin, CDN, and service-worker caches disclose no premium payload.
5. Complete a Stripe test-mode checkout, duplicate/out-of-order webhook, cancellation, renewal failure, refund/dispute, timeout, and reconciliation journey. Test events must remain isolated from any live account.
6. Submit a signed synthetic qualified fixture that can exist only in the isolated staging namespace. Confirm pending revision invisibility, routed-hash verification, promotion, expiry, withdrawal, notification suppression, and no upstream lock/storage writes.
7. Exercise worker crash/restart, dead-letter recovery, database backup/restore, and application rollback. Verify restore does not repeat provider operations or revive revoked access.
8. Run load/accessibility/browser checks and the read-only launch verifier:

   ```powershell
   python scripts/verify_paid_launch.py --environment staging --json-output outputs/launch/launch-gate-report.json --markdown-output outputs/launch/launch-readiness.md
   ```

Exit `0` means every configured gate and evidence item passed; `2` means blocked/unverified; `1` means an integrity/execution error. The verifier cannot enable sales, markets, or billing.

## Release and rollback

- An operator submits a reviewed package explicitly with `scripts/submit_customer_release.py`. Do not attach this action to an existing lock button.
- Release retry operates only on the staged subscriber revision. It never runs the inherited lock/publication retry.
- Application rollback changes image/routing only. Do not roll customer data backward. Migrations in this release are additive.
- Pause alerts or suspend/withdraw a release before rollback when current content integrity is uncertain. Keep cancellation and account/help access available.
- After database restoration, reconcile Stripe before processing queued financial jobs. Treat uncertain email submissions as `UNKNOWN_OUTCOME`; do not blindly resend.

## Incident priorities

Immediately stop the affected function and retain evidence for premium data exposure, wrong-user entitlement, an authority failure that may have allowed publication, or unintended live payment behavior. Do not erase release/billing/audit history during containment.
