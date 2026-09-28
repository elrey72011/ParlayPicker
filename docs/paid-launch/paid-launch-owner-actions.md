# Paid-launch owner actions and blockers

The implementation intentionally does not invent commercial facts or turn on sales. The completion label for this delivery is `IMPLEMENTED_LOCAL_VERIFIED`; paid launch remains blocked.

## Decisions and evidence required from the owner

1. Set the selling entity, monthly price, currency, exact product name, terms/renewal/cancellation/refund/privacy versions, support contact, eligible customer jurisdictions, age policy, and responsible-use resources.
2. Identify at least one exact sport/market/model/calibration/validation/policy scope with current independent `QUALIFIED` authority. Research rankings, positive EV, saved locks, legacy publish success, or an owner-written allowlist do not qualify a market.
3. Obtain and record counsel-reviewed jurisdiction/recurring-billing/claims decisions, provider-specific content/data rights, and the actual processor support decision for the accurately described business.
4. Configure a supported OIDC tenant. Require MFA for owner/operator accounts and verify issuer, callback, group, recovery, disabled-user, logout, and session revocation behavior in staging.
5. Configure Stripe test mode only and complete the recorded sandbox journey. Do not supply a live key or set `PAID_LIVE_BILLING_ENABLED=true` in this PR.
6. Select/configure an email provider and validate bounce, complaint, suppression, uncertain outcome, and dead-letter recovery.
7. Provision staging PostgreSQL, TLS routing, secret management, monitoring, alerts, backups/PITR, and an OIDC/MFA front door for the entire inherited operator UI.
8. Approve and review a separate, narrow legacy-publication containment change if current premium recommendations can still appear in embedded page data, public JSON/downloads, `previous.html`, Netlify/cPanel output, previews, artifacts, caches, or direct origins. This PR deliberately leaves those inherited publishers untouched.
9. Run unauthorized-origin/CDN/service-worker tests, source/runtime/hosted revision verification, PostgreSQL integration, load/LCP, alert-latency, restore, rollback, and hosted-content checks in staging.
10. Complete a non-billed pilot covering 14 consecutive observed days and at least two active slates. Elapsed days and operational incidents cannot be simulated or backfilled as PASS.
11. Review pilot evidence and make an explicit owner go-live decision. Merging code is not that decision and must not enable live billing.

## Exact current blockers

- `NO_COMMERCIALLY_ENABLED_QUALIFIED_MARKET`
- `COMMERCIAL_TERMS_NOT_CONFIGURED`
- `MISSING_APPROVAL_PRODUCT_CLAIMS`
- `MISSING_APPROVAL_DATA_RIGHTS`
- `MISSING_APPROVAL_PROCESSOR_SUPPORT`
- `MISSING_APPROVAL_LEGAL_JURISDICTION`
- `MISSING_APPROVAL_TERMS_PRIVACY_RENEWALS`
- `MISSING_APPROVAL_RESPONSIBLE_USE`
- `OIDC_MFA_STAGING_NOT_VERIFIED`
- `STRIPE_SANDBOX_JOURNEY_NOT_VERIFIED`
- `EMAIL_DELIVERY_AND_RECOVERY_NOT_VERIFIED`
- `OPERATOR_FRONT_DOOR_NOT_VERIFIED`
- `LEGACY_PUBLICATION_LEAK_PATH`
- `STAGING_ROUTE_AND_UNAUTHORIZED_ACCESS_NOT_VERIFIED`
- `SUBSCRIBER_LOAD_AND_MOBILE_LCP_NOT_VERIFIED`
- `BACKUP_RESTORE_AND_ROLLBACK_NOT_VERIFIED`
- `REMOTE_BENCHMARK_NOT_RUN_EXTERNAL_BLOCKER`
- `HOSTED_CONTENT_VERIFICATION_NOT_RUN`
- `PILOT_14_DAYS_AND_TWO_SLATES_NOT_COMPLETE`
- `OWNER_GO_LIVE_REVIEW_NOT_GRANTED`
- `LIVE_BILLING_DISABLED`
