# Paid-launch architecture

## Boundary

The paid service is additive. It never imports the Streamlit application, `app_core`, research storage, odds fetchers, model inference, or lock code. The private research plane remains the authority that creates a reviewed package; the subscriber service may only reject, stage, verify, promote, suspend, or withdraw that immutable package.

```text
existing operator/research plane
        | explicit signed reviewed package
        v
subscriber API + PostgreSQL ----> durable worker ----> Stripe test / SMTP
        |
        | session + entitlement + current authority
        v
subscriber shell (contains no embedded picks)
```

Subscriber requests cannot trigger analysis, Drive discovery/restore, provider odds fetches, lock creation, public publication, or model registration. The enforcement is structural: `services/subscriber` has no import path to those packages, and CI parses its imports to preserve that property.

## Trust boundaries

- OIDC uses Authorization Code with PKCE. State, nonce, and verifier are held server-side. Identity is issuer plus subject; owner/operator role requires a server-recognized group and an MFA authentication-method claim.
- Sessions are opaque, hashed in PostgreSQL, rotated per login, revocable, time-bound, and carried in Secure/HttpOnly cookies outside local test mode. State changes require a separate CSRF value whose hash is stored with the session.
- Billing uses hosted Stripe Checkout. Client input cannot select price, customer, amount, role, or arbitrary return URL. Browser redirects never create entitlements. Raw-body signed webhooks are durably deduplicated by provider/account/environment/event and projected asynchronously from reconciled provider state.
- Release submissions are HMAC-authenticated over the exact versioned package. The review hash binds the recommendation, model/calibration/validation/activation authority, price, line, timestamps, and content-rights references.
- A staged revision is hidden. The worker retrieves it through a privileged service probe, compares the routed content hash, rechecks current authority and expiry, then transactionally advances the active pointer. Failed retries do not call any lock/publication path.
- Private responses are `private, no-store`; the static shell contains no current recommendation data. The application applies authorization to each object lookup.

## Data ownership

The `subscriber` PostgreSQL schema is customer-only. Its migration does not touch research evidence, public-history data, performance caches, or lock schemas. Immutable release and settlement references preserve their upstream hashes. Corrections append versions; they do not rewrite original customer releases.

Core records include customer, session, product version, subscription, billing event, entitlement, approval, release revision/event/pointer, result projection, notification outbox, job queue, consent, and admin audit.

## Launch gates

The single `launch_gate` policy separates engineering, commercial, market, sales, publication, and recommendation state. Every unknown defaults to blocked. Checkout requires staging verification, complete owner-configured terms, current commercial approvals, an independently qualified and enabled exact market, and explicit owner sales enablement. Live Stripe keys are rejected while live billing is disabled.

Release access additionally rechecks entitlement, active pointer, recommendation expiry, and the current upstream authority window. `NO_CURRENT_VERIFIED_RELEASE`, data delay, and authority failure are not reported as a normal no-pick day.

## Existing public delivery containment

Repository inspection found current public delivery paths for embedded page data, `board-data.json`, downloaded `public-board.json`, `previous.html`, Netlify deployments, cPanel/SFTP deployment, and direct public site URLs. This PR does not edit or disable those inherited paths. Therefore `LEGACY_PUBLICATION_LEAK_PATH` remains a paid-launch blocker until the owner approves and staging proves a narrowly scoped containment/routing change. Paid payloads in this service never use those paths.

## Affected paths

| Path | Classification | Protection |
|---|---|---|
| `services/subscriber/**` | New feature | Isolated dependency file, import-boundary test, PostgreSQL integration job |
| `integrations/subscriber_release/**` | New read-only dependency adapter | Exact signature/review/authority contract tests |
| `web/subscriber/**` | New feature | No embedded premium payload; direct-access tests |
| `deploy/paid-launch/**` | Deployment/configuration | Sandbox defaults, live billing false, private internal route |
| `scripts/check_launch_change_scope.py` | New verification tooling | Baseline/tooling hashes and base-diff guard |
| `scripts/submit_customer_release.py` | New explicit operator action | Import-isolation test; no lock/publication imports |
| `scripts/verify_paid_launch.py` | New read-only verifier | Exit-code contract; cannot mutate gates |
| `tests/paid_launch/**` | New tests | Dedicated PostgreSQL CI job |
| `docs/paid-launch/**` | New contracts, runbooks, evidence | Honest PASS/BLOCKED/NOT_RUN statuses |

No existing source file, protected file, existing test expectation, research dependency file, or root requirement is modified.
