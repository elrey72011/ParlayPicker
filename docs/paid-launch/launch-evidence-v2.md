# Paid-launch evidence contract v2

`scripts/verify_paid_launch.py` accepts hosted evidence only after
`scripts/paid_launch_evidence.py` validates the version-2 contract. A top-level
`PASS` label is not evidence.

Every report binds its evidence kind and environment to one release candidate's
source, deployed, and served 40-character revisions. It identifies a supported
hosted execution kind, a trusted execution reference, completion interval,
non-fixture/non-synthetic status, an owner review reference, all required
executed scenarios, and content-addressed supporting artifacts. The canonical
payload digest excludes only its own `payload_sha256` field.

Freshness is set per evidence class in `EVIDENCE_POLICY`: three days for load
and pilot evidence, seven days for customer journeys and publication behavior,
and fourteen days for isolated restore evidence. Pilot intervals must have
actually ended, cover at least fourteen reported consecutive days, and include
at least two observed active slates.

The trusted generation routes are GitHub Actions, the owner-authorized staging
runner, and the deployment platform. Review references must come from GitHub,
an approved deployment change, or an owner approval record. These identities
and digests make replacement and mismatches detectable; they do not prove that
an arbitrary author told the truth. Independent hosted review must verify the
upstream run and artifact identities before owner sign-off.

The validator returns separate `structural_status` and `hosted_status` fields.
A deliberately marked fixture may pass structural tests but is always blocked
as hosted proof. Local execution is likewise not a hosted environment.

The verifier is read-only. It cannot deploy, write production data, enable
billing or sales, activate a market, replace calibration, or place a wager.
