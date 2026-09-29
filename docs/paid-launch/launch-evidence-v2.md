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

The validator returns separate `structural_status`, `document_status`,
`execution_provenance_status`, `hosted_scenarios_status`, and `hosted_status`
fields. A valid document without an out-of-band resolver result reports
`EVIDENCE_STRUCTURE_VALID`, `EXECUTION_PROVENANCE_UNVERIFIED`, and
`HOSTED_SCENARIOS_UNVERIFIED`; it remains blocked as hosted proof. A
deliberately marked fixture may pass structural tests but is always blocked.
Local execution is likewise not a hosted environment.

Independent proof is supplied through the immutable
`IndependentVerification` adapter result. The resolver must retrieve or verify
the execution outside the submitted evidence document and bind the expected
kind, environment, source revision, provider/execution identity, artifact
hashes, scenario set, verification time, and attestation identity. The command
line verifier has no default resolver, so locally authored evidence remains
structurally checkable but hosted-unverified. Self-declared provider, reviewer,
or execution IDs cannot create an `IndependentVerification` automatically.

The verifier is read-only. It cannot deploy, write production data, enable
billing or sales, activate a market, replace calibration, or place a wager.
