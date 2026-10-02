# Stale research carrier follow-up

Base: `39351a7580c5e38e49e122495a3e89137bb8ce76` (includes #2370 and #2371).
Review: #2371 comment `4161837391`, posted after that source merge.

This correction reconciles a retained display-only source carrier against
explicit current declarations before canonical projection and immutable capture.
An unsupported current probability contract, contradictory/invalid push, or
failed/unavailable inference cannot be hidden by the old carrier. Original
negative facts remain sticky; matching and malformed carrier bytes are retained.
Repeated normalization cannot turn a recorded rejection into success.

The source guard also checks current semantics during export. Existing
conditional-to-unconditional price conversion remains unchanged. A zero-push
market may be expressed conditionally by capture without changing probability
mass; a nonzero-push reversal cannot reinterpret an unconditional source.
No probability or EV calculations, target matching, protected capture, gate,
calibration/validation requirement, exposure rule or stake policy is changed.

The complete retained-upload control exposed two separately authorized retention
gaps: the normalizer had no `ml target` -> `ml_target` alias, and canonical
projection dropped ordinary probability semantics, push and model status despite
keeping their display carrier. Preserve only these explicitly supplied fields;
missing targets/status/semantics remain missing. Do not infer model availability
or a successful run. The upload still uses the existing probability calculations.

Ordinary metadata retention is an actual producer-input correction, so its
authority effects require a paired replay rather than a blanket display-only
claim. The synthetic frozen replay proves identical complete final contracts,
stage decisions, stakes, normalized probability and recorded EV. Strict
conservative probability/EV remain null; PASS stake remains zero.

An unresolved final line must not lend its PASS contract to another exact
selection. When a new research export's separate provenance-bound display
object establishes research provenance or an explicit source rejection and no
matching wager/trial contract, public probability/EV remain null in legacy fields.
The research object alone carries supported estimates. This also
prevents history's legacy estimate reader from recording unsupported research.
Existing approved/trial contracts and legacy inputs with no recorded target,
provenance or estimate retain their saved-value behavior; their separate research
object reports missing evidence. Public authority values are identical in every
view in the paired full-upload replay.

Two unchanged application expectations caught an overbroad legacy clearing
guard in the first pushed revision: genuinely absent half-point source semantics
remain compatible, and total-quality diagnostics must not erase a recorded
legacy estimate. Both tests pass on actual main `39351a7`, fail on `096413b`,
and pass after this correction without changing their expectations. Inspect the
validated saved display reason before public identity checks replace a missing
target/provenance reason; explicit unsupported semantics, invalid push and
failed/unavailable inference never qualify for that legacy exception. Missing
target metadata cannot mask an original unsupported source contract.

The existing producer writes `Market Score Model` into `model_status` to name
its model type. Treat that exact existing label as neutral type metadata only
in that field. It must not suppress separately recorded successful inference,
and must not manufacture success when inference is absent. Failed, conflicting
and unknown statuses remain rejected; original failures remain sticky. Real-path
fake-transport controls reproduce the regression on `096413b` and its absence
on current main, including complete saved contracts and zero-stake PASS outcomes.

Fresh review of `15b822f` also identified that the literal recorded status
`unknown` must reject research availability. Only genuinely absent status and
the existing model-type label are neutral. Explicit `unknown` now rejects both
with and without another recorded success, and remains sticky through repeated
normalization. Existing authority fields and stakes are unaffected.

Fresh review of `b02fa473` identified a missing-target exception that could hide
an unconverted conditional relabel. A separately reproduced non-probability-first
export also dropped invalid source push facts after the same early missing-target
return. Check source probability/push mass against the exported mass before that
return, using the same existing conversion and price math. The legacy exception
checks nonzero-push conversion against a separately retained raw source value;
a declaration carrier alone cannot prove a missing conversion. Actual `.575`
conditional mass with `.1` push still exports `.5175` and EV `.135`; merely
relabeling `.575` and impossible half-point push mass reject. Missing source
proof stays unavailable. No probability calculation or authority rule changes.
The same missing-target bypass also let Boolean source probabilities become
numeric zero/one before history read them. Reject the retained Boolean source
before that return; genuine numeric zero and its negative EV remain unchanged.
The complete pre-target probability checks now also preserve the existing
missing/nonfinite/out-of-range reasons instead of hiding them behind missing
metadata. A missing probability cannot retain an orphaned legacy EV. Boolean
recorded EV is signaled before that return, while genuine numeric zero EV stays
unchanged.

Fresh review of `9da3aade` identified the equivalent unconditional-source
legacy bypass. Every explicitly retained source contract now needs matching
source/export probability mass and compatible recorded EV, including when
target metadata is missing. A separate malformed-carrier finding showed that
Python accepts nonfinite JSON VALUE tokens before strict serialization rejects
them. Reject those carriers structurally, keeping their original bytes unchanged
even when a current conflicting status would otherwise reserialize them.

Fresh review of `a6b90359` identified a copied saved display identity that could
still preserve unrelated legacy values. The legacy exception now requires the
recorded saved identity facts to equal the export identity. Existing legacy
adapters may add a start time that was absent in the producer identity; that
one-way missing-time enrichment preserves historical metrics only. The separate
research object still rejects the changed identity, and any recorded identity
conflict rejects the legacy exception. The unchanged quality regression and a
paired different-event control verify this narrow compatibility. The pre-target self-review
also reproduced conflicting quote/period aliases and a selection/line conflict
hidden by missing metadata; the existing identity checks now run before that
return. Regression controls cover every saved identity field and retain the
matching legacy case. These checks do not create missing target evidence.

Parent review independently reproduced an EV-order gap through the actual
non-probability-first source/export path, without reinjecting original fields:
explicit unconditional probability `.6` with zero push and recorded EV `.8`
could survive the early missing-target return after source facts were dropped.
Check known source EV price basis before that return and retain its rejecting
value reason through projection. Ordinary export Boolean/nonfinite EV types
also reject without requiring any source carrier. Explicit export semantics
and push declarations require EV to match the existing price-value calculation;
an absent old basis remains unknown rather than being invented. Numeric zero,
compatible negative EV and genuinely absent EV remain covered. No recorded
producer evidence or probability math is overwritten.

Fresh review of `6e43c63b` identified explicit malformed producer EV values
projected to null and mistaken for absent evidence. Before projection, every
non-absent raw EV that the existing numeric parser rejects now records
`INVALID_RECORDED_EV`. Actual source/export controls cover both production and
calibrated probability paths, Boolean/nonfinite/nonnumeric EV, genuine absent
sentinels, zero, compatible positive and negative EV. No source reinjection,
new price math, authority or model success is introduced.

Fresh review of `a34eeb32` located an earlier loss boundary: evidence schema
`project()` replaces NaN with null before display validation. Its canonical
`expected_value` normalization also replaces infinity; the separate
`production_expected_value` alias retains infinity at this boundary. Preserve
only an allowlisted invalid numeric reason before that normalization, through
the existing display provenance carrier. Version 2 adds negative type facts for
the four supported probability sources and two EV sources; valid version 1
carriers remain unchanged. No raw invalid payload, replacement estimate, new
authority field, capture behavior or probability calculation is introduced.
The project hook is two lines; evidence schema fields and payload hashing remain
unchanged. Existing protected immutable capture and transport files are intact.

New frozen regressions start with the original object-typed producer rows,
then use authority projection, schema projection, immutable capture, Overall,
Sides and Totals exports, package validation, serialized assets and the actual
browser. The same tests on exact A34 source fail 14 cases and pass 39 controls;
the correction passes all 53. This supersedes the earlier direct-export test's
coverage claim: it did not exercise the pre-display NaN-erasure boundary.

Three unrelated application tests reproduced on unchanged main after their fixed
September 29 evidence crossed the existing three-day load/pilot expiry. The
separately approved test-only correction binds `_configure_verifier` to that
fixture's `NOW` through the existing `validate_evidence(now=...)` parameter.
Both verifier test modules pass; production freshness policy and every existing
assertion remain unchanged. No actual evidence is restamped or revalidated.

The upload fixture supplies its fake exact book before normalization through the
existing supported `quote_bookmaker` field; no metadata or fixture is inserted
after the export boundary. Other pre-existing upload omissions (model source,
market push, line provenance and quote-source aliases) are recorded privately
and left untouched. They cannot create authority or be silently defaulted.

Regressions exercise real upload, producer selection, terminal PASS gate,
immutable capture, all three export views, package builder, validator,
serialization, browser, history, Top 10 and parlay consumers using fake transport
and a frozen clock. Matching, stale, malformed, missing, boolean/nonfinite push
and invalid/failed status cases cannot manufacture a wager. Valid conditional
normalization, approved/trial fixtures, legacy packages, and zero/negative value
remain covered by existing actual-path tests.

An early normalization control merely relabeled conditional .575 with .1 push
as unconditional without changing the value. Both baseline and correction
correctly reject that inconsistent input. Its expectation was corrected into an
explicit rejection control; valid conditional export conversion remains .5175
with EV .135. The original failed control and investigation are retained in the
private evidence bundle; no price math or production expectation was weakened.

Historical October 1 first-loss causation remains UNVERIFIED: exact producer
exports and candidate/model/gate/allocation artifacts are unavailable. Synthetic
reproductions and the separately identified current public package are not
incident proof. No provider acquisition, paid model call, model training,
calibration fit, deployment, live publication, release or wager is performed.

After a separately authorized owner publication, retrieve only the existing HTML
and its referenced board/version assets, verify their common build/hash and
source identity, and compare exact selected-row/quote identities. Confirm research
labels and unavailable reasons, conservative contract values, saved decisions,
stakes, and current-wager filters. This PR does not refresh or publish that board.
