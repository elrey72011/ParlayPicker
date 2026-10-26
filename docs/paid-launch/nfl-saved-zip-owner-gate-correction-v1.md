# Private saved-ZIP owner-gate cleanup correction

The owner authorized the exact prepared two-line correction, directly necessary
regressions/bindings and ONE additional conditional complete final-head CI cycle
for draft PR #2443. Authorization was first observed at 2026-10-26 00:11:35 UTC;
the message's author timestamp is not exposed. No approval time is backdated.

Preserve failed head `d46528beb86752c81846c81aa944f603ff985a64`, tree
`5cf5876d9af71f52aa02cb566dafcb0a6f3dc593`, and continue linearly on the existing
branch. Keep its prior policy, failure logs, original artifacts and spent cycle.
The proposed patch SHA-256 is
`bd8829057e72c4de09e73477a688a0d3e531eb6759d6b9f488dd7499a74dc9b5`.

The completed first cycle had 21 successful checks and two failed checks: shard 4
and its required aggregate. One existing test failed:
`test_source_evidence_intake.py::test_owner_token_guards_actual_upload_panel`.
The new cleanup accessed `st.session_state` on a lightweight Streamlit stub that
does not have it. All 9,029 identities completed (9,026 passed, one failed, two
inherited skips); all 43 new ZIP tests passed. Do not replace or hide this failure.

Both locked-panel paths now pass `getattr(st, 'session_state', {})` to the unchanged
cleanup helper. An absent state has nothing to clear. A real state still removes
every `nfl_saved_` key, including uploads, selections and results, while preserving
unrelated keys. Cleanup errors propagate. Token comparison and unauthorized early
returns are unchanged. No old test, stub, assertion, timeout, reader, model,
archive parser, trust catalog, database behavior or prospective rule changes.

Validate the complete source-evidence-intake file together with both complete
remote-canonical files and affected saved-viewer/ZIP tests. Include the original
ordered owner-panel context, absent/empty/populated state, missing/short/wrong/
correct token, actual post-load revocation and genuine cleanup-error propagation.
Invalid authorization must parse no archive and access no database/provider/
predictor. Recheck wrong-token and missing/short-config revocation through actual
private browser controls, retaining original files and clocks, with tripwires.

Record the exact linear successor in the existing source-binding/policy controls.
All inherited files outside the exact correction set, including the first ZIP
policy and original model/replay readers, remain immutable. Finish inventory,
capacity and local correctness gates before publishing one final head. Use only
the newly authorized complete CI cycle; no retries or additional diagnostic jobs.

Authentic deployed-hosted upload verification remains pending owner merge,
release clearance, deployment and upload. No merge/deployment, prediction,
provider acquisition, database change, training or operational action is allowed.
Preserve both authentic results, Monday's NFL preparation, Delaware–WKU and Chrome
file-access permissions. Authentic bodies remain outside Git and hosted artifacts.
