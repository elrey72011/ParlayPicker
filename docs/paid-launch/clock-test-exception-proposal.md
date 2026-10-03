# Proposed exact exception for PR #2373

This is an unapproved policy proposal, separate from the provider and quote-age fix. The current manifest says: "No exception is encoded. Any protected-file or existing-test change requires separately approved, exact-scope integration work." The original guard, baseline manifest, paid-launch workflow and subscriber assertion remain unchanged and failing for #2373.

The owner explicitly requested that future timestamps remain disqualifying while being counted separately from actual expiry. Restoring the old `expired == 1` expectation would misrepresent that behavior. This proposal recognizes only the complete before/after Git blob pair of `tests/test_board_diagnostics.py` at the exact #2373 head. It covers the corrected expectation and its added regression cases. It does not grant permission for subsequent edits to that test or any other existing test.

Proposed binding: base `d8f580734c28b712f93e0e4a647e9b21ab1f2928`, head `dc211cc9438390c73848ce1a43d512ada00338d8`, before blob `cfa07b6b6c622f083cb7d2d7e3a0713780758013`, after blob `e610143aff5611235f9cfb44da13e2d54e1c6b48`. A CI merge is recognized only with those exact parents in base/head order and a tree equal to the authorized head. Existing baseline/tooling SHA-256 values are verified unchanged against Git blobs.

`tools/review_clock_scope_exception.py --repo <checkout>` produces `MATCHED_FOR_REVIEW` or `REJECTED`, retaining the complete original guard result. It always exits 2, has no gate authority, and is not wired into CI. There is no command-line policy override. The original guard still reports FAIL. This proposal cannot make either PR mergeable or authorize scientific qualification.

The assessor separately diagnoses LF-to-CRLF checkout conversion only when committed bytes match the original expected digest and checkout bytes differ solely by that conversion. It never resets hashes or removes the original raw-byte failure. Substantive changes, dirty tracked work, additional existing-test changes, protected changes and dependency shadows reject the proposal.

## Review and integration sequence

1. Review this exact blob-bound policy and its negative tests independently from #2373.
2. If accepted, approve a separately specified integration that teaches the original guard to report this single correction as an approved exception while retaining all other rejection checks. Preserve the original baseline and record the approval's reference, exact candidate identities and exception scope. Review the implementation and resulting tooling digest change as guard-policy work; do not reset the baseline or silently substitute current checkout hashes.
3. Require both protected-scope and the unchanged subscriber scope assertion to pass through that approved integration, including unrelated-test, stale-expectation, protected-file, baseline-tampering and CI-identity negative regressions. Any successor head or policy integration commit needs its own reviewed identity binding. Do not activate a blanket file/path allowlist.

This proposal implements the review assessor and coverage, not step 2. Offline verification: `python -B -m pytest -q tests/test_clock_scope_exception_proposal.py`. The tests use real temporary Git repositories and the verified original guard. No network, fitting, recovery, activation or financial action is involved.
