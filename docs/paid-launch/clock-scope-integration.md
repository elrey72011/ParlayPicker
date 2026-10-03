# Exact clock-test policy integration, version 2

The owner approved the policy in draft #2374 at `051baf59c07060576fbf7a169f8ef7984e7a7672` for preparation of this reviewable draft integration. This is not approval to merge, deploy, activate a market or wager. The combined draft includes #2373's exact shipping head `dc211cc9438390c73848ce1a43d512ada00338d8` and the unchanged reviewed proposal files.

The original `launch-baseline-manifest.json` remains byte-identical and authoritative for its protected files and recorded ancestry. Version 2 is a new policy file, not a replacement baseline. It records the original manifest SHA-256, original tooling evidence, successor tooling hashes, exact implementation commit/tree and before/after integration blobs. The existing paid-launch workflow and subscriber assertions are unchanged.

Only `tests/test_board_diagnostics.py`'s complete reviewed blob correction is an approved existing-test exception: before `cfa07b6b6c622f083cb7d2d7e3a0713780758013`, after `e610143aff5611235f9cfb44da13e2d54e1c6b48`. The current test must match the latter exactly. The report lists this in `approved_exceptions`; `existing_test_changes` contains only unauthorized edits. The original guard result remains available as `original_guard_report`.

## Identity without a circular commit hash

1. Commit A contains the integration implementation, tests/documentation and exact reviewed proposal files, with the exact shipping head as its single parent. Its allowed change set is fixed in code and every file is bound by before/after Git blobs.
2. Commit B has A as its single parent and adds only `launch-scope-policy-v2.json`. That file identifies A's already known commit/tree and tooling hashes; it does not contain B's own hash. This seals the content that reviewers inspect.
3. The candidate is B, or a CI merge with exact audited-main/B parents and a tree identical to B. Any later commit, altered tree, extra seal change, different clock blob, changed baseline or unauthorized integration path fails. A commit with identical sealed content and parents but different harmless commit metadata has the same policy identity; its actual Git SHA is still reported by the guard.

The policy is a review artifact within the draft, not a cryptographic signature or independent permission service. Repository review and branch protection remain the external authority for accepting the declared integration/tooling bindings. Copying or generating a policy file is not human approval. Future changes require an explicitly reviewed successor policy; this version does not accept additive successors automatically.

The original guard's `run` checks remain unchanged. Version 2 validates the complete seal first, then removes only the proven clock-test rejection and explicitly bound guard-tool change from the unauthorized findings. Other original reasons remain failures. LF/CRLF conversion is accepted only when committed SHA-256 and checkout bytes (after that conversion alone) match the explicit tooling binding; the original baseline hashes are never reset.

Offline tests use real Git repositories and exercise the original guard: exact local/CI identity, unchanged subscriber contract, line endings, unrelated dirty/staged/committed tests, protected code, calibration changes, altered original manifest, unauthorized workflow/guard tooling, policy tampering, runtime shadows, wrong merge parents/tree, extra commits and alternate comparison-base/manifest attempts. No existing assertion is skipped or weakened.

No model, calibration, validation threshold/calendar, authority or financial behavior changes. A technical scope PASS does not qualify a market; application output must remain PASS/zero stake when no qualified positive-value selection exists.
