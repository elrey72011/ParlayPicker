# NCAAF coverage correction follow-up

This separate draft corrects the two late reviews on merged #2377:
[revision facts](https://github.com/elrey72011/ParlayPicker/pull/2377#discussion_r4174610799)
and [unpriced receipts](https://github.com/elrey72011/ParlayPicker/pull/2377#discussion_r4174610802).
It starts at main `ab30a6aa74a9fddd23aa207cd0a49baf15e1793f`, tree
`c7c41cd049702a9ef53975577d844b0e0999d8d4`. Completed DFS draft #2378 stays at
`546ed5a42a92970f15c1279a92ac9eab5b33fe3f` and remains unmerged.

A later usable event revision fills missing canonical team names, IDs, aliases,
kickoff and status. Known contradictory facts are never overwritten; a different
known school ID cannot add aliases to the original school. Division, competition
and kickoff revisions remain retained. A recovered or incomplete revision keeps
the canonical conflict marker and PARTIAL inventory. It may improve display and
unresolved reconciliation but cannot silently become MATCHED or qualified.
Original source records remain untouched; the six-request budget is unchanged.

Quote coverage requires a finite American price with absolute value at least 100,
consistent with the existing price contract. Boolean, blank, malformed, nonfinite,
overflow, zero and unsupported small values do not count as priced quotes. A real
price without a valid provider update time still counts as quoted but not timestamped.
Future, at-kickoff and post-kickoff times do not count as pregame timestamped evidence.
Fetch observation time never supplies provider update time. Receipt production and
strict wager-contract validation remain unchanged.

Identical receipts copied through provider games and candidate rows count once.
Each row reconciles `quote_receipt_count = quote_count + invalid_price_quote_count`,
with `timestamped_quote_count <= quote_count`. Provider identity evidence retains
unpriced receipts. Aggregate quoted and timestamped counts count canonical events,
not receipt rows. The UI and CSV project the same coverage report.

The mixed offline fixture has 3 scheduled, 2 matched, 1 quoted, 1 timestamped,
2 ranked and 0 qualified events. Its three rows have receipt/priced/invalid/timestamped
counts of `(4, 2, 2, 1)`, `(1, 0, 1, 0)` and `(0, 0, 0, 0)`. The repaired third
event stays conflicted; no later information reconstructs historical predictions.

The correction has a distinct `paid-launch-ncaaf-coverage-v2` exact successor seal.
Implementation A has the starting main as its only parent and exactly six paths:
the schedule diagnostic module, new behavior regressions, prior v1 fixture-source
reconstruction, scope guard, new successor fixtures and this document. Seal B adds
only `docs/paid-launch/launch-scope-policy-ncaaf-v2.json`. The seal binds base and
implementation commits/trees, all before/after blobs, original manifest digest,
prior NCAAF policy and clock blobs, tooling hashes and immutable evidence. Module
replacements and three reviewed blobs are also independently frozen. Complete
successor guard bytes have a SHA-256 binding with its digest normalized to 64 zeroes.
CI must have parents `[starting main, B]` and B's exact tree.

Original/v2/v3/NCAAF-v1 guard logic before the CLI remains byte-identical. Prior
NCAAF policy tests reconstruct their original v1 module and guard inputs by reversing
only exact reviewed replacements and restoring the frozen prior CLI. Their assertions
are unchanged; the v1 seal is never rebound to corrected code. New fixtures reject
re-seals, dirty/staged/committed immutable changes, incorrect ancestry, changed
workflows, altered authority/science, runtime shadows and nested-entry-point edits.

No model, calibration, feature, selection, staking, eligibility, evidence-production,
wager authority or deployment behavior is changed. This PR requires review before
merge. Only after that reviewed correction is merged may #2378 receive new exact
bindings against the resulting main and rerun CI. #2378 remains draft/unmerged;
there is no deployment authorization.
