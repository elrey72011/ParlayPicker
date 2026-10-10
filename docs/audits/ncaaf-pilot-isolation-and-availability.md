# Pilot isolation and retained availability

Base main `497dfe6e9dfce5c5b7d81b99ce36e1f170c1ab9f`, ordered #2408 merge parents
`00434dc8d14227648bb19795d8e05bb9304bca99` and
`3779efd1749e260e96a9202b648e4296f088864e`, tree
`55883950dd39485e504b54d59b2d5b80adc883d9`, has all 13 required post-merge checks
passing. Final-head CI for this successor is separate. Hosted revision/current
availability is UNKNOWN; no new analysis, provider request, Drive operation or
historical numerical reconstruction was performed.

## Reproduced isolation boundaries

Network-blocked fakes through `football_stage1_cycle.run_cycle` observe Drive
sync, NFL schedule/odds, then NCAAF schedule/odds. The test stops before store
creation. Existing normal analysis observes NCAAF schedule, Novig recovery,
ESPN FCS fallback and model/external enrichment. The generic Odds API client
retries a fake 429 and follows fake pagination: at least three requests from one
call. These are measured synthetic call traces, not real service activity.
Neither entrypoint is used by the adjacent pilot.

The pilot regressions measure exact plan requests, zero forbidden entrypoint
calls, full original-byte read-back, interruption/no-resume, sanitized errors,
zero numerical calls on rejection, and four separately accepted synthetic
spread/total directions through caller/capture/export/display. Browser uses local
synthetic HTML only. No authentic successful observation is claimed.

## Static inventory and reconstruction limits

The private row-level `ncaaf-pilot-private-availability.json` and summary retain
exact archive/source hashes and row references, canonical identity if recorded,
provider identity, market/side/line, original clocks, first recorded rejection
stage/code, missing fields, next action/role, and separate probability/value/
wagering status. Raw artifacts/receipts remain outside Git. Existing assertions
and original files were not modified. The adjacent static inventory helper does
not calculate probabilities or guess skipped gate results.

Four unique October 4–6 archives contain 152 producer rows and 148 surviving
candidate rows. Duplicate ZIP copies are counted once. Rows overlap across runs
and markets; these are not 152 independent games. The latest retained supplied
run, October 6 `20261006T172545.230406Z`, contains 12 provider events / 48 producer
and candidate rows: nine NHL events (36 rows), two MLB (eight), one NCAAF (four),
zero NFL. Its CSV export label `20261006T172526Z` is distinct, not a run conflict.
No independent schedule denominator is retained in these old packages, so full
schedule coverage and **current** hosted decision state remain UNKNOWN. The
new lane retains an explicitly supplied independent inventory and each scheduled
event, including unresolved markets and FCS policy exclusions.

| Priority/scope | Retained evidence / first boundary | Exact next action and role |
|---|---|---|
| NCAAF half-point spreads, home/away | Old four-row packet records no configured market model. Merged compatible/custody software now exists, but no complete admitted original-response chain is established. Southern Miss +10 is integer/unvalidated. | Owner/source reviewers resolve the blocked Army plan's event/listing/rights/advance terms and histories; owner then separately authorizes capture. No model fitting required for software acceptance. |
| NCAAF half-point totals, over/under | Separate target software exists. Retained Novig 51.5 is excluded; no qualifying independently admitted total packet. | Source reviewer must bind a supported non-Novig full-game half-point total; owner needs a separate total plan, not automatic spread scope expansion. |
| NFL spread home/away | Retained New Orleans −1.5, Novig +102 has raw `0.51662958106965` and original blend `0.5046440022813474`. Target schedule crosswalk, listing/product, full-game/overtime rules, FVS applicability, clock meaning and account uses are unaccepted/unrecorded; UI-refresh receipt absent. | Owner obtains the already prepared offer-specific feed/Novig response; independent source reviewer binds exact applicable evidence. Source confirmation does not qualify the predictor. |
| NFL totals over/under | Original retained numbers do not demonstrate applicable source/settlement/value contracts or scientific qualification. | Reviewer verifies exact original target/model/offer and source contract first; do not transfer spread evidence or assume genuine push mass. |
| NHL ±1.5 home/away | Latest retained nine games/36 rows have unavailable market inference; compatible cover-input packets/artifacts and lineage not demonstrated. Snapshot counts cannot establish their presence. | Evidence custodian supplies the exact already-retained cover artifact/dependency references if accessible; otherwise UNKNOWN storage. NHL remains next in queue, no winner-model substitution. |
| NHL totals | Separate target; no authentic total chain established in inspected evidence. | Model/source reviewers establish independent total artifact, feature/goalie missingness, lineage, full-game OT/shootout semantics. No implementation expansion here. |
| NBA/NCAAB spreads and totals | No current supplied package inspected for these markets; their current reasons are UNKNOWN. | Owner supplies the existing exact private trace, only if these scopes are pursued later. No fresh analysis requested. |
| WNBA spreads and totals | October 4 archive has four rows, outside current supplied run. Current artifact/input/source availability UNKNOWN. | Custodian/reviewer inspect the exact existing trace/target bindings before diagnosing code. No expansion here. |
| MLB home/away Run Lines and totals | Eight successful October 6 inference rows exist; per-game trace's first source-contract failure is `SOURCE_MARKET_LISTING_BINDING_NOT_VERIFIED`, with absent period/rules/listing/product reviews. Home qualification does not extend to away/totals; Novig restrictions remain. | Paused by owner. Preserve original numbers and fixes; external offer-specific declarations remain required. No retrieval/development here. |
| All qualification/stakes | Research probabilities, value display, funded approval and stakes are distinct. No archive/fixture/CI supplies qualification or current authority. | Independent scientific reviewer and owner must complete governing routes separately. Unqualified stays PASS zero stake. |

Software blockers addressed here: bounded isolation, original-body transport and
separate admitted local retention path. External blockers: exact event/source/
listing/product/rights acceptance. Operational blockers: durable custody,
credential/account budget and actual reviewed hosted revision. Scientific
blockers: accepted prospective role/protocol, calibration where required and
independent qualification. The highest-priority owner action is source/mapping/
permission review of the blocked half-point proposal before any collection
authorization. Requests remain unsent.
