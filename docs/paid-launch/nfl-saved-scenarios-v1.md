# Read-only saved NFL scenarios

Owner engineering authorization, October 25, 2026: implement a saved-results
control in the existing owner-token-protected Streamlit panel, using the
configured private store and released static readers. Reconcile against main
`792abe61a8e00992290abe752eb17825b7bbd0b6`, tree
`de0f0762b784321cdda4fc8a72a6bfc773a96f13`. Authorizes isolated implementation,
focused regressions, required exact scope/binding metadata, a draft PR, and one
conditional final-head CI cycle after local validation. No merge or deployment.
No new predictions, provider acquisition, normal Refresh, model changes,
training, scheduling, wagering, public publishing, or operational changes.

## Owner workflow

1. Open **Workspace → Preview & Publish** and enter the publishing token.
2. Expand **Private NFL sporting scenario**.
3. Under **View saved NFL scenarios**, select the existing records you want.
4. Click **Load saved results**. Both opposing saved scenarios can appear
   together; changing the selection clears the old display until loaded.
5. Download each original private evidence ZIP if needed. A fresh session
   lists the configured store again; no packet upload or new Run is needed.

Saved viewing is separate from the enable/stage/confirm/Run controls. It never
sets their state or calls their action. The existing outer token gate remains
unchanged. The viewer shows the configured store path; if another deployment
uses a different store, it reports what is available there and never moves
records or creates a database. At most 100 records are indexed and four are
selected per load. Invalid records remain untouched and unavailable.

The index connects with SQLite `mode=ro`, checks the existing schema/triggers,
and reconstructs namespaced receipts from existing export/input hashes. Each
record is authenticated by `research_replay.read_scenario` and
`nfl_scenario_research.read_result`. Downloads reuse `download_scenario`.
There is no migration, setup, write connection, new archive or new inference.
The original model, reader, replay, clocks, permissions and result bytes are
unchanged. None of the authentic reference hashes is hardcoded in the UI.

The display includes selected team/handicap, original native cover probability,
selected-team-minus-opponent margin projection before handicap, residual scale,
original inference time, saved result hash and separately labelled viewing time.
Uncalibrated private scenario / PASS / zero stake labels remain. Static replay
checks the original inference time, so viewing after kickoff does not relax
prospective execution gates or pretend that the result is a fresh offer.
These rows are not public packages or ordinary offer-backed candidates.

## Binding and validation boundary

The UI module is outside the saved scenario's hashed model/replay reader graph.
Changes are limited to it, two viewer helpers, new tests/docs and exact scope
metadata. The inherited source graph authenticates the changed UI and retains
its exact previous bytes for historical readers. All existing tests, workflows,
runtime limits, policy seals, models and replay readers remain unchanged.
No historical result is rewritten or re-signed. A separate implementation and
policy-only seal bind the exact source and unchanged files.

Focused tests exercise real selection/Load controls, both results, switching,
fresh sessions, postgame replay, corruption/missing input/oversize/missing store,
store changes and unauthorized access. During viewer actions, predictor calls,
network and database writes fail immediately. Synthetic fixture creation occurs
before these tripwires; its saved numbers are labelled synthetic, not predictions.
Authentic read-only verification is recorded separately outside the repository;
browser interaction and renderer-only verification must be reported distinctly.

Preserve all source evidence, Monday's NFL backup, Delaware–WKU controls/window
and the MLB pause. No operational allowance is spent by this change.
