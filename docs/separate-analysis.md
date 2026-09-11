# Separate game and player-prop analysis

Run Game Analysis fetches game odds, model predictions and optional Gemini game
reviews, produces game picks/parlays, and saves game prediction evidence. It does
not fetch or review player props. Its former name was Run Master Analysis.

Run Player Props separately collects the selected supported leagues and applies
the existing prop eligibility, Gemini and staking rules. Saved game results remain
available. Prop-only parlays appear with the prop results, independently of game
parlays. A prop failure retains previous results; a successful empty run replaces
old props with an empty card. The run uses the current sport, bankroll, prop ledger
and Gemini controls.

The progress panel lists each stage with elapsed time at stage transitions. The
final timings are available in Diagnostics, separately for games and props. The
Save prediction evidence stage remains visible during the game run. No background
job or continuous elapsed-time ticker is introduced.

Game refreshes preserve prior props. Prop exports retain their own run IDs and
original analysis times; opening another tab or exporting does not refresh them.
Preview & Publish shows both analysis times and warns when saved props exceed the
15-minute freshness window. Include saved player props is optional. The public
page displays separate game/prop timestamps and retains per-selection stale rules.

Typical daily flow: Run Game Analysis, review games, optionally Run Player Props,
restore public history, build/review the publication, and publish. Grading history
and DraftKings lineup generation remain separate explicit actions. Saved analysis
is session state, as before; publishing history is still backed up to Drive.

The split removes the prop stage from the critical path for game results. It does
not make provider requests, Gemini reviews or evidence storage themselves faster;
actual savings depend on the slate. No live API benchmark was run in validation.
