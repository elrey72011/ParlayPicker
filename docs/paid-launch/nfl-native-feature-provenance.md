# Native NFL spread feature dependencies

This bounded successor extends the existing private `nfl_inputs` packet, immutable snapshots, capture/export/source receipts and owner replay download. It creates no storage tables or new acquisition path. Existing event-specific dependency contracts retain their interpretation; the new aggregate scope is explicit.

| Consumed feature | Native aggregate / mapping |
|---|---|
| `feature_home_ppg`, `feature_away_ppg` | `fetch_nfl_stats`: completed prior team points sum / count; named team mapping and float conversion |
| `feature_home_oppg`, `feature_away_oppg` | Opponent points sum / count; named team mapping and float conversion |
| `feature_home_games_played`, `feature_away_games_played` | Completed prior-game count |
| `feature_home_win_pct`, `feature_away_win_pct` | Strict score wins / count; existing [0,1] clamp |
| `feature_diff_last5` | Each team's last five date-ordered strict score wins / count; existing safe home-minus-away difference |
| `feature_home_recent_point_margin`, `feature_away_recent_point_margin` | Each team's mean score margin over the last five completed prior games |

`enrich_with_model_features` remains the producing mapper. Numerical formulas, defaults, weights and predictor behavior are unchanged. Missing/default-only inputs do not gain invented native dependencies. The adapter observes and retains the typed schedule-frame projection actually consumed, original game IDs, source module/version/function identity, named team/season/as-of scope, original rows and aggregation/mapping/code identities. The returned frame is not represented as original HTTP/CSV wire bytes.

The native aggregate receipt has prior-game source IDs distinct from the Odds API target event. Mapping binds the exact original provider namespace/event, named home/away/kickoff and feature slot. An unrelated team, contradictory target event, invalid window, completed member dated after its original observation, supplied invalid/future member availability, altered aggregate or transformation rejects even if nested hashes are consistently rewritten. Hashes establish consistency, not source authentication or rights.

`available_at` in the new contract means the original adapter's first observed local availability, explicitly labeled `original_adapter_first_observation`. It never means publisher first-publication. Publisher availability and earlier historical availability remain UNKNOWN unless separately supplied and verified; the new observation does not retroactively admit old predictions. Cached source observations keep their original clock, and the later mapping has a separate observation clock. Missing source/member/availability/coverage facts remain UNKNOWN/INCOMPLETE and cannot replay as COMPLETE.

The offline reader recomputes the recorded aggregates and mapping from retained typed rows without executing saved code or acquiring sources. Replay is software integrity evidence only. Research estimates remain distinct from calibrated/conservative probabilities. Existing source-contract protections, integer-line push limits, started-game guards, qualification gates, authority and PASS-at-zero-stake behavior remain intact. Complete native evidence does not fill missing product/rule/rights bindings.

Historical evidence is not rewritten, recomputed or upgraded. Pilot collection, its source permissions, model/protocol freeze, outcome custody, fitting, scientific qualification, accepted readers and activation/exposure authority require separate review and authorization. The private pilot decision package is not a public repository artifact.
