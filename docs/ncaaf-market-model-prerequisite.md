# NCAAF market-model prerequisite

The independent spread/total score-distribution model currently implements MLB and WNBA. NCAAF team-stat resolution does not imply that an NCAAF probability model is configured. Missing NCAAF output must remain unavailable until a model is trained and validated.

To build that model, obtain historical NCAAF games with stable event/team IDs, kickoff timestamps, final scores, and scoring features as they were available before kickoff. Features need observation timestamps and season/game counts. Historical line/price timestamps are needed for execution and EV evaluation. Do not train historical examples using today's aggregate team statistics.

Fit margin and total distributions on earlier seasons, calibrate on a later disjoint period, and evaluate on an untouched subsequent period. Estimate integer-score push mass explicitly. Compare against market and simple scoring baselines using calibration, Brier/log loss, coverage, and realized returns. Freeze the resulting artifact and input schema before prospective evaluation. A proposed parameter table alone is not a validated NCAAF model.

The current fix reports the actual unsupported-model cause and prevents exported false feature-eligibility flags from producing forecasts. It does not enable NCAAF wagering or substitute another sport's parameters.
