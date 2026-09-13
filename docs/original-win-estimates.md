# Original estimates versus outcomes

Published game and MLB prop results now carry `original_win_estimate` from the original archived leg's `win_estimate`. Owner locks use the immutable locked leg. A later analysis, publication or score correction cannot replace that original estimate. Selection ranking scores and implied sportsbook probabilities are never substituted.

The public board labels model win estimates as not yet validated. Results details and today's locks show the original estimate. Under Results, expand **Estimated versus actual win rates** to compare 5-percentage-point bands, separated by category, league and market. The existing period, selection and prop filters apply. Overall, sides, totals and Top 10 overlap and are never pooled. Spread directions share a spread group; over/under directions share a total group.

Each band compares the mean saved probability and actual win rate for the same WIN/LOSS records. Coverage counts disclose settled records missing an estimate and excluded pushes, pending and review cases. Imported recaps and parlays are excluded. These descriptive comparisons, especially small samples, do not establish calibration or profitability; live probabilities, rankings and wager eligibility are unchanged.

After deployment, restore original public history if the session has older report rows, then **Update results and publish** to regenerate reports and publish the display. Historical estimates can be recovered only when original archived or locked legs contain them. Missing or invalid historical estimates display **Not recorded**, with no inference from today's analysis. The additive result fields preserve compatibility with earlier public packages; immutable source records and their IDs are not rewritten.

## Display order

Overall picks, sides, totals and player props are listed from highest to lowest saved win estimate. Missing or invalid estimates are last; ties use stable pick identity, not estimated value or composite selection scores. Current tables show rank and league. Locked picks sort by their original saved probability, never a later analysis. League and market filters retain this ordering. The per-game selector now follows the probability-first policy described in `probability-first-selection.md`. Original Top 10 tracking and locks stay immutable; estimates are still unvalidated.
