# Independent model direction selection

The selector now prevents a ranking overlay from reversing a strong independent
model preference within an exact opposite market pair. It uses the existing
Precision Card floor (60%), not a threshold fitted to yesterday's winners.

Only resolved MLB score-distribution predictions qualify. Both candidates must
have matching model provenance, correct prediction targets, complementary
probabilities and the same total or opposite signed spread. Incomplete pairs,
duplicate alternatives, mismatched lines, missing models and weaker preferences
retain their existing ranking. The guard runs before family and global ranking.
It lowers only the opposed candidate's ranking score below its supported opposite.
It does not change probabilities, quoted odds, expected value or stake authority.
Pre-guard scores, applied flags, penalties and reasons are retained in the candidate
audit so future results can be compared against the unchanged selector.

## Retrospective evidence

Eight saved candidate-audit slates cover August 29 through September 5, 2026.
The fixed-candidate replay preserves original candidates and supplied grades;
it does not reconstruct historical features or API responses. Louisiana's duplicate
fallback event is excluded consistently from both comparison arms.

| Date | Existing record | Updated record |
|---|---|---|
| August 29 | 30-18 | 30-18 |
| August 30 | 12-6 | 12-6 |
| August 31 | 9-3 | 9-3 |
| September 1 | 7-8 | 7-8 |
| September 2 | 7-7 | 8-6 |
| September 3 | 12-10 | 12-10 |
| September 4 | 12-10 | 12-10 |
| September 5 | 50-49 | 50-49 |
| Total | 139-111 (55.6%) | 140-110 (56.0%) |

The one changed selection is September 2 Texas-Athletics: Over 12.5 (loss)
becomes Under 12.5 (win), supported by a 69.4% independent model estimate.
MLB alone moves from 68-41 (62.4%) to 69-40 (63.3%), with unchanged coverage.
NCAAF and props are unaffected.

An initial version that trusted every greater-than-50% model preference showed
no net improvement over these eight slates and was rejected. It would have made
September 5 MLB 12-3, but that result does NOT apply to the final conservative rule.
The final rule leaves September 5 at 50-49 overall and 11-4 for MLB.

This is development evidence: the data were inspected while designing the rule,
and one additional win is not statistically persuasive. No independent holdout
improvement or 75% accuracy is claimed. The final rule needs prospective evaluation
on newly exported slates, including losses, coverage and realized return at offered
prices. An all-games 75% target remains unsupported, especially for NCAAF, where
these exports lack independent spread/total probabilities.

## Reproduction

Run `python scripts/replay_model_direction.py PATH_TO_GRADED_CANDIDATE_AUDIT.csv`.
The replay strips outcome labels before applying the production guard and restores
them only to compute the comparison. The rule consumes no outcomes. Use one
snapshot per slate; repeated downloads are not independent evidence.
