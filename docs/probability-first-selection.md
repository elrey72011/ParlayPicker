# Probability-first per-game picks

For new analysis runs, the selector chooses the candidate with the highest available win estimate after the existing market, identity and line-integrity gates. The overall winner and each side/total winner use the same rule. Public quote selection continues to apply the configured Novig and named-sportsbook fallback rules.

The input is the oriented `calibrated_probability` saved on the candidate, before legacy status defaults. Exact, unambiguous complementary spread/total pairs at matching lines are normalized together so opposite estimates sum to one. Mismatched or ambiguous pairs are not normalized. This is not evidence of successful calibration: the estimates still need forward outcome validation.

`best_available_probability`, its source, pair-normalization flag and `probability-first-v1` policy are preserved on all candidate-audit rows and selected exports. The best-available score and winner/runner-up gap now describe that selection probability. Missing, nonfinite, boolean and out-of-range inputs do not become a fabricated 50% forecast. If an entire game lacks an estimate, a deterministic coverage row remains, marked unavailable and unfunded. A repaired line loses the old line's estimate.

Legacy empirical blends, family/regime penalties and component-direction scores remain available for diagnostics and existing conservative wager checks. They do not select a lower-probability candidate. Exact probability ties use market, pick and quote identity, with no EV or composite-score override. Wager approval still requires the existing calibrated price, integrity, portfolio and funding checks; a most-likely selection can be a PASS.

Public per-game boards display the same candidate probability used for selection, and calculate the displayed estimated edge and EV at that ticket's exact odds. Conservative production estimates continue to govern funding independently. Existing archived/locked selections are not recalculated or replaced.

After merging and deployment: Refresh picks, review the new selections, lock eligible picks if desired, and publish the board. Updating results alone does not rerun selection. The original-estimate results comparison provides forward evidence; this change implements a selection objective and does not claim a proven increase in win rate.
