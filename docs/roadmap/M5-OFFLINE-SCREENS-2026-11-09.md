# M5 offline screens: stored graded data against its prices

Written 2026-11-09 for `docs/roadmap/VALIDATION-ROADMAP-2026-11-09.md`, milestone M5, item 2.
Produced by `python3 scripts/offline_edge_screens.py` from data already in the repository. No
network and no credits. Every number is in-sample and nothing here grants authority.

## Reading

1. **MLB props do not beat their prices.** 741 graded props won 58.6%, against a mean break-even of
   61.3% at the logged odds. Flat ROI is −4.4% (z −1.8, clustered by slate). The audit's "58.6% on
   741, never compared with price" is now compared: the win rate is high because the prices are
   short, and it is below what those prices need.
2. **Novig's prop prices are fair; BetMGM's cost the most.** At Novig the props won 61.6% against a
   61.8% break-even (ROI 0.0%). At BetMGM they won 52.6% against 60.7% (ROI −13.9%, z −2.2). The model
   adds nothing at a fair price and loses the vig elsewhere.
3. **The model's own edge does not sort props.** Props where the model saw a positive edge returned
   −0.7%; the "Qualified" status returned −7.6%.
4. **Graded best picks are at break-even overall** (52.8% on 995, z +0.2 at an assumed −110), and
   the picks the model liked most did worse: staked picks 46.6%, model edge of 3% or more 47.8%.
5. **Against Kalshi's price for the same side the picks lose 1.6 points** (53.1% against 54.7%,
   z −1.0). When Kalshi agreed, the picks won 55.7% and Kalshi had priced them at 57.5%.
6. **No segment reaches |z| of 2 in the profitable direction.** The best-looking rows (batter hits
   over, model 6% above Kalshi) are under 1 standard error with 118 and 48 picks.

So this data adds no candidate to the roadmap's list. It supports two things already decided: stake
only at low-vig prices, and do not spend on props. The Pinnacle/Novig lag stays the only candidate
with a measured signal.

## Limits

- The props cover 10 slates in July and August 2026; the best picks cover 68 slates, 89% MLB.
- The best-pick exports carry no price. The −110 table assumes one; the Kalshi table does not.
- `kalshi_probability` is read as the probability of the pick's side. Rows at 0 (no market) or
  outside 5% to 95% are left out.
- Not screened here, because the data is not in the repository: price-screen outcomes by book and
  timing (in the research store on Drive), TheOver purple-star and sharp-money signals (5 stored
  rows), and Gemini agree against disagree (not in the graded exports).

## MLB props against their logged prices

741 graded picks, 2026-07-22 to 2026-08-13, 10 slates. Break-even is 1 / decimal odds at the logged price.

| Segment | Picks | Days | Win rate | Break-even | Excess | z | Flat ROI |
|---|---|---|---|---|---|---|---|
| All graded props | 741 | 10 | 58.6% | 61.3% | -2.7% | -1.8 | -4.4% |
| Book: betmgm | 196 | 10 | 52.6% | 60.7% | -8.1% | -2.2 | -13.9% |
| Book: draftkings | 115 | 10 | 57.4% | 60.5% | -3.1% | -0.6 | -4.4% |
| Book: novig | 430 | 10 | 61.6% | 61.8% | -0.2% | -0.1 | -0.0% |
| Side: over | 331 | 10 | 57.7% | 60.8% | -3.1% | -1.4 | -4.7% |
| Side: under | 410 | 10 | 59.3% | 61.7% | -2.4% | -1.5 | -4.1% |
| Type: batter | 643 | 10 | 58.8% | 61.7% | -2.9% | -2.0 | -4.6% |
| Type: pitcher | 98 | 10 | 57.1% | 58.8% | -1.7% | -0.5 | -3.2% |
| Market: batter_hits_over | 118 | 10 | 65.3% | 61.1% | +4.2% | +0.9 | +7.9% |
| Market: batter_hits_under | 100 | 10 | 58.0% | 63.5% | -5.5% | -1.4 | -8.3% |
| Market: batter_total_bases_over | 167 | 10 | 53.3% | 60.7% | -7.4% | -1.9 | -12.6% |
| Market: batter_total_bases_under | 258 | 10 | 59.7% | 61.8% | -2.1% | -1.0 | -3.6% |
| Model edge above 0 | 137 | 10 | 62.0% | 63.0% | -1.0% | -0.3 | -0.7% |
| Status: Qualified | 200 | 3 | 56.5% | 61.6% | -5.1% | -1.4 | -7.6% |
| Price -200 or shorter | 110 | 10 | 63.6% | 70.2% | -6.6% | -1.8 | -9.0% |
| Price longer than -200 | 631 | 10 | 57.7% | 59.7% | -2.1% | -1.4 | -3.6% |

## Graded best picks at an assumed -110

995 graded picks, 2026-06-02 to 2026-08-28, 68 slates. The exports carry no price, so break-even is 52.4%.

| Segment | Picks | Days | Win rate | Break-even | Excess | z | Flat ROI |
|---|---|---|---|---|---|---|---|
| All graded picks (-110 assumed) | 995 | 68 | 52.8% | 52.4% | +0.4% | +0.2 | +0.7% |
| League: MLB | 884 | 68 | 54.3% | 52.4% | +1.9% | +1.1 | +3.7% |
| League: NFL | 37 | 7 | 45.9% | 52.4% | -6.4% | -0.7 | -12.3% |
| League: WNBA | 61 | 23 | 41.0% | 52.4% | -11.4% | -1.8 | -21.8% |
| Kalshi: Agrees | 562 | 64 | 55.7% | 52.4% | +3.3% | +1.7 | +6.3% |
| Kalshi: Disagrees | 161 | 47 | 52.2% | 52.4% | -0.2% | -0.1 | -0.4% |
| Kalshi: Neutral | 226 | 60 | 47.3% | 52.4% | -5.0% | -1.6 | -9.6% |
| Kalshi: No Kalshi | 46 | 17 | 45.7% | 52.4% | -6.7% | -0.8 | -12.8% |
| Staked (Kelly above 0) | 161 | 23 | 46.6% | 52.4% | -5.8% | -1.7 | -11.1% |
| Model edge 3% or more | 295 | 62 | 47.8% | 52.4% | -4.6% | -1.5 | -8.8% |
| Model edge 6% or more | 124 | 28 | 47.6% | 52.4% | -4.8% | -1.1 | -9.2% |

## Graded best picks against Kalshi's price for the same side

| Segment | Picks | Days | Win rate | Kalshi | Excess | z |
|---|---|---|---|---|---|---|
| All picks with a Kalshi price | 948 | 68 | 53.1% | 54.7% | -1.6% | -1.0 |
| Kalshi: Agrees | 562 | 64 | 55.7% | 57.5% | -1.8% | -0.9 |
| Kalshi: Disagrees | 160 | 47 | 51.9% | 48.4% | +3.5% | +0.9 |
| Kalshi: Neutral | 226 | 60 | 47.3% | 52.2% | -4.9% | -1.6 |
| Model at least 3% above Kalshi | 126 | 46 | 47.6% | 46.6% | +1.0% | +0.2 |
| Model at least 6% above Kalshi | 48 | 22 | 47.9% | 41.7% | +6.2% | +0.9 |

