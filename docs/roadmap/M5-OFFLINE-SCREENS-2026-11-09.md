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
- Not screened, because the data is not in the repository: TheOver purple-star and sharp-money
  signals (5 stored rows), and Gemini agree against disagree (not in the graded exports).

## Price screen by book and by quote timing (research store)

Run 2026-11-09 at 4:03 PM ET from the read-only Market readiness workflow (run 42698680900), which
restores the research store and uses no credits. A candidate is the best price across books for one
side of one game at one line; "book" is the book that had that best price. CLV is against the same
book's certified close at the same line. Standard errors are in brackets.

**Reading**

1. **No book and no timing bucket shows a profit beyond noise.** Every ROI is within 2 standard
   errors of the −2% to −3% that the best price loses to the vig on average. The largest groups:
   LowVig NCAAB spreads −2.2% (6.7%) on 208 settled; William Hill NCAAB totals −3.5% (9.6%) on 101.
2. **The best NCAAB spread price does beat its own book's close, by a little.** Mean CLV is +0.8% on
   110 comparable closes: +0.8% (0.3%) for quotes 3 to 6 hours out and +0.8% (0.3%) for 6 to 24 hours,
   each about 2.7 standard errors. By book it is largest at the retail books: DraftKings +1.9% (0.8%)
   on 14, BetRivers +2.0% (0.9%) on 12, BetMGM +2.8% (0.5%) on 3. At LowVig it is +0.2% (0.2%) on 45.
3. **That is not an edge by itself.** The same candidates average −2.6% EV against the no-vig
   consensus and returned −2.7%. Beating a retail book's close by 1% to 2% does not cover its vig of
   about 4.5%. A comparable close also exists for only 110 of 552 candidates, and only where the
   line did not move, so the +0.8% is measured on a selected fifth of the picks.
4. **Novig has the cheapest prices and no results yet.** Its best prices average −0.7% EV on spreads
   and −1.4% on totals, against −2.5% to −4.3% elsewhere, but it has 0 settled picks and 0 closes:
   its capture began days ago.
5. **Timing cannot be judged yet.** 87% of candidates were quoted 6 to 24 hours before the start
   (research capture prices each game once). The under-3-hour rows hold 10 to 26 picks each. The
   price watch that started today fills those rows.

What it changes: nothing in the ranking. It does support the plan design from another side. Stale
retail prices beat their close, which is the lag the Pinnacle/Novig plan is built on, and the only
book where beating the close could clear the vig is the low-vig one.

| NCAAB spread, by book | Picks | Mean EV | Closes | Mean CLV (SE) | Settled | ROI (SE) |
|---|---|---|---|---|---|---|
| LowVig | 240 | −2.5% | 45 | +0.2% (0.2%) | 208 | −2.2% (6.7%) |
| William Hill | 61 | −3.7% | 11 | 0.0% (0.0%) | 56 | −18.3% (12.7%) |
| DraftKings | 51 | −3.2% | 14 | +1.9% (0.8%) | 45 | −9.4% (14.6%) |
| FanDuel | 51 | −2.8% | 15 | +0.3% (0.3%) | 43 | +17.3% (14.6%) |
| Fanatics | 46 | −3.1% | 4 | +1.1% (1.4%) | 40 | +6.6% (15.4%) |
| BetRivers | 29 | −1.8% | 12 | +2.0% (0.9%) | 22 | +7.6% (21.5%) |
| Bovada | 29 | −2.0% | 3 | +1.5% (1.5%) | 24 | −10.3% (20.3%) |
| Novig | 24 | −0.7% | 0 | | 0 | |
| BetMGM | 17 | −2.3% | 3 | +2.8% (0.5%) | 14 | −17.3% (26.5%) |

| By quote timing | Picks | Closes | Mean CLV (SE) | Settled | ROI (SE) |
|---|---|---|---|---|---|
| NCAAB spread, under 1 h | 10 | 2 | −1.7% (1.7%) | 10 | −4.2% (31.9%) |
| NCAAB spread, 1–3 h | 22 | 5 | 0.0% (0.3%) | 20 | −3.2% (22.2%) |
| NCAAB spread, 3–6 h | 112 | 50 | +0.8% (0.3%) | 112 | −2.5% (9.2%) |
| NCAAB spread, 6–24 h | 408 | 53 | +0.8% (0.3%) | 314 | −2.7% (5.5%) |
| NCAAB total, 6–24 h | 452 | 26 | +0.4% (0.2%) | 340 | −3.4% (5.3%) |
| NBA spread, 6–24 h | 96 | 7 | −0.4% (0.6%) | 74 | −2.5% (11.4%) |
| NBA total, 6–24 h | 112 | 6 | 0.0% (0.8%) | 82 | −3.2% (10.6%) |

NBA, NCAAB totals and NHL by book are in the run's log; no group there has more than 45 settled
picks apart from LowVig and William Hill NCAAB totals, and NHL has no certified closes at all.

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

