# NFL market tracking

This workflow uses only The Odds API. It collects market observations for later analysis, not independent predictions, selected bets or approved wagers. No new API key or model freeze is required.

## Operation

1. Merge NFL scheduler support. Keep the existing GitHub Odds and Drive secrets.
2. Set `RESEARCH_SPORTS` to `MLB,NCAAF,NFL`. Alternatively, verify NFL alone using the manual workflow's sports override before changing scheduled configuration.
3. Check the Actions summary for `sports.NFL`, shared budget use, errors and exclusions. Successful backup requires remote read-back verification.
4. In Streamlit's Data Maintenance section, expand **NFL Market Tracking**, select **Restore / back up NFL evidence**, then download the market report and records. These controls do not call paid sports endpoints.

The scheduler retains the existing 30-minute cadence and 11:45 a.m.–2:30 a.m. Eastern window. A cycle discovers NFL events using the free `/events` endpoint. If uncaptured events start within two hours, it makes one US-region request for `h2h,spreads,totals`, costing three reserved credits. It records both outcomes of each complete same-book market pair, with exact market update times no older than 15 minutes. It does not use bookmaker-level timestamps as exact-market proof. Missing/stale/malformed pairs are excluded and counted. Starts are checked again when saving.

Only the first saved snapshot for each event contributes to the report. Later runs skip captured events; this is not opening-line, continuous line-movement or closing-line collection. All game IDs and team names must match exactly. Postponements or changed schedule times during grading require review rather than an assumed match. The provider's scheduled start is recorded; actual kickoff timing is not independently verified.

Pending captured games are checked starting three hours after their scheduled start. One batched `/scores?daysFrom=3` request costs two reserved credits. Only `completed=true` games with exactly two valid, named integer final scores and matching event identity/start are saved. In-progress games remain pending. Games unresolved more than three days after the captured start are reported for operator review and no longer trigger paid polling. Provider availability after an outage or postponement is not guaranteed. The first accepted final score is retained; correction reconciliation is a future enhancement.

## Storage and interpretation

Records use an append-only local `nfl-market.sqlite3` and the isolated Drive prefix `parlaypicker/nfl-market-v1/`. Hash verification and read-back guard restoration and backup. Raw API keys and request URLs with credentials are never stored. The scheduler must restore successfully before any work; failed backup marks the run unsuccessful. The existing workflow concurrency group serializes shared budget reservations.

The report shows captured/graded game counts and quote-level comparisons. A quoted team spread wins when its final score plus its spread exceeds its opponent's final score; totals use combined final scores. Equality is a push; a tied moneyline is labeled tie. These are score comparisons, not verified sportsbook settlements. Overtime treatment, cancellations, refunds and bookmaker house rules are not independently checked. No paper or real-money ROI, combined quote hit rate, win probability, model accuracy, calibrated edge or 75% claim is reported. Quote rows from the same game are not independent observations.

## API budget

NFL and NCAAF draw from **one 7,500-credit rolling-31-UTC-day allowance**, under the unchanged **200-credit UTC daily cap**. This is automation's allocation from the user's stated 15,000 monthly credits. CFBD stays at 25 requests/day and 500 per rolling 31 days. Raising the ceiling preserves existing ledger usage and does not reset it. Prior usage, Streamlit runs and other clients are not counted; check provider dashboards for the actual remaining account balance.

Paid reservations are verified in Drive before requests and not refunded for failures or empty responses. Unexpected usage-header costs stop subsequent paid work for review. Budget pauses may reduce NFL/NCAAF coverage; a larger monthly account limit does not override daily protection.

Provider documentation: [The Odds API v4](https://the-odds-api.com/liveapi/guides/v4/) documents free event discovery, market/region odds costs, shared event IDs and the three-day recent-score endpoint with two-credit cost. NFL score coverage is listed in [supported sports](https://the-odds-api.com/sports-odds-data/sports-apis.html).
