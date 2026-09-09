# Public Results tracker

## Daily workflow

In Workspace > Preview & Publish, unlock publishing and expand Public results history. Click Restore public history from Drive. Select a date and click Grade published picks for selected date. Build a fresh preview, review Results, then publish and promptly check deployment status. A local preview or download does not count as a public publication.

The existing Netlify site ID and Shared Drive configuration are required. No new secrets are introduced. New public publications are blocked until history is restored, preventing an accidentally empty replacement tracker. Empty history on the first restore is valid. Local preview/download remains available without a restored history.

The Results tab separates Overall, Sides, Totals, and complete Parlays; Approved and Research are separate filters. Yesterday uses Eastern time. Last 7/30 days include today and the previous 6/29 calendar dates. Win percentage is wins / (wins + losses). Pending and pushes are excluded; settled count includes pushes. Categories overlap and must not be summed. No actual betting returns are claimed. Props and DFS are outside this tracker.

## Publication evidence

Under parlaypicker/public-history-v1/<site-id>/, Drive stores create-only packages by content hash, submitted deployment IDs, confirmation receipts, and final-score revisions. Every write is read back. Packages are archived before a Netlify upload; a failed upload or an unconfirmed deployment is never counted. Netlify must confirm the deployment is the current published website before recording it.

Confirmation time is the time this application observes verified publication, not a backdated build time. Check status immediately: a selection counts only if confirmation is before its start, after its analysis timestamp, and within 15 minutes of analysis. Restoring submitted deployment records can recover confirmation after a Streamlit restart if Netlify still reports that deployment as published. Late recovery does not retroactively claim pregame publication. An upload with an uncertain response and no deployment ID needs dashboard investigation; it is excluded.

The earliest eligible confirmation per category, league, team matchup and Eastern game date wins. Later changes to lines, approval, or kickoff time cannot replace that entry. Because public exports lack a provider game ID, this conservative identity keeps only the first matchup on doubleheader dates; second games are not separately counted. Parlays are deduplicated by their constituent matchups; cross-date tickets are excluded. All currently generated parlays remain Research.

## Grading

An explicit one-date action fetches the existing public ESPN scoreboard endpoints, once per relevant league/view with a 10-second request timeout and no automatic retries. No CFBD, Odds API or Gemini credits are used. No grading or Drive restore runs simply from navigation. This PR does not schedule grading or automatically redeploy Netlify.

Only completed final events with nonnegative numeric scores are accepted. Exact normalized teams and a start-time tolerance of 30 minutes are required. Ambiguous events, changed schedules, unavailable final scores, unsupported markets, and unknown pick text stay pending. No fuzzy match forces a result. Original lines and odds remain unchanged. Later saved score revisions can correct outcomes. The scoreboard's final state is authoritative; void/postponed games remain pending for investigation.

A parlay waits for all legs. Any losing leg then makes the ticket a loss; otherwise any pushed leg makes it a push, excluded from win rate. This is a selection-history convention, not sportsbook-specific payout settlement. The public report contains original picks, published single-leg odds, final scores, confirmation time, and outcome; it exposes no storage credentials.

## Verification and limits

29 focused tests passed, including publication deduplication, late exclusion, immutable restore/site isolation, score corrections, ambiguity/push cases, explicit no-fetch reruns, and existing publishing compatibility. A mobile-sized browser check verified calculated percentages, detail rows, filter isolation and no JavaScript errors. No live deployment, Drive writes, or provider calls were made during validation.

Prior CSVs and old deployments are not imported as verified public history. Use their separate research recap. Keep original Drive files: deleting evidence can remove it from the restored report. History loading is explicit and scales with accumulated records; no archival compaction or background synchronization is added here.

## Importing historical per-game recaps

After restoring history, upload Overall, Sides and Totals per-game CSVs from one run under Import an older recap. Click Import historical recap to Drive, select their game date and click Grade picks for selected date. Build and publish a new preview. Imports are immutable, deduplicated and restored from the site-specific imports prefix. They are always labeled Imported research, including source rows that say APPROVED. No publication timestamp or verified pregame provenance is invented. Rows whose export time is after their listed start, missing markets and missing odds are excluded. The earliest export per category/matchup/date is retained; no historical parlay tickets are synthesized.

Imported records match exact normalized teams and date only when one final event exists for that matchup/date. Multiple events remain pending. Verified publications retain the 30-minute time tolerance. MLB's bare Seattle alias is explicitly resolved to Seattle Mariners instead of Seattle University.

Results now defaults to All time and a group with settled data, or a group with pending data if none have settled. Each group remains separate. Empty-filter messages name the alternatives. The private panel lists available game dates so yesterday is not confused with today's pending picks.

Validation: 33 focused tests passed. The user's September 8 CSVs were checked against live ESPN finals: Overall 11–4 (73.3%), Sides 9–3 (75.0%; three unavailable), Totals 8–7 (53.3%). Mobile browser checks verified populated defaults and displayed rates. No user CSVs or live results are committed, and no Drive write or production deployment was performed.
