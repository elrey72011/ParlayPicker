# Lock eligibility diagnostics

The lock panel reports one status per current overall-board row and offers a downloadable CSV under **Why games cannot be locked**. The breakdown uses a single check time. Existing locks are compared by the same identity as locking; they keep their original selections and prices. Locks outside the current board are not counted in this board breakdown.

A started game means its recorded scheduled start is at or before the check time; this is not an independent live-score confirmation. Other exclusions distinguish missing quotes, stale quote or analysis timestamps, invalid timing or markets, other game dates, and duplicate board entries. The existing locking rules remain authoritative, including ambiguity rejection and the 30-minute limit.

## September 12 feed diagnosis

The owner-provided candidate export `best_picks_candidate_audit - 2026-09-12T133726.357.csv` contains 482 candidates for 127 matchup IDs (15 MLB, 112 NCAAF), from analysis run `20260912T173644.527516Z`.

At the export's 1:37:26 PM Eastern timestamp, 20 games had recorded start times that had passed, 74 upcoming games had at least one exact supported timestamped quote, and 33 upcoming college games had no verifiable quote. These categories sum to 127; quote availability alone does not prove a game is newly lockable.

All 33 missing-quote games had DraftKings offers from `espn_ncaaf_fcs_scoreboard` with null `recorded_at` values. The public ESPN scoreboard response checked during diagnosis also omitted quote-update timestamps. The application cannot verify their age, even though prices exist. The board now explains this specific failure instead of implying that the sportsbook has no line. No retrieval timestamp is substituted for a provider quote timestamp.

The candidate CSV does not contain saved lock records. It cannot establish the exact relationship to the earlier screenshot's 81 locks. The new panel performs that comparison using the owner's saved history. Older public-board snapshots are not substitutes for the current owner session.

Timestamped coverage from a supported provider is still required to make the 33 untimestamped games lockable. Refreshing alone cannot repair absent provider timestamps. This diagnostic change does not supply that missing feed coverage or alter saved locks.
