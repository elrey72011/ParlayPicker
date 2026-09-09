# Local daily-board publishing

This is the first publishing milestone: an explicit local build → preview →
publish workflow. It does not create another Streamlit app, call a provider,
connect a public host, or change predictions. Continue running the complete
ParlayPicker analysis with current TheOver and odds inputs in your private
workspace. TheOver CSVs alone are not sufficient to produce this public board.

## Prepare

Export Overall, Sides, and Totals from the same analysis run. Optionally export
**PRIMARY — Export Combined Player Props for Next-Day Grading** and the
DraftKings **top5_lineups** CSV (not the salary CSV or position-only entry file).
Run from the repository with its Python dependencies installed:

```powershell
python scripts/publish_board.py build --overall "C:/path/overall-per-game.csv" --sides "C:/path/sides-per-game.csv" --totals "C:/path/totals-per-game.csv" --output outputs/public-board-draft
```

Add `--props "C:/path/player_props_all_export.csv"` for props. If the export
has neither prediction_generated_at nor export_run_id, supply its actual original
analysis time using `--props-as-of "2026-09-08T16:00:00-04:00"`. Never use a new
time to make old prices look fresh. Unknown start times remain non-actionable.

For a saved DFS lineup export, add:

```powershell
--dfs "C:/path/draftkings_nfl_classic_top5_lineups.csv" --dfs-sport NFL --dfs-slate "Sunday main slate" --dfs-start "2026-09-13T13:00:00-04:00"
```

The DFS adapter supports one MLB or NFL Classic slate per package. It checks
complete unique roster slots and the $50,000 salary cap. It does not revalidate
player eligibility, current availability, projections, or exact contest rules.
Keep the optimizer's projection-source label, including historical-average
fallbacks. No contest-entry download or automatic submission is provided.

## Review

Open `outputs/public-board-draft/preview.html` in a browser. It works offline.
Game Picks contains all three tables; Player Props includes sport/market filters;
DraftKings DFS displays the chosen lineup export. Omitted inputs produce empty
tabs and do not retain yesterday's props or lineups.

Only allowlisted display fields enter `public-board.json`. Bankroll, stake
amounts, credentials, internal identifiers and diagnostics are excluded. Full
exports remain private. Check all dates, picks, labels and missing-data states.
The browser marks stale/started picks and locked slates; it never calls APIs.
Freshness uses the original analysis timestamp, not the preview build time.
Browser-clock status is a display aid, not server-side wagering authorization.

## Publish locally

After reviewing the preview:

```powershell
python scripts/publish_board.py publish --draft outputs/public-board-draft --destination outputs/public-board-site
```

This validates the saved package and atomically replaces the destination's
`index.html`. The page embeds its sanitized data, avoiding mismatched HTML/JSON
versions. The previous page is kept as `previous.html`. This command does not
publish to the internet. For a local HTTP preview, serve only the output directory:

```powershell
python -m http.server 8080 --bind 127.0.0.1 --directory outputs/public-board-site
```

Open `http://127.0.0.1:8080`. Never serve the repository root, secrets, or raw CSVs.

Rollback:

```powershell
python scripts/publish_board.py rollback --destination outputs/public-board-site
```

## Next milestone

Choose a public host/domain and connect the sanitized publication output. Public
data must be permitted by provider terms. Add authenticated delivery before
placing paid content behind a subscription: static HTML embeds readable data.
Multiple DFS slates, historical Results, and subscriptions are subsequent milestones.


## Streamlit Preview & Publish panel

Open **Workspace → Preview & Publish**. Configure an owner-only token in
Streamlit secrets (or the local environment) first:

```toml
PARLAYPICKER_PUBLISH_TOKEN = "replace-with-a-long-random-private-token"
```

Use at least 16 characters. Enter this token in the password field to unlock the
panel. This protects the publication controls; it is not a complete customer
account system and does not make the rest of Streamlit private.

Run Master Analysis as usual. The panel uses the finalized game card and its
matching candidate audit from that render. Optionally include the current prop
card. Generate DFS lineups under Full Pick Board to make that sport selectable;
then supply the exact slate label and timezone-aware lock time. Missing sections
remain empty rather than reusing an earlier slate.

Click **Build preview**, review the embedded three-tab page, then click
**Publish reviewed board locally**. Changing source data or inclusion options
invalidates the preview. HTML and sanitized JSON downloads are available without
publication. No CSV round trip or terminal command is required.

The default destination is `outputs/public-board-site` on the computer running
Streamlit. Optional `PARLAYPICKER_PUBLICATION_DIR` sets a different output folder.
On Streamlit Cloud, this is the server filesystem, NOT your laptop, and may be
lost during redeployment. Download the HTML to keep a portable copy. Publishing
still does not upload to a public host or Google Drive. The existing local CLI
rollback remains available to restore the previous publication.

## Optional public hosting

Netlify public publication is now available below the local publish controls.
It is disabled until a dedicated site ID and token are configured. See
`docs/netlify-publishing.md` for setup and the explicit public publish workflow.
The local button still writes only to the local server filesystem.

## Public parlay board

New previews include a Parlays tab with at most three two-leg combinations drawn from Overall Best Picks. Teams cannot recur anywhere in the set, including as opponents. Both game start times and analysis timestamps must be available; started games, future analysis timestamps, and analysis older than 15 minutes are excluded at build time. The browser marks saved pairs expired when their legs age or start.

Pairs with two individually approved legs rank first, then pairs rank by the product of leg probabilities. Remaining pairs may include PASS selections and are explicitly research only. All tickets remain research only: multiplied single-leg decimal prices and independence-based joint probabilities are illustrative, not verified bookmaker parlay quotes or ticket approval. No stakes are recommended and no extra API calls are made.

Version 2 public packages retain the chosen legs and calculated estimates; older version 1 drafts remain readable. Download the public JSON to retain a copy. This change does not add durable publication history or automatic parlay settlement; those are follow-up work. Build a fresh preview and explicitly publish it to update Netlify.
