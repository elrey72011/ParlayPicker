# Public website redesign

Implemented sequentially: Phase A (visual system), B (publication sidecars), C (automatic updates), then D (results). The private Streamlit theme is outside this change.

## Presentation

- Shared mint/navy theme, responsive navigation, pick cards, explicit status badges, summary metrics, and expandable explanations.
- Results show separate category summaries; overlapping categories are never summed. Existing one-unit hypothetical returns remain in a labeled disclosure, not presented as actual ROI.
- Results load 25 rows at a time, newest date first. Mobile result cards preserve original odds, estimates, scores/statistics, and outcomes.
- Existing qualification, probability order, stale-price rules, research classification, locked selections and history calculations are unchanged.

## Publication contract

`build_public_assets` validates and applies the existing current-record filter to the public package, then serializes canonical UTF-8 JSON and calculates its full SHA-256. The manifest contains `build_id`, `board_hash`, and `published_at`. JSON has the same saved data as the embedded fallback, without an additional model run.

Local publishing and SFTP atomically replace each artifact in this order: HTML, CSS, JavaScript, board JSON, version JSON. Netlify uploads the same allowlisted files as one atomic deployment, plus its existing cache headers. SFTP readiness verifies HTML and both JSON sidecars, so an interrupted publication cannot be confirmed from HTML alone.

Browser polling checks relative same-origin JSON URLs every 45 seconds using `no-store`, a cache-busting query and a 12-second timeout. Payload SHA-256 must match the advertised version before acceptance. Older manifests are rejected during partial publication; mismatches or failures retain the current board and retry on the ordinary interval. The manifest identity changes only after successful rendering. Timestamps never reset saved quote/analysis age or grant approval.

The browser initially displays embedded data, then accepts the current external publication when it differs. Embedded fallback remains intentionally enabled until production validation. CSS and the update client are also embedded for downloaded/Streamlit preview compatibility; published standalone copies are generated from the same sources. Previews do not poll. The shared shell CSS is mirrored in the checked-in board for older cached renderers; the regression suite checks this parity.

Refreshing preserves selected page, all select values, open disclosures, expanded results count, and scroll position where practical. Filter options are refreshed without resetting selections. Failed rendering restores the prior data.

CLI rollback republishes the previous embedded package with a fresh publication timestamp and matching sidecars. Immutable saved history is not modified.

## Verification and deployment

Run the existing public/history/results/qualification/publishing pytest suites. On this Windows workstation, use UTF-8 mode, a workspace `--basetemp`, and `.test-deps` (Streamlit 1.59.1); the system Streamlit lacks `st.iframe`. An installed third-party `tests` package may shadow this repository's namespace.

`tests/public_refresh.cjs` checks the update client in Node. `tests/public_site_browser.cjs PATH_TO_GENERATED_INDEX_HTML` checks live browser polling, update failures, state preservation, pagination, mobile widths and stale approval. Set `PLAYWRIGHT_MODULE` to the installed Playwright package and optionally `BROWSER_CHANNEL`; Windows defaults to Edge.

A local review build and screenshots are in `outputs/redesign-preview/`, generated from an existing saved public snapshot. Its records are historical; it is not a new analysis.

No production upload was performed. Deploy using the existing reviewed publishing workflow so HTML and all sidecars ship together. Verify same-origin `version.json` and `board-data.json` return JSON over HTTPS, check one new publication in an already-open tab, and confirm caching does not serve old JSON. Netlify's existing no-cache headers are retained. No Apache configuration was installed or assumed; any server cache-rule change requires checking the actual host. Keep the embedded fallback during production validation.

## Completed validation

- Phase A: 49 targeted public-board, research-parlay, probability-order and Top 10 tests passed.
- Phase B: 76 tests passed including payload consistency, secret exclusion and publishing paths.
- Phase C: 77 tests passed including polling/failure behavior.
- Phase D and integration: 279 public/results/history/qualification/publishing regressions passed with the compatible project dependencies.
- Final card refinement: 50 targeted tests passed again; real Edge browser checks passed for desktop/mobile layout, in-place updates, retry/fallback, filter/tab/disclosure/scroll preservation, pagination, and stale approval.
- `git diff --check` passed. No unresolved test failures in these checks.

Principal changed files: `app_core/public_site_shell.py`, `publishing/board.html`, new `publishing/site.css` and `publishing/site.js`, `scripts/publish_board.py`, `app_core/sftp_publishing.py`, and `app_core/netlify_publishing.py`. Methodology pages inherit the updated shared shell through their existing generator. Added payload/refresh/browser regressions and updated only structural assumptions in the existing rendering tests. Generated JSON is produced per publication, not committed as a static sample.

## V2 supersedes the original visual system

The warm light editorial identity and analysis-first freshness semantics now supersede the mint/navy presentation above. See `docs/public-site-v2-brand-separation.md` for the final design, conservative mixed-analysis clock handling, 285-test integration result, screenshots and deployment notes. The original publication transport remains intact.
