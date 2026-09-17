# Public site V2 — brand separation

Implemented V2-A → V2-B → V2-C → V2-D sequentially on top of the prior redesign.

## Visible changes

- Warm ivory canvas, white scorecards, dark ink, burnt-orange logo/links and underlined active navigation. No mint/teal/purple brand palette; green is reserved for positive semantics. Locked records use a neutral badge because locking is not approval.
- The board leads with the current approved-play count and its approved selections, followed by the existing Top 10 in a disclosure. Zero-approved boards use a sentence rather than a giant zero. Games reviewed and Not approved are secondary; Not approved is total overall rows minus currently qualifying rows, including expired/unavailable saved approvals.
- Neutral odds, restrained status badges, compact metric labels, and a parlay title/status row with ticket assumptions in a disclosure.
- Results retain category summaries, collapsed details, 25-row pagination, mobile cards and clearly labeled hypothetical returns.
- One site header landmark; page headings are labeled sections; content sections no longer claim tabpanel roles. Orange focus rings remain visible.
- The methodology shell and sandboxed visualization share the public light palette, including when the visitor's browser prefers dark mode. Source content and interactive calculations are preserved.

## Freshness decision

Analysis and publication are separate clocks. The header uses saved row `as_of` timestamps, never quote timestamps or a newer package-building time when analysis rows exist. For mixed cohorts it conservatively uses the oldest represented analysis, preventing fresh props from masking old games. An invalid or future row analysis timestamp produces Analysis time unavailable; a board without any rows falls back to `built_at`.

This deliberately tightens the document's illustrative `max(built_at, row timestamps)` helper: that example could still label old analysis as fresh when rebuilding a package. A separate site-publication age appears when the two clocks differ by over ten minutes. Poll failures append “update check unavailable” without replacing the analysis age. Republish and polling do not grant approval or modify saved history.

## Preserved behavior

Same-origin JSON fetches, 45-second polling, no-store/cache-busting, timeouts, SHA-256 verification, schema checks, older-manifest rejection, embedded fallback, state preservation, version-last publishing, and failure rollback remain in place. Model outputs, qualification, stale-price rules, historical results and locked records are unchanged. No browser provider requests were added.

Optional V2-E (targeted timer refresh) is deferred. The existing 30-second render/state-preservation path remains for this first visual release. The private Streamlit theme is unchanged.

## Files

- `publishing/site.css`: canonical public design system.
- `app_core/public_site_shell.py`: shared CSS and analysis-freshness header.
- `publishing/board.html`: mirrored CSS, hero, analysis helpers, cards and semantic markup.
- `publishing/site.js`: analysis-first freshness display; existing verified transport retained.
- `scripts/prepare_methodology_page.py`: matching shell and sandbox palette.
- `tests/test_public_brand.py`: brand, contrast, semantics, methodology, mixed/invalid analysis times and current eligibility counts.
- `tests/public_refresh.cjs`, `tests/public_site_browser.cjs`, `tests/test_probability_display_order.py`: updated transport/browser assertions and display test harness.

## Verification

- V2-A and V2-B: 49 targeted regressions passed after each phase.
- V2-C: 50 targeted/transport tests passed.
- Integration: 285 public/results/history/qualification/publishing regressions passed using `.test-deps` (compatible Streamlit), UTF-8 mode and workspace temporary directories.
- Final UI refinements: 56 targeted tests passed; real Edge browser tests passed again.
- Browser coverage: desktop/mobile, forced dark browser preference with light public output, approved and zero-approved hero, old analysis republished recently, failures/retries, page/filter/disclosure/scroll preservation, pagination, stale approval and same-origin-only requests.
- Methodology browser checks: light outer shell and sandbox, interactive selection, mobile width and no script errors.
- `git diff --check` passed.
- Contrast testing found the proposed neutral badge pair below AA normal-text contrast. Neutral badge text uses the darker muted-ink token instead; all tested body/link/status pairs pass 4.5:1.

## Review artifacts and remaining deployment work

`outputs/brand-v2-preview/index.html` is a local preview generated from an existing historical public snapshot, with matching JSON sidecars. `outputs/brand-v2-preview/how-picks-work/index.html` is the themed methodology preview.

Screenshots under `outputs/brand-v2-preview/screenshots/` cover desktop/mobile picks, results, research parlays, zero-approved and old-analysis states, plus methodology. Board screenshots use deterministic synthetic browser fixtures to exercise those states; they are not recommendations or newly published analysis.

No production upload was performed. Publish through the existing reviewed workflow so the HTML, CSS, JavaScript and sidecars ship together. Retain the embedded fallback while verifying the production host's JSON caching and a live update in an already-open tab.
