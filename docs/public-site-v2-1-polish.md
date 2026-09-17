# Public site V2.1 — editorial polish

Completed the constrained V2.1-A → B → C → D sequence.

## Changes

- Brand orange `#A84A16`, dark orange `#8F3D12`, and soft orange `#F1D9C8` replace the V2 brand tokens. Danger remains `#B42318`. Required text/background contrast pairs pass 4.5:1.
- System-serif display headings are limited to page H1s and the board hero. Navigation, controls, scorecards, odds, metrics and explanatory text remain sans-serif. No fonts or dependencies were added.
- Card shadows use the flatter two-layer values from the specification.
- One Overall / Sides / Totals research grid renders at a time. Overall is the default for a new visitor. Valid saved view preferences restore safely; blocked storage does not prevent switching.
- The selected research view uses the existing probability order, league filter and qualifiedPick predicate. Empty states name the selected view. Approved Picks remain all-league and separate. Top 10 remains collapsed.
- Research-view state is captured/restored during updates, including focus on the switcher button; existing filter, disclosure and scroll preservation remains.

## Files changed for V2.1

- `publishing/site.css`: brand/shadow tokens, font stacks, display headings and responsive switcher.
- `publishing/board.html`: CSS mirror, single-view renderer, state and controls.
- `tests/test_public_brand.py`: colors, contrast, typography and presentation contracts.
- `tests/test_probability_display_order.py`: verifies one research view plus props and unchanged ordering across all three views.
- `tests/test_public_board.py`: concise qualified-empty-state assertion.
- `tests/public_site_browser.cjs`: view switching, state persistence, filters, storage fallback and screenshots.

The shared shell and methodology generator consume the canonical CSS without additional production changes.

## Verification

- Token/contrast phase: 6 tests passed.
- Typography phase: desktop/mobile browser checks passed.
- Research-view phase: 56 targeted tests passed.
- Expanded brand/order tests: 9 passed.
- Integration: 287 public/results/history/qualification/publishing tests passed using the existing `.test-deps` test environment.
- Browser: one research grid, Overall/Sides/Totals, league and qualified-only filtering, unchanged Approved Picks, collapsed Top 10, update failures/retries, selected-view/filter/focus/disclosure/scroll preservation, reload preference, blocked storage, pagination, stale status, old-analysis freshness, desktop/mobile widths and same-origin-only requests passed.
- Protected code comparison confirmed qualifiedPick, state, supportedQuote, probabilityOrder, Top 10 ranking, analysisTimestamp/analysisFreshness and `publishing/site.js` are unchanged from V2.
- `git diff --check` passed. No unresolved failures.

## Review and deployment

Local preview: `outputs/v21-preview/index.html` (saved historical public data, not a new analysis).
Methodology: `outputs/v21-preview/how-picks-work/index.html`.
Screenshots: `outputs/v21-preview/screenshots/`, covering approved/zero-approved desktop Picks, Sides, Totals, mobile Overall/Sides, Results, Parlays and old analysis. Browser screenshots use deterministic synthetic fixtures, not new recommendations.

No production deployment was performed. Publish through the existing workflow. The 30-second rendering timer is intentionally unchanged; its optional optimization remains separate from V2.1.
