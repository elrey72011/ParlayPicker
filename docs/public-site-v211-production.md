# V2.1.1 production promotion — September 17, 2026

The completed public redesign was present only as uncommitted workspace changes. GitHub/main still served the legacy template. This promotion tracks the canonical template, shared CSS, publication client, renderer, and local/SFTP/Netlify publishing paths together.

Production files:
- `publishing/board.html`: warm ivory/white/ink/orange design, approved-first hero, analysis freshness, one labeled Overall/Sides/Totals research view, heading cleanup and accessibility.
- `publishing/site.css`: canonical shared public stylesheet, mirrored in the standalone template and checked for parity.
- `publishing/site.js`: existing verified polling, hash checking, fallback and preserved state.
- `app_core/public_site_shell.py`: shared public shell and canonical stylesheet.
- `scripts/publish_board.py`: production rendering, verified sidecars, ordered atomic writes and rollback; fingerprint all rendering sources.
- `app/ui/publish_panel.py`: invalidate saved previews when any production rendering source changes, including CSS/client.
- `app_core/sftp_publishing.py`, `app_core/netlify_publishing.py`: deploy the matching publication bundle; SFTP verifies sidecars and writes the version marker last.
- `scripts/prepare_methodology_page.py`: matching shared public identity.

Qualification, probability ranking, saved results/history, quote expiration and analysis timestamps are preserved. No analysis refresh was performed.

## Current-package verification

Source: the embedded package from https://picks.cmsvconsulting.com/, retrieved September 17. This is the freshly published September 17 analysis, not the historical September 12 fixture.

- Analysis: `2026-09-17T19:11:02.749212+00:00` (3:11:02 p.m. Eastern).
- Package built: `2026-09-17T19:20:03.711501+00:00`.
- 16 games per view, 76 props, no DFS lineups, 839 result records.
- Preview: `outputs/production-v211-sept17/index.html`.
- The decoded embedded package and `board-data.json` equal the source package without modification.
- Embedded manifest equals `version.json`; its build ID and board hash both equal SHA-256 of the exact JSON bytes: `3b97ec831f844030ccb1feab22106a95f3ca5c4530d8a627931b11e490714b33`.
- `verification.json` and actual desktop/mobile screenshots are next to the preview (local artifacts, not committed).

## Validation

- 298 affected Python regressions passed, including Streamlit preview/local publish, qualification, ranking, history, freshness, SFTP/Netlify and sidecar integrity.
- `tests/public_site_browser.cjs`: passed live refresh, fallback/retry, preserved state, pagination, desktop/mobile layout, stale approval and all public navigation.
- `tests/public_bundle_browser.cjs`: passed against the actual generated September 17 bundle without substituting fixture data; verifies embedded data/manifest and desktop/mobile warm design and research view controls.
- `git diff --check`: clean.
- Production template contains none of `color-scheme: dark`, `#0b111b`, or `#6fe1ba` (case insensitive).

The generated local preview does not replace the live hosted board. After the production source is updated, Streamlit Publish board generates the new identity for the currently loaded package.
