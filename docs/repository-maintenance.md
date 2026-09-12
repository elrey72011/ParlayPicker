# Repository maintenance

## Maintained entry points

Use `streamlit_app.py` for Streamlit, `python -m pytest` for the automated suite,
and the documented scripts referenced by the two GitHub workflows. The unused
`app/legacy_streamlit_app.py` was removed after checking tracked references,
including application imports and deployment workflows. Git history retains it.
Root-level diagnostics remain available for explicit manual use; relocating them
requires checking imports and relative data paths first.

## Local outputs versus tracked inputs

The root `output/`, `outputs/`, `.codex-export-audit/`, `test-results/`, and runtime
`data/parlay_log/` folders are ignored. This is an ignore-policy change, not a
removal or migration of their contents. Tracked editor settings under `.idea/`
were removed from the Git index; local editor files are retained.

The September 12 inventory found about 691 MB under `outputs/` and 169 MB under
`output/`. Most was research material: two AI Analysis capture folders accounted
for about 671 MB, with additional MLB collection and evaluation datasets. Do not
blindly delete these directories. Review and back up useful datasets and reports
before deleting individually identified temporary outputs. Test scratch folders
are reproducible, but research results may not be.

Keep curated `data/`, `models/`, calibration JSON files, test fixtures, and stored
history. The current broad JSON/model ignore rules have explicit exceptions for
calibration inputs; changing them needs a separate inventory so neither private
runtime data nor new model artifacts are mishandled.

The legacy-named `PARLAYPICKER_NETLIFY_SITE_ID` remains the history namespace used
by Namecheap/SFTP deployments. Do not rename or remove it as cosmetic cleanup.
See [Namecheap setup](namecheap-publishing.md). Archived publication/lock schema
support likewise remains necessary for restoring existing records.

## Follow-up work, in separate changes

1. Establish a tested dependency set for the deployment platform and CI. Resolve
   the full dependency graph in a clean environment, check supported native
   NumPy/XGBoost/scikit-learn combinations, and run CI before tightening versions
   or splitting runtime/research requirements. Do not remove packages merely
   because their imports occur only in optional provider paths.
2. Extract coherent portions of `core/streamlit_pipeline.py` incrementally. Start
   with one boundary (such as provider collection) while preserving import
   compatibility, ranking behavior, saved evidence, and existing tests. Measure
   runtime before claiming a performance improvement.
3. Inventory root diagnostics and historical exports for later organization;
   check consumers before moving files or changing paths.

This cleanup does not change model weights, pick ranking, quote freshness,
publication/locking semantics, grading, dependency versions, or CI job gates.
The September 12 CI timing work already split the full suite across isolated
runners. Untracked local output removal does not make GitHub CI faster.
