# Readiness dashboard report schemas

This bounded display correction starts from actual main
`d955ceb9843c9bc320b63b29b7da9c7aa2305441`, which includes the merged coverage,
CI scheduling and recovered NCAAF compatibility work (#2401, #2403, #2402).
It changes no formulas, thresholds, sources, stakes, scientific acceptance,
wagering gates, providers, workflows or existing test assertions.

The actual dashboard previously indexed slate coverage decisions using legacy
candidate readiness columns, raising a KeyError at its dataframe call. An early
return also hid retained coverage when candidate evidence was empty. Candidate
rendering assumed an `issues` column even for an empty candidate table.

The renderer now handles independent slate coverage and legacy game readiness
explicitly. Coverage preserves canonical event/team identities, inventory/run
clocks, decision states, blocker codes, explanations and nested market evidence.
Candidate game readiness remains separately visible, including candidate games
outside the independent slate. All recorded candidate details remain available.
Missing scalar fields display `Unavailable`; no probability, pick, approval,
gate result or observation is synthesized. The JSON and Markdown report builders
are unchanged. Each CSV uses the same complete table as its dataframe, with
stable headers even when empty. Saved snapshots do not borrow current coverage.

Offline regressions invoke `render_readiness_dashboard` with only the Streamlit
surface replaced by a recorder; the actual report builder, coverage validation
and renderer execute. Inputs are labelled synthetic. The pre-change renderer
fails 7 checks and passes 1, including the reproduced hosted KeyError. These same
checks cover legacy reports, coverage, absent/empty candidates, empty reports,
display/CSV consistency, saved snapshots and malformed authority declarations.
Four additional regressions use the real Streamlit AppTest surface for both
schemas and empty inputs, including a column mixing recorded and unavailable
probabilities. No Arrow conversion fallback is needed for these tables.

The successor policy binds this exact main tree, reviewed implementation blobs,
the exact unchanged predecessor guard and prior policy, and all retained protected
files and existing tests. The implementation has one parent: the verified main.
A separate commit adds only the new policy seal. CI merge validation requires
ordered base/seal parents and an unchanged implementation tree. Unreviewed edits,
resealed changes, policy tampering and runtime shadowing fail closed. Exact
predecessor source readers retain all historical guard assertions. No existing
test exception is added. No old seal is reused. A new base requires fresh bindings,
review and CI.

Only offline checks and normal pull-request CI are authorized. No live refresh,
provider call, restoration, operational workflow, merge or deployment is part
of this correction.
