# Live producer provenance trace — September 18

The latest downloaded scheduled report (run 35380052694) contains zero eligible games. Its latest MLB cohort has 60 candidate rows missing model/calibration provenance, verified identity, conservative probability, evidence/policy versions, and outcomes. These are candidate counts, not game counts.

Trace findings:
- attach_challenger in app_core/mlb_spread_total_model.py returns NOT_CONFIGURED without PARLAYPICKER_MLB_CHALLENGER_MODEL. Even a configured challenger is research-only and keeps its model facts inside mlb_challenger_result. They cannot truthfully be assigned to baseline probabilities.
- core/streamlit_pipeline.py records calibration provenance only for the calibration path actually consumed. Stale bucket history bypasses that path.
- authority_projection and project preserve supplied model/calibration/policy fields. Regression coverage now proves this and verifies that absent evidence remains absent.
- selection_probability_source was not part of the canonical FIELDS allowlist. Preserve the supplied basis in new evidence projections so producer diagnostics can distinguish the probability path without inferring facts. Historical rows are untouched.
- ranking rebuild already runs after scheduled grading. It now separates model configuration, baseline provenance, calibration, identity, uncertainty, policy/evidence, and settlement requirements. No active ranking or wager gate changes.

Remaining work requires real artifacts: train and verify a model from admissible receipts, evaluate matching calibration prospectively, and bind approved producer outputs to the exact selection. A missing model cannot be fixed by copying challenger metadata or a code hash. Fresh outcomes alone do not satisfy these prerequisites. The current stale warning remains accurate.
