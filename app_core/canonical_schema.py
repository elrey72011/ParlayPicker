"""Versioned, side-effect-free contract for canonical remote row identities."""

from __future__ import annotations


CANONICAL_SCHEMA_VERSION = 1

# Every prospective table currently emitted by ``prospective_remote`` must be
# declared here.  Consumers fail closed for undeclared future tables instead
# of silently skipping evidence they do not understand.
CANONICAL_PRIMARY_KEYS = {
    "prospective_event": ("event_id",),
    "prospective_quote": ("quote_id",),
    "prospective_close": ("close_id",),
    "prospective_result": ("result_id",),
    "prospective_model": ("model_id",),
    "prospective_model_training_result": ("model_id", "result_id"),
    "prospective_calibration": ("calibration_id",),
    "prospective_calibration_result": ("calibration_id", "result_id"),
    "prospective_prediction": ("observation_id",),
    "prospective_validation_plan": ("validation_plan_id",),
    "prospective_validation_artifact": ("artifact_id",),
    "prospective_deployment_review": ("deployment_id",),
    "prospective_football_event": ("version_id",),
    "prospective_football_quote": ("quote_id",),
    "prospective_football_result": ("result_id",),
    "prospective_football_settlement": ("settlement_id",),
    "prospective_football_training_row": ("training_row_id",),
    "prospective_football_team_identity": ("identity_id",),
    "prospective_football_theover": ("research_row_id",),
    "prospective_football_coverage": ("coverage_id",),
    "prospective_football_cycle_coverage": ("coverage_id",),
    "prospective_reconciled_source": ("source_key",),
    "prospective_reconciled_fact": ("fact_id",),
}

RECONCILED_RESEARCH_TABLES = frozenset({
    "prospective_reconciled_source",
    "prospective_reconciled_fact",
})


def primary_key_for(table: object) -> tuple[str, ...] | None:
    """Return the exact v1 identity columns for a supported canonical table."""

    if not isinstance(table, str):
        return None
    return CANONICAL_PRIMARY_KEYS.get(table)
