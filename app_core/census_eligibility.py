"""Ephemeral eligibility projection of verified canonical object facts.

No evidence database is opened, initialized, restored or migrated. The SQL below
is the canonical football view/manifest rule, verified against writer databases
by real run_census regressions. Non-football qualification remains explicit.
"""
from contextlib import closing
import sqlite3

VERSION = "census-active-manifest-v1"
SELECTION_RULE = "football-one-observation-v1"
PROJECTION = {
    "prospective_football_training_row": (
        "training_row_id", "game_id", "sport", "market_family", "quote_id",
        "result_id", "settlement_id", "training_row_status"),
    "prospective_football_result": (
        "result_id", "game_id", "home_score", "away_score", "available_at"),
    "prospective_football_quote": (
        "quote_id", "event_version_id", "quote_verified", "observed_at",
        "capture_horizon", "sportsbook", "selection"),
    "prospective_football_event": ("version_id", "game_id", "scheduled_start"),
    "prospective_football_settlement": ("settlement_id", "quote_id", "result_id"),
}
ACTIVE_SQL = """
SELECT t.* FROM prospective_football_training_row t
JOIN prospective_football_result r ON r.result_id=t.result_id
WHERE t.training_row_status='TRAINING_READY'
AND NOT EXISTS (
    SELECT 1 FROM prospective_football_result revision
    WHERE revision.game_id=t.game_id
      AND (revision.home_score<>r.home_score OR revision.away_score<>r.away_score)
)
"""
ELIGIBLE_SQL = """
SELECT t.*, ROW_NUMBER() OVER (
    PARTITION BY t.sport, t.market_family, t.game_id
    ORDER BY CASE q.capture_horizon WHEN 'FINAL_LEGAL_PREGAME' THEN 0
             WHEN 'MID_PREGAME' THEN 1 WHEN 'EARLY_RESEARCH' THEN 2 ELSE 3 END,
             q.observed_at DESC, q.sportsbook COLLATE BINARY,
             q.selection COLLATE BINARY, q.quote_id COLLATE BINARY,
             t.training_row_id COLLATE BINARY
) AS choice_rank
FROM active t
JOIN prospective_football_quote q ON q.quote_id=t.quote_id
JOIN prospective_football_result r ON r.result_id=t.result_id
JOIN prospective_football_settlement s ON s.settlement_id=t.settlement_id
JOIN prospective_football_event e ON e.version_id=q.event_version_id
WHERE q.quote_verified=1 AND q.observed_at<e.scheduled_start
  AND r.available_at>q.observed_at
  AND s.quote_id=q.quote_id AND s.result_id=r.result_id
  AND NOT EXISTS (
      SELECT 1 FROM prospective_football_event revision
      WHERE revision.game_id=t.game_id AND revision.scheduled_start<=q.observed_at
  )
"""


def project(row, table):
    columns = PROJECTION.get(table)
    return {key: row.get(key) for key in columns} if columns else None


def eligibility_by_scope(facts, *, canonical_complete):
    """Numbers are known only for a complete, current canonical projection."""
    tables = {table: [] for table in PROJECTION}
    missing = False
    for fact in facts:
        table = fact.get("record_type")
        if table in tables:
            row = fact.get("eligibility_row")
            if not isinstance(row, dict):
                missing = True
            else:
                tables[table].append(row)
    if not canonical_complete or missing:
        return {}, ("CANONICAL_ELIGIBILITY_PROJECTION_MISSING" if missing
                    else "CANONICAL_CENSUS_INCOMPLETE")
    output = {}
    with closing(sqlite3.connect(":memory:")) as db:
        db.row_factory = sqlite3.Row
        for table, columns in PROJECTION.items():
            # Preserve the writer's TEXT/INTEGER affinities and binary ordering.
            types = {"quote_verified": "INTEGER", "home_score": "INTEGER", "away_score": "INTEGER"}
            definition = ",".join(f'"{key}" {types.get(key, "TEXT")}' for key in columns)
            db.execute(f'CREATE TABLE "{table}" ({definition})')
            db.executemany(f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in columns)})',
                           [tuple(row.get(key) for key in columns) for row in tables[table]])
        db.execute("CREATE VIEW active AS " + ACTIVE_SQL)
        db.execute("CREATE VIEW eligible AS " + ELIGIBLE_SQL)
        for sport in ("NFL", "NCAAF"):
            for market in ("SPREAD", "TOTAL"):
                parameters = (sport, market)
                where = "WHERE sport=? AND market_family=?"
                raw = db.execute("SELECT game_id FROM prospective_football_training_row "
                                 + where + " AND training_row_status='TRAINING_READY'", parameters).fetchall()
                active = db.execute("SELECT game_id FROM active " + where, parameters).fetchall()
                eligible = db.execute("SELECT game_id FROM eligible " + where, parameters).fetchall()
                independent = db.execute(
                    "SELECT training_row_id FROM eligible " + where
                    + " AND choice_rank=1 ORDER BY training_row_id", parameters).fetchall()
                output[f"{sport}/{market}"] = {
                    "stored_ready_rows": len(raw),
                    "stored_ready_games": len({row[0] for row in raw}),
                    "active_training_rows": len(active),
                    "active_eligible_rows": len(eligible),
                    "active_eligible_games": len({row[0] for row in eligible}),
                    "independent_eligible_games": len(independent),
                    "manifest_ids": [SELECTION_RULE + ":" + row[0] for row in independent],
                }
    return output, None


def scope_eligibility(scope, results, reason):
    sport = scope.split("/")[0]
    supported = sport in {"NFL", "NCAAF"}
    reason = reason if supported else "NO_REMOTE_CANONICAL_TRAINING_MANIFEST_READER_FOR_SCOPE"
    counts = results.get(scope, {})
    fields = ("stored_ready_rows", "stored_ready_games", "active_training_rows",
              "active_eligible_rows", "active_eligible_games", "independent_eligible_games")
    reader = ("prospective_football_training_manifest" if supported else
              "app_core.prospective_evidence.market_readiness")
    return {
        "counts": {**{key: counts.get(key) for key in fields},
                   "independent_eligible_reason": reason},
        "eligibility": {
            "projection_version": VERSION,
            "status": "KNOWN" if counts else "UNKNOWN",
            "reason": reason,
            "reader": reader,
            "selection_rule_version": SELECTION_RULE if supported else None,
            "manifest_ids": counts.get("manifest_ids"),
            "follow_up": None if counts else
                "OWNER_AUTHORIZED_READ_ONLY_EXACT_SCOPE_QUALIFICATION_REPORT",
            "active_training_rows_definition": "canonical active view before quote/settlement/event chronology",
            "active_eligible_rows_definition": "canonical manifest joins and exclusions before independent selection",
        },
        "cohorts": {
            name: {"independent_games": None, "status": "UNKNOWN",
                   "reason": "BOUND_COHORT_NOT_EVALUATED_FROM_REMOTE_INVENTORY",
                   "reader": ("app_core.football_stage2.chronological_partitions" if supported and name in
                              {"training", "selection", "calibration"} else
                              "app_core.prospective_evidence.evaluate_validation_plan"),
                   "follow_up": "OWNER_AUTHORIZED_READ_ONLY_BOUND_MODEL_PLAN_COHORT_REPORT"}
            for name in ("training", "selection", "calibration", "untouched_evaluation")
        },
        "qualification_binding": {
            "status": "NOT_VERIFIED", "production_eligible": False,
            "reader": "app_core.prospective_evidence.deployment_state",
            "reason": "INVENTORY_PRESENCE_IS_NOT_EXACT_MODEL_CALIBRATION_PLAN_VALIDATION_BINDING",
        },
    }

