"""Qualification progress (roadmap C3). SYNTHETIC readiness report; no network."""
from streamlit.testing.v1 import AppTest

from core import qualification_progress as qp

REPORT = {
    "research_bindings": [
        {"sport": "NCAAB", "market_family": "SPREAD", "most_training_events": 130, "candidate_decided": 40,
         "games_per_day_14d": 28.5, "proposed_window_days": 11, "bound_plan_id": "plan-ncaab-v2",
         "next_blocker": "VALIDATION_IN_PROGRESS"},
        {"sport": "NBA", "market_family": "SPREAD", "most_training_events": 30, "candidate_decided": 0,
         "games_per_day_14d": 6.0, "proposed_window_days": 45, "bound_plan_id": None,
         "next_blocker": "CANDIDATE_NEEDS_120_TRAINING_EVENTS"},
        {"sport": "NHL", "market_family": "PUCK_LINE", "next_blocker": "SCORE_SETTLEMENT_NOT_ENABLED"},
    ],
    "plans": [{"validation_plan_id": "plan-ncaab-v2", "status": "VALIDATION_PENDING",
               "validation": {"scored_observations": 50, "unique_events": 55}}],
    "price_edge": [{"sport": "NCAAB", "market_family": "SPREAD",
                    "by_threshold": {"0.02": {"picks": 12, "beat_close_share": 0.58}}}],
}


def test_rows_report_each_gate_as_a_fraction():
    rows = {(r["sport"], r["market"]): r for r in qp.rows(REPORT)}
    ncaab = rows[("NCAAB", "SPREAD")]
    assert ncaab["stage"] == "Validating" and ncaab["training_progress"] == 1.0
    assert ncaab["calibration_progress"] == 1.0 and ncaab["validation_progress"] == 0.25
    assert ncaab["price_screen_picks"] == 12 and ncaab["price_screen_beat_close"] == 0.58
    nba = rows[("NBA", "SPREAD")]
    assert nba["stage"] == "Collecting training games" and nba["training_progress"] == 0.25
    assert rows[("NHL", "PUCK_LINE")]["stage"] == "Settlement off"
    assert qp.summary(qp.rows(REPORT)).startswith("Closest to qualifying: NCAAB SPREAD, validating")
    assert qp.rows({}) == [] and qp.summary([]) == "No research markets in this report."


def test_panel_renders_a_loaded_report():
    import json
    script = ("import json\nimport streamlit as st\n"
              "from app.ui.qualification_progress import render_qualification_progress\n"
              "from core.qualification_progress import rows\n"
              f"st.session_state['qualification_progress'] = rows(json.loads({json.dumps(json.dumps(REPORT))}))\n"
              "render_qualification_progress()\n")
    at = AppTest.from_string(script)
    at.run(timeout=30)
    assert not at.exception
    assert any("Closest to qualifying: NCAAB SPREAD" in m.value for m in at.markdown)
    assert len(at.dataframe) == 1


def test_latest_report_reads_the_newest_artifact_zip():
    import io
    import json
    import zipfile

    def archive(report):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as bundle:
            bundle.writestr("market-readiness.json", json.dumps(report))
            bundle.writestr("market-readiness.md", "| table |")
        return buffer.getvalue()

    class Response:
        def __init__(self, payload=None, content=b""):
            self.payload, self.content = payload, content

        def raise_for_status(self):
            pass

        def json(self):
            return self.payload

    calls = []

    def get(url, **kwargs):
        calls.append((url, kwargs.get("headers", {}).get("Authorization")))
        if url.endswith("/artifacts"):
            return Response({"artifacts": [
                {"created_at": "2026-11-04T11:20:00Z", "archive_download_url": "old", "expired": False},
                {"created_at": "2026-11-05T11:20:00Z", "archive_download_url": "new", "expired": False},
                {"created_at": "2026-11-06T11:20:00Z", "archive_download_url": "gone", "expired": True}]})
        return Response(content=archive({"research_bindings": [], "from": url}))

    report, created = qp.latest_report("owner/repo", "t0ken", get)
    assert report["from"] == "new" and created == "2026-11-05T11:20:00Z"
    assert all(auth == "Bearer t0ken" for _, auth in calls)
