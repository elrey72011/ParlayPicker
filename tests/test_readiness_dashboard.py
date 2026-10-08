"""Exercise the actual dashboard with retained SYNTHETIC reports, fully offline."""
from contextlib import nullcontext
from copy import deepcopy
from io import StringIO
import json

import pandas as pd
import pytest

from app.ui import readiness_dashboard as panel
from core.run_readiness import build_readiness
from test_prediction_evidence import fixture_frames
from test_slate_coverage import inventory, report as coverage_report


class UI:
    def __init__(self, source="Current run", saved=None):
        self.session_state = {"readiness_snapshots": saved or []}
        self.source = source
        self.frames = []
        self.downloads = {}
        self.messages = []

    def expander(self, *args, **kwargs):
        return nullcontext()

    def button(self, *args, **kwargs):
        return False

    def selectbox(self, label, options, **kwargs):
        return self.source if label == "Readiness source" else options[0]

    def dataframe(self, frame, **kwargs):
        self.frames.append(pd.DataFrame(frame).copy(deep=True))

    def download_button(self, label, data, file_name=None, mime=None, **kwargs):
        self.downloads[file_name] = data

    def __getattr__(self, name):
        return lambda *args, **kwargs: self.messages.append((name, args))


def render(monkeypatch, audit=None, final=None, diagnostics=None, ui=None):
    ui = ui or UI()
    monkeypatch.setattr(panel, "st", ui)
    panel.render_readiness_dashboard(audit, final, diagnostics)
    return ui


def csv_frame(ui, filename):
    return pd.read_csv(StringIO(ui.downloads[filename]), keep_default_na=False)


def assert_csv_matches_display(ui, filename, identity):
    downloaded = csv_frame(ui, filename)
    displayed = next(f for f in ui.frames if identity in f.columns)
    assert list(downloaded.columns) == list(displayed.columns)
    # Compare through CSV parsing to account for serialized types and empty cells.
    pd.testing.assert_frame_equal(downloaded, pd.read_csv(
        StringIO(displayed.to_csv(index=False)), keep_default_na=False))
    return displayed


def test_legacy_renderer_preserves_readiness_and_all_candidate_details(monkeypatch):
    audit, final = fixture_frames()
    audit["snapshot_id"] = final["snapshot_id"] = "SYNTHETIC-snapshot"
    before_audit, before_final = audit.copy(deep=True), final.copy(deep=True)
    expected = build_readiness(audit, final)
    ui = render(monkeypatch, audit, final)
    games = assert_csv_matches_display(ui, "game-readiness.csv", "readiness")
    candidates = assert_csv_matches_display(ui, "candidate-readiness.csv", "issues")
    assert games["matchup_id"].tolist() == ["game-1"]
    assert games["selected_pick"].tolist() == ["Over 8.5"]
    assert games["wager_decision"].tolist() == ["approved"]
    assert games["production_probability"].tolist() == ["Unavailable"]
    assert candidates["pick"].tolist() == ["Over 8.5", "Under 8.5"]
    assert "independent_model_probability" in candidates
    assert "ml_unavailable_reason" in candidates
    assert json.loads(ui.downloads["run-readiness.json"]) == expected
    assert "Read-only diagnostics" in ui.downloads["run-readiness.md"]
    pd.testing.assert_frame_equal(audit, before_audit)
    pd.testing.assert_frame_equal(final, before_final)


def test_coverage_renderer_preserves_identities_states_reasons_and_readiness(monkeypatch):
    audit, final = fixture_frames()
    audit["snapshot_id"] = final["snapshot_id"] = "SYNTHETIC-snapshot"
    coverage = coverage_report()
    before = deepcopy(coverage)
    ui = render(monkeypatch, audit, final, {"slate_coverage": coverage})
    games = assert_csv_matches_display(ui, "game-readiness.csv", "canonical_event_id")
    assert games["canonical_event_id"].tolist() == ["nfl:one"]
    assert games["coverage_decision_state"].tolist() == ["UNVERIFIED"]
    assert json.loads(games.iloc[0]["blocker_codes"]) == coverage["decisions"][0]["blocker_codes"]
    assert games.iloc[0]["home_team_id"] == "h"
    assert json.loads(games.iloc[0]["market_results"]) == coverage["decisions"][0]["market_results"]
    # Candidate readiness remains visible even for games outside the independent slate.
    readiness = assert_csv_matches_display(ui, "candidate-game-readiness.csv", "readiness")
    assert readiness["matchup_id"].tolist() == ["game-1"]
    assert_csv_matches_display(ui, "candidate-readiness.csv", "issues")
    metrics = json.loads(ui.downloads["run-readiness.json"])
    assert metrics["slate_coverage"] == coverage == before
    assert metrics["production_changes"] is False
    assert any("Coverage decisions" in str(args) for _, args in ui.messages)


@pytest.mark.parametrize("audit", [None, pd.DataFrame()])
def test_coverage_with_empty_candidates_still_renders_and_downloads(monkeypatch, audit):
    coverage = coverage_report()
    ui = render(monkeypatch, audit, diagnostics={"slate_coverage": coverage})
    games = assert_csv_matches_display(ui, "game-readiness.csv", "canonical_event_id")
    assert games["canonical_event_id"].tolist() == ["nfl:one"]
    assert "selected_pick" not in games and "wager_decision" not in games
    assert csv_frame(ui, "candidate-readiness.csv").empty
    assert "issues" in csv_frame(ui, "candidate-readiness.csv")
    metrics = json.loads(ui.downloads["run-readiness.json"])
    assert metrics["games"] == metrics["candidates"] == []
    assert metrics["counts"]["approved_wagers"] == 0
    assert metrics["slate_coverage"] == coverage
    assert any("No candidate evidence" in str(args) for _, args in ui.messages)


@pytest.mark.parametrize("coverage", [None] + [
    coverage_report(inventory(events=[], status=status))
    for status in ("COMPLETE", "PARTIAL", "UNAVAILABLE")
])
def test_empty_reports_keep_typed_headers_and_honest_counts(monkeypatch, coverage):
    diagnostics = {"slate_coverage": coverage} if coverage is not None else None
    ui = render(monkeypatch, pd.DataFrame(), diagnostics=diagnostics)
    metrics = json.loads(ui.downloads["run-readiness.json"])
    assert metrics["status"] == "no_candidate_evidence"
    assert metrics["counts"] == {"games": 0, "ready_for_grading": 0, "approved_wagers": 0}
    identity = "canonical_event_id" if coverage is not None else "readiness"
    assert assert_csv_matches_display(ui, "game-readiness.csv", identity).empty
    assert csv_frame(ui, "candidate-readiness.csv").empty
    if coverage is not None:
        assert any("Scheduled events: 0" in str(args) for _, args in ui.messages)
        assert metrics["slate_coverage"]["inventory_status"] == coverage["inventory_status"]
        assert metrics["slate_coverage"]["fully_reconciled"] is (coverage["inventory_status"] == "COMPLETE")


def test_empty_independent_slate_keeps_nonempty_candidate_evidence(monkeypatch):
    audit, final = fixture_frames()
    coverage = coverage_report(inventory(events=[]))
    ui = render(monkeypatch, audit, final, {"slate_coverage": coverage})
    assert assert_csv_matches_display(ui, "game-readiness.csv", "canonical_event_id").empty
    readiness = assert_csv_matches_display(ui, "candidate-game-readiness.csv", "readiness")
    assert not readiness.empty
    assert len(assert_csv_matches_display(ui, "candidate-readiness.csv", "issues")) == len(audit)
    metrics = json.loads(ui.downloads["run-readiness.json"])
    assert metrics["counts"]["games"] == 2
    assert metrics["slate_coverage"]["counts"]["scheduled_events"] == 0
    assert any("Candidate games: 2" in str(args) for _, args in ui.messages)
    assert any("Scheduled events: 0" in str(args) for _, args in ui.messages)


def test_saved_legacy_snapshot_does_not_borrow_current_coverage(monkeypatch):
    audit, final = fixture_frames()
    ui = UI(saved=[("SYNTHETIC-old", audit, final)])
    render(monkeypatch, audit, final, {"slate_coverage": coverage_report()}, ui=ui)
    assert "slate_coverage" in json.loads(ui.downloads["run-readiness.json"])
    ui.source = "Saved snapshot"
    ui.frames.clear()
    ui.downloads.clear()
    render(monkeypatch, diagnostics={"slate_coverage": coverage_report()}, ui=ui)
    metrics = json.loads(ui.downloads["run-readiness.json"])
    assert "slate_coverage" not in metrics
    assert "canonical_event_id" not in csv_frame(ui, "game-readiness.csv")


def test_malformed_coverage_still_fails_validation(monkeypatch):
    coverage = coverage_report()
    coverage["wagering_authority"] = True
    with pytest.raises(ValueError, match="COVERAGE_AUTHORITY_FORBIDDEN"):
        render(monkeypatch, diagnostics={"slate_coverage": coverage})


@pytest.mark.parametrize("scenario,rows", [
    ("legacy", [2, 2]), ("coverage", [1, 2, 2]),
    ("coverage_without_candidates", [1, 0, 0]), ("empty", [0, 0]),
])
def test_real_streamlit_surface_renders_both_schemas_and_empty_reports(scenario, rows):
    from streamlit.testing.v1 import AppTest
    source = "\n".join([
        "from test_prediction_evidence import fixture_frames",
        "from test_slate_coverage import report as coverage_report",
        "from app.ui.readiness_dashboard import render_readiness_dashboard",
        "audit, final = fixture_frames()" if scenario in {"legacy", "coverage"} else "audit = final = None",
        "diagnostics = {'slate_coverage': coverage_report()}" if scenario.startswith("coverage") else "diagnostics = None",
        "render_readiness_dashboard(audit, final, diagnostics)",
    ])
    app = AppTest.from_string(source).run(timeout=30)
    assert not app.exception
    assert [len(frame.value) for frame in app.dataframe] == rows
    if scenario in {"legacy", "coverage"}:
        games = app.dataframe[0 if scenario == "legacy" else 1].value
        # One synthetic game lacks selection evidence; its missing probability
        # must remain unavailable next to the other game's recorded value.
        assert games["market_probability"].tolist() == ["0.5", "Unavailable"]
        assert games["selected_pick"].tolist() == ["Over 8.5", "Unavailable"]
