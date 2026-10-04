"""Supplied prospective transport through actual inference/candidate/export/browser."""
from copy import deepcopy
from io import StringIO
import json
import pandas as pd
import pytest
from app_core import prediction_evidence as evidence
from app_core.market_probability_model import predict_market_probabilities
from app_core.producer_provenance import DERIVED_NAMESPACE, diagnose
from app_core.research_estimate_trace import carry_origin_columns, origin_rejection
from app_core.per_game_boards import per_game_board
from app_core.public_board import build_package
from app_core.research_replay import retain_export, read_export, frame_from_payload
from scripts.benchmark_drive_history_loading import blocked_network
from test_research_probability_producer import forecast, real_path
from test_research_probability_display import NOW, ANALYSIS, QUOTE, START

INFERENCE = (pd.Timestamp(ANALYSIS) - pd.Timedelta(seconds=10)).isoformat()


def transport(*, period="full_game", rules="includes_overtime", quote_id=None, start=START,
              home="Home", away="Away", event_id="synthetic-event", kind="spreads", line=3.5, price=104):
    market = dict(key=kind, last_update=QUOTE, outcomes=[dict(name=home if kind == "spreads" else "Over", point=line, price=price)])
    if period is not None: market["period"] = period
    if rules is not None: market["settlement_rules"] = rules
    if quote_id is not None: market["outcomes"][0]["quote_id"] = quote_id
    return dict(id=event_id, home_team=home, away_team=away, commence_time=start,
                bookmakers=[dict(key="novig", last_update="2026-10-01T18:00:00Z", markets=[market])])


def supplied(monkeypatch, *, source_changes=None, transport_changes=None, before_inference=None):
    raw = forecast(league="NFL", market_type="spread_home", best_pick="Home +3.5", spread_line=3.5,
        live_spread_line=3.5, total_line=None, live_total_line=None, odds_american=104,
        calibrated_probability=.54, model_probability=.54, expected_value=.54*2.04-1,
        feature_home_ppg=24., feature_away_ppg=22., feature_home_oppg=21., feature_away_oppg=23.,
        feature_home_games_played=7, feature_away_games_played=7, ml_feature_eligible=True,
        stats_resolution_status="resolved", feature_home_win_pct=.55, feature_away_win_pct=.45,
        feature_diff_last5=.1, model_status="Market Score Model", matchup_id="AWAY|HOME|2026-10-01")
    for k in ("quote_id", "market_period", "period", "settlement_rules", "inference_status", "push_probability", "market_push_probability", "probability_semantics", "prediction_generated_at"):
        raw.pop(k, None)
    raw.update(source_changes or {})
    game = transport(**(transport_changes or {}))
    raw["provider_quotes"] = evidence.provider_quotes(game)
    if before_inference: before_inference(raw)
    monkeypatch.setattr("app_core.research_estimate_trace.generated_time", lambda: INFERENCE)
    with blocked_network():
        output = predict_market_probabilities(pd.DataFrame([raw]))
        # Exercise the exact copy helper used by run_analysis_pipeline.
        merged = pd.DataFrame([raw])
        carry_origin_columns(merged, output)
    result = merged.iloc[0].to_dict()
    result.update({k:output.iloc[0][k] for k in output if k not in result})
    result.update(ml_probability=output.iloc[0].ml_probability, ml_target=output.iloc[0].ml_target,
                  ml_probability_source=output.iloc[0].ml_probability_source)
    return raw, result, output


def pass_only(result):
    row = result["package"]["games"]["overall"][0]
    assert row["status"] == "PASS"
    assert row.get("wager_contract", {}).get("production_bet_amount", 0) == 0
    assert result["card"].iloc[0].wager_contract["production_bet_amount"] == 0
    assert not result["captured"].production_eligible.fillna(False).any()
    return row["research_display"]


@pytest.mark.parametrize("provider_quote", [None, "synthetic-provider-quote"])
def test_complete_prospective_contract_and_original_clock_through_entire_path(monkeypatch, tmp_path, provider_quote):
    original, raw, output = supplied(monkeypatch, transport_changes=dict(quote_id=provider_quote))
    untouched = deepcopy(original)
    metadata = json.loads(raw["ml_estimate_metadata"])
    contract = metadata["producer_contract"]
    assert metadata["version"] == 2 and contract["inference_time"] == INFERENCE
    assert contract["target_period"] == "full_game"
    assert contract["offer"]["quote_kind"] == ("provider_issued" if provider_quote else "locally_derived")
    assert contract["offer"]["quote_namespace"] == ("odds_api" if provider_quote else DERIVED_NAMESPACE)
    assert contract["offer"]["source_time"] == pd.Timestamp(QUOTE).isoformat()
    assert raw["quote_id"] == provider_quote if provider_quote else raw["quote_id"].startswith(DERIVED_NAMESPACE+":")
    legacy_input = pd.DataFrame([dict(original, provider_quotes="[]")])
    with blocked_network():
        legacy = predict_market_probabilities(legacy_input)
        assert output.ml_probability.tolist() == legacy.ml_probability.tolist()
        result = real_path(monkeypatch, tmp_path, raw)
    display = pass_only(result)
    assert display["availability_reason"] == "AVAILABLE"
    assert display["probability"] == pytest.approx(.54) and display["ev"] == pytest.approx(.54*2.04-1)
    assert display["inference_status"] == "UNKNOWN"  # Blend is not independent inference.
    for name in ("authority", "captured", "card"):
        frame = result[name]
        assert frame.iloc[0].prediction_generated_at == INFERENCE
        assert frame.iloc[0].ml_estimate_metadata == raw["ml_estimate_metadata"]
        assert frame.iloc[0].ml_probability == output.iloc[0].ml_probability != .54
    for frame in result["frames"][:2]:
        row = pd.read_csv(StringIO(frame.to_csv(index=False))).iloc[0]
        assert row.prediction_generated_at == INFERENCE
        trace = json.loads(row.research_estimate_trace)
        assert trace["origin"]["missing_fields"] == trace["origin"]["conflicting_fields"] == []
        assert trace["first_rejection_stage"] is None
        assert trace["source"]["ml_probability"]["value"] != trace["source"]["best_available_probability"]["value"]
    with blocked_network():
        receipt = retain_export(result["frames"], result["package"], result["card"], result["captured"], path=tmp_path/"isolated-evidence.sqlite3")
        saved, sources = read_export(receipt["export_id"], path=tmp_path/"isolated-evidence.sqlite3")
        source = next(iter(sources.values()))
        frames = [per_game_board(frame_from_payload(source["captured_card"]), frame_from_payload(source["captured_candidates"]), family=f, novig_only=True) for f in ("overall", "sides", "totals")]
        assert build_package(*frames) == saved["package"]
    from test_research_probability_browser import inspect_browser
    browser = inspect_browser(result["package"], tmp_path/"browser", NOW, rendered_html=result["html"])
    assert browser["initial"]["shown"][0]["probability"] == pytest.approx(.54)
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0
    assert browser["initial"]["saved"] == [dict(probability=None, ev=None, status="PASS", stake=0)]
    assert browser["expired"]["current"] == browser["started"]["current"] == 0
    assert original == untouched
    assert "producer_contract" not in json.dumps(result["package"])


@pytest.mark.parametrize("field,change,missing", [
    ("period", dict(period=None), "offer.period"), ("rules", dict(rules=None), "offer.rules"),
    ("provider_event_id", dict(event_id=None), "offer.quote_id")])
def test_missing_original_facts_remain_unknown(monkeypatch, tmp_path, field, change, missing):
    _, raw, _ = supplied(monkeypatch, transport_changes=change)
    with blocked_network(): result = real_path(monkeypatch, tmp_path, raw)
    display = pass_only(result)
    assert display["availability_reason"] == "ESTIMATE_PROVENANCE_NOT_RECORDED"
    trace = json.loads(result["frames"][0].iloc[0].research_estimate_trace)
    assert missing in trace["origin"]["missing_fields"]
    assert trace["origin"]["stage"] == "producer_contract"
    assert trace["first_rejection_stage"] == "per_game_export.research_display"
    assert display["probability"] is None and display["ev"] is None


@pytest.mark.parametrize("field,value", [
    ("quote_id", ""), ("market_period", ""), ("settlement_rules", ""),
    ("home_team", "Away"), ("away_team", "Home"), ("provider_event_id", "unrelated"),
    ("provider_namespace", "other-provider"), ("prediction_generated_at", ANALYSIS),
    ("prospective_quote_id", "conflicting")])
def test_downstream_missing_or_conflicting_facts_never_repair(monkeypatch, tmp_path, field, value):
    _, raw, _ = supplied(monkeypatch)
    raw[field] = value
    with blocked_network(): result = real_path(monkeypatch, tmp_path, raw)
    display = pass_only(result)
    assert display["availability_reason"] != "AVAILABLE"
    assert display["probability"] is None and display["ev"] is None


def test_unordered_keys_never_assign_sides_and_v1_interpretation_is_unchanged(monkeypatch):
    _, raw, _ = supplied(monkeypatch)
    for key in ("HOME|AWAY|2026-10-01", "AWAY|HOME|2026-10-01", "2026-10-01|home|away", "2026-10-01|away|home"):
        assert origin_rejection(dict(raw, matchup_id=key)) is None
    swapped = dict(raw, home_team="Away", away_team="Home")
    assert origin_rejection(swapped) == "ESTIMATE_IDENTITY_MISMATCH"
    assert origin_rejection(dict(raw, matchup_id="OTHER|HOME|2026-10-01")) == "ESTIMATE_IDENTITY_MISMATCH"
    metadata = json.loads(raw["ml_estimate_metadata"])
    metadata.pop("producer_contract"); metadata["version"] = 1
    metadata["identity"]["matchup_id"] = {"state":"VALUE", "value":"HOME|AWAY|2026-10-01"}
    assert origin_rejection(dict(raw, matchup_id="2026-10-01|away|home", ml_estimate_metadata=json.dumps(metadata))) == "ESTIMATE_IDENTITY_MISMATCH"


@pytest.mark.parametrize("changes", [dict(best_pick="Away +3.5"), dict(spread_line=-3.5), dict(market_type="spread_away")])
def test_exact_selected_side_and_signed_line_never_reinterpret_origin(monkeypatch, changes):
    _, raw, _ = supplied(monkeypatch)
    corrupted = dict(raw, **changes)
    assert origin_rejection(corrupted) == "ESTIMATE_IDENTITY_MISMATCH"
    diagnostic = diagnose(corrupted, json.loads(raw["ml_estimate_metadata"]))
    assert diagnostic["conflicting_fields"]


def test_doubleheaders_and_consistently_rehashed_quote_conflicts_reject(monkeypatch, tmp_path):
    def duplicate(raw):
        quotes = json.loads(raw["provider_quotes"])
        other = dict(quotes[0], provider_event_id="synthetic-second-game", event_start_utc="2026-10-02T00:00:00Z")
        raw["provider_quotes"] = json.dumps(quotes+[other])
    _, raw, _ = supplied(monkeypatch, before_inference=duplicate)
    diagnostic = diagnose(raw, json.loads(raw["ml_estimate_metadata"]))
    assert "producer_contract.matched_offer_count" in diagnostic["conflicting_fields"]
    assert origin_rejection(raw) == "ESTIMATE_IDENTITY_MISMATCH"
    # The actual candidate constructor has already rejected the ambiguous quote;
    # its empty card cannot be captured or exported as a selected estimate.
    with blocked_network(), pytest.raises(ValueError, match="empty candidate audit or final card"):
        real_path(monkeypatch, tmp_path, raw)


@pytest.mark.parametrize("changes", [dict(spread_line=3., live_spread_line=3., best_pick="Home +3"), dict(league="NHL")])
def test_integer_push_and_unsupported_nhl_protections(monkeypatch, tmp_path, changes):
    transport_changes = dict(line=3.) if changes.get("spread_line") else {}
    _, raw, _ = supplied(monkeypatch, source_changes=changes, transport_changes=transport_changes)
    with blocked_network(): result = real_path(monkeypatch, tmp_path, raw)
    display = pass_only(result)
    assert display["availability_reason"] != "AVAILABLE"
    if changes.get("league") == "NHL": assert raw["ml_inference_status"] == "unavailable"


def test_mixed_legacy_rows_do_not_lose_facts(monkeypatch):
    _, _, output = supplied(monkeypatch)
    frame = pd.DataFrame([dict(quote_id="old"), dict(quote_id="new")])
    predictions = output.reindex([0, 1])
    carry_origin_columns(frame, predictions)
    assert frame.iloc[1].quote_id == "new"


def test_original_provider_identity_survives_canonical_upload_boundary(monkeypatch):
    from core.streamlit_pipeline import _coerce_export_to_canonical
    _, raw, _ = supplied(monkeypatch)
    with blocked_network(): frame = _coerce_export_to_canonical(pd.DataFrame([raw]), ["NFL"])
    for field in ("provider_namespace", "provider_event_id", "ml_estimate_metadata", "prediction_generated_at"):
        assert frame.iloc[0][field] == raw[field]


@pytest.mark.parametrize("period", ["first_half", "first_five_innings"])
def test_complete_quote_of_an_unsupported_model_period_rejects(monkeypatch, tmp_path, period):
    _, raw, _ = supplied(monkeypatch, transport_changes=dict(period=period))
    with blocked_network(): result = real_path(monkeypatch, tmp_path, raw)
    display = pass_only(result)
    assert display["availability_reason"] == "TARGET_MISMATCH"
    assert display["probability"] is None and display["ev"] is None
    trace = json.loads(result["frames"][0].iloc[0].research_estimate_trace)
    assert trace["origin"]["conflicting_fields"] == ["offer.period_target"]


def test_missing_clock_is_unknown_rather_than_an_invented_identity_conflict(monkeypatch):
    _, raw, _ = supplied(monkeypatch, transport_changes=dict(start=None))
    diagnostic = diagnose(raw, json.loads(raw["ml_estimate_metadata"]))
    assert diagnostic["reason"] == "ESTIMATE_PROVENANCE_NOT_RECORDED"
    assert "event.start" in diagnostic["missing_fields"]
    assert diagnostic["conflicting_fields"] == []
    _, complete, _ = supplied(monkeypatch)
    complete["prediction_generated_at"] = ""
    diagnostic = diagnose(complete, json.loads(complete["ml_estimate_metadata"]))
    assert diagnostic["reason"] == "ESTIMATE_PROVENANCE_NOT_RECORDED"
    assert "prediction_generated_at" in diagnostic["missing_fields"]
    assert diagnostic["conflicting_fields"] == []


def test_mlb_sorted_pair_actual_producer_to_renderer_and_negative_ev(monkeypatch, tmp_path):
    _, raw, _ = supplied(monkeypatch, source_changes=dict(league="MLB", home_team="Zulu", away_team="Alpha",
        market_type="total_over", best_pick="Over 7.5", total_line=7.5, live_total_line=7.5,
        spread_line=None, live_spread_line=None, odds_american=-120, expected_value=.54*(1+100/120)-1,
        matchup_id="ALPHA|ZULU|2026-10-01", feature_home_ppg=4.5, feature_away_ppg=4.5,
        feature_home_oppg=4.5, feature_away_oppg=4.5),
        transport_changes=dict(home="Zulu", away="Alpha", kind="totals", line=7.5, price=-120))
    with blocked_network(): result = real_path(monkeypatch, tmp_path, raw)
    display = pass_only(result)
    assert display["availability_reason"] == "AVAILABLE"
    assert display["probability"] == pytest.approx(.54)
    assert display["ev"] == pytest.approx(-.01)
    contract = json.loads(result["captured"].iloc[0].ml_estimate_metadata)["producer_contract"]
    assert contract["event"]["home"] != contract["event"]["away"]
    assert contract["offer"]["side"] == "over"
    from test_research_probability_browser import inspect_browser
    browser = inspect_browser(result["package"], tmp_path/"browser", NOW, rendered_html=result["html"])
    assert browser["initial"]["shown"][0]["probability"] == pytest.approx(.54)
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0


def test_rejected_display_cannot_expose_legacy_numeric_package_fields(monkeypatch, tmp_path):
    _, raw, _ = supplied(monkeypatch, transport_changes=dict(rules=None))
    with blocked_network(): result = real_path(monkeypatch, tmp_path, raw)
    assert pass_only(result)["availability_reason"] == "ESTIMATE_PROVENANCE_NOT_RECORDED"
    # A legacy export has populated numeric fields and no saved authority
    # contract. Build/validate its package normally; the renderer must still
    # honor the rejecting research object, without reading the legacy numbers.
    frames = [f.drop(columns=["wager_contract"], errors="ignore") for f in result["frames"]]
    package = build_package(*frames)
    row = package["games"]["overall"][0]
    assert row["win_estimate"] == pytest.approx(.54) and row["ev"] is not None
    assert row["research_display"]["probability"] is None and row["status"] == "PASS"
    from scripts.publish_board import render
    from test_research_probability_browser import inspect_browser
    browser = inspect_browser(package, tmp_path/"browser", NOW, rendered_html=render(package))
    assert browser["initial"]["shown"][0]["probability"] is None
    assert browser["initial"]["current"] == browser["initial"]["top"] == 0
