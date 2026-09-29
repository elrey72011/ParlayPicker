"""Writer/publisher entrypoint regressions. All evidence here is synthetic."""
from contextlib import closing
from copy import deepcopy
from datetime import timedelta
import sqlite3

import pytest

from app_core import football_stage1 as stage1
from app_core.prospective_evidence import connect
from app_core.prospective_remote import _schema, _encode
from app_core.read_only_census import run_census
from app_core.release_preflight import ReleasePreflightError
from core.exposure_ledger import append
from test_football_stage1 import NOW as CAPTURE, START, nfl_event, odds_event
from test_post2357_read_only_census import FakeReadOnlyDrive, all_source_objects
from test_post2362_external_verification import _current_source, _freeze_release_clocks
from test_current_wagers_trace_and_release import NOW


def _ready(path, *, pending=False, duplicate=False):
    schedule, _ = stage1.append_schedule(path, "NFL", nfl_event(), CAPTURE)
    stage1.append_offers(path, schedule, odds_event(), CAPTURE, "capture-1")
    if duplicate:
        stage1.append_offers(path, schedule, odds_event(price=-115),
                             CAPTURE + timedelta(minutes=1), "capture-2")
    if not pending:
        _result(path, schedule)
    return schedule


def _result(path, schedule, *, home_score=24, offset=3):
    provider = nfl_event(completed=True)
    provider["competitions"][0]["competitors"][0]["score"] = str(home_score)
    result, _ = stage1.append_result(
        path, schedule,
        {"provider_event_id": "401", "home_team_id": "8", "away_team_id": "9",
         "home_score": home_score, "away_score": 20, "status": "FINAL",
         "provider_response": provider},
        START + timedelta(hours=offset), source="ESPN")
    stage1.settle_game(path, schedule, result, START + timedelta(hours=offset))


def _objects(path, *, omitted=()):
    objects = all_source_objects()
    with closing(connect(path)) as db:
        for table, (columns, primary) in _schema(db).items():
            if table in omitted:
                continue
            for row in db.execute(f'SELECT * FROM "{table}"'):
                name, raw = _encode(table, columns, primary, tuple(row))
                objects[name] = raw
    return objects


def _census(path, **kwargs):
    client = FakeReadOnlyDrive(_objects(path, omitted=kwargs.pop("omitted", ())))
    report = run_census(client, source_revision="writer-fixture", max_objects=10000,
                        deadline_seconds=60, **kwargs)
    assert not client.mutation_calls
    return report


def _scope(report, scope="NFL/SPREAD"):
    return next(item for item in report["scopes"] if item["scope"] == scope)


def _reference(path, market="SPREAD"):
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as db:
        return db.execute(
            "SELECT manifest_id FROM prospective_football_training_manifest "
            "WHERE sport='NFL' AND market_family=? ORDER BY manifest_id", (market,)
        ).fetchall()


def test_d03_run_census_conflicting_result_revision_matches_manifest(tmp_path):
    path = tmp_path / "writer.sqlite3"
    schedule = _ready(path)
    assert len(_reference(path)) == 1
    assert _scope(_census(path))["counts"]["independent_eligible_games"] == 1
    _result(path, schedule, home_score=27, offset=4)
    assert _reference(path) == []
    report = _census(path)
    assert _scope(report)["counts"]["independent_eligible_games"] == 0


def _commit(settings, contract, *, amount=50, bet_id="unrelated", game_id="other",
            team_ids=("other-home", "other-away"), source_snapshot_id="unrelated-review"):
    return append(
        settings["PARLAYPICKER_EXPOSURE_LEDGER"],
        {"status": "COMMITTED", "bet_id": bet_id, "source_snapshot_id": source_snapshot_id,
         "sportsbook": contract["sportsbook"], "stake_dollars": amount,
         "legs": [{"sport": contract["sport"], "game_id": game_id,
                   "team_ids": list(team_ids), "market": contract["market_type"],
                   "selection": contract["selection"], "line": contract["line"],
                   "odds": contract["odds"]}]},
        confirmed=True, now=NOW + timedelta(seconds=30))


def _publisher(tmp_path, monkeypatch):
    package, activation, settings = _current_source(tmp_path, monkeypatch)
    _freeze_release_clocks(monkeypatch)
    for name, value in settings.items():
        monkeypatch.setenv(name, value)
    from app_core import netlify_publishing
    calls = []
    monkeypatch.setattr(netlify_publishing, "api_call",
                        lambda *args, **kwargs: calls.append(args) or
                        {"id": "deploy-fixture", "site_id": "site-fixture",
                         "state": "processing"})
    return package, activation, settings, calls, netlify_publishing


def test_e01_actual_publisher_holds_after_unrelated_commitment(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    original = deepcopy(package)
    _commit(settings, package["games"]["overall"][0]["wager_contract"])
    with pytest.raises(ReleasePreflightError):
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert calls == []
    assert package == original



@pytest.mark.parametrize("case", ["baseline", "event_revision", "pending", "duplicate",
                                  "missing_quote", "missing_result", "missing_settlement",
                                  "missing_event", "chronology"])
def test_d04_d05_run_census_agrees_with_canonical_query(tmp_path, case):
    from app_core.prospective_remote import _decode
    import json
    path = tmp_path / "writer.sqlite3"
    _ready(path, pending=case == "pending", duplicate=case == "duplicate")
    if case == "event_revision":
        stage1.append_schedule(path, "NFL", nfl_event(start=CAPTURE - timedelta(minutes=1)),
                               START + timedelta(hours=4))
    omitted = {
        "missing_quote": ("prospective_football_quote",),
        "missing_result": ("prospective_football_result",),
        "missing_settlement": ("prospective_football_settlement",),
        "missing_event": ("prospective_football_event",),
    }.get(case, ())
    objects = _objects(path, omitted=omitted)
    if case == "chronology":
        # A serialized legacy chronology defect is never promoted. Re-encode
        # the immutable local test object with the actual writer.
        for name, raw in list(objects.items()):
            item = json.loads(raw)
            if item.get("table") == "prospective_football_quote":
                values = list(item["row"])
                values[item["columns"].index("observed_at")] = (START + timedelta(hours=4)).isoformat()
                key, encoded = _encode(item["table"], item["columns"], ("quote_id",), values)
                objects[key] = encoded
    # Independently execute the existing canonical views over the exact verified
    # fixture inventory. This is scratch reference data; the writer DB is intact.
    with closing(connect(":memory:")) as reference:
        reference.execute("PRAGMA foreign_keys=OFF")
        schema = _schema(reference)
        for name, raw in objects.items():
            if not name.startswith("parlaypicker/canonical-prospective-v1/"):
                continue
            table, row = _decode(name, raw, schema)
            reference.execute(f'INSERT INTO "{table}" VALUES ({",".join("?" for _ in row)})', row)
        expected = {}
        for market in ("SPREAD", "TOTAL"):
            expected[market] = [row[0] for row in reference.execute(
                "SELECT manifest_id FROM prospective_football_training_manifest "
                "WHERE sport='NFL' AND market_family=? ORDER BY manifest_id", (market,))]
    report = run_census(FakeReadOnlyDrive(objects), source_revision="writer-fixture")
    for market in expected:
        scope = _scope(report, "NFL/" + market)
        assert scope["eligibility"]["manifest_ids"] == expected[market]
        assert scope["counts"]["independent_eligible_games"] == len(expected[market])
        assert scope["counts"]["stored_ready_rows"] >= len(expected[market])
    if case == "duplicate":
        scope = _scope(report)
        assert scope["counts"]["stored_ready_rows"] > 1
        assert scope["counts"]["active_eligible_rows"] > 1
        assert scope["counts"]["independent_eligible_games"] == 1


def test_d06_all_twelve_scopes_counts_or_explicit_unknowns(tmp_path):
    path = tmp_path / "writer.sqlite3"
    _ready(path)
    report = _census(path)
    assert len(report["scopes"]) == 12
    for scope in report["scopes"]:
        assert scope["eligibility"]["reader"]
        assert scope["qualification_binding"]["production_eligible"] is False
        assert scope["qualification_binding"]["status"] == "NOT_VERIFIED"
        for cohort in scope["cohorts"].values():
            assert cohort["independent_games"] is None
            assert cohort["reader"] and cohort["follow_up"]
        if not scope["scope"].startswith(("NFL/", "NCAAF/")):
            assert scope["counts"]["independent_eligible_games"] is None
            assert scope["eligibility"]["status"] == "UNKNOWN"
            assert scope["eligibility"]["reason"]
        elif scope["scope"].startswith("NCAAF/"):
            assert scope["counts"]["independent_eligible_games"] == 0


def test_d02_partial_canonical_count_is_unknown_and_checkpoint_resumes(tmp_path):
    path = tmp_path / "writer.sqlite3"
    _ready(path, duplicate=True)
    checkpoint = tmp_path / "checkpoint.json"
    client = FakeReadOnlyDrive(_objects(path))
    partial = run_census(client, source_revision="writer-fixture", max_objects=1,
                         checkpoint_path=checkpoint)
    assert _scope(partial)["counts"]["independent_eligible_games"] is None
    assert _scope(partial)["counts"]["independent_eligible_reason"] == "CANONICAL_CENSUS_INCOMPLETE"
    complete = run_census(client, source_revision="writer-fixture", checkpoint_path=checkpoint)
    assert complete["metrics"]["verified_reused_objects"] == 1
    assert _scope(complete)["eligibility"]["manifest_ids"] == [x[0] for x in _reference(path)]


def _private_snapshot(tmp_path, package, team_sets=None):
    import pandas as pd
    import hashlib
    from pathlib import Path
    from app_core.prediction_evidence import connect as writer_connect
    path = tmp_path / "private"
    path.mkdir(parents=True, exist_ok=True)
    candidates = []
    decisions = []
    unique = {}
    for row in package["games"]["overall"]:
        c = row["wager_contract"]
        unique[c["game_id"]] = c
    for game, c in unique.items():
        candidates.append({
            "sport": c["sport"], "game_id": game, "market_type": c["market_type"],
            "selection": c["selection"], "book": c["sportsbook"], "quote_time": c["quote_timestamp"],
            "line": c["line"], "odds_american": c["odds"],
            "team_ids": (team_sets or {}).get(game, [game + "-home", game + "-away"]),
        })
        decisions.append({"wager_contract": c})
    raw = [pd.DataFrame(candidates).to_csv(index=False),
           pd.DataFrame(decisions).to_csv(index=False), pd.DataFrame().to_csv(index=False)]
    sid = next(iter(unique.values()))["evidence_version"]
    with closing(writer_connect(path / "evidence.sqlite3")) as db, db:
        db.execute("INSERT OR IGNORE INTO bundles VALUES (?,?,?)", ("fixture", NOW.isoformat(), "{}"))
        db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?)",
                   (sid, "fixture", NOW.isoformat(), *raw, hashlib.sha256("\0".join(raw).encode()).hexdigest()))
    return str(path)


def _add_ticket(package, monkeypatch, game="1", *, amount=None):
    from test_current_wagers_trace_and_release import _approved_source, _package
    contracts = [deepcopy(row["wager_contract"]) for row in package["games"]["overall"]]
    contract = deepcopy(contracts[0])
    contract["game_id"] = contract["matchup_id"] = game
    contract["selection"] = f"Home{game} -2.5"
    if amount is not None:
        contract["production_bet_amount"] = amount
    contracts.append(contract)
    sources = []
    for c in contracts:
        g = c["game_id"]
        sources.append(_approved_source(
            candidate_id="candidate-" + g, matchup_id=g, matchup=f"Away{g} at Home{g}",
            Home="Home" + g, Away="Away" + g, wager_contract=c,
            pick=c["selection"], best_pick=c["selection"], Play_Stake=c["production_bet_amount"]))
    package.clear()
    package.update(_package(monkeypatch, sources))


def test_e03_package_total_and_duplicate_views(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    private = _private_snapshot(tmp_path, package)
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", private)
    before = deepcopy(package)
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert len(calls) == 2  # Same-ticket retries create no ledger events.
    assert package == before
    from core.exposure_ledger import events
    assert len(events(settings["PARLAYPICKER_EXPOSURE_LEDGER"])) == 1

    _add_ticket(package, monkeypatch)
    # Rewrite only the isolated private fixture in a new directory.
    private = _private_snapshot(tmp_path / "second", package)
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", private)
    _commit(settings, dict(package["games"]["overall"][0]["wager_contract"], sport="NBA"), amount=35)
    with pytest.raises(ReleasePreflightError) as exc:
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert exc.value.report["capacity_check"]["status"] == "BLOCKED"
    assert len(exc.value.report["capacity_check"]["tickets"]) == 2
    assert "total" in [key for item in exc.value.report["capacity_check"]["tickets"] for key in item["failed_limits"]]
    assert len(calls) == 2


@pytest.mark.parametrize("limit", ["total", "daily", "weekly", "game", "team", "sport"])
def test_e02_each_existing_limit_at_actual_publisher(tmp_path, monkeypatch, limit):
    package, activation, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    from pathlib import Path
    import json
    from core.exposure_ledger import digest
    caps = dict(bankroll=1000.0, unit_value=1.0, currency="USD",
                total_cap=1., daily_cap=1., weekly_cap=1., game_cap=1., team_cap=1.)
    if limit != "sport":
        caps[limit + "_cap"] = .01
    append(settings["PARLAYPICKER_EXPOSURE_LEDGER"],
           dict(status="CONFIGURED", **caps), confirmed=True, now=NOW + timedelta(seconds=10))
    activation["exposure_limits"] = {key: value for key, value in caps.items() if key.endswith("_cap")}
    activation["policy"]["sport_exposure_cap"] = .01 if limit == "sport" else 1.
    activation["activation_hash"] = digest({k: v for k, v in activation.items() if k != "activation_hash"})
    Path(settings["PARLAYPICKER_MARKET_ACTIVATIONS_DIR"], "NFL-SPREAD.json").write_text(json.dumps(activation))
    private = _private_snapshot(tmp_path, package, {"0": ["team-home", "team-away"]})
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", private)
    contract = package["games"]["overall"][0]["wager_contract"]
    _commit(settings, contract, amount=10, game_id="0" if limit == "game" else "other",
            team_ids=("team-home", "elsewhere") if limit == "team" else ("elsewhere1", "elsewhere2"))
    with pytest.raises(ReleasePreflightError) as exc:
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    failed = [key for ticket in exc.value.report["capacity_check"]["tickets"] for key in ticket["failed_limits"]]
    assert any(key == limit or key.startswith(limit + ":") for key in failed)
    assert calls == []


def test_e04_committed_display_and_retry_preserve_history(tmp_path, monkeypatch):
    from core.exposure_ledger import digest, events
    from pathlib import Path
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    contract = package["games"]["overall"][0]["wager_contract"]
    _commit(settings, contract, amount=contract["production_bet_amount"], bet_id="original-ticket",
            game_id=contract["game_id"], team_ids=("home", "away"),
            source_snapshot_id=digest(contract))
    _commit(settings, contract, amount=50, bet_id="capacity-consumer")
    before = deepcopy(package)
    path = Path(settings["PARLAYPICKER_EXPOSURE_LEDGER"])
    ledger_before = path.read_bytes()
    history_before = events(path)
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert len(calls) == 2 and package == before
    assert path.read_bytes() == ledger_before
    assert events(path) == history_before


def test_e04_lookalike_commitment_does_not_exempt_new_ticket(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    contract = package["games"]["overall"][0]["wager_contract"]
    _commit(settings, contract, amount=50, game_id=contract["game_id"],
            source_snapshot_id=contract["evidence_version"])
    with pytest.raises(ReleasePreflightError):
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert calls == []



def test_e03_multiple_new_tickets_fit_once_per_view(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    _add_ticket(package, monkeypatch)
    private = _private_snapshot(tmp_path, package)
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", private)
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert len(calls) == 1


def test_e02_parlay_commitment_consumes_underlying_game_and_team(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    contract = package["games"]["overall"][0]["wager_contract"]
    private = _private_snapshot(tmp_path, package, {"0": ["home", "away"]})
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", private)
    first = {"sport": "NFL", "game_id": "0", "team_ids": ["home", "away"],
             "market": contract["market_type"], "selection": contract["selection"],
             "line": contract["line"], "odds": contract["odds"]}
    second = dict(first, game_id="other", team_ids=["other-home", "other-away"])
    append(settings["PARLAYPICKER_EXPOSURE_LEDGER"],
           {"status": "COMMITTED", "bet_id": "unrelated-parlay", "parlay_id": "parlay-fixture",
            "source_snapshot_id": "different-review", "sportsbook": contract["sportsbook"],
            "stake_dollars": 10, "legs": [first, second]}, confirmed=True,
           now=NOW + timedelta(seconds=30))
    with pytest.raises(ReleasePreflightError) as exc:
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    failed = exc.value.report["capacity_check"]["tickets"][0]["failed_limits"]
    assert "game:NFL:0" in failed
    assert "team:NFL:home" in failed
    assert calls == []


def test_e05_settlement_does_not_restore_daily_turnover(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    c = package["games"]["overall"][0]["wager_contract"]
    _commit(settings, c)
    append(settings["PARLAYPICKER_EXPOSURE_LEDGER"],
           {"status": "SETTLED", "bet_id": "unrelated"}, confirmed=True,
           now=NOW + timedelta(seconds=40))
    with pytest.raises(ReleasePreflightError) as exc:
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert "daily" in exc.value.report["capacity_check"]["tickets"][0]["failed_limits"]
    assert calls == []


def test_e04_exact_immutable_snapshot_binding_for_committed_ticket(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    private = _private_snapshot(tmp_path, package)
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", private)
    c = package["games"]["overall"][0]["wager_contract"]
    _commit(settings, c, amount=c["production_bet_amount"], game_id=c["game_id"],
            source_snapshot_id=c["evidence_version"])
    _commit(settings, c, amount=50, bet_id="unrelated-second")
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert len(calls) == 1



def test_e04_trial_retry_reuses_existing_reservation_read_only(tmp_path, monkeypatch):
    import json
    from pathlib import Path
    from app_core.trial_authority import record_consent, consent_status, reserve_recommendation
    from core.exposure_ledger import snapshot
    from test_controlled_trial_integration import contract, trial_board_row
    from test_current_wagers_trace_and_release import _package
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    consent_path = tmp_path / "consent.sqlite3"
    reservation_path = tmp_path / "reservations.sqlite3"
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIAL_CONSENT_LEDGER", str(consent_path))
    monkeypatch.setenv("PARLAYPICKER_CONTROLLED_TRIAL_RESERVATION_LEDGER", str(reservation_path))
    record_consent(consent_path, "GRANTED", "fixture-owner", confirmed=True,
                   expires_at=(NOW + timedelta(minutes=20)).isoformat(), now=NOW)
    consent, _ = consent_status(consent_path, now=NOW)
    c = contract(quote_timestamp=(NOW - timedelta(minutes=1)).isoformat(),
                 start=(NOW + timedelta(hours=3)).isoformat())
    identity = {key: c[key] for key in (
        "candidate_id", "game_id", "sport", "market_type", "selection", "line",
        "odds", "sportsbook", "quote_timestamp", "estimated_probability", "push_probability")}
    identity["team_ids"] = ["detroit", "washington"]
    exposure = snapshot(settings["PARLAYPICKER_EXPOSURE_LEDGER"], now=NOW)
    amount, _, status = reserve_recommendation(
        identity, c["recommended_bet_amount"], consent=consent, exposure=exposure,
        external_recommended={}, slate_key="fixture-slate", sport="MLB", game_id="g1",
        team_ids=identity["team_ids"], now=NOW, path=reservation_path)
    assert amount == 2.5 and status == "RESERVED"
    source = trial_board_row().to_dict()
    source.update(controlled_trial_contract=c, quote_time=c["quote_timestamp"],
                  export_run_id=NOW.isoformat(),
                  odds_recorded_at=c["quote_timestamp"], game_start_utc=c["start"])
    package = _package(monkeypatch, [source])
    assert package["games"]["overall"][0]["status"] == "TRIAL"
    original = deepcopy(package)
    reservation_before = reservation_path.read_bytes()
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert len(calls) == 2
    assert reservation_path.read_bytes() == reservation_before and package == original
    _commit(settings, dict(c, sport="NFL"))
    with pytest.raises(ReleasePreflightError) as exc:
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert "CURRENT_EXPOSURE_CAPACITY_INSUFFICIENT" in exc.value.report["blocker_counts"]
    assert reservation_path.read_bytes() == reservation_before and len(calls) == 2


def test_e03_duplicate_views_cannot_change_saved_stake(tmp_path, monkeypatch):
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    package["games"]["sides"][0]["wager_contract"]["production_bet_amount"] = 9.0
    # Ensure independent view copies for this adversarial duplicate.
    original = deepcopy(package)
    with pytest.raises(ReleasePreflightError) as exc:
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert "CURRENT_EXPOSURE_DUPLICATE_DECISION_CONFLICT" in exc.value.report["blocker_counts"]
    assert package == original and calls == []


def test_e04_tampered_private_snapshot_cannot_exempt_capacity(tmp_path, monkeypatch):
    from pathlib import Path
    package, _, settings, calls, publisher = _publisher(tmp_path, monkeypatch)
    private = _private_snapshot(tmp_path, package)
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", private)
    c = package["games"]["overall"][0]["wager_contract"]
    _commit(settings, c, amount=c["production_bet_amount"], game_id=c["game_id"],
            source_snapshot_id=c["evidence_version"])
    _commit(settings, c, amount=50, bet_id="other-consumer")
    # An invalid stored checksum is never trusted as an immutable decision.
    corrupt = tmp_path / "corrupt"
    corrupt.mkdir()
    with closing(sqlite3.connect(corrupt / "evidence.sqlite3")) as db, db:
        db.execute("CREATE TABLE snapshots(snapshot_id,candidates,decisions,inputs,payload_hash)")
        db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?)", (c["evidence_version"], "", "", "", "invalid"))
    monkeypatch.setenv("PARLAYPICKER_EVIDENCE_DIR", str(corrupt))
    with pytest.raises(ReleasePreflightError):
        publisher.deploy(package, "site-fixture", "unused-fixture-token")
    assert calls == []

