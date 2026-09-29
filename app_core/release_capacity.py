"""Current, read-only package capacity using the existing allocation policy.

This check neither allocates a new stake nor writes recommendations/reservations.
Unknown private team identities use a disclosed worst-case team bound; they are
never invented from display names. Committed display requires exact evidence.
"""
from contextlib import closing
import ast
import csv
from io import StringIO
import math
from pathlib import Path
import sqlite3

from core.exposure_ledger import digest, events, snapshot, verify_snapshot
from core.wager_decisions import allocation_limits, allocation_headroom, finite


def _teams(contract, evidence_path):
    """Read the existing immutable snapshot format without its restore/writer."""
    if not Path(evidence_path).is_file():
        return None
    identities = set()
    try:
        with closing(sqlite3.connect(Path(evidence_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            rows = db.execute("SELECT snapshot_id,candidates,decisions,inputs,payload_hash FROM snapshots")
            for sid, candidates, decisions, inputs, expected in rows:
                import hashlib
                if hashlib.sha256("\0".join((candidates, decisions, inputs)).encode()).hexdigest() != expected:
                    raise ValueError("Immutable prediction snapshot hash mismatch")
                if contract.get("evidence_version") and sid != contract["evidence_version"]:
                    continue
                for row in csv.DictReader(StringIO(candidates)):
                    pairs = (
                        (row.get("sport") or row.get("league"), contract.get("sport")),
                        (row.get("game_id") or row.get("matchup_id"), contract.get("game_id")),
                        (row.get("market_type"), contract.get("market_type")),
                        (row.get("selection") or row.get("best_pick"), contract.get("selection")),
                        (row.get("book") or row.get("quote_source") or row.get("odds_source"), contract.get("sportsbook")),
                        (row.get("quote_time") or row.get("odds_recorded_at"), contract.get("quote_timestamp")),
                    )
                    if any(str(a or "") != str(b or "") for a, b in pairs):
                        continue
                    line = row.get("line") or row.get("market_line_used") or row.get("spread_line") or row.get("total_line")
                    if finite(line) != finite(contract.get("line")) or finite(
                            row.get("odds_american") or row.get("odds")) != finite(contract.get("odds")):
                        continue
                    teams = ast.literal_eval(row.get("team_ids") or "None")
                    if (isinstance(teams, (list, tuple)) and len(teams) == 2
                            and len(set(teams)) == 2 and all(isinstance(t, str) and t.strip() for t in teams)):
                        identities.add(tuple(sorted(teams)))
    except (OSError, sqlite3.Error, ValueError, SyntaxError, TypeError):
        return None
    return list(identities.pop()) if len(identities) == 1 else None


def _bound_snapshot_ids(contract, evidence_path):
    if not Path(evidence_path).is_file():
        return set()
    import hashlib
    found = set()
    try:
        with closing(sqlite3.connect(Path(evidence_path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
            for sid, candidates, decisions, inputs, expected in db.execute(
                    "SELECT snapshot_id,candidates,decisions,inputs,payload_hash FROM snapshots"):
                if hashlib.sha256("\0".join((candidates, decisions, inputs)).encode()).hexdigest() != expected:
                    continue
                for row in csv.DictReader(StringIO(decisions)):
                    try:
                        saved = ast.literal_eval(row.get("wager_contract") or "None")
                    except (SyntaxError, ValueError):
                        continue
                    if saved == contract:
                        found.add(sid)
    except (OSError, sqlite3.Error, TypeError):
        pass
    return found


def _committed(contract, history, evidence_path):
    source_ids = {digest(contract)} | _bound_snapshot_ids(contract, evidence_path)
    matches = []
    for event in history:
        if event.get("status") != "COMMITTED" or event.get("source_snapshot_id") not in source_ids:
            continue
        legs = event.get("legs", [])
        if len(legs) != 1:
            continue
        leg = legs[0]
        if (event.get("sportsbook") == contract.get("sportsbook")
                and finite(event.get("stake_dollars")) == finite(contract.get("production_bet_amount", contract.get("recommended_bet_amount")))
                and all(str(leg.get(a) or "") == str(contract.get(b) or "") for a, b in
                        (("sport", "sport"), ("game_id", "game_id"), ("market", "market_type"),
                         ("selection", "selection")))
                and finite(leg.get("line")) == finite(contract.get("line"))
                and finite(leg.get("odds")) == finite(contract.get("odds"))):
            matches.append(event)
    return matches


def _reservations(path):
    if not Path(path).is_file():
        return []
    with closing(sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        return [dict(row) for row in db.execute(
            "SELECT r.* FROM reservations r LEFT JOIN releases x "
            "ON x.reservation_id=r.reservation_id WHERE x.reservation_id IS NULL")]


def _reservation_for(contract, authority, reservations):
    import json
    matches = []
    for row in reservations:
        teams = json.loads(row["team_ids"])
        identity = {key: contract[key] for key in (
            "candidate_id", "game_id", "sport", "market_type", "selection", "line",
            "odds", "sportsbook", "quote_timestamp", "estimated_probability", "push_probability")}
        identity["team_ids"] = teams
        expected = digest(dict(identity=identity, consent_id=authority["authority_id"]))
        if expected == row["reservation_id"] and finite(row["stake_dollars"]) == finite(
                contract.get("recommended_bet_amount")):
            matches.append(row)
    return matches


def check_capacity(package, authorities, *, exposure_path, evidence_path, reservation_path, now):
    from app_core.release_preflight import ACTIONABLE, row_identity
    rows = {}
    conflicts = set()
    for values in package.get("games", {}).values():
        for row in values:
            if row.get("status") not in ACTIONABLE:
                continue
            identity = row_identity(row)
            contract = row.get("wager_contract") or row.get("controlled_trial_contract")
            if identity in rows and rows[identity][1] != contract:
                conflicts.add(identity)
            rows[identity] = (row, contract)
    if not rows:
        return {"status": "NOT_APPLICABLE", "reason_codes": [], "tickets": []}
    try:
        history = events(exposure_path)
        exposure = verify_snapshot(snapshot(exposure_path, now=now, history=history), now=now)
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        return {"status": "BLOCKED", "reason_codes": ["CURRENT_EXPOSURE_UNAVAILABLE"], "tickets": []}
    used = dict(exposure["committed"])
    import json
    try:
        reservations = _reservations(reservation_path)
        for reserved in reservations:
            if finite(reserved["bankroll"]) != exposure["bankroll"]:
                raise ValueError("Reservation bankroll binding mismatch")
            fraction = float(reserved["stake_dollars"]) / exposure["bankroll"]
            keys = {"total", "daily", "weekly", f"game:{reserved['sport']}:{reserved['game_id']}",
                    f"sport:{reserved['sport']}"}
            keys.update(f"team:{reserved['sport']}:{team}" for team in json.loads(reserved["team_ids"]))
            for key in keys:
                used[key] = used.get(key, 0.0) + fraction
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError):
        return {"status": "BLOCKED", "reason_codes": ["CURRENT_TRIAL_RESERVATION_UNAVAILABLE"], "tickets": []}
    baseline_used = dict(used)
    sport_caps = {}
    for identity, (row, contract) in rows.items():
        policy = authorities.get(identity, {}).get("allocation_policy", {})
        cap = finite(policy.get("sport_cap"))
        if row["status"] == "APPROVED" and cap is not None:
            sport = contract["sport"]
            sport_caps[sport] = min(sport_caps.get(sport, cap), cap)
    report = []
    seen_contracts = set()
    # A conservative all-team upper bound handles legacy public contracts
    # without stable IDs and interactions with other tickets in this package.
    team_bounds = {}
    for identity, (row, contract) in sorted(rows.items()):
        reasons = []
        contract_id = digest(contract)
        authority = authorities.get(identity, {})
        policy = authority.get("allocation_policy")
        if contract_id in seen_contracts:
            report.append({"row_id": identity, "status": "DUPLICATE_TICKET_VIEW", "reason_codes": []})
            continue
        seen_contracts.add(contract_id)
        if identity in conflicts:
            reasons.append("CURRENT_EXPOSURE_DUPLICATE_DECISION_CONFLICT")
        trial = row["status"] == "TRIAL"
        reserved = []
        if trial:
            try:
                reserved = _reservation_for(contract, authority, reservations)
            except (ValueError, KeyError, TypeError):
                pass
            if len(reserved) != 1:
                reasons.append("CURRENT_TRIAL_EXACT_RESERVATION_UNAVAILABLE")
        committed = _committed(contract, history, evidence_path)
        if len(committed) > 1:
            reasons.append("CURRENT_EXPOSURE_COMMITMENT_AMBIGUOUS")
        if len(committed) == 1 and not reasons:
            report.append({"row_id": identity, "status": "ALREADY_COMMITTED_DISPLAY",
                           "ledger_event_id": committed[0]["ledger_event_id"], "reason_codes": []})
            continue
        if not isinstance(policy, dict):
            reasons.append("CURRENT_EXPOSURE_POLICY_UNAVAILABLE")
        elif policy.get("configuration") != {key: exposure[key] for key in
                                            ("bankroll", "unit_value", "currency", "total_cap",
                                             "daily_cap", "weekly_cap", "game_cap", "team_cap")}:
            reasons.append("CURRENT_EXPOSURE_CONFIGURATION_CHANGED")
        teams = json.loads(reserved[0]["team_ids"]) if len(reserved) == 1 else _teams(contract, evidence_path)
        amount = finite(contract.get("production_bet_amount", contract.get("recommended_bet_amount")))
        bankroll = exposure["bankroll"]
        if amount is None or amount <= 0:
            reasons.append("CURRENT_EXPOSURE_ALLOCATION_INVALID")
        failed_limits = []
        if not reasons:
            requested = amount / bankroll
            sport = contract["sport"]
            limits = allocation_limits(
                sport, contract["game_id"], teams, total_cap=exposure["total_cap"],
                game_cap=exposure["game_cap"], team_cap=exposure["team_cap"],
                sport_cap=sport_caps.get(sport, 0.0), daily_cap=exposure["daily_cap"],
                weekly_cap=exposure["weekly_cap"])
            if trial:
                limits.pop(f"sport:{sport}")
            current_used = dict(used)
            if trial:
                for key in limits:
                    current_used[key] = max(0.0, current_used.get(key, 0) - requested)
            if teams is None or team_bounds.get(sport, {}).get("unknown"):
                maximum = max((value for key, value in baseline_used.items()
                               if key.startswith(f"team:{sport}:")), default=0.0)
                pending = team_bounds.get(sport, {}).get("pending", 0.0)
                limits[f"unknown_team_bound:{sport}"] = exposure["team_cap"]
                current_used[f"unknown_team_bound:{sport}"] = maximum + pending
            headroom = allocation_headroom(limits, current_used)
            if requested > headroom and not math.isclose(requested, headroom, rel_tol=0, abs_tol=1e-12):
                failed_limits = [key for key, cap in limits.items()
                                 if requested + current_used.get(key, 0) > cap + 1e-12]
                reasons.append("CURRENT_EXPOSURE_CAPACITY_INSUFFICIENT")
            else:
                if not trial:
                    for key in limits:
                        if not key.startswith("unknown_team_bound:"):
                            used[key] = used.get(key, 0) + requested
                bound = team_bounds.setdefault(sport, {"pending": 0.0, "unknown": False})
                bound["pending"] += requested
                bound["unknown"] |= teams is None
        report.append({"row_id": identity, "status": "BLOCKED" if reasons else
                       "EXISTING_TRIAL_RESERVATION_CAPACITY_VERIFIED" if trial else "NEW_RECOMMENDATION_CAPACITY_VERIFIED",
                       "stake_dollars": amount, "team_identity_status": "EXACT_TRIAL_RESERVATION" if trial and teams else
                       "EXACT_IMMUTABLE_SNAPSHOT" if teams else
                       "UNKNOWN_CONSERVATIVE_ALL_TEAM_BOUND", "failed_limits": failed_limits,
                       "reason_codes": reasons})
    reasons = sorted({reason for item in report for reason in item["reason_codes"]})
    return {"status": "BLOCKED" if reasons else "VERIFIED", "reason_codes": reasons,
            "as_of": now.isoformat(), "snapshot_hash": exposure["snapshot_hash"],
            "ledger_hash": exposure["ledger_hash"], "tickets": report,
            "writes": 0, "reservations": 0, "decision_changes": 0}

