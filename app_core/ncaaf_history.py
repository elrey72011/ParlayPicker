"""Resumable CFBD history collection and conservative research feature audit."""
from copy import deepcopy
from datetime import datetime, timezone, timedelta
import csv
import hashlib
import io
import json
import math
import zipfile
import requests

BASE = "https://api.collegefootballdata.com"
GAME_FIELDS = "id season week seasonType startDate startTimeTBD completed neutralSite homeId homeTeam homePoints awayId awayTeam awayPoints".split()


def now():
    return datetime.now(timezone.utc).isoformat()


def integer(v):
    return isinstance(v, int) and not isinstance(v, bool)


def timestamp(value):
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc) if dt.tzinfo else None
    except (ValueError, TypeError):
        return None


def new_collection(current_year=None):
    year = current_year or datetime.now(timezone.utc).year
    return {"schema": 1, "years": list(range(year - 3, year)), "batches": []}


def _clean(kind, records):
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("Invalid response shape")
    json.dumps(records, allow_nan=False)
    if kind == "games":
        if any(isinstance(r.get(k), (dict, list)) for r in records for k in GAME_FIELDS):
            raise ValueError("Invalid game fields")
        return [{k: r.get(k) for k in GAME_FIELDS} for r in records]
    result = []
    for r in records:
        teams = r.get("teams", [])
        if not isinstance(teams, list):
            raise ValueError("Invalid team statistics")
        clean_teams = []
        for team in teams:
            if not isinstance(team, dict) or not isinstance(team.get("stats", []), list):
                raise ValueError("Invalid team statistics")
            clean_teams.append({"teamId": team.get("teamId"), "points": team.get("points"),
                                "stats": [{"category": s.get("category"), "stat": s.get("stat")}
                                          for s in team.get("stats", []) if isinstance(s, dict)]})
        result.append({"id": r.get("id"), "teams": clean_teams})
    return result


def pending_requests(state):
    done = [b["request"] for b in state["batches"]]
    queue = [{"kind": "games", "year": y} for y in state["years"]]
    for b in state["batches"]:
        if b["request"]["kind"] != "games":
            continue
        for g in b["records"]:
            if (g.get("season") == b["request"]["year"] and integer(g.get("week"))
                and 0 <= g["week"] <= 30 and g.get("seasonType") in ("regular", "postseason")):
                req = {"kind": "stats", "year": g["season"], "week": g["week"], "season_type": g["seasonType"]}
                if req not in queue:
                    queue.append(req)
    return [r for r in queue if r not in done]


def collect_batch(state, token, *, get=None):
    """Return an updated checkpoint; six requests maximum, stop on first error."""
    state = deepcopy(state)
    token = str(token or "").strip()
    if token.lower().startswith("bearer "):
        token = token[7:].strip()
    if not token or any(c.isspace() for c in token):
        return state, "missing_or_invalid_key"
    get = get or requests.get
    for _ in range(6):
        queue = pending_requests(state)
        if not queue:
            break
        req = queue[0]
        params = {"year": req["year"]}
        if req["kind"] == "games":
            path = "/games"
            params.update(seasonType="both", classification="fbs")
        else:
            path = "/games/teams"
            params.update(week=req["week"], seasonType=req["season_type"])
        try:
            response = get(BASE + path, params=params, headers={"Authorization": "Bearer " + token},
                           timeout=6, allow_redirects=False)
            if response.status_code != 200:
                return state, "http_" + str(int(response.status_code))
            records = _clean(req["kind"], response.json())
        except requests.RequestException:
            return state, "connection_failed"
        except (ValueError, TypeError):
            return state, "invalid_response"
        state["batches"].append({"request": req, "retrieved_at": now(), "records": records})
    return state, "requests_complete" if not pending_requests(state) else "batch_saved"


def checkpoint_bytes(state):
    return json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def load_checkpoint(raw):
    """Import only our schema and fixed request shapes; never accept arbitrary URLs."""
    try:
        if len(raw) > 40_000_000:
            raise ValueError()
        value = json.loads(raw)
        if not isinstance(value, dict) or set(value) != {"schema", "years", "batches"} or value["schema"] != 1:
            raise ValueError()
        years = value["years"]
        if (not isinstance(years, list) or len(years) != 3 or any(not integer(y) for y in years)
            or years != list(range(years[0], years[0] + 3))
            or not 2000 <= years[0] <= datetime.now(timezone.utc).year - 3):
            raise ValueError()
        if not isinstance(value["batches"], list) or len(value["batches"]) > 189:
            raise ValueError()
        clean = {"schema": 1, "years": years, "batches": []}
        for batch in value["batches"]:
            if set(batch) != {"request", "retrieved_at", "records"}:
                raise ValueError()
            if batch["request"] not in pending_requests(clean) or timestamp(batch["retrieved_at"]) is None:
                raise ValueError()
            clean["batches"].append({"request": batch["request"], "retrieved_at": batch["retrieved_at"],
                                    "records": _clean(batch["request"]["kind"], batch["records"])})
        checkpoint_bytes(clean)
        return clean
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        raise ValueError("Invalid NCAAF collection checkpoint") from None


def _valid_game(g, year):
    return (g.get("season") == year and integer(g.get("id"))
            and integer(g.get("homeId")) and integer(g.get("awayId")) and g["homeId"] != g["awayId"]
            and all(integer(g.get(k)) and g[k] >= 0 for k in ("homePoints", "awayPoints"))
            and g.get("completed") is True and g.get("startTimeTBD") is False
            and timestamp(g.get("startDate")) is not None
            and integer(g.get("week")) and 0 <= g["week"] <= 30
            and g.get("seasonType") in ("regular", "postseason"))


def _yards(team):
    values = [s.get("stat") for s in team["stats"] if s.get("category") == "totalYards"]
    try:
        result = float(values[0]) if len(values) == 1 else None
        return result if result is not None and math.isfinite(result) and result >= 0 else None
    except (ValueError, TypeError):
        return None


def build_dataset(state):
    games, stats, issues, seen, conflicts = {}, {}, [], {}, set()
    for b in state["batches"]:
        req = b["request"]
        if req["kind"] != "games":
            continue
        for g in b["records"]:
            gid = g.get("id")
            if not _valid_game(g, req["year"]):
                issues.append({"season": req["year"], "game_id": gid, "issue": "invalid_or_incomplete_game"})
                continue
            if gid in seen:
                issues.append({"season": req["year"], "game_id": gid, "issue": "duplicate_game"})
                if seen[gid] != g:
                    conflicts.add(gid)
            seen[gid] = g
    games = {gid: g for gid, g in seen.items() if gid not in conflicts}
    for gid in conflicts:
        issues.append({"season": seen[gid]["season"], "game_id": gid, "issue": "conflicting_game_excluded"})
    bad_stats = set()
    for b in state["batches"]:
        req = b["request"]
        if req["kind"] != "stats":
            continue
        for item in b["records"]:
            gid = item.get("id")
            if not integer(gid) or gid not in games:
                continue  # weekly endpoint also returns games outside the FBS slate
            g = games[gid]
            if (g["season"], g["week"], g["seasonType"]) != (req["year"], req["week"], req["season_type"]):
                issues.append({"season": g["season"], "game_id": gid, "issue": "stats_request_mismatch"})
                continue
            for t in item["teams"]:
                tid = t.get("teamId")
                if tid not in (g["homeId"], g["awayId"]):
                    continue
                key = (gid, tid)
                score = g["homePoints"] if tid == g["homeId"] else g["awayPoints"]
                if (not integer(t.get("points")) or t["points"] != score or not any(
                    isinstance(s.get("category"), str) and s["category"].strip()
                    and isinstance(s.get("stat"), (str, int, float))
                    and str(s["stat"]).strip() for s in t["stats"])):
                    bad_stats.add(key)
                if key in stats and stats[key] != t:
                    bad_stats.add(key)
                stats[key] = t
    for key in bad_stats:
        stats.pop(key, None)
    ordered = sorted(games.values(), key=lambda g: (timestamp(g["startDate"]), g["id"]))
    history = {}
    for g in ordered:
        for tid in (g["homeId"], g["awayId"]):
            history.setdefault((g["season"], tid), []).append(g)
    rows = []
    for g in ordered:
        gid = g["id"]
        if not all((gid, g[s + "Id"]) in stats for s in ("home", "away")):
            issues.append({"season": g["season"], "game_id": gid, "issue": "missing_or_conflicting_team_stats"})
        cutoff = timestamp(g["startDate"]) - timedelta(days=7)
        row = {"game_id": gid, "season": g["season"], "kickoff": g["startDate"],
               "home_id": g["homeId"], "away_id": g["awayId"], "neutral_site": g["neutralSite"],
               "feature_cutoff": cutoff.isoformat(), "availability_basis": "seven_day_lag_proxy",
               "historical_publication_time_verified": False, "production_eligible": False}
        for side in ("home", "away"):
            tid = g[side + "Id"]
            prior = [p for p in history[(g["season"], tid)]
                     if timestamp(p["startDate"]) < cutoff]
            own, against, yards = [], [], []
            for p in prior:
                home = p["homeId"] == tid
                own.append(p["homePoints"] if home else p["awayPoints"])
                against.append(p["awayPoints"] if home else p["homePoints"])
                t = stats.get((p["id"], tid))
                y = _yards(t) if t else None
                if y is not None:
                    yards.append(y)
            row.update({side + "_prior_games": len(prior),
                        side + "_prior_game_ids": ";".join(str(p["id"]) for p in prior),
                        side + "_ppg": sum(own) / len(own) if own else None,
                        side + "_oppg": sum(against) / len(against) if against else None,
                        side + "_yards_games": len(yards),
                        side + "_yards_per_game": sum(yards) / len(yards) if yards else None})
        row["scoring_features_available"] = min(row["home_prior_games"], row["away_prior_games"]) >= 3
        rows.append(row)
    summaries = []
    for y in state["years"]:
        batches = [b for b in state["batches"] if b["request"] == {"kind": "games", "year": y}]
        season_games = [g for g in ordered if g["season"] == y]
        summaries.append({"season": y, "schedule_fetched": bool(batches),
                          "returned_games": sum(len(b["records"]) for b in batches),
                          "usable_games": len(season_games),
                          "games_with_both_team_stats": sum(all((g["id"], g[s + "Id"]) in stats for s in ("home", "away")) for g in season_games),
                          "rows_with_scoring_features": sum(r["scoring_features_available"] for r in rows if r["season"] == y)})
    audit = {"schema": 1, "requests_finished": len(state["batches"]), "requests_remaining": len(pending_requests(state)),
             "seasons": summaries, "issues": issues,
             "limitations": ["Completed requests do not prove full provider schedule coverage.",
                             "Historical publication and correction timestamps are unavailable; seven-day lag is a research assumption.",
                             "Features use same-season prior games only; early-season rows remain sparse.",
                             "No historical odds, model training, validation, or wagering approval is included."]}
    targets = [{"game_id": g["id"], "home_score": g["homePoints"], "away_score": g["awayPoints"],
                "home_margin": g["homePoints"] - g["awayPoints"], "total_points": g["homePoints"] + g["awayPoints"]} for g in ordered]
    return audit, rows, targets


def archive_bytes(state):
    audit, features, targets = build_dataset(state)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("checkpoint.json", checkpoint_bytes(state))
        archive.writestr("coverage-audit.json", json.dumps(audit, indent=2))
        for name, rows in (("research-features.csv", features), ("targets.csv", targets)):
            text = io.StringIO(newline="")
            if rows:
                writer = csv.DictWriter(text, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            archive.writestr(name, text.getvalue())
    return out.getvalue()


def backup_checkpoint(state, *, client=None, folder=None):
    from app_core.evidence_remote import settings
    from app_core.evidence_drive import DriveStore, AlreadyExists
    folder = folder or settings()[0]
    client = client or DriveStore(folder)
    raw = checkpoint_bytes(state)
    key = "parlaypicker/ncaaf-history-v1/" + hashlib.sha256(raw).hexdigest() + ".json"
    try:
        client.put_object(Bucket=folder, Key=key, Body=raw, ContentType="application/json", IfNoneMatch="*")
    except AlreadyExists:
        pass
    with client.get_object(Bucket=folder, Key=key)["Body"] as body:
        if body.read() != raw:
            raise ValueError("NCAAF backup verification failed")
    return key
