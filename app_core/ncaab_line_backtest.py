"""Market-anchored NCAAB model screened on archived lines (roadmap C5, the NCAAB analogue of D18).

NCAAB spread is the first research market (D14), but its in-season models wait for 60
captured games. This asks the question early, on history: given the closing-era
archived line, can simple team form predict what the line misses?

* Data: CollegeBasketballData ``/games`` and ``/lines`` for two seasons, in date windows
  (about 30 calls) so the 3,000-row cap per request never truncates a season.
* Features, strictly as of each game: each team's average margin, points for and
  points against over its earlier games that season (shrunk toward 0 with 5 pseudo
  games), the home-minus-away differences, a neutral-site flag, and the line itself.
* Targets: home cover margin (margin + home spread) and total error (points - line).
* Ridge fit on the earlier season; bias and sigma from the first half of the later
  season; scored once on its second half at -110.

With no real signal P(cover) stays near 50% and few picks clear a threshold. Archived
lines carry no quote timestamp, so this screens the idea and is not plan evidence.
Research only: writes nothing to the evidence store.
"""
from __future__ import annotations

from datetime import datetime
import math

import numpy as np

BASE_URL = "https://api.collegebasketballdata.com"
PROVIDER_ORDER = ("consensus", "ESPN Bet", "DraftKings", "Bovada", "William Hill (New Jersey)", "teamrankings")
SHRINK = 5.0
MIN_PRIOR_GAMES = 5
ALPHA = 25.0
THRESHOLDS = (0.52, 0.55, 0.58)
WIN = 100 / 110
FEATURES = ("margin_diff", "points_for_sum", "points_against_sum", "home_margin", "away_margin", "neutral")


ROW_CAP = 3000          # CBBD returns at most this many rows per request
MAX_CALLS = 80


def _request(endpoint, params, token, get):
    response = get(f"{BASE_URL}/{endpoint}", params=params,
                   headers={"Authorization": "Bearer " + token}, timeout=60, allow_redirects=False)
    if response.status_code != 200:
        raise ValueError(f"CBBD {endpoint} HTTP {int(response.status_code)}")
    value = response.json()
    if not isinstance(value, list):
        raise ValueError(f"CBBD {endpoint} response is not a list")
    return value


def fetch(endpoint, season, token, get):
    """Every row of a season, requested in date windows so the per-request row cap never truncates it.

    Windows start monthly (October to April); a window that returns ``ROW_CAP`` rows
    is split in half and fetched again. Rows are de-duplicated by game id.
    """
    from datetime import date, timedelta
    pending = [(date(season - 1, 10, 1), date(season - 1, 11, 1))]
    first = date(season - 1, 11, 1)
    while first < date(season, 5, 1):
        nxt = date(first.year + (first.month == 12), first.month % 12 + 1, 1)
        pending.append((first, nxt))
        first = nxt
    rows, seen, calls = [], set(), 0
    while pending:
        start, end = pending.pop(0)
        calls += 1
        if calls > MAX_CALLS:
            raise ValueError(f"CBBD {endpoint}: more than {MAX_CALLS} requests for season {season}")
        chunk = _request(endpoint, {"season": season, "startDateRange": start.isoformat(),
                                    "endDateRange": (end - timedelta(days=1)).isoformat()}, token, get)
        if len(chunk) >= ROW_CAP:
            if (end - start).days <= 1:
                raise ValueError(f"CBBD {endpoint}: a single day returned {len(chunk)} rows")
            middle = start + (end - start) // 2
            pending[:0] = [(start, middle), (middle, end)]
            continue
        for row in chunk:
            key = _first(row, "id", "gameId")
            if key is None or key not in seen:
                seen.add(key)
                rows.append(row)
    return rows


def _first(row, *keys):
    for key in keys:
        if row.get(key) is not None:
            return row[key]
    return None


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _when(value):
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def parse_games(rows):
    """Finished games: [{id, start, home, away, home_points, away_points, neutral}] in start order."""
    out = []
    for row in rows:
        game = {"id": _first(row, "id", "gameId"), "start": _when(_first(row, "startDate", "start_date")),
                "home": _first(row, "homeTeam", "home_team"), "away": _first(row, "awayTeam", "away_team"),
                "home_points": _number(_first(row, "homePoints", "home_points")),
                "away_points": _number(_first(row, "awayPoints", "away_points")),
                "neutral": bool(_first(row, "neutralSite", "neutral_site"))}
        if None in (game["id"], game["start"], game["home"], game["away"], game["home_points"], game["away_points"]):
            continue
        out.append(game)
    return sorted(out, key=lambda g: (g["start"], str(g["id"])))


def select_lines(rows):
    """{game_id: {spread, total, provider}}: one archived line per game, home perspective."""
    rank = {name: i for i, name in enumerate(PROVIDER_ORDER)}
    out = {}
    for row in rows:
        game_id = _first(row, "gameId", "id")
        lines = sorted((l for l in row.get("lines") or [] if isinstance(l, dict)),
                       key=lambda l: rank.get(l.get("provider"), len(PROVIDER_ORDER)))
        spread = total = provider = None
        for line in lines:
            spread = spread if spread is not None else _number(_first(line, "spread"))
            total = total if total is not None else _number(_first(line, "overUnder", "over_under", "total"))
            provider = provider or line.get("provider")
            if spread is not None and total is not None:
                break
        if game_id is not None and (spread is not None or total is not None):
            out[game_id] = {"spread": spread, "total": total, "provider": provider}
    return out


def features(games):
    """{game_id: feature dict} from each team's earlier games only (needs MIN_PRIOR_GAMES each)."""
    form, out = {}, {}

    def rating(team):
        n, margin, pf, pa = form.get(team, (0, 0.0, 0.0, 0.0))
        scale = 1.0 / (n + SHRINK)
        return n, margin * scale, pf * scale, pa * scale

    for game in games:
        hn, hm, hpf, hpa = rating(game["home"])
        an, am, apf, apa = rating(game["away"])
        if hn >= MIN_PRIOR_GAMES and an >= MIN_PRIOR_GAMES:
            out[game["id"]] = {"margin_diff": hm - am, "points_for_sum": hpf + apf,
                               "points_against_sum": hpa + apa, "home_margin": hm, "away_margin": am,
                               "neutral": 1.0 if game["neutral"] else 0.0}
        margin = game["home_points"] - game["away_points"]
        for team, m, pf, pa in ((game["home"], margin, game["home_points"], game["away_points"]),
                                (game["away"], -margin, game["away_points"], game["home_points"])):
            n, tm, tpf, tpa = form.get(team, (0, 0.0, 0.0, 0.0))
            form[team] = (n + 1, tm + m, tpf + pf, tpa + pa)
    return out


def rows(games, feats, lines, market):
    """(feature vector incl. line, target) per game with a line and features."""
    out = []
    for game in games:
        line = (lines.get(game["id"]) or {}).get("spread" if market == "SPREAD" else "total")
        if line is None or game["id"] not in feats:
            continue
        margin = game["home_points"] - game["away_points"]
        total = game["home_points"] + game["away_points"]
        target = margin + line if market == "SPREAD" else total - line
        out.append(([feats[game["id"]][k] for k in FEATURES] + [line], target, game))
    return out


def fit(train, calibration, market):
    if len(train) < 100 or len(calibration) < 50:
        return None
    x = np.array([r[0] for r in train], dtype=float)
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    z = (x - mean) / scale
    y = np.array([r[1] for r in train], dtype=float)
    coef = np.linalg.solve(z.T @ z + ALPHA * np.eye(z.shape[1]), z.T @ (y - y.mean()))
    model = {"market": market, "mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(),
             "intercept": float(y.mean()), "train_games": len(train)}
    predicted = predict(model, calibration)
    actual = np.array([r[1] for r in calibration], dtype=float)
    model["bias"] = float(np.mean(actual - predicted))
    model["sigma"] = max(1.0, float(np.sqrt(np.mean((actual - predicted - model["bias"]) ** 2))))
    model["calibration_games"] = len(calibration)
    return model


def predict(model, data):
    x = np.array([r[0] for r in data], dtype=float)
    z = (x - np.array(model["mean"])) / np.array(model["scale"])
    return z @ np.array(model["coef"]) + model["intercept"] + model.get("bias", 0.0)


def evaluate(model, holdout):
    """Home/Over when P(target > 0) >= 0.5, else away/Under; graded at -110 per threshold."""
    if not holdout:
        return {"games": 0}
    centres = predict(model, holdout)
    graded = []
    for (_, target, _), mu in zip(holdout, centres):
        p = 0.5 * (1 + math.erf(mu / (model["sigma"] * math.sqrt(2))))
        side_a = p >= 0.5
        outcome = "PUSH" if abs(target) < 1e-9 else ("WIN" if (target > 0) == side_a else "LOSS")
        graded.append((p if side_a else 1 - p, p, target, outcome))
    decided = [(p, int(t > 0)) for _, p, t, o in graded if o != "PUSH"]
    out = {"games": len(graded), "brier": sum((p - y) ** 2 for p, y in decided) / len(decided) if decided else None,
           "mean_abs_centre": float(np.mean(np.abs(centres))), "by_threshold": {}}
    for threshold in THRESHOLDS:
        picks = [g for g in graded if g[0] >= threshold]
        wins = sum(1 for g in picks if g[3] == "WIN")
        losses = sum(1 for g in picks if g[3] == "LOSS")
        out["by_threshold"][f"{threshold:.2f}"] = {
            "picks": len(picks), "wins": wins, "losses": losses,
            "hit_rate": wins / (wins + losses) if wins + losses else None,
            "roi_per_pick": (wins * WIN - losses) / len(picks) if picks else None}
    return out


def run(train_games, test_games, train_lines, test_lines):
    """Fit on the earlier season, calibrate on the first half of the later one, score its second half."""
    train_feats, test_feats = features(train_games), features(test_games)
    report = {}
    for market in ("SPREAD", "TOTAL"):
        train = rows(train_games, train_feats, train_lines, market)
        later = rows(test_games, test_feats, test_lines, market)
        half = len(later) // 2
        model = fit(train, later[:half], market)
        if model is None:
            report[market] = {"games": 0, "blocker": "TOO_FEW_LINED_GAMES",
                              "train_rows": len(train), "later_rows": len(later)}
            continue
        report[market] = dict(evaluate(model, later[half:]), train_games=model["train_games"],
                              calibration_games=model["calibration_games"], sigma=model["sigma"])
    return report


def markdown(result):
    lines = ["## NCAAB market-anchored model vs archived lines", "",
             f"- seasons: train {result['train_season']}, calibrate and score {result['test_season']} "
             f"(first half calibrates, second half scores)",
             f"- finished games: {result['train_games']} / {result['test_games']}; with a line: "
             f"{result['train_lined']} / {result['test_lined']}",
             f"- break-even hit rate at -110: {110 / 210:.1%}", "",
             "| market | scored games | mean |predicted edge| (pts) | min P(cover) | picks | hit rate | ROI/pick | Brier |",
             "|---|---|---|---|---|---|---|---|"]
    for market, entry in result["report"].items():
        if entry.get("blocker"):
            lines.append(f"| {market} | 0 | | | | | | {entry['blocker']} |")
            continue
        for threshold, row in entry["by_threshold"].items():
            hit = "" if row["hit_rate"] is None else f"{row['hit_rate']:.1%}"
            roi = "" if row["roi_per_pick"] is None else f"{row['roi_per_pick']:+.3f}"
            brier = "" if entry["brier"] is None else f"{entry['brier']:.3f}"
            lines.append(f"| {market} | {entry['games']} | {entry['mean_abs_centre']:.2f} | {threshold} | "
                         f"{row['picks']} | {hit} | {roi} | {brier} |")
    lines += ["", "Archived lines have no quote timestamp; this screens the model and is not validation evidence."]
    return "\n".join(lines) + "\n"
