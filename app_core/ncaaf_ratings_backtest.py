"""NCAAF ratings vs archived opening and closing lines (owner 2026-11-09, research only).

D18 found no edge for the frozen NCAAF score model against archived closing lines, and the NCAAB
ratings screen (``ncaab_ratings_backtest``, 2026-11-08) found none against closing lines either
(k = 0). It could not test opening lines, because CBBD has them only from January 2025. CFBD
has opening spreads and totals for several football seasons, so this asks the open question for
football: **are weekly power ratings ahead of the opening line?**

Two models, each from data available before the game:

* ``ratings``: the opponent-adjusted ridge ratings of ``ncaab_ratings_backtest.RatingBook`` with
  football settings (``PARAMS``), carried from season to season (regressed by ``carry``).
  Gives a margin and a total.
* ``elo``: CFBD's own pregame Elo on each game (``homePregameElo``/``awayPregameElo``), turned into
  a margin at ``ELO_PER_POINT`` Elo points per point plus ``ELO_HOME_POINTS`` at home. Spreads only.

For each model, market and line type (opening, closing) the shrink ``k`` and ``sigma`` are fitted
on the calibration seasons, then each scored season is graded once, split into August-September
(early season, where openers are softest and ratings lean most on last year) and October on.
Prices are assumed at -110; archived lines carry no quote timestamp, so this screens the idea and
is not plan evidence. Research only: it never writes evidence, plans or wagers.
"""
from __future__ import annotations

from datetime import datetime
import math

from app_core import ncaab_ratings_backtest as core
from app_core.ncaaf_line_backtest import PROVIDER_ORDER

BASE_URL = "https://api.collegefootballdata.com"
PARAMS = {"margin_cap": 35.0, "carry": 0.6, "prior_weight": 3.0, "hca_prior": 2.5, "hca_weight": 20.0,
          "team_points": 28.0}
ELO_PER_POINT = 25.0
ELO_HOME_POINTS = 2.5
SEGMENTS = (("Aug-Sep", (8, 9)), ("Oct-Jan", (10, 11, 12, 1)))
MODELS = {"SPREAD": ("ratings", "elo"), "TOTAL": ("ratings",)}


def _get(path, params, token, get):
    response = get(f"{BASE_URL}{path}", params=params, headers={"Authorization": "Bearer " + token},
                   timeout=30, allow_redirects=False)
    if response.status_code != 200:
        raise ValueError(f"CFBD {path} HTTP {int(response.status_code)}")
    value = response.json()
    if not isinstance(value, list):
        raise ValueError(f"CFBD {path} response is not a list")
    return value


def fetch(kind, year, token, get):
    """One season of ``games`` (regular and postseason, games with an FBS team) or ``lines``."""
    if kind == "games":
        return _get("/games", {"year": year, "seasonType": "both", "classification": "fbs"}, token, get)
    return [row for season_type in ("regular", "postseason")
            for row in _get("/lines", {"year": year, "seasonType": season_type}, token, get)]


def _number(value):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def parse_games(rows):
    """Finished games in start order: {id, start, home, away, home_points, away_points, neutral, *_elo}."""
    out = []
    for row in rows:
        try:
            start = datetime.fromisoformat(str(row.get("startDate")).replace("Z", "+00:00"))
        except ValueError:
            continue
        game = {"id": row.get("id"), "start": start, "home": row.get("homeTeam"), "away": row.get("awayTeam"),
                "home_points": _number(row.get("homePoints")), "away_points": _number(row.get("awayPoints")),
                "neutral": bool(row.get("neutralSite")),
                "home_elo": _number(row.get("homePregameElo")), "away_elo": _number(row.get("awayPregameElo"))}
        if row.get("completed") is False or None in (game["id"], game["home"], game["away"],
                                                     game["home_points"], game["away_points"]):
            continue
        out.append(game)
    return sorted(out, key=lambda g: (g["start"], str(g["id"])))


def select_lines(rows):
    return core.select_lines(rows, order=PROVIDER_ORDER)


def elo_margin(game):
    if game["home_elo"] is None or game["away_elo"] is None:
        return None
    return (game["home_elo"] - game["away_elo"]) / ELO_PER_POINT + (0.0 if game["neutral"] else ELO_HOME_POINTS)


def predictions(seasons):
    """{model: {game_id: (margin, total)}} for every season in order; ratings carry across seasons."""
    out = {"ratings": {}, "elo": {}}
    prior = None
    for year in sorted(seasons):
        season, prior = core.season_predictions(seasons[year], prior, PARAMS)
        out["ratings"].update(season)
        for game in seasons[year]:
            margin = elo_margin(game)
            if margin is not None:
                out["elo"][game["id"]] = (margin, None)
    return out


def _cases(games, predicted, lines, market, line_key):
    if market == "TOTAL":
        predicted = {gid: p for gid, p in predicted.items() if p[1] is not None}
    return core._cases(games, predicted, lines, market, line_key)


def run(seasons, lines, calibration_years, scored_years):
    """Fit k and sigma on ``calibration_years``, grade each of ``scored_years`` once, by segment."""
    predicted = predictions(seasons)
    calibration_games = [g for y in calibration_years for g in seasons.get(y, [])]
    calibration_lines = {gid: v for y in calibration_years for gid, v in lines.get(y, {}).items()}
    report = {}
    for market, models in MODELS.items():
        close_key, open_key = ("spread", "spread_open") if market == "SPREAD" else ("total", "total_open")
        for model in models:
            entry = {"fits": {}, "lines": {}}
            for label, key in (("closing", close_key), ("opening", open_key)):
                fit = core.calibrate(_cases(calibration_games, predicted[model], calibration_lines, market, key))
                entry["fits"][label] = fit
                entry["lines"][label] = {}
                for year in scored_years:
                    cases = _cases(seasons.get(year, []), predicted[model], lines.get(year, {}), market, key)
                    scored = {"games": len(cases)}
                    if fit is None:
                        scored["blocker"] = "TOO_FEW_CALIBRATION_GAMES"
                    else:
                        for name, months in SEGMENTS:
                            part = [c for c in cases if c[2]["start"].month in months]
                            scored[name] = core.score(part, fit) if part else {"games": 0}
                    entry["lines"][label][str(year)] = scored
            report[f"{market} {model}"] = entry
    return report


def markdown(result):
    out = ["## NCAAF ratings and Elo vs archived opening and closing lines (CFBD)", "",
           f"- seasons: ratings from {min(result['seasons'])}, calibrate {result['calibration_years']}, "
           f"score {result['scored_years']}",
           f"- finished games by season: {result['games']}",
           f"- games with a closing / opening spread by season: {result['lined']}",
           f"- break-even hit rate at -110: {110 / 210:.1%}", "",
           "| market model | line | season | segment | games | Brier | P >= | picks | hit rate | ROI/pick |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for name, entry in result["report"].items():
        for label, by_year in entry["lines"].items():
            fit = entry["fits"].get(label) or {}
            out.append(f"| {name} | {label} | fit | k = {fit.get('k', 0):.2f}, sigma = {fit.get('sigma', 0):.1f} "
                       f"({fit.get('games', 0)} calibration games) | | | | | | |")
            for year, scored in by_year.items():
                if scored.get("blocker"):
                    out.append(f"| {name} | {label} | {year} | {scored['blocker']} | {scored['games']} | | | | | |")
                    continue
                for segment, _ in SEGMENTS:
                    part = scored.get(segment) or {}
                    if not part.get("games"):
                        out.append(f"| {name} | {label} | {year} | {segment} | 0 | | | | | |")
                        continue
                    brier = "" if part["brier"] is None else f"{part['brier']:.3f}"
                    for threshold, row in part["by_threshold"].items():
                        hit = "" if row["hit_rate"] is None else f"{row['hit_rate']:.1%}"
                        roi = "" if row["roi_per_pick"] is None else f"{row['roi_per_pick']:+.3f}"
                        out.append(f"| {name} | {label} | {year} | {segment} | {part['games']} | {brier} | "
                                   f"{threshold} | {row['picks']} | {hit} | {roi} |")
    out += ["", "A pick is the side whose probability clears the threshold, graded at -110. Archived lines have "
                "no quote timestamp; this screens the idea and is not plan evidence."]
    return "\n".join(out) + "\n"
