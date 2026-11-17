"""NCAAB possession-based model vs archived opening and closing lines (owner 2026-11-17, research only).

Stage 1 of the college basketball model (roadmap C5). ``ncaab_ratings_backtest`` rates teams on
points. This rates them the way the public college-basketball systems do, from team box scores
(CollegeBasketballData ``/games/teams``):

* **Tempo**: possessions per 40 minutes, ``tau + t_home + t_away``.
* **Efficiency**: points per possession, ``mu + o_team - d_opponent``, plus a home edge ``a`` for the
  home side and minus it for the visitor (nothing on a neutral floor).
* **Prediction**: each side's points = tempo x efficiency x the season's overtime factor; the margin
  and the total follow.

Everything is as of each game day (``PossessionBook``): ridge least squares on that season's earlier
games, each team pulled toward a share of last season's final rating. Three things the points
ratings do not do:

* teams with under ``MIN_GAMES`` games in the season (opponents from outside Division I) are pooled
  into one team, so beating one by 40 is not read as beating an average team by 40;
* a blowout can count for at most ``margin_cap`` points, by pulling the two efficiencies together;
* how strong the prior is, how much of last season carries over and whether blowouts are capped are
  chosen from a small grid on the calibration season (``tune``), never on the scored one.

A day runs from 10:00 UTC to 10:00 UTC (5 AM Eastern), so a late West Coast game never sees a game
that was still being played when it tipped off.

``run`` sets the model beside the points ratings on the same games: how far each misses the final
margin and total next to how far the lines themselves miss (``accuracy``), then the readings of
``ncaab_ratings_backtest.evaluate`` (fitted, by disagreement, line moves), the line moves again by
conference strength (``by_tier``), since a soft opening number is likelier in a small conference,
which side the model took (``by_side``) and how the lines lean with no model at all (``market_lean``).
Archived lines carry no quote timestamp: this screens the idea and is not plan evidence. Research
only: nothing here reaches a pick.
"""
from __future__ import annotations

from collections import Counter
from datetime import timedelta
import math

import numpy as np

from app_core import ncaab_line_backtest as data
from app_core import ncaab_ratings_backtest as ratings
from app_core.ncaab_line_backtest import _first, _number
from app_core.ncaab_ratings_backtest import SEGMENTS, _accumulate

POOLED = "__other__"            # every team with too few games to rate
MIN_GAMES = 15                  # a Division I team plays about 30; an outside opponent appears a handful of times
DAY_STARTS_UTC_HOUR = 10        # 5 AM Eastern: no game is in progress
PARAMS = {"carry": 0.7, "carry_tempo": 0.8, "prior_weight": 4.0, "margin_cap": 25.0,
          "home_edge": 0.022, "home_weight": 40.0}
# Chosen on the calibration season: how much of last season carries over, how many games the prior counts
# for, and whether a blowout is capped (0 = every point counts).
GRID = tuple({"carry": carry, "prior_weight": weight, "margin_cap": cap}
             for carry in (0.5, 0.7, 0.9) for weight in (2.0, 4.0, 8.0) for cap in (25.0, 0.0))
LEAGUE = {"efficiency": 1.04, "tempo": 68.0, "overtime": 1.0075}     # only until a season of games says otherwise
OVERTIME_PRIOR_GAMES = 200.0
TOP_CONFERENCES = 7


def parse_games(rows):
    """``ncaab_line_backtest.parse_games`` plus each side's conference and the number of periods played."""
    games = data.parse_games(rows)
    raw = {_first(row, "id", "gameId"): row for row in rows}
    for game in games:
        row = raw.get(game["id"]) or {}
        game["home_conference"], game["away_conference"] = row.get("homeConference"), row.get("awayConference")
        game["periods"] = len(row.get("homePeriodPoints") or [])
    return games


def _side_possessions(stats):
    """One side's possessions: the provider's figure, or the box-score estimate where it is missing."""
    if not isinstance(stats, dict):
        return None
    value = _number(stats.get("possessions"))
    if value is not None:
        return value
    try:
        parts = (_number(stats["fieldGoals"]["attempted"]), _number(stats["rebounds"]["offensive"]),
                 _number(stats["turnovers"]["total"]), _number(stats["freeThrows"]["attempted"]))
    except (KeyError, TypeError):
        return None
    if None in parts:
        return None
    attempts, offensive_rebounds, turnovers, free_throws = parts
    return attempts - offensive_rebounds + turnovers + 0.475 * free_throws


def _box_minutes(row):
    """Game length in minutes (40, 45, ...) as the box-score row gives it, or None when it does not."""
    minutes = _number(row.get("gameMinutes"))
    if minutes is not None and 190.0 <= minutes <= 360.0:      # recorded as player minutes: five on the floor
        minutes /= 5.0
    return minutes if minutes is not None and 38.0 <= minutes <= 72.0 else None


def parse_box(rows):
    """{game_id: {"possessions", "minutes", "pace"}} from ``/games/teams`` rows. One row per game is enough.

    A row carries its own team's numbers and its opponent's; the game's possessions are the average
    of the two sides. A game outside 40 to 120 possessions is left out (bad data).
    """
    out = {}
    for row in rows:
        game_id = _first(row, "gameId", "id")
        if game_id is None or game_id in out:
            continue
        sides = [value for value in (_side_possessions(row.get("teamStats")),
                                     _side_possessions(row.get("opponentStats"))) if value is not None]
        if not sides:
            continue
        possessions = sum(sides) / len(sides)
        if 40.0 <= possessions <= 120.0:
            out[game_id] = {"possessions": possessions, "minutes": _box_minutes(row), "pace": _number(row.get("pace"))}
    return out


def game_minutes(game, entry=None):
    """Minutes played: the box score's figure, else two halves plus five minutes for each further period,
    else what the box score's pace (possessions per 40 minutes) implies, else 40."""
    entry = entry or {}
    if entry.get("minutes") is not None:
        return entry["minutes"]
    periods = game.get("periods") or 0
    if periods >= 2:
        return 40.0 + 5.0 * (periods - 2)
    pace, possessions = entry.get("pace"), entry.get("possessions")
    if pace is not None and possessions is not None and 50.0 <= pace <= 95.0:
        implied = 5.0 * round(40.0 * possessions / pace / 5.0)
        if 40.0 <= implied <= 70.0:
            return implied
    return 40.0


def game_tempo(game, box):
    """(possessions per 40 minutes, minutes played) for a game with a box score, else None."""
    entry = box.get(game["id"])
    if not entry:
        return None
    minutes = game_minutes(game, entry)
    return entry["possessions"] * 40.0 / minutes, minutes


def rated_teams(games):
    """Teams with at least ``MIN_GAMES`` games in the season's list; the rest are pooled as one team."""
    played = Counter()
    for game in games:
        played[game["home"]] += 1
        played[game["away"]] += 1
    return {team for team, count in played.items() if count >= MIN_GAMES}


class PossessionBook:
    """Tempo and efficiency ratings accumulated game by game; ``solve`` gives them as of now."""

    def __init__(self, teams, prior=None, params=None):
        self.p = {**PARAMS, **(params or {})}
        self.index = {team: i for i, team in enumerate(sorted(teams))}
        n = self.n = len(self.index)
        prior = prior or {}
        league = {**LEAGUE, **(prior.get("__league__") or {})}
        # efficiency: [o_0..o_n-1, d_0..d_n-1, mu, a]; tempo: [t_0..t_n-1, tau]
        self.e_mat, self.e_vec, self.e_prior = np.zeros((2 * n + 2, 2 * n + 2)), np.zeros(2 * n + 2), np.zeros(2 * n + 2)
        self.t_mat, self.t_vec, self.t_prior = np.zeros((n + 1, n + 1)), np.zeros(n + 1), np.zeros(n + 1)
        for team, i in self.index.items():
            offence, defence, tempo = prior.get(team) or (0.0, 0.0, 0.0)
            self.e_prior[i], self.e_prior[n + i] = self.p["carry"] * offence, self.p["carry"] * defence
            self.t_prior[i] = self.p["carry_tempo"] * tempo
        self.e_prior[2 * n], self.e_prior[2 * n + 1] = league["efficiency"], self.p["home_edge"]
        self.t_prior[n] = league["tempo"]
        self.overtime_prior = league["overtime"]
        self.games = self.boxed = 0
        self.minutes = self.tempo_sum = 0.0

    def average_tempo(self):
        return self.tempo_sum / self.boxed if self.boxed else float(self.t_prior[self.n])

    def add(self, game, tempo=None):
        """One finished game. ``tempo`` is ``game_tempo`` output; a game without one is given the average pace so far."""
        n = self.n
        h, a = self.index[game["home"]], self.index[game["away"]]
        site = 0.0 if game["neutral"] else 1.0
        per40, minutes = tempo if tempo is not None else (self.average_tempo(), game_minutes(game))
        possessions = per40 * minutes / 40.0                   # the game's own possessions, overtime included
        home, away = game["home_points"] / possessions, game["away_points"] / possessions
        margin, cap = game["home_points"] - game["away_points"], self.p["margin_cap"]
        if cap and abs(margin) > cap:
            # A blowout counts for ``cap`` points: pull the two efficiencies together, keeping their sum.
            excess = math.copysign((abs(margin) - cap) / possessions / 2.0, margin)
            home, away = home - excess, away + excess
        _accumulate(self.e_mat, self.e_vec, ((h, 1.0), (n + a, -1.0), (2 * n, 1.0), (2 * n + 1, site)), home)
        _accumulate(self.e_mat, self.e_vec, ((a, 1.0), (n + h, -1.0), (2 * n, 1.0), (2 * n + 1, -site)), away)
        if tempo is not None:
            _accumulate(self.t_mat, self.t_vec, ((h, 1.0), (a, 1.0), (n, 1.0)), per40)
            self.boxed += 1
            self.minutes += minutes
            self.tempo_sum += per40
        self.games += 1

    def solve(self):
        n, weight = self.n, self.p["prior_weight"]
        ew = np.full(2 * n + 2, weight)
        ew[2 * n], ew[2 * n + 1] = 1e-3, self.p["home_weight"]
        efficiency = np.linalg.solve(self.e_mat + np.diag(ew), self.e_vec + ew * self.e_prior)
        tw = np.full(n + 1, weight)
        tw[n] = 1e-3
        tempo = np.linalg.solve(self.t_mat + np.diag(tw), self.t_vec + tw * self.t_prior)
        overtime = ((self.minutes / 40.0 + OVERTIME_PRIOR_GAMES * self.overtime_prior)
                    / (self.boxed + OVERTIME_PRIOR_GAMES))
        return efficiency, tempo, overtime

    def predict(self, game, efficiency, tempo, overtime):
        """(home margin, total) for a game between two teams in the book."""
        n = self.n
        h, a = self.index[game["home"]], self.index[game["away"]]
        site = 0.0 if game["neutral"] else 1.0
        mu, edge = efficiency[2 * n], efficiency[2 * n + 1]
        possessions = (tempo[n] + tempo[h] + tempo[a]) * overtime
        home = possessions * (mu + site * edge + efficiency[h] - efficiency[n + a])
        away = possessions * (mu - site * edge + efficiency[a] - efficiency[n + h])
        return float(home - away), float(home + away)

    def finals(self):
        efficiency, tempo, overtime = self.solve()
        n = self.n
        out = {team: (float(efficiency[i]), float(efficiency[n + i]), float(tempo[i])) for team, i in self.index.items()}
        out["__league__"] = {"efficiency": float(efficiency[2 * n]), "tempo": float(tempo[n]), "overtime": float(overtime)}
        return out


def _day(game):
    return (game["start"] - timedelta(hours=DAY_STARTS_UTC_HOUR)).date()


def season_predictions(games, box, prior=None, params=None):
    """({game_id: (margin, total)}, final ratings, coverage) with each game predicted from earlier days only.

    Only games between two rated teams are predicted. A game against a pooled team still teaches the
    book about the rated one; a game between two pooled teams is left out.
    """
    rated = rated_teams(games)
    book = PossessionBook(rated | {POOLED}, prior, params)
    out, day, solved = {}, None, None
    for game in games:
        home = game["home"] if game["home"] in rated else POOLED
        away = game["away"] if game["away"] in rated else POOLED
        if home == away:
            continue
        today = _day(game)
        if today != day:
            day, solved = today, book.solve()       # games of the same day never see each other
        named = {**game, "home": home, "away": away}
        if POOLED not in (home, away):
            out[game["id"]] = book.predict(named, *solved)
        book.add(named, game_tempo(game, box))
    return out, book.finals(), {"games": book.games, "with_box_score": book.boxed, "rated_teams": len(rated)}


def prediction_error(games, predictions):
    """Root mean square miss of the margin and of the total over the predicted games."""
    margin, total = [], []
    for game in games:
        if game["id"] in predictions:
            predicted_margin, predicted_total = predictions[game["id"]]
            margin.append(predicted_margin - (game["home_points"] - game["away_points"]))
            total.append(predicted_total - (game["home_points"] + game["away_points"]))
    if not margin:
        return {"games": 0, "margin_rmse": None, "total_rmse": None}
    return {"games": len(margin), "margin_rmse": float(np.sqrt(np.mean(np.square(margin)))),
            "total_rmse": float(np.sqrt(np.mean(np.square(total))))}


def tune(prior_games, prior_box, calibration_games, calibration_box, grid=GRID):
    """(the grid entry that predicts the calibration season best, every entry's error).

    Best is the smallest margin error plus total error. The first season has no prior to carry, so
    it is run once for each setting of everything but the carry.
    """
    first, rows = {}, []
    for entry in grid:
        rest = {key: value for key, value in entry.items() if not key.startswith("carry")}
        key = tuple(sorted(rest.items()))
        if key not in first:
            first[key] = season_predictions(prior_games, prior_box, None, rest)[1]
        predictions, _, _ = season_predictions(calibration_games, calibration_box, first[key], entry)
        rows.append({**entry, **prediction_error(calibration_games, predictions)})
    scored = [row for row in rows if row["margin_rmse"] is not None]
    if not scored:
        return dict(grid[0]), rows
    best = min(scored, key=lambda row: row["margin_rmse"] + row["total_rmse"])
    return {key: best[key] for key in grid[0]}, rows


def accuracy(games, models, lines):
    """Mean absolute miss of each model next to the opening and the closing line, on the same games.

    ``models`` is ``{name: predictions}``. A game counts only when every model predicted it and it
    has both lines, so the columns can be read against each other.
    """
    out = {}
    for market, open_key, close_key in (("SPREAD", "spread_open", "spread"), ("TOTAL", "total_open", "total")):
        misses = {name: {} for name in ("all",) + tuple(name for name, _ in SEGMENTS)}
        for game in games:
            line = lines.get(game["id"]) or {}
            opened, closed = line.get(open_key), line.get(close_key)
            if opened is None or closed is None or any(game["id"] not in predictions for predictions in models.values()):
                continue
            spread = market == "SPREAD"
            actual = (game["home_points"] - game["away_points"]) if spread else (game["home_points"] + game["away_points"])
            row = {name: abs(predictions[game["id"]][0 if spread else 1] - actual) for name, predictions in models.items()}
            row["opening_line"] = abs((-opened if spread else opened) - actual)
            row["closing_line"] = abs((-closed if spread else closed) - actual)
            for segment in ("all",) + tuple(name for name, months in SEGMENTS if game["start"].month in months):
                for column, miss in row.items():
                    misses[segment].setdefault(column, []).append(miss)
        out[market] = {segment: {"games": len(next(iter(columns.values()), [])),
                                 **{column: float(np.mean(values)) for column, values in columns.items()}}
                       for segment, columns in misses.items()}
    return out


def conference_tiers(games, finals, top=TOP_CONFERENCES):
    """{conference: "high" or "other"}: the ``top`` conferences by their teams' average rating last season.

    A team's rating is offence plus defence (both in points per possession above average). Membership
    is read from this season's games, which is known before it starts.
    """
    members = {}
    for game in games:
        for team, conference in ((game["home"], game.get("home_conference")), (game["away"], game.get("away_conference"))):
            if conference:
                members.setdefault(conference, set()).add(team)
    strength = {}
    for conference, teams in members.items():
        rated = [finals[team][0] + finals[team][1] for team in teams if isinstance(finals.get(team), tuple)]
        if len(rated) >= 4:
            strength[conference] = sum(rated) / len(rated)
    high = set(sorted(strength, key=strength.get, reverse=True)[:top])
    return {conference: ("high" if conference in high else "other") for conference in members}


def game_tier(game, tiers):
    """"high" when both sides are in a top conference, "low" when neither is, else "mixed"."""
    count = sum(1 for key in ("home_conference", "away_conference") if tiers.get(game.get(key)) == "high")
    return ("low", "mixed", "high")[count]


def by_tier(games, predictions, lines, pairs, tiers):
    """Per market and tier: the ratings' side at the opening line (no fit) and the line-move reading."""
    out = {}
    for market, open_key in (("SPREAD", "spread_open"), ("TOTAL", "total_open")):
        cases = ratings._cases(games, predictions, lines, market, open_key)
        moves = ratings.line_moves(games, predictions, pairs, market)
        out[market] = {}
        for tier in ("high", "mixed", "low"):
            part = [move for move in moves if game_tier(move[2], tiers) == tier]
            out[market][tier] = {
                "opening_disagreement": ratings.by_disagreement([c for c in cases if game_tier(c[2], tiers) == tier]),
                "line_moves": ratings.move_summary(part), "line_move_slope": ratings.move_slope(part)}
    return out


def by_side(games, predictions, lines):
    """The model's side at each opening line (no fit), split by which side it took.

    ``over_or_home`` is the Over for a total and the home side for a spread. A model that wins on
    one side only is leaning on a lean in the lines, not on what it knows about the two teams.
    """
    out = {}
    for market, open_key in (("SPREAD", "spread_open"), ("TOTAL", "total_open")):
        cases = ratings._cases(games, predictions, lines, market, open_key)
        out[market] = {"over_or_home": ratings.by_disagreement([c for c in cases if c[0] > 0]),
                       "under_or_away": ratings.by_disagreement([c for c in cases if c[0] < 0])}
    return out


def market_lean(games, lines):
    """With no model at all: how often the Over and the home side beat each line. Half is no lean."""
    out = {}
    for market, keys in (("SPREAD", (("closing", "spread"), ("opening", "spread_open"))),
                         ("TOTAL", (("closing", "total"), ("opening", "total_open")))):
        out[market] = {}
        for label, key in keys:
            plus = minus = push = 0
            for game in games:
                line = (lines.get(game["id"]) or {}).get(key)
                if line is None:
                    continue
                target = ((game["home_points"] - game["away_points"] + line) if market == "SPREAD"
                          else (game["home_points"] + game["away_points"] - line))
                if abs(target) < 1e-9:
                    push += 1
                elif target > 0:
                    plus += 1
                else:
                    minus += 1
            out[market][label] = {"over_or_home": plus, "under_or_away": minus, "push": push,
                                  "share": plus / (plus + minus) if plus + minus else None}
    return out


def run(prior, calibration, scored, calibration_lines, scored_lines, scored_pairs=None, grid=GRID):
    """The possession model and the points ratings on the same three seasons.

    ``prior``, ``calibration`` and ``scored`` are ``(games, box)`` for each season, ``games`` from
    ``parse_games`` here and ``box`` from ``parse_box``. The first season only sets the priors, the
    second picks the grid entry and fits ``k`` and ``sigma``, the third is scored.
    """
    pairs = ratings._pairs_from_lines(scored_lines) if scored_pairs is None else scored_pairs
    (prior_games, prior_box), (calibration_games, calibration_box), (scored_games, scored_box) = prior, calibration, scored
    chosen, grid_rows = tune(prior_games, prior_box, calibration_games, calibration_box, grid)
    _, first, first_cover = season_predictions(prior_games, prior_box, None, chosen)
    calibration_predictions, second, second_cover = season_predictions(calibration_games, calibration_box, first, chosen)
    scored_predictions, _, scored_cover = season_predictions(scored_games, scored_box, second, chosen)
    _, points_first = ratings.season_predictions(prior_games)
    points_calibration, points_second = ratings.season_predictions(calibration_games, points_first)
    points_scored, _ = ratings.season_predictions(scored_games, points_second)
    tiers = conference_tiers(scored_games, second)
    return {
        "tuning": {"chosen": chosen, "grid": grid_rows},
        "coverage": {"prior": first_cover, "calibration": second_cover, "scored": scored_cover},
        "accuracy": accuracy(scored_games, {"possession": scored_predictions, "ratings": points_scored}, scored_lines),
        "high_conferences": sorted(conference for conference, tier in tiers.items() if tier == "high"),
        "models": {
            "possession": ratings.evaluate(calibration_games, calibration_predictions, scored_games, scored_predictions,
                                           calibration_lines, scored_lines, pairs),
            "ratings": ratings.evaluate(calibration_games, points_calibration, scored_games, points_scored,
                                        calibration_lines, scored_lines, pairs)},
        "by_tier": by_tier(scored_games, scored_predictions, scored_lines, pairs, tiers),
        "by_side": by_side(scored_games, scored_predictions, scored_lines),
        "market_lean": market_lean(scored_games, scored_lines)}


def markdown(result):
    """The run as text: tuning, accuracy against the market, then each model's readings."""
    report = result["report"]
    chosen = report["tuning"]["chosen"]
    lines = ["## NCAAB possession model vs the points ratings and the lines (CBBD)", "",
             f"- seasons: priors {result['prior_season']}, calibrate {result['calibration_season']}, "
             f"score {result['scored_season']}",
             "- chosen on the calibration season: " + ", ".join(f"{key} {value:g}" for key, value in chosen.items())]
    for season, cover in report["coverage"].items():
        lines.append(f"- {season} season: {cover['games']} games used, {cover['with_box_score']} with a box score, "
                     f"{cover['rated_teams']} rated teams")
    lines += ["", "### Tuning grid (calibration season; smaller is better)", "",
              "| " + " | ".join(chosen) + " | games | margin RMSE | total RMSE |",
              "|" + "---|" * (len(chosen) + 3)]
    for row in report["tuning"]["grid"]:
        margin = "" if row["margin_rmse"] is None else f"{row['margin_rmse']:.3f}"
        total = "" if row["total_rmse"] is None else f"{row['total_rmse']:.3f}"
        lines.append("| " + " | ".join(f"{row[key]:g}" for key in chosen) + f" | {row['games']} | {margin} | {total} |")
    lines += ["", "### How far each misses the final result (mean absolute miss, points; same games)", "",
              "| market | segment | games | possession model | points ratings | opening line | closing line |",
              "|---|---|---|---|---|---|---|"]
    for market, by_segment in report["accuracy"].items():
        for segment, row in by_segment.items():
            cells = " | ".join("" if row.get(column) is None else f"{row[column]:.2f}"
                               for column in ("possession", "ratings", "opening_line", "closing_line"))
            lines.append(f"| {market} | {segment} | {row['games']} | {cells} |")
    lines += ["", "A model that misses by more than the line knows less than the line does. It can still be "
                  "ahead of an opening number that the close later corrects, which the tables below test.",
              "", f"### By conference strength (possession model; top conferences: "
                  f"{', '.join(report['high_conferences']) or 'none'})", "",
              "| market | tier | games with both lines | share the close gave back (SE) | t | "
              "4+ points apart at the open: picks, wins - losses, hit rate |", "|---|---|---|---|---|---|"]
    for market, by_level in report["by_tier"].items():
        for tier, row in by_level.items():
            slope = row["line_move_slope"]
            share = "" if slope["slope"] is None else f"{slope['slope']:+.3f} ({slope['se']:.3f})"
            t = "" if slope["t"] is None else f"{slope['t']:+.1f}"
            four = row["opening_disagreement"]["4"]
            hit = "" if four["hit_rate"] is None else f"{four['hit_rate']:.1%}"
            lines.append(f"| {market} | {tier} | {slope['games']} | {share} | {t} | "
                         f"{four['picks']}, {four['wins']} - {four['losses']}, {hit} |")
    lines += ["", "### Which side the possession model took at the opening line (no fit)", "",
              "| market | side | points apart | picks | wins - losses | hit rate |", "|---|---|---|---|---|---|"]
    for market, sides in report["by_side"].items():
        for side, rows in sides.items():
            name = {"SPREAD": {"over_or_home": "home", "under_or_away": "away"},
                    "TOTAL": {"over_or_home": "Over", "under_or_away": "Under"}}[market][side]
            for points, row in rows.items():
                hit = "" if row["hit_rate"] is None else f"{row['hit_rate']:.1%}"
                lines.append(f"| {market} | {name} | {points}+ | {row['picks']} | {row['wins']} - {row['losses']} | {hit} |")
    lines += ["", "### With no model: how often the Over and the home side beat each line", "",
              "| market | line | Over or home | Under or away | push | share |", "|---|---|---|---|---|---|"]
    for market, by_line in report["market_lean"].items():
        for label, row in by_line.items():
            share = "" if row["share"] is None else f"{row['share']:.1%}"
            lines.append(f"| {market} | {label} | {row['over_or_home']} | {row['under_or_away']} | {row['push']} | {share} |")
    text = "\n".join(lines) + "\n"
    for name, title in (("possession", "Possession model vs archived lines"), ("ratings", "Points ratings vs archived lines")):
        text += "\n" + ratings.markdown({**result, "report": report["models"][name]}, title=title)
    return text
