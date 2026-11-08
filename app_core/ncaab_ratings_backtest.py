"""NCAAB opponent-adjusted ratings vs archived opening and closing lines (owner 2026-11-08, research only).

``app_core.ncaab_line_backtest`` screened raw team form against archived closing lines and found
no edge (2024-25 second half: about 50% at every threshold). This asks a sharper question with
CollegeBasketballData (CBBD) data, the kind of reason a model could disagree with the market:

* **Ratings as of each game day** (``RatingBook``), from that season's earlier games only:
  - margin: ``home margin = hca * home + r_home - r_away`` (capped at ``MARGIN_CAP``);
  - totals: each team's points = ``mu + hca_t * home + offence_team - defence_opponent``.
  Ridge least squares, each team pulled toward a prior: the previous season's final rating times
  ``CARRY`` (so early-season ratings start from last season, regressed to the mean).
* **Edge vs the line**: model cover margin ``margin + spread`` and total error ``total - line``,
  shrunk by ``k`` and turned into a probability with ``sigma``, both fitted on the calibration
  season (never the scored one).
* **Scored season**, split into November-December and January on, against **closing** lines and,
  where CBBD has them, **opening** lines (early lines are where public ratings can still be ahead).

Three seasons: the first only sets the priors, the second calibrates ``k`` and ``sigma``, the
third is scored. Prices are assumed at -110 (archived lines have spreads and totals, not their
prices). Archived lines carry no quote timestamp, so this screens the idea; it is not plan evidence.
"""
from __future__ import annotations

import math

import numpy as np

from app_core.ncaab_line_backtest import PROVIDER_ORDER, _first, _number

MARGIN_CAP = 25.0
CARRY = 0.6            # share of last season's rating kept as this season's prior
PRIOR_WEIGHT = 4.0     # the prior counts like this many games
HCA_PRIOR, HCA_WEIGHT = 3.0, 20.0
THRESHOLDS = (0.52, 0.55, 0.58)
WIN = 100 / 110
SEGMENTS = (("Nov-Dec", (11, 12)), ("Jan-Apr", (1, 2, 3, 4)))


def select_lines(rows):
    """{game_id: {spread, total, spread_open, total_open, provider}} with one provider per game."""
    rank = {name: i for i, name in enumerate(PROVIDER_ORDER)}
    out = {}
    for row in rows:
        game_id = _first(row, "gameId", "id")
        found = {"spread": None, "total": None, "spread_open": None, "total_open": None, "provider": None}
        for line in sorted((l for l in row.get("lines") or [] if isinstance(l, dict)),
                           key=lambda l: rank.get(l.get("provider"), len(PROVIDER_ORDER))):
            for key, names in (("spread", ("spread",)), ("total", ("overUnder", "over_under", "total")),
                               ("spread_open", ("spreadOpen", "spread_open")),
                               ("total_open", ("overUnderOpen", "over_under_open", "totalOpen"))):
                if found[key] is None:
                    found[key] = _number(_first(line, *names))
            found["provider"] = found["provider"] or line.get("provider")
        if game_id is not None and (found["spread"] is not None or found["total"] is not None):
            out[game_id] = found
    return out


def _accumulate(matrix, vector, terms, y):
    """Add one observation ``y = sum(value * param[index])`` to the normal equations (sparse terms)."""
    for i, vi in terms:
        if vi == 0:
            continue
        vector[i] += vi * y
        for j, vj in terms:
            if vj:
                matrix[i, j] += vi * vj


class RatingBook:
    """Ridge ratings accumulated game by game; ``solve`` gives the ratings as of now."""

    def __init__(self, teams, prior=None):
        self.index = {t: i for i, t in enumerate(sorted(teams))}
        n = len(self.index)
        self.n = n
        prior = prior or {}
        # margin: [r_0..r_n-1, hca]
        self.m_mat, self.m_vec = np.zeros((n + 1, n + 1)), np.zeros(n + 1)
        self.m_prior = np.zeros(n + 1)
        # totals: [o_0..o_n-1, d_0..d_n-1, mu, hca_t]
        self.t_mat, self.t_vec = np.zeros((2 * n + 2, 2 * n + 2)), np.zeros(2 * n + 2)
        self.t_prior = np.zeros(2 * n + 2)
        for team, i in self.index.items():
            r, o, d = (prior.get(team) or (0.0, 0.0, 0.0))
            self.m_prior[i], self.t_prior[i], self.t_prior[n + i] = CARRY * r, CARRY * o, CARRY * d
        self.m_prior[n] = HCA_PRIOR
        self.t_prior[2 * n] = prior.get("__mu__", (0.0, 70.0, 0.0))[1]
        self.games = 0

    def add(self, game):
        h, a = self.index[game["home"]], self.index[game["away"]]
        home = 0.0 if game["neutral"] else 1.0
        margin = max(-MARGIN_CAP, min(MARGIN_CAP, game["home_points"] - game["away_points"]))
        n = self.n
        _accumulate(self.m_mat, self.m_vec, ((h, 1.0), (a, -1.0), (n, home)), margin)
        for scorer, defender, points, at_home in ((h, a, game["home_points"], home),
                                                  (a, h, game["away_points"], 0.0)):
            _accumulate(self.t_mat, self.t_vec,
                        ((scorer, 1.0), (n + defender, -1.0), (2 * n, 1.0), (2 * n + 1, at_home)), points)
        self.games += 1

    def solve(self):
        n = self.n
        mw = np.full(n + 1, PRIOR_WEIGHT)
        mw[n] = HCA_WEIGHT
        margin = np.linalg.solve(self.m_mat + np.diag(mw), self.m_vec + mw * self.m_prior)
        tw = np.full(2 * n + 2, PRIOR_WEIGHT)
        tw[2 * n], tw[2 * n + 1] = 1e-3, HCA_WEIGHT
        prior = self.t_prior.copy()
        prior[2 * n + 1] = HCA_PRIOR / 2
        totals = np.linalg.solve(self.t_mat + np.diag(tw), self.t_vec + tw * prior)
        return margin, totals

    def predict(self, game, margin, totals):
        n = self.n
        h, a = self.index[game["home"]], self.index[game["away"]]
        home = 0.0 if game["neutral"] else 1.0
        mu, hca_t = totals[2 * n], totals[2 * n + 1]
        home_pts = mu + hca_t * home + totals[h] - totals[n + a]
        away_pts = mu + totals[a] - totals[n + h]
        return margin[h] - margin[a] + margin[n] * home, home_pts + away_pts

    def finals(self):
        margin, totals = self.solve()
        out = {t: (float(margin[i]), float(totals[i]), float(totals[self.n + i])) for t, i in self.index.items()}
        out["__mu__"] = (0.0, float(totals[2 * self.n]), 0.0)
        return out


def season_predictions(games, prior=None):
    """{game_id: (model margin, model total)} for each game, from the ratings as of its day."""
    teams = {g["home"] for g in games} | {g["away"] for g in games}
    book = RatingBook(teams, prior)
    out, day, solved = {}, None, None
    for game in games:
        today = game["start"].date()
        if today != day:
            day, solved = today, book.solve()   # games of the same day never see each other
        out[game["id"]] = book.predict(game, *solved)
        book.add(game)
    return out, book.finals()


def _cases(games, predictions, lines, market, line_key):
    """(model edge in points, actual target, game) for games with that line."""
    out = []
    for game in games:
        line = (lines.get(game["id"]) or {}).get(line_key)
        if line is None or game["id"] not in predictions:
            continue
        margin, total = predictions[game["id"]]
        actual_margin = game["home_points"] - game["away_points"]
        if market == "SPREAD":
            out.append((margin + line, actual_margin + line, game))
        else:
            out.append((total - line, game["home_points"] + game["away_points"] - line, game))
    return out


def calibrate(cases):
    """k (shrink of the model edge) and sigma from a calibration season's cases."""
    if len(cases) < 200:
        return None
    edge = np.array([c[0] for c in cases])
    target = np.array([c[1] for c in cases])
    k = float(np.clip(edge @ target / max(edge @ edge, 1e-9), 0.0, 1.0))
    sigma = float(max(1.0, np.sqrt(np.mean((target - k * edge) ** 2))))
    return {"k": k, "sigma": sigma, "games": len(cases)}


def score(cases, fit):
    """Pick the side with P >= threshold; graded at -110."""
    graded = []
    for edge, target, game in cases:
        p = 0.5 * (1 + math.erf(fit["k"] * edge / (fit["sigma"] * math.sqrt(2))))
        side = p >= 0.5
        outcome = "PUSH" if abs(target) < 1e-9 else ("WIN" if (target > 0) == side else "LOSS")
        graded.append((max(p, 1 - p), p, target, outcome, game))
    decided = [(p, int(t > 0)) for _, p, t, o, _ in graded if o != "PUSH"]
    out = {"games": len(graded), "brier": (sum((p - y) ** 2 for p, y in decided) / len(decided)) if decided else None,
           "by_threshold": {}}
    for threshold in THRESHOLDS:
        picks = [g for g in graded if g[0] >= threshold]
        wins = sum(1 for g in picks if g[3] == "WIN")
        losses = sum(1 for g in picks if g[3] == "LOSS")
        out["by_threshold"][f"{threshold:.2f}"] = {
            "picks": len(picks), "wins": wins, "losses": losses,
            "hit_rate": wins / (wins + losses) if wins + losses else None,
            "roi_per_pick": (wins * WIN - losses) / len(picks) if picks else None}
    return out


def run(prior_games, calibration_games, scored_games, calibration_lines, scored_lines):
    """Priors from the first season, k and sigma from the second, the third scored by segment and line."""
    _, priors = season_predictions(prior_games)
    calibration_predictions, calibration_finals = season_predictions(calibration_games, priors)
    scored_predictions, _ = season_predictions(scored_games, calibration_finals)
    report = {}
    for market, close_key, open_key in (("SPREAD", "spread", "spread_open"), ("TOTAL", "total", "total_open")):
        fit = calibrate(_cases(calibration_games, calibration_predictions, calibration_lines, market, close_key))
        if fit is None:
            report[market] = {"blocker": "TOO_FEW_CALIBRATION_GAMES"}
            continue
        entry = {"fit": fit, "lines": {}}
        for label, key in (("closing", close_key), ("opening", open_key)):
            cases = _cases(scored_games, scored_predictions, scored_lines, market, key)
            entry["lines"][label] = {"games": len(cases)}
            for name, months in SEGMENTS:
                part = [c for c in cases if c[2]["start"].month in months]
                entry["lines"][label][name] = score(part, fit) if part else {"games": 0}
        report[market] = entry
    return report


def markdown(result):
    lines = ["## NCAAB opponent-adjusted ratings vs archived lines (CBBD)", "",
             f"- seasons: priors {result['prior_season']}, calibrate {result['calibration_season']}, "
             f"score {result['scored_season']}",
             f"- finished games: {result['games']}; scored-season games with a closing / opening line: "
             f"{result['closing_lined']} / {result['opening_lined']}",
             f"- break-even hit rate at -110: {110 / 210:.1%}", "",
             "| market | line | segment | games | Brier | P >= | picks | hit rate | ROI/pick |",
             "|---|---|---|---|---|---|---|---|---|"]
    for market, entry in result["report"].items():
        if entry.get("blocker"):
            lines.append(f"| {market} | | | | {entry['blocker']} | | | | |")
            continue
        for label, by_line in entry["lines"].items():
            for segment, _ in SEGMENTS:
                part = by_line.get(segment) or {}
                if not part.get("games"):
                    lines.append(f"| {market} | {label} | {segment} | 0 | | | | | |")
                    continue
                brier = "" if part["brier"] is None else f"{part['brier']:.3f}"
                for threshold, row in part["by_threshold"].items():
                    hit = "" if row["hit_rate"] is None else f"{row['hit_rate']:.1%}"
                    roi = "" if row["roi_per_pick"] is None else f"{row['roi_per_pick']:+.3f}"
                    lines.append(f"| {market} | {label} | {segment} | {part['games']} | {brier} | {threshold} | "
                                 f"{row['picks']} | {hit} | {roi} |")
        fit = entry.get("fit") or {}
        lines.append(f"| {market} | fit | k = {fit.get('k', 0):.2f}, sigma = {fit.get('sigma', 0):.1f} "
                     f"({fit.get('games')} games) | | | | | | |")
    lines += ["", "A pick is the side whose probability clears the threshold, graded at -110. Archived lines "
                  "have no quote timestamp; this screens the idea and is not plan evidence."]
    return "\n".join(lines) + "\n"
