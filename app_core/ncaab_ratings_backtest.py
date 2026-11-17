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
  season (never the scored one), separately for closing and opening lines.
* **Scored season**, split into November-December and January on, against **closing** lines and,
  where CBBD has them, **opening** lines (early lines are where public ratings can still be ahead).

Two more readings need no fitted shrink at all (2026-11-17), so a ``k`` of zero cannot hide them:

* **By disagreement** (``by_disagreement``): when the ratings sit 2, 4 or 6 points or more from the
  line, how the ratings' side of that line finished.
* **Line moves** (``line_moves``): for games with both lines from one provider (``select_pairs``), how
  far the closing line moved in favour of the side the ratings took at the open. A line that moves
  toward the ratings is a better number than the close, the same thing closing-line value measures,
  and it is far less noisy than results. ``move_slope`` puts it in one number: of each point the
  ratings sat from the opening line, the share of a point the close then moved their way.

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
DISAGREEMENTS = (2.0, 4.0, 6.0)     # points between the ratings and the line


def select_lines(rows, order=PROVIDER_ORDER):
    """{game_id: {spread, total, spread_open, total_open, provider}} with one provider per game."""
    rank = {name: i for i, name in enumerate(order)}
    out = {}
    for row in rows:
        game_id = _first(row, "gameId", "id")
        found = {"spread": None, "total": None, "spread_open": None, "total_open": None, "provider": None}
        for line in sorted((l for l in row.get("lines") or [] if isinstance(l, dict)),
                           key=lambda l: rank.get(l.get("provider"), len(order))):
            for key, names in (("spread", ("spread",)), ("total", ("overUnder", "over_under", "total")),
                               ("spread_open", ("spreadOpen", "spread_open")),
                               ("total_open", ("overUnderOpen", "over_under_open", "totalOpen"))):
                if found[key] is None:
                    found[key] = _number(_first(line, *names))
            found["provider"] = found["provider"] or line.get("provider")
        if game_id is not None and (found["spread"] is not None or found["total"] is not None):
            out[game_id] = found
    return out


def select_pairs(rows, order=PROVIDER_ORDER):
    """{game_id: {"SPREAD": (open, close), "TOTAL": (open, close)}}, each pair from ONE provider.

    ``select_lines`` may take a game's opening number from one book and its closing number from
    another: fine for grading a bet at either line, wrong for measuring how far a line moved. Here
    each market's pair comes from the first provider in ``order`` that quotes both numbers.
    """
    rank = {name: i for i, name in enumerate(order)}
    out = {}
    for row in rows:
        game_id = _first(row, "gameId", "id")
        found = {}
        for line in sorted((l for l in row.get("lines") or [] if isinstance(l, dict)),
                           key=lambda l: rank.get(l.get("provider"), len(order))):
            for market, close_names, open_names in (
                    ("SPREAD", ("spread",), ("spreadOpen", "spread_open")),
                    ("TOTAL", ("overUnder", "over_under", "total"), ("overUnderOpen", "over_under_open", "totalOpen"))):
                if market in found:
                    continue
                opened, closed = _number(_first(line, *open_names)), _number(_first(line, *close_names))
                if opened is not None and closed is not None:
                    found[market] = (opened, closed)
        if game_id is not None and found:
            out[game_id] = found
    return out


def _pairs_from_lines(lines):
    """The ``select_pairs`` shape from ``select_lines`` output, for a caller with nothing better."""
    out = {}
    for game_id, line in lines.items():
        found = {market: (line.get(open_key), line.get(close_key))
                 for market, open_key, close_key in (("SPREAD", "spread_open", "spread"), ("TOTAL", "total_open", "total"))
                 if line.get(open_key) is not None and line.get(close_key) is not None}
        if found:
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


# Basketball defaults; another sport passes its own (``ncaaf_ratings_backtest.PARAMS``).
PARAMS = {"margin_cap": MARGIN_CAP, "carry": CARRY, "prior_weight": PRIOR_WEIGHT,
          "hca_prior": HCA_PRIOR, "hca_weight": HCA_WEIGHT, "team_points": 70.0}


class RatingBook:
    """Ridge ratings accumulated game by game; ``solve`` gives the ratings as of now."""

    def __init__(self, teams, prior=None, params=None):
        self.p = {**PARAMS, **(params or {})}
        carry = self.p["carry"]
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
            self.m_prior[i], self.t_prior[i], self.t_prior[n + i] = carry * r, carry * o, carry * d
        self.m_prior[n] = self.p["hca_prior"]
        self.t_prior[2 * n] = prior.get("__mu__", (0.0, self.p["team_points"], 0.0))[1]
        self.games = 0

    def add(self, game):
        h, a = self.index[game["home"]], self.index[game["away"]]
        home = 0.0 if game["neutral"] else 1.0
        cap = self.p["margin_cap"]
        margin = max(-cap, min(cap, game["home_points"] - game["away_points"]))
        n = self.n
        _accumulate(self.m_mat, self.m_vec, ((h, 1.0), (a, -1.0), (n, home)), margin)
        for scorer, defender, points, at_home in ((h, a, game["home_points"], home),
                                                  (a, h, game["away_points"], 0.0)):
            _accumulate(self.t_mat, self.t_vec,
                        ((scorer, 1.0), (n + defender, -1.0), (2 * n, 1.0), (2 * n + 1, at_home)), points)
        self.games += 1

    def solve(self):
        n = self.n
        mw = np.full(n + 1, self.p["prior_weight"])
        mw[n] = self.p["hca_weight"]
        margin = np.linalg.solve(self.m_mat + np.diag(mw), self.m_vec + mw * self.m_prior)
        tw = np.full(2 * n + 2, self.p["prior_weight"])
        tw[2 * n], tw[2 * n + 1] = 1e-3, self.p["hca_weight"]
        prior = self.t_prior.copy()
        prior[2 * n + 1] = self.p["hca_prior"] / 2
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


def season_predictions(games, prior=None, params=None):
    """{game_id: (model margin, model total)} for each game, from the ratings as of its day."""
    teams = {g["home"] for g in games} | {g["away"] for g in games}
    book = RatingBook(teams, prior, params)
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


def by_disagreement(cases):
    """With no fitted shrink: how the ratings' side finished where they sit N or more points from the line.

    ``cases`` are ``_cases`` tuples. Graded at -110; a push is neither a win nor a loss.
    """
    out = {}
    for points in DISAGREEMENTS:
        picks = [(edge, target) for edge, target, _ in cases if abs(edge) >= points]
        wins = sum(1 for edge, target in picks if abs(target) > 1e-9 and (target > 0) == (edge > 0))
        losses = sum(1 for edge, target in picks if abs(target) > 1e-9 and (target > 0) != (edge > 0))
        out[f"{points:g}"] = {"picks": len(picks), "wins": wins, "losses": losses,
                              "hit_rate": wins / (wins + losses) if wins + losses else None,
                              "roi_per_pick": (wins * WIN - losses) / len(picks) if picks else None}
    return out


def line_moves(games, predictions, pairs, market):
    """[(points the ratings sit from the opening line, points the close moved in their side's favour, game)].

    ``pairs`` is ``select_pairs`` output: only games with an opening and a closing number. For a
    spread the ratings take the home side when their margin beats the opening spread; a home bet at
    the open has the better number when the home spread closed lower. For a total they take the Over
    when their total is above the opening line; an Over at the open has the better number when the
    total closed higher.
    """
    out = []
    for game in games:
        pair = (pairs.get(game["id"]) or {}).get(market)
        if pair is None or game["id"] not in predictions:
            continue
        opened, closed = pair
        margin, total = predictions[game["id"]]
        edge = margin + opened if market == "SPREAD" else total - opened
        if edge == 0:
            continue
        toward_home_or_over = (opened - closed) if market == "SPREAD" else (closed - opened)
        out.append((abs(edge), toward_home_or_over if edge > 0 else -toward_home_or_over, game))
    return out


def move_slope(moves):
    """Of each point the ratings sat from the opening line, the share of a point the close moved their way.

    Least squares through the origin of the move on the disagreement, with its standard error. 0 says
    the opening line already held everything the ratings know; 1 says the close moved all the way to them.
    """
    n = len(moves)
    if n < 3:
        return {"games": n, "slope": None, "se": None, "t": None}
    edge = np.array([m[0] for m in moves], dtype=float)
    moved = np.array([m[1] for m in moves], dtype=float)
    sxx = float(edge @ edge)
    slope = float(edge @ moved) / sxx
    residual = moved - slope * edge
    se = math.sqrt(float(residual @ residual) / (n - 1) / sxx)
    return {"games": n, "slope": slope, "se": se, "t": slope / se if se > 0 else None}


def move_summary(moves):
    """By disagreement at the open: games, mean points the close moved the ratings' way (with SE), and the split."""
    out = {}
    for points in (0.0,) + DISAGREEMENTS:
        moved = np.array([m for edge, m, _ in moves if edge >= points], dtype=float)
        n = len(moved)
        out[f"{points:g}"] = {
            "games": n, "mean_points": float(moved.mean()) if n else None,
            "se": float(moved.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
            "toward": int((moved > 0).sum()), "against": int((moved < 0).sum()), "unchanged": int((moved == 0).sum())}
    return out


def run(prior_games, calibration_games, scored_games, calibration_lines, scored_lines, scored_pairs=None):
    """Priors from the first season, k and sigma from the second, the third scored by segment and line.

    ``scored_pairs`` (``select_pairs``) gives the line-move reading one provider's opening and closing
    number per game; without it the pairs are read from ``scored_lines``.
    """
    pairs = _pairs_from_lines(scored_lines) if scored_pairs is None else scored_pairs
    _, priors = season_predictions(prior_games)
    calibration_predictions, calibration_finals = season_predictions(calibration_games, priors)
    scored_predictions, _ = season_predictions(scored_games, calibration_finals)
    report = {}
    for market, close_key, open_key in (("SPREAD", "spread", "spread_open"), ("TOTAL", "total", "total_open")):
        entry = {"fits": {}, "lines": {}}
        for label, key in (("closing", close_key), ("opening", open_key)):
            # Each line type gets its own k and sigma from the calibration season's lines of that type:
            # a model can add nothing to closing lines and still be ahead of opening ones.
            fit = calibrate(_cases(calibration_games, calibration_predictions, calibration_lines, market, key))
            entry["fits"][label] = fit
            cases = _cases(scored_games, scored_predictions, scored_lines, market, key)
            entry["lines"][label] = {"games": len(cases)}
            # Needs no fit, so it is reported even where the calibration season had too few such lines.
            entry.setdefault("disagreement", {})[label] = {
                "all": by_disagreement(cases),
                **{name: by_disagreement([c for c in cases if c[2]["start"].month in months]) for name, months in SEGMENTS}}
            if fit is None:
                entry["lines"][label]["blocker"] = "TOO_FEW_CALIBRATION_GAMES"
                continue
            for name, months in SEGMENTS:
                part = [c for c in cases if c[2]["start"].month in months]
                entry["lines"][label][name] = score(part, fit) if part else {"games": 0}
        moves = line_moves(scored_games, scored_predictions, pairs, market)
        by_segment = {"all": moves,
                      **{name: [m for m in moves if m[2]["start"].month in months] for name, months in SEGMENTS}}
        entry["line_moves"] = {name: move_summary(part) for name, part in by_segment.items()}
        entry["line_move_slope"] = {name: move_slope(part) for name, part in by_segment.items()}
        if not any(entry["fits"].values()):
            report[market] = {"blocker": "TOO_FEW_CALIBRATION_GAMES", "disagreement": entry["disagreement"],
                              "line_moves": entry["line_moves"], "line_move_slope": entry["line_move_slope"]}
            continue
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
            fit = (entry.get("fits") or {}).get(label) or {}
            lines.append(f"| {market} | {label} | fit: k = {fit.get('k', 0):.2f}, sigma = {fit.get('sigma', 0):.1f} "
                         f"({fit.get('games', 0)} calibration games) | | | | | | |")
            if by_line.get("blocker"):
                lines.append(f"| {market} | {label} | {by_line['blocker']} | | | | | | |")
                continue
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
    lines += ["", "A pick is the side whose probability clears the threshold, graded at -110. Archived lines "
                  "have no quote timestamp; this screens the idea and is not plan evidence."]
    lines += ["", "### Where the ratings disagree with the line (no fitted shrink)", "",
              "| market | line | segment | points apart | picks | wins - losses | hit rate | ROI/pick |",
              "|---|---|---|---|---|---|---|---|"]
    for market, entry in result["report"].items():
        for label, by_segment in (entry.get("disagreement") or {}).items():
            for segment, rows in by_segment.items():
                for points, row in rows.items():
                    hit = "" if row["hit_rate"] is None else f"{row['hit_rate']:.1%}"
                    roi = "" if row["roi_per_pick"] is None else f"{row['roi_per_pick']:+.3f}"
                    lines.append(f"| {market} | {label} | {segment} | {points}+ | {row['picks']} | "
                                 f"{row['wins']} - {row['losses']} | {hit} | {roi} |")
    lines += ["", "### Did the closing line move toward the ratings? (side taken at the opening line)", "",
              "| market | segment | points apart at the open | games | mean move their way (SE) | toward / against / unchanged |",
              "|---|---|---|---|---|---|"]
    for market, entry in result["report"].items():
        for segment, rows in (entry.get("line_moves") or {}).items():
            for points, row in rows.items():
                mean = "" if row["mean_points"] is None else f"{row['mean_points']:+.2f}"
                se = "" if row["se"] is None else f" ({row['se']:.2f})"
                lines.append(f"| {market} | {segment} | {points}+ | {row['games']} | {mean}{se} | "
                             f"{row['toward']} / {row['against']} / {row['unchanged']} |")
    lines += ["", "| market | segment | games | share of the disagreement the close gave back (SE) | t |",
              "|---|---|---|---|---|"]
    for market, entry in result["report"].items():
        for segment, row in (entry.get("line_move_slope") or {}).items():
            slope = "" if row["slope"] is None else f"{row['slope']:+.3f} ({row['se']:.3f})"
            t = "" if row["t"] is None else f"{row['t']:+.1f}"
            lines.append(f"| {market} | {segment} | {row['games']} | {slope} | {t} |")
    lines += ["", "A positive mean move says a bet on the ratings' side at the opening line held a better number "
                  "than the close. Results are noisy; line moves are not, so read these two tables first. The "
                  "share is of each point the ratings sat from the opening line: 0 means the opening line already "
                  "held everything the ratings know. Both numbers of a game come from one provider. An opening "
                  "number is the first one posted, often at low limits and soon gone, so this is the most a bet "
                  "placed later could have had, not what it would have had."]
    return "\n".join(lines) + "\n"
