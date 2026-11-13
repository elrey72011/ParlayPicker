"""The value of one point of line in college basketball, measured (owner 2026-11-12; research only).

The D30 proposal scores a bet whose line moved by carrying the closing probability to the bet's
line with a value per point. The report assumed 3.6% a point for a side and 2.4% for a total, from
a normal margin. This measures it instead, from archived closing lines and final scores
(CollegeBasketballData ``/games`` and ``/lines``; the same data as ``ncaab_line_backtest``).

For a side, ``d`` is the home cover margin (home margin + home spread); for a total, the points
minus the line. A bet at a line ``x`` points better than the close wins when ``d + x > 0``. With a
push counted as half a win, ``W(x) = P(d + x > 0) + P(d + x = 0) / 2``, and the value per point
over a move of ``k`` points either way is ``(W(k) - W(-k)) / (2k)``: the share of games that land
within ``k`` points of the line, spread over the ``2k`` points. It is measured for several ``k``
because the value is not constant: the further the move, the thinner the margin distribution.

Archived lines carry no timestamp and one provider per game, so this is a property of the game
outcomes around a closing line, not of any book's price. Results are read here by design; nothing
is written to the evidence store and nothing here is read by a pick, a stake or a plan.
"""
from __future__ import annotations

import math

MOVES = (0.5, 1.0, 1.5, 2.0, 3.0)
ASSUMED = {"SPREAD": 0.036, "TOTAL": 0.024}       # what the readiness report assumed until measured
SPREAD_BANDS = ((0, 5, "under 5"), (5, 10, "5 to 10"), (10, 20, "10 to 20"), (20, None, "20 or more"))
TOTAL_BANDS = ((0, 135, "under 135"), (135, 150, "135 to 150"), (150, None, "150 or more"))
MIN_GAMES = 200


def differences(games, lines):
    """{"SPREAD": [(d, |spread|)], "TOTAL": [(d, total)]} for finished games with an archived line."""
    out = {"SPREAD": [], "TOTAL": []}
    for game in games:
        line = lines.get(game["id"])
        if not line:
            continue
        if line.get("spread") is not None:
            out["SPREAD"].append((game["home_points"] - game["away_points"] + line["spread"], abs(line["spread"])))
        if line.get("total") is not None:
            out["TOTAL"].append((game["home_points"] + game["away_points"] - line["total"], line["total"]))
    return out


def win_share(values, shift):
    """W(shift): the share of games a bet ``shift`` points better than the line wins, a push as half."""
    if not values:
        return None
    won = sum(1.0 if d + shift > 1e-9 else 0.5 if abs(d + shift) <= 1e-9 else 0.0 for d in values)
    return won / len(values)


def point_value(values, move):
    """(value per point over +/- ``move`` points, its standard error, games), or (None, None, n)."""
    n = len(values)
    if n < MIN_GAMES:
        return None, None, n
    inside = sum(1.0 if abs(d) < move - 1e-9 else 0.5 if abs(abs(d) - move) <= 1e-9 else 0.0 for d in values) / n
    # W(move) - W(-move) is the share landing inside the band (edges as half), a binomial share.
    return inside / (2 * move), math.sqrt(max(inside * (1 - inside), 0.0) / n) / (2 * move), n


def _row(label, values):
    spread = math.sqrt(sum((d - sum(values) / len(values)) ** 2 for d in values) / (len(values) - 1)) if len(values) > 1 else None
    row = {"group": label, "games": len(values), "sd": spread,
           "pushes": sum(1 for d in values if abs(d) <= 1e-9) / len(values) if values else None, "by_move": {}}
    for move in MOVES:
        value, se, _ = point_value(values, move)
        row["by_move"][f"{move:g}"] = {"value_per_point": value, "se": se}
    return row


def measure(games, lines):
    """Per market: the value per point overall and by line size, against the assumed value."""
    found = differences(games, lines)
    out = {}
    for family, bands in (("SPREAD", SPREAD_BANDS), ("TOTAL", TOTAL_BANDS)):
        pairs = found[family]
        rows = [_row("all games", [d for d, _ in pairs])]
        for low, high, label in bands:
            rows.append(_row(label, [d for d, size in pairs if size >= low and (high is None or size < high)]))
        out[family] = {"assumed": ASSUMED[family], "rows": rows}
    return out


def markdown(result):
    lines = ["## NCAAB value of a point (research only)", "",
             f"Seasons {', '.join(str(s) for s in result.get('seasons', []))}: {result.get('games', 0)} finished games, "
             f"{result.get('lined', 0)} with an archived closing line (providers: "
             f"{', '.join(result.get('providers', [])) or 'none'}).", "",
             "Value per point = the share of games landing within k points of the closing line, divided by 2k "
             "(a push counted as half). It is the change in a bet's chance of winning for each point of line, "
             "averaged over a move of k points either way. Standard error in brackets.", ""]
    for family, title in (("SPREAD", "Spread (home cover margin)"), ("TOTAL", "Total (points minus the line)")):
        found = (result.get("report") or {}).get(family)
        if not found:
            continue
        moves = [f"{m:g}" for m in MOVES]
        lines += [f"### {title}: assumed {found['assumed']:.1%} a point", "",
                  "| group | games | sd of the difference | pushes | " + " | ".join(f"k = {m}" for m in moves) + " |",
                  "|---|---|---|---|" + "---|" * len(moves)]
        for row in found["rows"]:
            cells = []
            for move in moves:
                cell = row["by_move"][move]
                cells.append("" if cell["value_per_point"] is None else f"{cell['value_per_point']:.2%} ({cell['se']:.2%})")
            sd = "" if row["sd"] is None else f"{row['sd']:.1f}"
            pushes = "" if row["pushes"] is None else f"{row['pushes']:.1%}"
            lines.append(f"| {row['group']} | {row['games']} | {sd} | {pushes} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["Groups with fewer than 200 games show no value. Archived lines have no timestamp and one provider per "
              "game: this is how game outcomes fall around a closing line, not any book's price.", ""]
    return "\n".join(lines)
