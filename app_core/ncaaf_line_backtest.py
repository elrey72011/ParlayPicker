"""Backtest the frozen NCAAF research model against archived betting lines (roadmap D18, step 1).

``ncaaf_research`` trains on CFBD 2023, calibrates on 2024 and evaluates 2025, but
only on score error: "No historical odds or returns were evaluated." Before binding
that model to a validation plan, this measures it on the 2025 holdout against CFBD's
archived lines (``/lines``), using exactly the live policy's pricing
(``ncaaf_prospective.quote_candidates``): home covers when margin > -(home spread);
totals use the truncated total distribution.

Research only. It never writes evidence, plans or wagers. Archived lines carry no
quote timestamp, so this is a screen for whether prospective validation is worth
running, not evidence for one.
"""
from __future__ import annotations

import math

from app_core.ncaaf_research import centers, probabilities

LINES_URL = "https://api.collegefootballdata.com/lines"
# Prefer a consensus line; otherwise the first named book with both numbers.
PROVIDER_ORDER = ("consensus", "ESPN Bet", "DraftKings", "Bovada", "William Hill (New Jersey)", "teamrankings")
WIN_PAYOUT = 100 / 110  # -110 standard price
THRESHOLDS = (0.5, 0.55, 0.6)


def fetch_lines(year, token, get, *, season_types=("regular", "postseason")):
    out = []
    for season_type in season_types:
        response = get(LINES_URL, params={"year": year, "seasonType": season_type},
                       headers={"Authorization": "Bearer " + token}, timeout=20, allow_redirects=False)
        if response.status_code != 200:
            raise ValueError(f"CFBD lines HTTP {int(response.status_code)}")
        value = response.json()
        if not isinstance(value, list):
            raise ValueError("CFBD lines response is not a list")
        out.extend(value)
    return out


def _number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def select_lines(games):
    """{game_id: {"spread", "total", "provider"}}: one archived line per game, home perspective."""
    selected = {}
    for game in games:
        lines = [line for line in game.get("lines") or [] if isinstance(line, dict)]
        rank = {name: i for i, name in enumerate(PROVIDER_ORDER)}
        lines.sort(key=lambda line: rank.get(line.get("provider"), len(PROVIDER_ORDER)))
        spread = total = provider = None
        for line in lines:
            spread = spread if spread is not None else _number(line.get("spread"))
            total = total if total is not None else _number(line.get("overUnder"))
            provider = provider or line.get("provider")
            if spread is not None and total is not None:
                break
        if isinstance(game.get("id"), int) and (spread is not None or total is not None):
            selected[game["id"]] = {"spread": spread, "total": total, "provider": provider}
    return selected


def backtest(artifact, holdout, lines):
    """Per model kind and market: picks, hit rate, ROI at -110 and Brier, by confidence threshold."""
    report = {}
    for kind, models in artifact["models"].items():
        report[kind] = {}
        for market, target in (("SPREAD", "margin"), ("TOTAL", "total")):
            model = models[target]
            rows = [r for r in holdout if r["game_id"] in lines and
                    lines[r["game_id"]]["spread" if market == "SPREAD" else "total"] is not None]
            if not rows:
                report[kind][market] = {"games": 0}
                continue
            means = centers(model, rows, target)
            graded = []
            for row, mu in zip(rows, means):
                if market == "SPREAD":
                    spread = lines[row["game_id"]]["spread"]
                    p = probabilities(float(mu), model["sigma"], -spread)
                    side_a, actual = p["over"], row["y_margin"] + spread  # home side
                else:
                    total = lines[row["game_id"]]["total"]
                    p = probabilities(float(mu), model["sigma"], total, total=True)
                    side_a, actual = p["over"], row["y_total"] - total     # over side
                conditional = side_a / max(1e-12, 1 - p["push"])
                pick_a = conditional >= 0.5
                confidence = conditional if pick_a else 1 - conditional
                if abs(actual) < 1e-9:
                    result = "PUSH"
                else:
                    result = "WIN" if (actual > 0) == pick_a else "LOSS"
                graded.append((confidence, conditional, actual, result))
            decided = [(c, int(a > 0)) for _, c, a, r in graded if r != "PUSH"]
            brier = sum((c - y) ** 2 for c, y in decided) / len(decided) if decided else None
            entry = {"games": len(graded), "brier_side_a": brier, "brier_coin_flip": 0.25, "by_threshold": {}}
            for threshold in THRESHOLDS:
                picks = [g for g in graded if g[0] >= threshold]
                wins = sum(1 for g in picks if g[3] == "WIN")
                losses = sum(1 for g in picks if g[3] == "LOSS")
                units = wins * WIN_PAYOUT - losses
                entry["by_threshold"][f"{threshold:.2f}"] = {
                    "picks": len(picks), "wins": wins, "losses": losses,
                    "pushes": len(picks) - wins - losses,
                    "hit_rate": wins / (wins + losses) if wins + losses else None,
                    "break_even_hit_rate": 110 / 210,
                    "roi_per_pick": units / len(picks) if picks else None}
            report[kind][market] = entry
    return report


def markdown(result):
    lines = [f"## NCAAF frozen model vs archived 2025 lines", "",
             f"- artifact hash: `{result['artifact_hash']}` "
             f"({'matches' if result['artifact_matches_frozen'] else 'DIFFERS FROM'} the frozen cohort artifact)",
             f"- holdout games: {result['holdout_games']}; games with an archived line: {result['games_with_lines']}",
             f"- break-even hit rate at -110: {110 / 210:.1%}", "",
             "| model | market | min confidence | picks | hit rate | ROI/pick | Brier (coin flip 0.250) |",
             "|---|---|---|---|---|---|---|"]
    for kind, markets in result["backtest"].items():
        for market, entry in markets.items():
            for threshold, row in (entry.get("by_threshold") or {}).items():
                hit = "" if row["hit_rate"] is None else f"{row['hit_rate']:.1%}"
                roi = "" if row["roi_per_pick"] is None else f"{row['roi_per_pick']:+.3f}"
                brier = "" if entry.get("brier_side_a") is None else f"{entry['brier_side_a']:.3f}"
                lines.append(f"| {kind} | {market} | {threshold} | {row['picks']} | {hit} | {roi} | {brier} |")
    aware = result.get("line_aware") or {}
    if aware:
        lines += ["", "### Market-anchored model (line as input; fit 2023, calibrated 2024, scored 2025)", "",
                  "| market | games | mean |predicted edge| (pts) | min P(cover) | picks | hit rate | ROI/pick | Brier |",
                  "|---|---|---|---|---|---|---|---|"]
        for market, entry in aware.items():
            for threshold, row in (entry.get("by_threshold") or {}).items():
                hit = "" if row["hit_rate"] is None else f"{row['hit_rate']:.1%}"
                roi = "" if row["roi_per_pick"] is None else f"{row['roi_per_pick']:+.3f}"
                lines.append(f"| {market} | {entry['games']} | {entry['mean_abs_centre']:.2f} | {threshold} | "
                             f"{row['picks']} | {hit} | {roi} | {entry['brier']:.3f} |")
            if entry.get("blocker"):
                lines.append(f"| {market} | 0 | | | | | | {entry['blocker']} |")
    lines += ["", "Archived lines have no quote timestamp; this screens the model and is not validation evidence."]
    return "\n".join(lines) + "\n"
