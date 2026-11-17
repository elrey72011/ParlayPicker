"""NCAAB opponent-adjusted ratings vs archived opening and closing lines (CBBD, research only).

    python scripts/ncaab_ratings_backtest.py --output OUT.json [--scored-season 2025] [--probe]

Seasons: scored-2 sets the priors, scored-1 calibrates, scored is scored. About 45 CBBD calls
(games for three seasons, lines for two, in date windows). Needs CBBD_API_KEY (falls back to
CFBD_API_KEY: the same account serves both). No Odds API calls; writes nothing to Drive or the store.

``--probe`` (2026-11-17) adds one more call: a single day of team box scores, reduced to its field
names. The next model needs possessions and shooting per game, and this shows what the API really
returns before any code is written against it. The field names of the games and lines already
fetched are always recorded (``schema``); no values are kept, only names.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app_core import ncaab_line_backtest as data  # noqa: E402
from app_core import ncaab_ratings_backtest as bt  # noqa: E402


def field_names(value, depth=2):
    """The shape of an API value as names only: a dict's keys (nested to ``depth``), a list's first item. No values."""
    if isinstance(value, dict):
        return {str(key): (field_names(item, depth - 1) if depth > 0 and isinstance(item, (dict, list)) else type(item).__name__)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, list):
        return [field_names(value[0], depth)] if value else []
    return type(value).__name__


def _names(rows, key=None):
    """Every field name seen in ``rows`` (or in the dicts under ``key`` of each row), sorted."""
    names = set()
    for row in rows:
        items = [row] if key is None else [item for item in (row.get(key) or []) if isinstance(item, dict)]
        for item in items:
            names.update(str(name) for name in item)
    return sorted(names)


def probe_team_stats(token, season, get=None):
    """One day of ``/games/teams`` reduced to field names, or the error it gave. One CBBD call; never raises."""
    if get is None:
        import requests
        get = requests.get
    day = f"{season}-01-15"                       # mid-January: a full slate in any season
    try:
        rows = data._request("games/teams", {"season": season, "startDateRange": day, "endDateRange": day}, token, get)
        return {"endpoint": "games/teams", "day": day, "rows": len(rows), "fields": field_names(rows[:1], depth=3)}
    except ValueError as exc:                     # the status the API gave, e.g. "CBBD games/teams HTTP 403"
        return {"endpoint": "games/teams", "day": day, "error": str(exc)[:120]}
    except Exception as exc:                      # a probe must never cost the run its result; the type says enough
        return {"endpoint": "games/teams", "day": day, "error": type(exc).__name__}


def run(token, scored_season, get=None):
    if get is None:
        import requests
        get = requests.get
    prior_season, calibration_season = scored_season - 2, scored_season - 1
    raw_games = {s: data.fetch("games", s, token, get) for s in (prior_season, calibration_season, scored_season)}
    games = {s: data.parse_games(rows) for s, rows in raw_games.items()}
    raw_lines = {s: data.fetch("lines", s, token, get) for s in (calibration_season, scored_season)}
    lines = {s: bt.select_lines(rows) for s, rows in raw_lines.items()}
    scored_lines = lines[scored_season]
    return {"prior_season": prior_season, "calibration_season": calibration_season, "scored_season": scored_season,
            # Field names only, from rows already fetched: what a fuller model could read without another call.
            "schema": {"games": _names(raw_games[scored_season]), "lines": _names(raw_lines[scored_season]),
                       "line": _names(raw_lines[scored_season], "lines")},
            "games": {str(s): len(g) for s, g in games.items()},
            "closing_lined": sum(1 for v in scored_lines.values() if v["spread"] is not None),
            "opening_lined": sum(1 for v in scored_lines.values() if v["spread_open"] is not None),
            "providers": sorted({v["provider"] for v in scored_lines.values() if v["provider"]}),
            "report": bt.run(games[prior_season], games[calibration_season], games[scored_season],
                             lines[calibration_season], scored_lines, bt.select_pairs(raw_lines[scored_season]))}


def schema_markdown(result):
    """The field names found, for the log: what the next model can read."""
    schema = result.get("schema") or {}
    out = ["", "### Field names in the data (names only)", "",
           f"- games: {', '.join(schema.get('games') or []) or 'none'}",
           f"- lines rows: {', '.join(schema.get('lines') or []) or 'none'}",
           f"- one line: {', '.join(schema.get('line') or []) or 'none'}"]
    probe = result.get("team_stats_probe")
    if probe:
        detail = probe.get("error") or json.dumps(probe.get("fields"), sort_keys=True)
        out.append(f"- team box scores ({probe['endpoint']}, {probe['day']}, {probe.get('rows', 0)} rows): {detail}")
    return "\n".join(out) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scored-season", type=int, default=2025)
    parser.add_argument("--probe", action="store_true", help="also record the field names of one day of team box scores")
    args = parser.parse_args(argv)
    token = (os.getenv("CBBD_API_KEY", "") or os.getenv("CFBD_API_KEY", "")).strip()
    if not token:
        raise SystemExit("MISSING_CREDENTIALS: CBBD_API_KEY or CFBD_API_KEY")
    result = run(token, args.scored_season)
    if args.probe:
        result["team_stats_probe"] = probe_team_stats(token, args.scored_season)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = bt.markdown(result) + schema_markdown(result)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
