"""Does the closing line already price a missing key player? (NCAAB, CBBD history, research only.)

Owner 2026-11-20 ("Build it"). The ratings and possession models matched the closing line and did
not beat it, so the next question is whether information the models never saw, who actually played,
is in the line as well. This asks it on finished seasons before anything is built to use it.

* Data: CollegeBasketballData ``/games``, ``/lines`` and ``/games/players`` (one row per team per
  game, with every player's minutes).
* Key players, strictly as of each tip-off: a team's top ``TOP_N`` players by minutes per team game
  over its earlier games that season, each averaging at least ``KEY_MINUTES``. A team needs
  ``MIN_PRIOR`` earlier games with a box score before it is judged at all.
* Short-handed: at least one key player logged no minutes in the game. ``streak`` is how many of
  the team's games that player had already missed in a row, so 0 is the first game missed: the
  one where the news is freshest and the line is least likely to have it.
* Scored: games where exactly one side is short and the other is known to be at full strength,
  from the short side's view, against the closing spread. Betting against the short side at -110
  needs ``BREAK_EVEN`` to pay. The game's points against the closing total are reported beside it.

The verdict rule is fixed here, before any real season is read (``verdict``): the headline is the
first game missed, both seasons pooled. Every other split is description, not a second chance.

What this cannot see: why a player sat (injury, suspension, a coach's choice), and when the news
came out. An absence is read from the box score after the fact, so a positive result says only
that a forward test, with absences known before tip-off, would be worth building. Archived lines
carry no quote time. Research only: nothing here is read by selection, rating or staking, and
nothing is written to the evidence store.
"""
from __future__ import annotations

from datetime import date, timedelta
import math

from app_core import ncaab_line_backtest as data
from app_core import ncaab_possession_model as possession

TOP_N = 3                   # a team's key players: its top three by minutes
KEY_MINUTES = 20.0          # ... each averaging at least this many minutes per team game
MIN_PRIOR = 5               # earlier games with a box score before a team is judged
WIN = 100 / 110
BREAK_EVEN = 110 / 210      # 52.4%: what a side must win at -110
MIN_CASES = 200             # fewer first-missed games than this and the verdict is "too few"
SPLIT_AT = 1000             # a window returning this many rows may be cut off: halve it and ask again
MAX_CALLS = 80
WINDOW_DAYS = 7
STREAKS = (("first game missed", (0, 0)), ("second or third", (1, 2)), ("fourth or later", (3, 10 ** 6)))
SEGMENTS = (("Nov-Dec", (11, 12)), ("Jan-Apr", (1, 2, 3, 4)))


# ---- data ----

def parse_players(rows, into=None):
    """{(game id, is home): {player: minutes}} keeping only players who logged minutes."""
    out = {} if into is None else into
    for row in rows:
        game_id, is_home = data._first(row, "gameId", "id"), row.get("isHome")
        if game_id is None or is_home is None:
            continue
        minutes = {}
        for player in row.get("players") or []:
            if not isinstance(player, dict):
                continue
            who = data._first(player, "athleteId", "athleteSourceId", "name")
            played = data._number(player.get("minutes"))
            if who is not None and played is not None and played > 0:
                minutes[who] = minutes.get(who, 0.0) + played
        if minutes:
            out[(game_id, bool(is_home))] = minutes
    return out


def fetch_players(season, token, get):
    """Every team's player minutes for a season: ``parse_players`` over weekly windows.

    Both of a game's rows share its id, so rows are kept per team, never de-duplicated by game.
    A window that returns ``SPLIT_AT`` rows or more is halved and asked again, down to one day.
    Rows are reduced to minutes as they arrive: the raw season is far larger than what is kept.
    """
    pending, first = [], date(season - 1, 10, 28)
    while first < date(season, 4, 15):
        pending.append((first, first + timedelta(days=WINDOW_DAYS)))
        first += timedelta(days=WINDOW_DAYS)
    out, calls = {}, 0
    while pending:
        start, end = pending.pop(0)
        calls += 1
        if calls > MAX_CALLS:
            raise ValueError(f"CBBD games/players: more than {MAX_CALLS} requests for season {season}")
        chunk = data._request("games/players", {"season": season, "startDateRange": start.isoformat(),
                                                "endDateRange": (end - timedelta(days=1)).isoformat()}, token, get)
        if len(chunk) >= SPLIT_AT:
            if (end - start).days <= 1:
                raise ValueError(f"CBBD games/players: a single day returned {len(chunk)} rows")
            middle = start + (end - start) // 2
            pending[:0] = [(start, middle), (middle, end)]
            continue
        parse_players(chunk, out)
    return out


# ---- who was missing, as of each game ----

def availability(games, box):
    """{(game id, is home): {"key": n, "absent": n, "streak": n or None}} from earlier games only.

    ``games`` are in start order. A team is judged in a game only when that game has its box score,
    it has ``MIN_PRIOR`` earlier ones, and at least one player qualifies as key.
    """
    schedule = {}
    for game in games:
        schedule.setdefault(game["home"], []).append((game["id"], True))
        schedule.setdefault(game["away"], []).append((game["id"], False))
    out = {}
    for team_games in schedule.values():
        total, missed, counted = {}, {}, 0
        for key in team_games:
            played = box.get(key)
            if played is None:
                continue
            if counted >= MIN_PRIOR:
                ranked = sorted(total, key=lambda who: (-total[who], str(who)))[:TOP_N]
                regulars = [who for who in ranked if total[who] / counted >= KEY_MINUTES]
                absent = [who for who in regulars if who not in played]
                if regulars:
                    out[key] = {"key": len(regulars), "absent": len(absent),
                                "streak": min(missed[who] for who in absent) if absent else None}
            for who in set(total) | set(played):
                missed[who] = 0 if who in played else missed.get(who, 0) + 1
                total[who] = total.get(who, 0.0) + played.get(who, 0.0)
            counted += 1
    return out


def conference_tiers(games, prior_games):
    """{conference: "high" or "other"} from last season's average margins, this season's membership."""
    margins = {}
    for game in prior_games:
        margin = game["home_points"] - game["away_points"]
        margins.setdefault(game["home"], []).append(margin)
        margins.setdefault(game["away"], []).append(-margin)
    finals = {team: (sum(values) / len(values), 0.0) for team, values in margins.items() if len(values) >= 10}
    return possession.conference_tiers(games, finals)


def _streak_label(streak):
    return next(label for label, (low, high) in STREAKS if low <= streak <= high)


def cases(games, box, lines, tiers, season=None):
    """(short-handed cases, full-strength games) among games with both sides judged and a closing line.

    A case is a game where exactly one side is short, from that side's view: ``cover`` is its margin
    against the closing spread (positive = it covered), ``move`` is how many points the spread moved
    against it from open to close. Full-strength games (nobody short) are the baseline for totals.
    """
    status = availability(games, box)
    short, full = [], []
    for game in games:
        home, away, line = status.get((game["id"], True)), status.get((game["id"], False)), lines.get(game["id"])
        if home is None or away is None or line is None:
            continue
        margin = game["home_points"] - game["away_points"]
        total_error = None if line["total"] is None else game["home_points"] + game["away_points"] - line["total"]
        base = {"season": season, "game_id": game["id"], "month": game["start"].month, "total_error": total_error,
                "tier": possession.game_tier(game, tiers)}
        if not home["absent"] and not away["absent"]:
            full.append(base)
            continue
        if home["absent"] and away["absent"]:
            continue
        is_home = bool(home["absent"])
        side, sign = (home, 1.0) if is_home else (away, -1.0)
        spread, opened = line["spread"], line["spread_open"]
        short.append({**base, "site": "neutral" if game["neutral"] else ("home" if is_home else "away"),
                      "absent": side["absent"], "streak": side["streak"], "streak_label": _streak_label(side["streak"]),
                      "cover": None if spread is None else sign * (margin + spread),
                      "move": None if spread is None or opened is None else sign * (spread - opened)})
    return short, full


# ---- the record ----

def _wilson(wins, n, z=1.96):
    if not n:
        return None, None
    rate, denominator = wins / n, 1 + z * z / n
    centre = (rate + z * z / (2 * n)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denominator
    return centre - half, centre + half


def record(rows):
    """The short side against the closing spread, and what betting against it at -110 returned."""
    decided = [row["cover"] for row in rows if row.get("cover") is not None]
    covered, failed = sum(1 for c in decided if c > 0), sum(1 for c in decided if c < 0)
    n = covered + failed
    low, high = _wilson(failed, n)
    moves = [row["move"] for row in rows if row.get("move") is not None]
    return {"games": len(decided), "covered": covered, "failed": failed, "pushes": len(decided) - n,
            "fade_rate": failed / n if n else None, "fade_low": low, "fade_high": high,
            "fade_units": failed * WIN - covered,
            "mean_cover": sum(decided) / len(decided) if decided else None,
            "mean_move": sum(moves) / len(moves) if moves else None, "moves": len(moves)}


def totals_record(rows):
    """The game's points against the closing total."""
    errors = [row["total_error"] for row in rows if row.get("total_error") is not None]
    over, under = sum(1 for e in errors if e > 0), sum(1 for e in errors if e < 0)
    n = over + under
    low, high = _wilson(under, n)
    return {"games": len(errors), "over": over, "under": under, "pushes": len(errors) - n,
            "under_rate": under / n if n else None, "under_low": low, "under_high": high,
            "under_units": under * WIN - over, "mean_error": sum(errors) / len(errors) if errors else None}


def verdict(short):
    """The answer, by a rule fixed before any season was read.

    Headline: the first game a key player misses, seasons pooled. "worth a forward test" needs
    ``MIN_CASES`` decided games, the low end of the 95% range of the fade above ``BREAK_EVEN``,
    and the fade above half in every season on its own. Anything else with enough games is
    "priced in".
    """
    first = [row for row in short if row["streak"] == 0]
    pooled = record(first)
    decided = pooled["covered"] + pooled["failed"]
    seasons = {season: record([row for row in first if row["season"] == season])
               for season in sorted({row["season"] for row in first}, key=str)}
    if decided < MIN_CASES:
        answer = "too few games to say"
    elif pooled["fade_low"] > BREAK_EVEN and all((r["fade_rate"] or 0) > 0.5 for r in seasons.values()):
        answer = "worth a forward test"
    else:
        answer = "priced in"
    return {"answer": answer, "headline": pooled, "by_season": {str(k): v for k, v in seasons.items()},
            "rule": {"min_cases": MIN_CASES, "break_even": BREAK_EVEN}}


def summarise(short, full):
    def split(key, labels):
        return {label: record([row for row in short if row[key] == label]) for label in labels}

    first = [row for row in short if row["streak"] == 0]
    return {
        "verdict": verdict(short),
        "all_short": record(short),
        "by_streak": split("streak_label", [label for label, _ in STREAKS]),
        "first_missed_by_tier": {tier: record([r for r in first if r["tier"] == tier]) for tier in ("high", "mixed", "low")},
        "first_missed_by_site": {site: record([r for r in first if r["site"] == site]) for site in ("home", "away", "neutral")},
        "first_missed_by_part": {label: record([r for r in first if r["month"] in months]) for label, months in SEGMENTS},
        "first_missed_by_count": {"one key player out": record([r for r in first if r["absent"] == 1]),
                                  "two or more out": record([r for r in first if r["absent"] >= 2])},
        "totals": {"first game missed": totals_record(first), "all short-handed": totals_record(short),
                   "full strength (baseline)": totals_record(full)},
        "settings": {"top_n": TOP_N, "key_minutes": KEY_MINUTES, "min_prior_games": MIN_PRIOR},
    }


def run(seasons):
    """``seasons``: [(season, games, prior season's games, box, lines)] -> the report."""
    short, full, coverage = [], [], {}
    for season, games, prior_games, box, lines in seasons:
        season_short, season_full = cases(games, box, lines, conference_tiers(games, prior_games), season)
        short += season_short
        full += season_full
        lined = [game for game in games if (lines.get(game["id"]) or {}).get("spread") is not None]
        coverage[str(season)] = {
            "finished_games": len(games), "with_closing_spread": len(lined),
            "lined_with_both_box_scores": sum(1 for g in lined if (g["id"], True) in box and (g["id"], False) in box),
            "lined_with_one_box_score": sum(1 for g in lined if ((g["id"], True) in box) != ((g["id"], False) in box)),
            "short_handed_cases": len(season_short), "full_strength_games": len(season_full)}
    return {**summarise(short, full), "coverage": coverage}


# ---- the write-up ----

def _pct(value):
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _num(value, signed=True):
    return "n/a" if value is None else (f"{value:+.2f}" if signed else f"{value:.2f}")


def _row(label, r):
    return (f"| {label} | {r['games']} | {r['covered']}-{r['failed']}-{r['pushes']} | {_pct(r['fade_rate'])} | "
            f"{_pct(r['fade_low'])} to {_pct(r['fade_high'])} | {_num(r['fade_units'])} | {_num(r['mean_cover'])} | "
            f"{_num(r['mean_move'])} |")


def markdown(result):
    report, v = result["report"], result["report"]["verdict"]
    head = ("| Group | Games | Short side covered-failed-push | Betting against it won | 95% range | Units at -110 | "
            "Avg vs spread (pts) | Line moved against it, open to close (pts) |\n|---|---|---|---|---|---|---|---|")
    out = [f"## NCAAB: is a missing key player already in the closing line? (seasons {', '.join(map(str, result['seasons']))})",
           "",
           f"**Answer: {v['answer']}.** Rule set before the data was read: the first game a key player misses, "
           f"seasons pooled, needs {v['rule']['min_cases']} decided games, a 95% range wholly above "
           f"{_pct(v['rule']['break_even'])}, and more wins than losses in each season.",
           "",
           "Research only. An absence is read from the box score after the game, so this cannot say when "
           "the news was known. Not a betting record.", "", head,
           _row("**First game missed (headline)**", v["headline"])]
    out += [_row(f"First game missed, season {season}", r) for season, r in v["by_season"].items()]
    out += [_row(label.capitalize(), r) for label, r in report["by_streak"].items() if label != "first game missed"]
    out.append(_row("Every short-handed game", report["all_short"]))
    for title, key in (("conference strength", "first_missed_by_tier"), ("where the short side played", "first_missed_by_site"),
                       ("part of season", "first_missed_by_part"), ("how many were out", "first_missed_by_count")):
        out += ["", f"First game missed, by {title} (description only):", "", head]
        out += [_row(label, r) for label, r in report[key].items()]
    out += ["", "Points against the closing total:", "",
            "| Group | Games | Over-under-push | Under won | 95% range | Units on the under at -110 | Avg points vs total |",
            "|---|---|---|---|---|---|---|"]
    out += [f"| {label.capitalize()} | {r['games']} | {r['over']}-{r['under']}-{r['pushes']} | {_pct(r['under_rate'])} | "
            f"{_pct(r['under_low'])} to {_pct(r['under_high'])} | {_num(r['under_units'])} | {_num(r['mean_error'])} |"
            for label, r in report["totals"].items()]
    s = report["settings"]
    out += ["", f"- key players: a team's top {s['top_n']} by minutes averaging at least {s['key_minutes']:.0f} a game, "
                f"judged after {s['min_prior_games']} games, from earlier games only",
            f"- coverage by season: {report['coverage']}", ""]
    return "\n".join(out)
