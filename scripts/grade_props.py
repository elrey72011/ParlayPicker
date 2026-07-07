#!/usr/bin/env python3
"""Grade a strikeout-props card against actual results and append to the running log.

Usage
-----
    python3 scripts/grade_props.py <strikeout_props_export.csv> <YYYY-MM-DD>

Fetches each pitcher's actual strikeouts from the free MLB StatsAPI game log, grades every
``Actionable`` pick W/L, computes P&L at the staked Kelly, appends to
``data/prop_results/prop_results_log.csv`` (deduped by date+pitcher+pick), and prints the
slate result plus the running cumulative record, P&L, and ROI.

The grading logic (grade_side / grade_card) is pure and unit-tested; only fetch_actual_ks
touches the network.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

_BASE = "https://statsapi.mlb.com/api/v1"
_TIMEOUT = 10
LOG_PATH = Path("data/prop_results/prop_results_log.csv")


def _decimal(odds: float) -> float:
    o = float(odds)
    return 1 + (o / 100.0 if o > 0 else 100.0 / abs(o))


def _side_from_market(market_type, best_pick) -> str:
    m = str(market_type).lower()
    if "over" in m:
        return "over"
    if "under" in m:
        return "under"
    return "over" if "over" in str(best_pick).lower() else "under"


def grade_side(side: str, line: float, actual_ks) -> str | None:
    """WIN / LOSS / PUSH for a strikeout over/under given actual Ks. Pure.

    A .5 line never pushes; an integer line pushes on an exact match.
    """
    if actual_ks is None:
        return None
    line = float(line)
    if float(actual_ks) == line:
        return "PUSH"
    over = float(actual_ks) > line
    win = over if side == "over" else (not over)
    return "WIN" if win else "LOSS"


def _stat_for_market(market_type: object, best_pick: object) -> str:
    """Which counted stat a prop row grades against: ks | walks | outs.

    Order matters: "pitcher_strikeouts_*" contains the substring "outs", so
    strikeouts must be checked first.
    """
    # market_type is authoritative and checked FIRST — pick text is a fallback
    # only, with padded tokens. Two substring traps: "pitcher_strikeouts"
    # contains "outs" (strikeouts before outs), and pitcher NAMES can contain
    # stat words — "Walker Buehler Under 15.5 Outs" matched "walk" and graded
    # his OUTS pick against his WALKS (6 Jul).
    mt = str(market_type or "").lower()
    if "strikeout" in mt:
        return "ks"
    if "walk" in mt:
        return "walks"
    if "out" in mt:
        return "outs"
    text = f" {str(best_pick or '').lower()} "
    if " ks " in text:
        return "ks"
    if " bbs " in text:
        return "walks"
    if " outs " in text:
        return "outs"
    return "ks"


def grade_card(card: pd.DataFrame, date: str, name_to_id: dict, actual_ks_fn) -> list[dict]:
    """Grade a prop card DataFrame into result rows.

    ``actual_ks_fn(pitcher_id) -> int | dict | None``: an int is the strikeout
    count (historic single-market form); a dict carries {"ks", "walks", "outs"}
    and the row's market decides which stat grades it (4 Jul multi-market).
    """
    rows = []
    for _, r in card.iterrows():
        if str(r.get("Pick_Status", "")).strip() != "Actionable":
            continue
        name = str(r["pitcher"]).strip()
        side = _side_from_market(r.get("market_type"), r.get("best_pick"))
        line = float(r["line"])
        odds = float(r["odds_american"])
        stake = float(r.get("Kelly_Bet_Size") or 0.0)
        pid = name_to_id.get(name.lower())
        actual = actual_ks_fn(pid) if pid is not None else None
        if isinstance(actual, dict):
            ks = actual.get(_stat_for_market(r.get("market_type"), r.get("best_pick")))
        else:
            ks = actual
        res = grade_side(side, line, ks)
        if res == "WIN":
            profit = stake * (_decimal(odds) - 1)
        elif res == "LOSS":
            profit = -stake
        elif res == "PUSH":
            profit = 0.0
        else:
            profit = None
        rows.append({
            "date": date, "pitcher": name, "pick": r.get("best_pick"), "side": side,
            "line": line, "expected_ks": r.get("expected_ks"), "actual_ks": ks,
            "odds_american": odds, "stake": round(stake, 2), "result": res,
            "profit": round(profit, 2) if profit is not None else None,
        })
    return rows


def summarize(df: pd.DataFrame) -> dict:
    """Cumulative record / P&L / ROI over graded (WIN|LOSS) rows."""
    g = df[df["result"].isin(["WIN", "LOSS"])]
    w = int((g["result"] == "WIN").sum())
    l = int((g["result"] == "LOSS").sum())
    staked = float(pd.to_numeric(g["stake"], errors="coerce").fillna(0).sum())
    pnl = float(pd.to_numeric(g["profit"], errors="coerce").fillna(0).sum())
    return {"wins": w, "losses": l, "staked": staked, "pnl": pnl,
            "roi": (pnl / staked) if staked else 0.0,
            "win_rate": (w / (w + l)) if (w + l) else 0.0}


def append_log(rows: list[dict], log_path: Path = LOG_PATH) -> pd.DataFrame:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if log_path.exists():
        combined = pd.concat([pd.read_csv(log_path), new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date", "pitcher", "pick"], keep="last")
    else:
        combined = new
    combined.to_csv(log_path, index=False)
    return combined


# ── network (not unit-tested) ──
def _ip_to_outs(ip) -> int | None:
    """StatsAPI inningsPitched ("5.2") -> outs (17). The fraction digit IS the outs."""
    try:
        text = str(ip)
        whole, _, frac = text.partition(".")
        outs = int(frac) if frac in ("1", "2") else 0
        return int(whole) * 3 + outs
    except (TypeError, ValueError):
        return None


def fetch_actual_ks(pitcher_id, date: str, season: int, http_get=requests.get):
    """Actual pitching counts for the date: {"ks", "walks", "outs"} (or None)."""
    try:
        resp = http_get(f"{_BASE}/people/{pitcher_id}/stats",
                        params={"stats": "gameLog", "group": "pitching", "season": season},
                        timeout=_TIMEOUT)
        resp.raise_for_status()
        for sp in resp.json()["stats"][0]["splits"]:
            if sp.get("date") == date:
                stat = sp["stat"]
                return {
                    "ks": int(stat.get("strikeOuts", 0) or 0),
                    "walks": int(stat.get("baseOnBalls", 0) or 0),
                    "outs": _ip_to_outs(stat.get("inningsPitched")),
                }
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError):
        return None
    return None


def _build_name_to_id(date: str) -> dict:
    from app_core.mlb_pitcher_stats import fetch_schedule_probables
    ids = {}
    for g in fetch_schedule_probables(date) or []:
        for s in ("home", "away"):
            n, i = g.get(s + "_pitcher"), g.get(s + "_pitcher_id")
            if n and i:
                ids[str(n).strip().lower()] = i
    return ids


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    card_path, date = sys.argv[1], sys.argv[2]
    season = int(date[:4])
    card = pd.read_csv(card_path)
    name_to_id = _build_name_to_id(date)
    rows = grade_card(card, date, name_to_id, lambda pid: fetch_actual_ks(pid, date, season))

    slate = pd.DataFrame(rows)
    sl = summarize(slate)
    print(f"=== {date} slate ===")
    for r in rows:
        ks = "??" if r["actual_ks"] is None else r["actual_ks"]
        pl = "" if r["profit"] is None else f"${r['profit']:+.2f}"
        print(f"  {r['pitcher']:18s} {r['side']:5s} {r['line']:>4} -> {str(ks):>3} Ks  {str(r['result']):4}  {pl}")
    print(f"  slate: {sl['wins']}-{sl['losses']}   P&L ${sl['pnl']:+.2f}   ROI {sl['roi']:+.1%}")

    full = append_log(rows)
    cum = summarize(full)
    print(f"\n=== RUNNING TOTAL ({cum['wins'] + cum['losses']} graded props) ===")
    print(f"  record {cum['wins']}-{cum['losses']} ({cum['win_rate']:.1%})   "
          f"P&L ${cum['pnl']:+.2f} on ${cum['staked']:.0f} staked   ROI {cum['roi']:+.1%}")
    print(f"  log: {LOG_PATH}")


if __name__ == "__main__":
    main()
