"""Publish research picks (NBA, NCAAB, NHL) and their replayed record. Read-only restore.

    python scripts/research_picks.py --directory EMPTY_DIR --output OUT.json [--cache-dir DIR]

Restores the canonical store with every Drive upload refused, then builds the board
(``app_core.research_picks.board``) and the replayed record (``replay``) per sport and
market. Research only: no stake, no plan, no wager authority.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app_core import research_picks as rp  # noqa: E402
from app_core import selector_shadow_compare as shadow  # noqa: E402
from app_core import pinnacle_value as pv  # noqa: E402
from app_core.prospective_research_models import SCOPES  # noqa: E402

NOTICE = ("Research picks from two in-season models: the stored score model, and a line-anchored model "
          "that predicts the result against the line. Best expected value at the best available price. "
          "No market has qualified, so every pick is NO_BET with zero stake.")


def build(path, now=None):
    now = now or datetime.now(timezone.utc)
    boards, records, comparison = [], {}, {}
    for sport in rp.SPORTS:
        for market in SCOPES[sport]:
            for method in ("score", "anchored"):
                boards.append(rp.board(path, sport, market, now=now, method=method))
                graded = rp.replay(path, sport, market, now=now, method=method)
                records[f"{sport} {market} ({method})"] = dict(rp.record(graded), positive_ev=rp.record(
                    [g for g in graded if g["ev"] > 0]))
                if sport != "NHL":
                    comparison[f"{sport} {market} ({method})"] = shadow.summarize(
                        shadow.compare(path, sport, market, now=now, method=method))
    for sport in pv.PINNACLE_SPORTS:
        boards.append(pv.board(path, sport, "SPREAD", now=now))
        records[f"{sport} SPREAD ({pv.METHOD})"] = dict(pv.record(pv.replay(path, sport, "SPREAD", now=now)),
                                                       positive_ev=rp.record([]))
    return {"schema": "research-picks-v1", "as_of": now.isoformat(), "boards": boards, "records": records,
            "selector_comparison": comparison, "wager_authority": False}


def markdown(result):
    def pct(v):
        return "" if v is None else f"{v:.1%}"
    lines = ["## Research picks", "", NOTICE, ""]
    for b in result["boards"]:
        method = b.get("method", "score")
        head = f"### {b['sport']} {b['market_family']} ({'Pinnacle value' if method == pv.METHOD else method + ' model'})"
        if b["blocker"]:
            lines += [head, f"- {b['blocker']}", ""]
            continue
        trained = (" (prices that beat Pinnacle's no-vig line by 2%+)" if b.get("method") == pv.METHOD
                   else f" (model trained on {b['training_games']} games)")
        lines += [head + trained, "",
                  "| start (UTC) | game | pick | line | price | book | P(win) | EV | consensus |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for p in b["picks"]:
            lines.append(f"| {p['scheduled_start'][:16]} | {p['away_team']} @ {p['home_team']} | {p['selection']} | "
                         f"{p['line']:+g} | {p['american_odds']:+d} | {p['sportsbook']} | {pct(p['win_probability'])} | "
                         f"{p['ev']:+.1%} | {pct(p['consensus_probability'])} |")
        if not b["picks"]:
            lines.append("| | no upcoming games with prices | | | | | | | |")
        lines.append("")
    lines += ["### Replayed record (each finished game as of its first captured price)", "",
              "| market | picks | W-L-P | hit rate (95%) | ROI/pick | +EV picks | +EV W-L-P | +EV ROI |",
              "|---|---|---|---|---|---|---|---|"]
    for key, r in result["records"].items():
        e = r["positive_ev"]
        interval = f"{pct(r['hit_rate'])} ({pct(r['hit_rate_95_low'])}–{pct(r['hit_rate_95_high'])})" if r["hit_rate"] is not None else ""
        lines.append(f"| {key} | {r['picks']} | {r['wins']}-{r['losses']}-{r['pushes']} | {interval} | "
                     f"{pct(r['roi_per_pick'])} | {e['picks']} | {e['wins']}-{e['losses']}-{e['pushes']} | "
                     f"{pct(e['roi_per_pick'])} |")
    value = {k: r for k, r in result["records"].items() if pv.METHOD in k}
    if value:
        lines += ["", "### Pinnacle value: closing-line value against Pinnacle's last pregame no-vig line", "",
                  "| market | picks | W-L-P | ROI/pick | CLV measured | mean CLV | beat close |", "|---|---|---|---|---|---|---|"]
        for key, r in value.items():
            clv = "" if r.get("mean_clv") is None else f"{r['mean_clv']:+.1%}"
            lines.append(f"| {key} | {r['picks']} | {r['wins']}-{r['losses']}-{r['pushes']} | {pct(r['roi_per_pick'])} | "
                         f"{r.get('clv_measured', 0)} | {clv} | {pct(r.get('beat_close_share'))} |")
    comparison = result.get("selector_comparison") or {}
    if comparison:
        lines += ["", "### Selector shadow test (D7): current rule vs EV selector on the same games", "",
                  "Current: bet the best-EV side when EV >= 0. EV selector: market nudged toward the model "
                  "(at most 5 points), bet only with an edge of 2+ points. CLV is against the last captured "
                  "pregame consensus, not a certified close.", "",
                  "| market | games | rule | bets | W-L-P | ROI/pick | mean CLV | beat close | same side |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for key, c in comparison.items():
            for rule in shadow.RULES:
                r = c[rule]
                same = f"{c['same_side']}/{c['both_bet']}" if rule == "ev_selector" else ""
                clv = "" if r["mean_clv"] is None else f"{r['mean_clv']:+.1%}"
                lines.append(f"| {key} | {c['games']} | {rule} | {r['picks']} | {r['wins']}-{r['losses']}-{r['pushes']} | "
                             f"{pct(r['roi_per_pick'])} | {clv} | {pct(r['beat_close_share'])} | {same} |")
    lines += ["", "NHL is not graded here: its settlement needs the per-game official-score check."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args(argv)
    args.directory.mkdir(parents=True, exist_ok=True)
    if any(args.directory.iterdir()):
        raise SystemExit("the restore directory must start empty")
    from app_core.evidence_drive import DRIVE_CACHE_ENV, DriveStore
    from app_core.evidence_remote import settings
    from app_core.prospective_remote import sync
    from market_readiness_report import ReadOnlyDrive
    folder = settings()[0]
    if not folder or not os.getenv("PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT", "").strip():
        raise SystemExit("MISSING_CREDENTIALS: PARLAYPICKER_DRIVE_FOLDER_ID / PARLAYPICKER_GOOGLE_SERVICE_ACCOUNT")
    if args.cache_dir:
        os.environ[DRIVE_CACHE_ENV] = str(args.cache_dir)
    path = args.directory / "prospective-evidence.sqlite3"
    sync(path, ReadOnlyDrive(DriveStore(folder), args.cache_dir), folder, {})
    result = build(path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    text = markdown(result)
    args.output.with_suffix(".md").write_text(text, encoding="utf-8")
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as fh:
            fh.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
