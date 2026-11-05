"""Print the pick-quality scoreboard for a folder of graded exports (roadmap E1.3).

Usage:
    python scripts/scoreboard.py [DIR] [--min-n 150] [--json OUT.json] [--markdown OUT.md]
                                 [--evidence-db data/prediction_evidence/evidence.sqlite3]

DIR defaults to data/backtest_exports. Read-only: no stored pick is modified,
and no network is used.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app_core.evaluation import scoreboard  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('directory', nargs='?', default=str(ROOT / 'data' / 'backtest_exports'))
    parser.add_argument('--min-n', type=int, default=150)
    parser.add_argument('--json', type=Path)
    parser.add_argument('--markdown', type=Path)
    parser.add_argument('--evidence-db', type=Path,
                        help='prediction-evidence SQLite (read-only) with closing observations, for CLV')
    args = parser.parse_args(argv)
    with contextlib.redirect_stdout(io.StringIO()):  # the loader prints a progress line
        rows = scoreboard.load(args.directory)
    board = scoreboard.scoreboard(rows)
    text = scoreboard.markdown(board, minimum=args.min_n)
    if args.evidence_db:
        from app_core.evaluation import clv
        clv_report = clv.report(clv.read_observations(args.evidence_db))
        board['clv'] = clv_report
        text += '\n' + clv.markdown(clv_report)
    if args.json:
        args.json.write_text(json.dumps(board, indent=2) + '\n', encoding='utf-8')
    if args.markdown:
        args.markdown.write_text(text, encoding='utf-8')
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
