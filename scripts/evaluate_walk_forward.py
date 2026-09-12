#!/usr/bin/env python3
"""Evaluate a probability column on a chronological holdout.

Usage:
  python scripts/evaluate_walk_forward.py EXPORT.csv game_date effective_win_probability result
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.walk_forward import walk_forward_evaluate, compare_by_league


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('path')
    parser.add_argument('date_col')
    parser.add_argument('prob_col')
    parser.add_argument('outcome_col')
    parser.add_argument('--market-probability-column')
    parser.add_argument('--league-column',default='league')
    parser.add_argument('--min-train-rows',type=int,default=100)
    args=parser.parse_args()
    if args.min_train_rows<1:
        parser.error('--min-train-rows must be positive')
    frame = pd.read_csv(args.path)
    if args.market_probability_column:
        result=compare_by_league(frame,args.date_col,args.prob_col,args.outcome_col,
                                 args.market_probability_column,league_col=args.league_column,
                                 min_train_rows=args.min_train_rows)
    else:
        result = walk_forward_evaluate(frame,args.date_col,args.prob_col,args.outcome_col,
                                       min_train_rows=args.min_train_rows)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
