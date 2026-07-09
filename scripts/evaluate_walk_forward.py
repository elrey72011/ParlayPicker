#!/usr/bin/env python3
"""Evaluate a probability column on a chronological holdout.

Usage:
  python scripts/evaluate_walk_forward.py EXPORT.csv game_date effective_win_probability result
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.walk_forward import walk_forward_evaluate


def main() -> int:
    if len(sys.argv) != 5:
        print(__doc__.strip())
        return 2
    path, date_col, prob_col, outcome_col = sys.argv[1:]
    frame = pd.read_csv(path)
    result = walk_forward_evaluate(frame, date_col, prob_col, outcome_col)
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
