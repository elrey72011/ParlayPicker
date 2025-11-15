"""Utility to verify NewsAPI-backed sentiment analysis.

Usage:
    python scripts/verify_sentiment.py --team "Los Angeles Lakers" --sport nba \
        --api-key $NEWS_API_KEY

If no API key is supplied via --api-key or the NEWS_API_KEY environment
variable, the script reports the neutral fallback values that the
Streamlit app will also use.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app_core.sentiment import RealSentimentAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check sentiment scores for a team")
    parser.add_argument("--team", required=True, help="Team name to analyze")
    parser.add_argument("--sport", required=True, help="Sport key (e.g. nba, nfl)")
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional NewsAPI key. Falls back to NEWS_API_KEY env var if omitted.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON response instead of emitting a compact string.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = args.api_key or os.environ.get("NEWS_API_KEY")

    analyzer = RealSentimentAnalyzer(api_key)
    result: Dict[str, Any] = analyzer.get_team_sentiment(args.team, args.sport)

    dump = json.dumps(result, indent=2 if args.pretty else None)
    print(dump)


if __name__ == "__main__":
    main()
