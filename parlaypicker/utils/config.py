"""Centralized application configuration."""
from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
MODEL_DIR = ROOT_DIR / "models"

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
KALSHI_API_KEY = os.getenv("KALSHI_API_KEY", "")
THEOVER_API_KEY = os.getenv("THEOVER_API_KEY", "")

EV_THRESHOLD = float(os.getenv("EV_THRESHOLD", "0.03"))
KELLY_MULTIPLIER = float(os.getenv("KELLY_MULTIPLIER", "0.5"))

MODEL_WEIGHTS = {
    "market": 0.45,
    "ml": 0.30,
    "kalshi": 0.15,
    "theover": 0.07,
    "sentiment": 0.03,
}

CACHE_DIR.mkdir(parents=True, exist_ok=True)
