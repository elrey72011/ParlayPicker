"""Local atomic request budget and short-lived cache for structured Gemini reviews.

This store survives process restarts, not loss of the deployment filesystem.
It covers batch reviews (including retries), not other Gemini entry points.
"""
import hashlib
import json
import os
import sqlite3
import time
from contextlib import closing


def connection():
    from app_core.prediction_evidence import database_path
    path = database_path().parent / 'gemini_review_usage.sqlite3'
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path, timeout=30)
    db.execute('CREATE TABLE IF NOT EXISTS requests (at REAL NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, at REAL, payload TEXT)')
    return db


def request_key(model, prompt):
    return hashlib.sha256((model + '\nreview-v2\n' + prompt).encode()).hexdigest()


def lookup(key):
    with closing(connection()) as db:
        row = db.execute('SELECT at,payload FROM cache WHERE key=?', (key,)).fetchone()
    return json.loads(row[1]) if row and 0 <= time.time()-row[0] < 600 else None


def reserve():
    # Invalid configuration disables calls rather than silently raising limits.
    try:
        raw = os.environ.get('PARLAYPICKER_GEMINI_DAILY_REQUESTS')
        if raw is None:
            import streamlit as st
            try:
                raw = st.secrets.get('PARLAYPICKER_GEMINI_DAILY_REQUESTS', 20)
            except (FileNotFoundError, KeyError):
                raw = 20
        cap = max(0, int(raw))
    except (ValueError, TypeError):
        cap = 0
    now = time.time()
    start = now - now % 86400
    with closing(connection()) as db:
        db.execute('BEGIN IMMEDIATE')
        count = db.execute('SELECT COUNT(*) FROM requests WHERE at>=?', (start,)).fetchone()[0]
        if count >= cap:
            db.rollback()
            return False
        db.execute('INSERT INTO requests VALUES (?)', (now,))
        db.commit()
    return True


def save(key, payload):
    with closing(connection()) as db:
        db.execute('INSERT OR REPLACE INTO cache VALUES (?,?,?)', (key, time.time(), json.dumps(payload)))
        db.commit()
