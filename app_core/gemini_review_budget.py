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
    db.execute(
        '''CREATE TABLE IF NOT EXISTS request_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            at REAL NOT NULL,
            request_key TEXT NOT NULL,
            model TEXT NOT NULL,
            batch_size INTEGER NOT NULL,
            duration_ms REAL NOT NULL,
            cache_hit INTEGER NOT NULL,
            prompt_tokens INTEGER,
            output_tokens INTEGER,
            thought_tokens INTEGER,
            cached_tokens INTEGER,
            total_tokens INTEGER,
            finish_reason TEXT NOT NULL,
            error_type TEXT NOT NULL
        )'''
    )
    return db


def request_key(model, prompt, configuration=''):
    return hashlib.sha256(
        (model + '\nreview-v3\n' + str(configuration) + '\n' + prompt).encode()
    ).hexdigest()


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


def record_metric(
    *,
    request_key,
    model,
    batch_size,
    duration_ms,
    cache_hit=False,
    prompt_tokens=None,
    output_tokens=None,
    thought_tokens=None,
    cached_tokens=None,
    total_tokens=None,
    finish_reason='',
    error_type='',
):
    """Append one provider attempt or cache lookup without storing prompt text."""
    values = (
        time.time(),
        str(request_key),
        str(model),
        max(0, int(batch_size)),
        max(0.0, float(duration_ms)),
        int(bool(cache_hit)),
        *[
            None if value is None else max(0, int(value))
            for value in (
                prompt_tokens,
                output_tokens,
                thought_tokens,
                cached_tokens,
                total_tokens,
            )
        ],
        str(finish_reason or ''),
        str(error_type or ''),
    )
    with closing(connection()) as db:
        db.execute(
            '''INSERT INTO request_metrics (
                at, request_key, model, batch_size, duration_ms, cache_hit,
                prompt_tokens, output_tokens, thought_tokens, cached_tokens,
                total_tokens, finish_reason, error_type
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            values,
        )
        db.commit()


def recent_metrics(limit=100):
    """Return recent redacted request metrics for diagnostics and tests."""
    columns = (
        'at', 'request_key', 'model', 'batch_size', 'duration_ms', 'cache_hit',
        'prompt_tokens', 'output_tokens', 'thought_tokens', 'cached_tokens',
        'total_tokens', 'finish_reason', 'error_type',
    )
    safe_limit = min(1000, max(1, int(limit)))
    with closing(connection()) as db:
        rows = db.execute(
            '''SELECT at, request_key, model, batch_size, duration_ms, cache_hit,
                      prompt_tokens, output_tokens, thought_tokens, cached_tokens,
                      total_tokens, finish_reason, error_type
               FROM request_metrics ORDER BY id DESC LIMIT ?''',
            (safe_limit,),
        ).fetchall()
    return [dict(zip(columns, row)) for row in rows]
