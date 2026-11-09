"""Is a research loop running or waiting? (audit 2026-11-09)

    python scripts/research_loop_active.py      # writes active=true|false to $GITHUB_OUTPUT

The research loop, the research scheduler and Football Stage 1 share the
``research-capture-and-grading`` concurrency group, and GitHub keeps only one *pending* run
per group: a newer queued run cancels the older pending one. The scheduler (every 30 minutes)
and Football Stage 1 (hourly) kept replacing a research loop that was waiting its turn
(cancelled loops on 2026-11-07 and 2026-11-08), so the slate lost its research cycles.

A scheduled scheduler or Stage 1 run calls this first, outside the group, and does nothing
while a loop is queued, waiting or running (the loop runs the same cycle every 30 minutes).
Manual runs are never skipped. An API error answers "not active" (the old behaviour).
"""
from __future__ import annotations

import os

LOOP_WORKFLOW = "research-loop.yml"
ACTIVE = ("in_progress", "queued", "waiting", "pending", "requested")


def loop_active(get=None, setting=os.environ.get) -> bool:
    if setting("GITHUB_EVENT_NAME", "") != "schedule":
        return False
    token, repo = setting("GITHUB_TOKEN", ""), setting("GITHUB_REPOSITORY", "")
    if not token or not repo:
        return False
    if get is None:
        import requests
        get = requests.get
    for status in ACTIVE:
        try:
            response = get(f"https://api.github.com/repos/{repo}/actions/workflows/{LOOP_WORKFLOW}/runs",
                           params={"status": status, "per_page": 1}, timeout=15,
                           headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
            if response.status_code == 200 and response.json().get("total_count", 0) > 0:
                return True
        except Exception:  # an API hiccup never blocks research
            continue
    return False


def main() -> int:
    active = loop_active()
    print(f"research loop active: {str(active).lower()}")
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"active={str(active).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
