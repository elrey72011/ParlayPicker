"""Per-market progress toward qualification, from a market-readiness report (roadmap C3).

Input: the JSON written by ``scripts/market_readiness_report.py`` (workflow "Market
readiness (read-only)"). Output: one row per research market with plain-language stage,
fractional progress toward each gate, and the next blocker. Pure and descriptive: it
reads a report and never authorizes a wager.
"""
from __future__ import annotations

TRAINING_TARGET = 120     # pinned-candidate minimum (prospective_calibration)
CALIBRATION_TARGET = 40   # decided calibration predictions (prospective_calibration)
VALIDATION_TARGET = 200   # frozen-plan minimum independent sample (v1 template)

STAGES = {
    "SCORE_SETTLEMENT_NOT_ENABLED": "Settlement off",
    "VALIDATION_IN_PROGRESS": "Validating",
    "READY_FOR_OWNER_REVIEW": "Ready for your review",
    "NO_RECENT_GAMES_TO_SIZE_WINDOW": "Waiting for games",
}


def _fraction(value, target):
    try:
        return max(0.0, min(1.0, float(value or 0) / target))
    except (TypeError, ValueError):
        return 0.0


def _stage(blocker):
    if not blocker:
        return "Unknown"
    if blocker.startswith("CANDIDATE_NEEDS_"):
        return "Collecting training games"
    if blocker.startswith("CALIBRATION_NEEDS_"):
        return "Calibrating"
    return STAGES.get(blocker, blocker.replace("_", " ").capitalize())


def rows(report):
    plans = {p.get("validation_plan_id"): p for p in report.get("plans") or []}
    edges = {(e.get("sport"), e.get("market_family")): e for e in report.get("price_edge") or []}
    out = []
    for binding in report.get("research_bindings") or []:
        key = (binding.get("sport"), binding.get("market_family"))
        plan = plans.get(binding.get("bound_plan_id")) or {}
        validation = plan.get("validation") if isinstance(plan.get("validation"), dict) else {}
        settled = validation.get("scored_observations", 0)
        edge = ((edges.get(key) or {}).get("by_threshold") or {}).get("0.02") or {}
        blocker = binding.get("next_blocker") or binding.get("evaluation_error")
        out.append({
            "sport": key[0], "market": key[1], "stage": _stage(blocker),
            "training_games": binding.get("most_training_events", 0),
            "training_progress": _fraction(binding.get("most_training_events"), TRAINING_TARGET),
            "calibration_decided": binding.get("candidate_decided", 0),
            "calibration_progress": _fraction(binding.get("candidate_decided"), CALIBRATION_TARGET),
            "validation_settled": settled,
            "validation_progress": _fraction(settled, VALIDATION_TARGET),
            "games_per_day": binding.get("games_per_day_14d"),
            "window_days": binding.get("proposed_window_days"),
            "price_screen_picks": edge.get("picks"),
            "price_screen_beat_close": edge.get("beat_close_share"),
            "plan_status": plan.get("status"),
            "next_blocker": blocker})
    return out


def summary(progress, generated_at=None):
    """A one-line headline: the market closest to qualification and what it waits for."""
    if not progress:
        return "No research markets in this report."
    order = {"Validating": 4, "Ready for your review": 3, "Calibrating": 2, "Collecting training games": 1}

    def score(row):
        return (order.get(row["stage"], 0), row["validation_progress"], row["calibration_progress"],
                row["training_progress"])
    best = max(progress, key=score)
    when = f" (report {generated_at})" if generated_at else ""
    return f"Closest to qualifying: {best['sport']} {best['market']}, {best['stage'].lower()}{when}."


ARTIFACT_NAME = "market-readiness"
REPORT_FILE = "market-readiness.json"
API = "https://api.github.com/repos/{repo}/actions/artifacts"


def latest_report(repo, token, get, *, max_bytes=20_000_000):
    """(report, created_at) from the newest unexpired market-readiness artifact (read-only).

    ``token`` needs only Actions read access. ``get`` is ``requests.get``-like.
    """
    import io
    import json as _json
    import zipfile
    headers = {"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json"}
    listing = get(API.format(repo=repo), params={"name": ARTIFACT_NAME, "per_page": 20},
                  headers=headers, timeout=20)
    listing.raise_for_status()
    artifacts = [a for a in listing.json().get("artifacts") or [] if not a.get("expired")]
    if not artifacts:
        raise ValueError("no market-readiness report has been published yet")
    newest = max(artifacts, key=lambda a: a.get("created_at") or "")
    if (newest.get("size_in_bytes") or 0) > max_bytes:
        raise ValueError("market-readiness artifact is unexpectedly large")
    archive = get(newest["archive_download_url"], headers=headers, timeout=60)
    archive.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        name = next((n for n in bundle.namelist() if n.rsplit("/", 1)[-1] == REPORT_FILE), None)
        if name is None:
            raise ValueError("the artifact has no market-readiness.json")
        return _json.loads(bundle.read(name).decode("utf-8")), newest.get("created_at")
