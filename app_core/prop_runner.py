"""End-to-end orchestration for the pitcher-strikeout prop slice.

Wires the four standalone pieces into one runnable card:

  1. list the day's MLB events (Odds API) and pull each event's strikeout props,
  2. pull the day's probable starters + team ids (StatsAPI schedule),
  3. resolve each propped pitcher to a StatsAPI id (-> recent form) and to the lineup he
     faces (-> opponent K rate for the projection),
  4. score every prop with the same no-edge-no-stake discipline as the main card.

Scoring stays in prop_pipeline (pure, unit-tested); this module is the thin, network-facing
wrapper plus the name<->id resolution that the Odds API alone can't provide. Every fetch
degrades to empty/None so a missing feed yields an honest empty card, never a crash.
"""
from __future__ import annotations

from typing import Any, Callable

from app_core.mlb_pitcher_stats import (
    fetch_pitcher_form,
    fetch_schedule_probables,
    fetch_team_k_rate,
)
from app_core.prop_odds_ingest import fetch_strikeout_props
from app_core.prop_pipeline import PROP_KS_DISPERSION, PROP_MIN_EDGE, evaluate_strikeout_props

MLB_SPORT_KEY = "baseball_mlb"


def _norm_name(s: object) -> str:
    return " ".join(str(s or "").strip().lower().split())


def build_resolvers(
    schedule_rows: list[dict],
    season: int,
    *,
    form_fetch: Callable = fetch_pitcher_form,
    team_k_fetch: Callable = fetch_team_k_rate,
):
    """Build the ``form_lookup`` / ``opp_k_lookup`` callables prop_pipeline expects.

    ``schedule_rows`` is :func:`mlb_pitcher_stats.parse_schedule_probables` output. A propped
    pitcher is matched by normalized name to his StatsAPI id (recent form) and to his
    OPPONENT's team id — the lineup he's striking out, whose K rate drives the projection.
    Per-id results are memoized so a slate of N games costs at most N form + N team-rate
    fetches. ``form_fetch`` / ``team_k_fetch`` are injectable so the resolvers run offline.
    """
    pitcher_id_by_name: dict[str, int] = {}
    opp_team_id_by_pitcher: dict[str, Any] = {}
    for row in schedule_rows:
        for side, other in (("home", "away"), ("away", "home")):
            name = _norm_name(row.get(f"{side}_pitcher"))
            pid = row.get(f"{side}_pitcher_id")
            if not name:
                continue
            if pid is not None:
                pitcher_id_by_name[name] = pid
            opp_team_id_by_pitcher[name] = row.get(f"{other}_team_id")

    form_cache: dict[Any, dict | None] = {}
    team_cache: dict[Any, float | None] = {}

    def form_lookup(pitcher: object) -> dict | None:
        pid = pitcher_id_by_name.get(_norm_name(pitcher))
        if pid is None:
            return None
        if pid not in form_cache:
            form_cache[pid] = form_fetch(pid, season)
        return form_cache[pid]

    def opp_k_lookup(prop_row: dict) -> float | None:
        team_id = opp_team_id_by_pitcher.get(_norm_name(prop_row.get("pitcher")))
        if team_id is None:
            return None
        if team_id not in team_cache:
            team_cache[team_id] = team_k_fetch(team_id, season)
        return team_cache[team_id]

    return form_lookup, opp_k_lookup


def build_strikeout_card(
    odds_client: Any,
    date: str,
    season: int,
    *,
    sport_key: str = MLB_SPORT_KEY,
    dispersion: float = PROP_KS_DISPERSION,
    min_edge: float = PROP_MIN_EDGE,
    list_events: Callable | None = None,
    props_fetch: Callable = fetch_strikeout_props,
    schedule_fetch: Callable = fetch_schedule_probables,
    form_fetch: Callable = fetch_pitcher_form,
    team_k_fetch: Callable = fetch_team_k_rate,
) -> list[dict]:
    """Score every pitcher-strikeout prop on ``date`` into a card (best edge first).

    ``odds_client`` is a :class:`TheOddsAPIClient` (carries BASE_URL / api_key / regions /
    bookmakers). The injection points (``list_events``, ``props_fetch``, ``schedule_fetch``)
    let the whole pipeline run offline against fixtures in tests; in production they default
    to the live Odds API + StatsAPI calls.
    """
    if list_events is None:
        def list_events(client, sk, d):  # noqa: ANN001
            return client.get_odds(sk, date=d) or []

    events = list_events(odds_client, sport_key, date)
    event_ids = [e.get("id") for e in events if isinstance(e, dict) and e.get("id")]

    props: list[dict] = []
    for event_id in event_ids:
        props.extend(props_fetch(odds_client, sport_key, event_id) or [])

    if not props:
        return []

    schedule_rows = schedule_fetch(date) or []
    form_lookup, opp_k_lookup = build_resolvers(
        schedule_rows, season, form_fetch=form_fetch, team_k_fetch=team_k_fetch
    )

    scored = evaluate_strikeout_props(
        props, form_lookup, opp_k_lookup, dispersion=dispersion, min_edge=min_edge
    )
    scored.sort(key=lambda r: (r.get("best_edge") is not None, r.get("best_edge") or 0.0), reverse=True)
    return scored
