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

import pandas as pd

from app_core.batter_prop_pipeline import (
    BATTER_DEFAULT_ENABLED_MARKETS,
    BATTER_PROP_SPECS,
    evaluate_batter_props,
)
from app_core.mlb_batter_stats import fetch_batter_form
from app_core.mlb_pitcher_stats import (
    fetch_pitcher_form,
    fetch_schedule_probables,
    fetch_team_k_rate,
)
from app_core.prop_odds_ingest import fetch_mlb_player_props, fetch_strikeout_props
from app_core.prop_calibration import (
    apply_prop_calibration,
    directional_market_key,
    load_prop_results_log,
    normalize_prop_results,
)
from app_core.prop_pipeline import (
    PITCHER_PROP_SPECS,
    PROP_KS_DISPERSION,
    PROP_MAX_PLAUSIBLE_EDGE,
    PROP_MIN_EDGE,
    PROP_MIN_STARTER_AVG_INNINGS,
    PROP_MIN_STARTER_GAMES,
    PROP_MIN_WIN_PROBABILITY,
    PROP_DEFAULT_ENABLED_MARKETS,
    PROP_MAX_DAYS_SINCE_LAST_START,
    evaluate_strikeout_props,
)

MLB_SPORT_KEY = "baseball_mlb"
MLB_PROP_DEFAULT_ENABLED_MARKETS = PROP_DEFAULT_ENABLED_MARKETS + BATTER_DEFAULT_ENABLED_MARKETS


def stamp_prop_export(
    card,
    pipeline_build: str,
    *,
    export_run_id: str | None = None,
):
    """Return a snapshot-identifiable prop export without mutating the scored card."""
    import pandas as pd

    if card is None:
        return pd.DataFrame(columns=["pipeline_build", "export_run_id"])

    out = card.copy()
    build = str(pipeline_build or "").strip()
    run_id = str(export_run_id or "").strip()
    if not run_id:
        run_id = pd.Timestamp.now(tz="UTC").strftime("%Y%m%dT%H%M%SZ")
    if "pipeline_build" in out.columns:
        out["pipeline_build"] = build
    else:
        out.insert(0, "pipeline_build", build)
    if "export_run_id" in out.columns:
        out["export_run_id"] = run_id
    else:
        out.insert(1, "export_run_id", run_id)
    ordered = ["pipeline_build", "export_run_id"] + [
        column for column in out.columns
        if column not in {"pipeline_build", "export_run_id"}
    ]
    out = out[ordered]
    return out


def _norm_name(s: object) -> str:
    return " ".join(str(s or "").strip().lower().split())


def build_resolvers(
    schedule_rows: list[dict],
    season: int,
    *,
    as_of_date: str | None = None,
    form_fetch: Callable = fetch_pitcher_form,
    team_k_fetch: Callable = fetch_team_k_rate,
):
    """Build the ``form_lookup`` / ``opp_k_lookup`` callables prop_pipeline expects.

    ``schedule_rows`` is :func:`mlb_pitcher_stats.parse_schedule_probables` output. A propped
    pitcher is matched by normalized name to his StatsAPI id (recent form) and to his
    OPPONENT's team id â€” the lineup he's striking out, whose K rate drives the projection.
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
            try:
                import inspect
                accepts_date = "as_of_date" in inspect.signature(form_fetch).parameters
            except (TypeError, ValueError):
                accepts_date = False
            form_cache[pid] = (
                form_fetch(pid, season, as_of_date=as_of_date)
                if accepts_date else form_fetch(pid, season)
            )
        return form_cache[pid]

    def opp_k_lookup(prop_row: dict) -> float | None:
        team_id = opp_team_id_by_pitcher.get(_norm_name(prop_row.get("pitcher")))
        if team_id is None:
            return None
        if team_id not in team_cache:
            team_cache[team_id] = team_k_fetch(team_id, season)
        return team_cache[team_id]

    return form_lookup, opp_k_lookup


def build_batter_form_lookup(
    season: int,
    *,
    as_of_date: str | None = None,
    form_fetch: Callable = fetch_batter_form,
):
    """Memoized batter-name -> form lookup for the supported batter markets."""
    cache: dict[str, dict | None] = {}

    def lookup(player: object) -> dict | None:
        name = _norm_name(player)
        if not name:
            return None
        if name not in cache:
            try:
                import inspect
                accepts_date = "as_of_date" in inspect.signature(form_fetch).parameters
            except (TypeError, ValueError):
                accepts_date = False
            cache[name] = (
                form_fetch(player, season, as_of_date=as_of_date)
                if accepts_date else form_fetch(player, season)
            )
        return cache[name]

    return lookup


def build_strikeout_card(
    odds_client: Any,
    date: str,
    season: int,
    *,
    sport_key: str = MLB_SPORT_KEY,
    dispersion: float = PROP_KS_DISPERSION,
    min_edge: float = PROP_MIN_EDGE,
    min_starter_games: int = PROP_MIN_STARTER_GAMES,
    min_starter_avg_innings: float = PROP_MIN_STARTER_AVG_INNINGS,
    max_plausible_edge: float = PROP_MAX_PLAUSIBLE_EDGE,
    enabled_markets: tuple[str, ...] = MLB_PROP_DEFAULT_ENABLED_MARKETS,
    max_days_since_last_start: int | None = PROP_MAX_DAYS_SINCE_LAST_START,
    diagnostics: dict | None = None,
    event_list_attempts: int = 2,
    list_events: Callable | None = None,
    props_fetch: Callable = fetch_mlb_player_props,
    schedule_fetch: Callable = fetch_schedule_probables,
    form_fetch: Callable = fetch_pitcher_form,
    batter_form_fetch: Callable = fetch_batter_form,
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

    # The main board may already be using cached odds while this separate prop
    # slice makes a fresh request. A single transient timeout used to escape
    # here and mark the entire prop card unavailable. Retry once and degrade to
    # an empty, diagnosed card only when every attempt fails.
    events = []
    event_error = None
    attempts = max(1, int(event_list_attempts))
    for attempt in range(attempts):
        try:
            events = list_events(odds_client, sport_key, date) or []
            event_error = None
            if diagnostics is not None:
                diagnostics["strikeout_prop_event_list_attempts"] = attempt + 1
            # The shared client returns [] for HTTP failures such as 429/5xx,
            # so an empty response also needs the bounded retry path.
            if events or attempt + 1 >= attempts:
                break
        except Exception as exc:  # network clients do not share one exception hierarchy
            event_error = exc
    if event_error is not None:
        if diagnostics is not None:
            diagnostics["strikeout_prop_feed_status"] = "event_list_failed"
            diagnostics["strikeout_prop_feed_error_type"] = type(event_error).__name__
            diagnostics["strikeout_prop_event_list_attempts"] = attempts
        return []

    event_ids = [e.get("id") for e in events if isinstance(e, dict) and e.get("id")]
    if diagnostics is not None:
        diagnostics["strikeout_prop_event_count"] = len(event_ids)
    if not event_ids:
        if diagnostics is not None:
            diagnostics["strikeout_prop_feed_status"] = "no_events"
        return []

    props: list[dict] = []
    prop_fetch_errors = 0
    for event_id in event_ids:
        try:
            props.extend(props_fetch(odds_client, sport_key, event_id) or [])
        except Exception:
            # Keep the remaining games usable when one event endpoint fails.
            prop_fetch_errors += 1

    if not props:
        if diagnostics is not None:
            diagnostics["strikeout_prop_feed_status"] = (
                "prop_fetch_failed" if prop_fetch_errors == len(event_ids) else "no_prop_markets"
            )
            diagnostics["strikeout_prop_event_fetch_errors"] = prop_fetch_errors
        return []
    if diagnostics is not None:
        diagnostics["strikeout_prop_raw_count"] = len(props)
        diagnostics["strikeout_prop_event_fetch_errors"] = prop_fetch_errors

    try:
        schedule_rows = schedule_fetch(date) or []
    except Exception as exc:
        schedule_rows = []
        if diagnostics is not None:
            diagnostics["strikeout_prop_schedule_error_type"] = type(exc).__name__
    form_lookup, opp_k_lookup = build_resolvers(
        schedule_rows, season, as_of_date=date,
        form_fetch=form_fetch, team_k_fetch=team_k_fetch
    )

    pitcher_props = [p for p in props if not str(p.get("market_key", "")).startswith("batter_")]
    batter_props = [p for p in props if str(p.get("market_key", "")).startswith("batter_")]
    scored = evaluate_strikeout_props(
        pitcher_props, form_lookup, opp_k_lookup, dispersion=dispersion, min_edge=min_edge,
        min_starter_games=min_starter_games,
        min_starter_avg_innings=min_starter_avg_innings,
        max_plausible_edge=max_plausible_edge,
        enabled_markets=enabled_markets,
        max_days_since_last_start=max_days_since_last_start,
    )
    batter_lookup = build_batter_form_lookup(
        season, as_of_date=date, form_fetch=batter_form_fetch
    )
    scored.extend(evaluate_batter_props(
        batter_props,
        batter_lookup,
        min_edge=min_edge,
        enabled_markets=enabled_markets,
    ))
    if diagnostics is not None:
        diagnostics["strikeout_prop_feed_status"] = "ready"
        diagnostics["strikeout_prop_scored_count"] = len(scored)
    scored.sort(key=lambda r: (r.get("best_edge") is not None, r.get("best_edge") or 0.0), reverse=True)
    return scored


def _decimal_odds(american: float) -> float:
    a = float(american)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def _resolve_prop_results_history(uploaded_results):
    """Use an explicit non-empty ledger, otherwise restore the repo baseline."""
    if isinstance(uploaded_results, pd.DataFrame):
        return (
            uploaded_results
            if not uploaded_results.empty
            else load_prop_results_log()
        )
    return load_prop_results_log(uploaded=uploaded_results)


def build_prop_card(
    odds_client: Any,
    date: str,
    season: int,
    bankroll: float,
    *,
    kelly_per_pick_pct: float,
    kelly_total_pct: float,
    kelly_fraction: float = 0.25,
    min_edge: float = PROP_MIN_EDGE,
    min_win_probability: float = PROP_MIN_WIN_PROBABILITY,
    prop_results_log=None,
    **card_kwargs: Any,
):
    """Production strikeout-prop card: the ACTIONABLE props with conservative stakes.

    Scores the slate (:func:`build_strikeout_card`), keeps only the recommended over/under
    picks (the +EV / min-edge gate already lives in scoring), and sizes each at fractional
    Kelly capped at ``kelly_per_pick_pct`` of bankroll, with the slate total capped at
    ``kelly_total_pct``. Sizing is deliberately small: the prop model is uncalibrated, so the
    caps bound the downside until a graded record exists. Returns a DataFrame (empty when the
    feed is missing or nothing clears the edge bar) so the caller can display + export it
    without touching the main best-picks card. ``card_kwargs`` forward the injectable feeds to
    :func:`build_strikeout_card` for offline testing.

    Win-probability-first (owner preference, 3 Jul): picks must be genuine favorites on
    the model's own number (``p_side >= min_win_probability``) and the card is ordered by
    win probability, not edge â€” near-coin-flip plus-money price plays no longer make the
    card even when their EV is the highest on the slate. The +EV/min-edge gate remains as
    the eligibility floor so a likely winner at a losing price is still never staked.
    """
    scored = build_strikeout_card(
        odds_client, date, season, min_edge=min_edge, **card_kwargs
    )
    rows = []
    for r in scored:
        side = r.get("best_side")
        if r.get("recommendation") not in ("over", "under") or side not in ("over", "under"):
            continue
        odds = r.get("over_odds") if side == "over" else r.get("under_odds")
        p_over = r.get("model_p_over")
        if odds is None or p_over is None:
            continue
        p_side = float(p_over) if side == "over" else 1.0 - float(p_over)
        if p_side < float(min_win_probability):
            continue
        market_key = str(r.get("market_key") or "pitcher_strikeouts")
        spec = BATTER_PROP_SPECS.get(market_key) or PITCHER_PROP_SPECS.get(
            market_key, PITCHER_PROP_SPECS["pitcher_strikeouts"]
        )
        label = spec["label"]
        player = r.get("player") or r.get("batter") or r.get("pitcher")
        participant_type = r.get("participant_type") or (
            "batter" if market_key.startswith("batter_") else "pitcher"
        )
        dec = _decimal_odds(odds)
        b = dec - 1.0
        kelly_frac = ((p_side * dec) - 1.0) / b if b > 0 else 0.0
        stake_pct = min(max(kelly_frac, 0.0) * kelly_fraction, kelly_per_pick_pct)
        rows.append({
            "league": "MLB",
            "game_date": date,
            "player": player,
            "participant_type": participant_type,
            "pitcher": player if participant_type == "pitcher" else None,
            "batter": player if participant_type == "batter" else None,
            "matchup": f"{r.get('away_team', '')} @ {r.get('home_team', '')}".strip(" @"),
            "market_type": f"{market_key}_{side}",
            "best_pick": f"{player} {side.capitalize()} {r.get('line')} {label}",
            "line": r.get("line"),
            "expected_stat": r.get("expected_stat"),
            "expected_count": r.get("expected_count"),
            "expected_ks": r.get("expected_ks") if market_key == "pitcher_strikeouts" else None,
            "expected_walks": r.get("expected_walks") if market_key == "pitcher_walks" else None,
            "expected_outs": r.get("expected_outs") if market_key == "pitcher_outs" else None,
            "expected_hits": r.get("expected_hits") if market_key == "batter_hits" else None,
            "expected_total_bases": r.get("expected_total_bases") if market_key == "batter_total_bases" else None,
            "WinProbability": round(p_side, 4),
            "expected_value": r.get("best_ev"),
            "edge": r.get("best_edge"),
            "odds_american": odds,
            "book": r.get("book"),
            "Pick_Status": "Actionable",
            "_stake_pct": stake_pct,
        })

    card = pd.DataFrame(rows)
    if card.empty:
        return card

    uploaded_results = prop_results_log
    if uploaded_results is None:
        try:
            import streamlit as st

            active_results = st.session_state.get("active_prop_results_log")
            generated_results = st.session_state.get("generated_prop_results_log")
            uploaded_results = (
                active_results
                if isinstance(active_results, pd.DataFrame) and not active_results.empty
                else generated_results
                if isinstance(generated_results, pd.DataFrame) and not generated_results.empty
                else st.session_state.get("prop_results_log")
            )
        except (ImportError, RuntimeError, AttributeError):
            uploaded_results = None
    results_history = _resolve_prop_results_history(uploaded_results)
    card = apply_prop_calibration(card, results_history, as_of_date=date)

    # Re-size from the conservative probability. Raw-model Kelly is intentionally
    # discarded so an overconfident projection cannot survive downstream caps.
    def _calibrated_stake_pct(row) -> float:
        try:
            probability = float(row["WinProbability"])
            decimal = _decimal_odds(float(row["odds_american"]))
            profit_multiple = decimal - 1.0
            full_kelly = (
                ((probability * decimal) - 1.0) / profit_multiple
                if profit_multiple > 0 else 0.0
            )
            return min(
                max(full_kelly, 0.0) * float(kelly_fraction),
                float(kelly_per_pick_pct),
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            return 0.0

    card["_stake_pct"] = card.apply(_calibrated_stake_pct, axis=1)
    bankroll = float(bankroll or 0.0)
    card["Kelly_Bet_Size"] = (card["_stake_pct"] * bankroll).round(2)
    total = float(card["Kelly_Bet_Size"].sum())
    cap_total = kelly_total_pct * bankroll
    if total > cap_total and total > 0:
        card["Kelly_Bet_Size"] = (card["Kelly_Bet_Size"] * (cap_total / total)).round(2)
    card = card.drop(columns=["_stake_pct"])
    # Per-market probation must use the same pre-slate cutoff as calibration.
    # Passing the full cumulative ledger here would let future results prove an
    # older market during replay/backtesting.
    clean_history = normalize_prop_results(results_history, as_of_date=date)
    try:
        card = apply_market_probation(
            card, market_records_from_log(clean_history, detailed=True)
        )
    except Exception:
        pass  # probation is protective, never card-breaking
    diagnostics = card_kwargs.get("diagnostics")
    if diagnostics is not None:
        diagnostics["prop_calibration_graded_count"] = int(len(clean_history))
        diagnostics["prop_calibration_status"] = (
            "ready"
            if card["CalibrationSource"].isin(["directional", "pooled"]).any()
            else "needs_graded_results"
        )
    card = apply_batter_probation_exposure_cap(card, bankroll)
    card = apply_probation_portfolio_guard(card, bankroll)
    card = apply_prop_participant_guard(card)
    card = apply_production_prop_gate(card)
    card = apply_controlled_prop_rollout(
        card,
        results_history,
        as_of_date=date,
    )
    card = apply_prop_game_guard(card)
    card = apply_prop_slate_guard(card)
    card = apply_novig_minimum_selection(
        card, bankroll, total_cap_pct=kelly_total_pct
    )
    card = apply_extended_prop_stake_cap(card)
    card = apply_probation_share_guard(card)
    card = apply_prop_stake_status(card)
    card = apply_prop_precision_shortlist(card)
    card = card.sort_values(
        ["Prop_Precision_Shortlist", "Prop_Precision_Rank", "WinProbability", "edge"],
        ascending=[False, True, False, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return card


# â”€â”€ Market probation (6 Jul): stake follows each market's GRADED record â”€â”€
# The outs market went 2-5 (28.6%) in its first week â€” the books price manager
# leashes better than a recent-workload average. Rather than benching (which
# would stop grading and freeze the record forever), an underwater market's
# picks stay on the card at a flat probation stake: the model keeps taking its
# swings and the ledger keeps filling, but the bleed is bounded until the
# market re-proves itself. Strikeouts (24-12) are untouched.
PROBATION_MIN_GRADED = 20     # require a credible sample before full staking
PROBATION_MIN_RATE = 0.55     # must clear typical vig, not merely break even
PROBATION_STAKE = 1.0         # flat $ per probation pick
PROVEN_PROP_MARKETS = frozenset()
BATTER_PROBATION_TOTAL_PCT = 0.0075  # max 0.75% bankroll across all unproven batter picks
PROBATION_PORTFOLIO_TOTAL_PCT = 0.008  # max eight $1 research tickets per $1,000
PROBATION_MAX_PER_MARKET = 2
PROBATION_MAX_PER_GAME = 1
PROBATION_MAX_FUNDED_TICKETS = 2
PROP_PRECISION_SHORTLIST_MAX = 8
PROBATION_MAX_PORTFOLIO_SHARE = 0.25
PROP_MAX_FUNDED_PER_PLAYER = 1
PROBATION_MIN_WIN_PROBABILITY = 0.62
PROBATION_MIN_EXPECTED_VALUE = 0.05
PROBATION_MIN_EDGE = 0.05
NOVIG_MINIMUM_BET = 1.0
PROP_PRODUCTION_MIN_EXPECTED_VALUE = 0.03
PROP_PRODUCTION_MIN_WIN_PROBABILITY = 0.62
PROP_EXTENDED_MIN_WIN_PROBABILITY = 0.60
PROP_PRODUCTION_MIN_PROJECTION_CUSHION = 0.50
PROP_PRODUCTION_MIN_AMERICAN_ODDS = -150
PROP_MAX_FUNDED_PER_GAME = 2
PROP_MAX_FUNDED_PER_SLATE = 5
PROP_EXTENDED_FLAT_STAKE = 1.0
PROP_PRODUCTION_BLOCKED_MARKETS = frozenset(
    {"batter_total_bases_over", "batter_total_bases_under"}
)
PROP_REQUIRED_IDENTITY_FIELDS = ("player", "matchup", "market_type", "best_pick")
PROP_CONTROLLED_PILOT_MARKETS = frozenset(
    {"batter_hits_over", "batter_hits_under"}
)
PROP_CONTROLLED_PILOT_START_DATE = "2026-08-02"
PROP_CONTROLLED_PILOT_MIN_POST_FIX_GRADED = 50
PROP_CONTROLLED_PILOT_MIN_RATE = 0.55
PROP_CONTROLLED_PILOT_STAKE = 1.0


def _market_of_pick(text: object, market_type: object = None) -> str:
    # Padded tokens only: pitcher NAMES can contain stat words ("Walker
    # Buehler Over 3.5 Ks" must not classify as a walks pick â€” 6 Jul).
    mt = str(market_type or "").lower()
    base = mt.removesuffix("_over").removesuffix("_under")
    if base in {*PITCHER_PROP_SPECS, *BATTER_PROP_SPECS}:
        return base
    t = f" {str(text or '').lower()} "
    if " tb " in t or " total bases " in t:
        return "batter_total_bases"
    if " hits " in t:
        return "batter_hits"
    if " outs " in t:
        return "pitcher_outs"
    if " bbs " in t:
        return "pitcher_walks"
    return "pitcher_strikeouts"


def market_records_from_log(log_df, *, detailed: bool = False) -> dict:
    """Directional and pooled market records from settled prop results.

    The default ``(wins, losses)`` shape remains compatible with older callers.
    ``detailed=True`` also includes flat-stake ROI so full staking requires a
    market to beat its actual prices, not merely an arbitrary hit-rate target.
    """
    import pandas as pd

    if log_df is None or len(log_df) == 0:
        return {}
    clean = log_df.copy()
    result_col = next(
        (column for column in ("result", "Outcome", "outcome") if column in clean.columns),
        None,
    )
    if result_col is None:
        return {}
    clean["result"] = clean[result_col].astype(str).str.upper().str.strip()
    clean = clean[clean["result"].isin(["WIN", "LOSS"])].copy()
    if clean.empty:
        return {}
    market_type = clean.get("market_type", pd.Series("", index=clean.index))
    pick = clean.get("pick", clean.get("best_pick", pd.Series("", index=clean.index)))
    clean["directional_market"] = [
        directional_market_key(market, selection)
        for market, selection in zip(market_type, pick)
    ]
    clean["outcome"] = clean["result"].eq("WIN").astype(int)
    clean["odds_american"] = pd.to_numeric(
        clean.get("odds_american", pd.Series(float("nan"), index=clean.index)),
        errors="coerce",
    )
    accum: dict[str, dict[str, float]] = {}
    for _, row in clean.iterrows():
        directional = str(row["directional_market"])
        family = directional.removesuffix("_over").removesuffix("_under")
        for key in {directional, family}:
            rec = accum.setdefault(
                key, {"wins": 0, "losses": 0, "profit": 0.0, "priced": 0}
            )
            won = int(row["outcome"]) == 1
            rec["wins" if won else "losses"] += 1
            try:
                decimal = _decimal_odds(float(row["odds_american"]))
                rec["profit"] += decimal - 1.0 if won else -1.0
                rec["priced"] += 1
            except (TypeError, ValueError, ZeroDivisionError):
                pass
    if not detailed:
        return {
            key: (int(rec["wins"]), int(rec["losses"]))
            for key, rec in accum.items()
        }
    return {
        key: {
            "wins": int(rec["wins"]),
            "losses": int(rec["losses"]),
            "n": int(rec["wins"] + rec["losses"]),
            "roi": (
                float(rec["profit"] / rec["priced"])
                if rec["priced"] else None
            ),
        }
        for key, rec in accum.items()
    }


def apply_market_probation(card, records: dict,
                           min_graded: int = PROBATION_MIN_GRADED,
                           min_rate: float = PROBATION_MIN_RATE,
                           probation_stake: float = PROBATION_STAKE):
    """Cap stakes for markets whose graded record is underwater. Pure."""
    if card is None or card.empty:
        return card
    out = card.copy()
    probation_markets = set()
    mk_col = out.apply(
        lambda row: directional_market_key(
            row.get("market_type"), row.get("best_pick")
        ),
        axis=1,
    )
    card_markets = set(mk_col.dropna().astype(str))
    for mk in card_markets:
        if mk in PROVEN_PROP_MARKETS:
            continue
        family = mk.removesuffix("_over").removesuffix("_under")
        record = records.get(mk)
        if record is None:
            family_record = records.get(family, (0, 0))
            # Detailed records came from the current ledger parser and therefore
            # preserve direction. Do not let a pooled family record prove an
            # ungraded side. Tuple-only inputs are legacy callers and retain the
            # historic family fallback.
            record = (0, 0) if isinstance(family_record, dict) else family_record
        if isinstance(record, dict):
            w = int(record.get("wins", 0) or 0)
            l = int(record.get("losses", 0) or 0)
            roi = record.get("roi")
        else:
            w, l = record
            roi = None
        n = w + l
        if (
            n < min_graded
            or (w / n) < min_rate
            or (roi is not None and float(roi) <= 0.0)
        ):
            probation_markets.add(mk)
    if not probation_markets:
        out["Market_Probation"] = False
        return out
    on_probation = mk_col.isin(probation_markets)
    out["Market_Probation"] = on_probation
    out.loc[on_probation, "Kelly_Bet_Size"] = out.loc[on_probation, "Kelly_Bet_Size"].clip(upper=float(probation_stake))
    return out


def apply_batter_probation_exposure_cap(
    card,
    bankroll: float,
    cap_pct: float = BATTER_PROBATION_TOTAL_PCT,
    minimum_bet: float = NOVIG_MINIMUM_BET,
):
    """Select executable $1 batter tickets within the aggregate probation cap."""
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    market_type = out.get("market_type", pd.Series("", index=out.index)).astype(str)
    participant = out.get("participant_type", pd.Series("", index=out.index)).astype(str)
    probation = out.get("Market_Probation", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    batter_probation = probation & (
        participant.eq("batter") | market_type.str.startswith("batter_")
    )
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    exposure_before = float(stakes[batter_probation].sum())
    exposure_cap = max(0.0, float(bankroll or 0.0) * float(cap_pct))
    minimum_bet = max(0.0, float(minimum_bet))
    candidates = out.index[batter_probation & stakes.gt(0)]
    max_tickets = int(exposure_cap // minimum_bet) if minimum_bet > 0 else 0
    probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    ranked = pd.DataFrame({
        "probability": probability.loc[candidates],
        "edge": edge.loc[candidates],
        "original_stake": stakes.loc[candidates],
    }).sort_values(
        ["probability", "edge", "original_stake"],
        ascending=[False, False, False],
    )
    selected = ranked.head(max_tickets).index
    out.loc[batter_probation, "Kelly_Bet_Size"] = 0.0
    if len(selected):
        out.loc[selected, "Kelly_Bet_Size"] = minimum_bet
    exposure_after = float(pd.to_numeric(
        out.loc[batter_probation, "Kelly_Bet_Size"], errors="coerce"
    ).fillna(0.0).sum())
    cap_applied = bool(
        (pd.to_numeric(out.loc[batter_probation, "Kelly_Bet_Size"], errors="coerce")
         .fillna(0.0) - stakes[batter_probation]).abs().gt(1e-9).any()
    )
    out["batter_probation_cap_applied"] = cap_applied
    out["batter_probation_exposure_cap"] = round(exposure_cap, 2)
    out["batter_probation_exposure_before"] = round(exposure_before, 2)
    out["batter_probation_exposure_after"] = round(exposure_after, 2)
    out["batter_probation_minimum_bet"] = round(minimum_bet, 2)
    out["batter_probation_selected_count"] = int(len(selected))
    return out


def apply_probation_portfolio_guard(
    card,
    bankroll: float,
    cap_pct: float = PROBATION_PORTFOLIO_TOTAL_PCT,
    max_per_market: int = PROBATION_MAX_PER_MARKET,
    max_per_game: int = PROBATION_MAX_PER_GAME,
    max_funded_tickets: int = PROBATION_MAX_FUNDED_TICKETS,
    min_probability: float = PROBATION_MIN_WIN_PROBABILITY,
    min_ev: float = PROBATION_MIN_EXPECTED_VALUE,
    min_edge: float = PROBATION_MIN_EDGE,
    minimum_bet: float = NOVIG_MINIMUM_BET,
):
    """Concentrate unproven props into a small, diversified research portfolio."""
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    probation = out.get(
        "Market_Probation", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    ev = pd.to_numeric(
        out.get("expected_value", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    market = out.get("market_type", pd.Series("", index=out.index)).astype(str)
    market_family = market.str.replace(r"_(over|under)$", "", regex=True)
    matchup = out.get("matchup", pd.Series("", index=out.index)).astype(str).str.lower().str.strip()
    eligible = (
        probation & stakes.gt(0)
        & probability.ge(float(min_probability))
        & ev.ge(float(min_ev))
        & edge.ge(float(min_edge))
    )
    ranked = pd.DataFrame({
        "probability": probability[eligible],
        "ev": ev[eligible],
        "edge": edge[eligible],
        "market_family": market_family[eligible],
        "matchup": matchup[eligible],
    }).sort_values(["probability", "ev", "edge"], ascending=False)

    cap = max(0.0, float(bankroll or 0.0) * float(cap_pct))
    cap_tickets = int(cap // float(minimum_bet)) if minimum_bet > 0 else 0
    max_tickets = min(cap_tickets, max(0, int(max_funded_tickets)))
    selected = []
    market_counts: dict[str, int] = {}
    game_counts: dict[str, int] = {}
    for index, row in ranked.iterrows():
        if len(selected) >= max_tickets:
            break
        family, game = str(row["market_family"]), str(row["matchup"])
        if market_counts.get(family, 0) >= int(max_per_market):
            continue
        if game and game_counts.get(game, 0) >= int(max_per_game):
            continue
        selected.append(index)
        market_counts[family] = market_counts.get(family, 0) + 1
        if game:
            game_counts[game] = game_counts.get(game, 0) + 1
    exposure_before = float(stakes[probation].sum())
    out.loc[probation, "Kelly_Bet_Size"] = 0.0
    if selected:
        out.loc[selected, "Kelly_Bet_Size"] = float(minimum_bet)
    out["probation_portfolio_cap"] = round(cap, 2)
    out["probation_portfolio_exposure_before"] = round(exposure_before, 2)
    out["probation_portfolio_exposure_after"] = round(float(len(selected)) * float(minimum_bet), 2)
    out["probation_portfolio_selected_count"] = int(len(selected))
    out["probation_portfolio_guard_applied"] = bool(probation.any())
    return out


def apply_prop_participant_guard(
    card,
    max_funded_per_player: int = PROP_MAX_FUNDED_PER_PLAYER,
):
    """Keep at most one funded prop for each player on a slate.

    Multiple markets on the same participant are not independent bankroll
    decisions. Rank the player's qualified rows by proven-market status, win
    probability, EV, edge, and original stake; retain only the strongest row.
    """
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    candidates = out.index[stakes.gt(0)]
    if len(candidates) == 0:
        return out
    player = out.get("player", pd.Series("", index=out.index)).fillna("").astype(str)
    batter = out.get("batter", pd.Series("", index=out.index)).fillna("").astype(str)
    pitcher = out.get("pitcher", pd.Series("", index=out.index)).fillna("").astype(str)
    participant = player.where(player.str.strip().ne(""), batter)
    participant = participant.where(participant.str.strip().ne(""), pitcher)
    participant = participant.str.lower().str.split().str.join(" ")
    probation = out.get(
        "Market_Probation", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    ev = pd.to_numeric(
        out.get("expected_value", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    ranked = pd.DataFrame({
        "participant": participant.loc[candidates],
        "proven": (~probation.loc[candidates]).astype(int),
        "probability": probability.loc[candidates],
        "ev": ev.loc[candidates],
        "edge": edge.loc[candidates],
        "stake": stakes.loc[candidates],
    }).sort_values(
        ["proven", "probability", "ev", "edge", "stake"],
        ascending=[False, False, False, False, False],
    )

    selected = []
    counts: dict[str, int] = {}
    limit = max(0, int(max_funded_per_player))
    for index, row in ranked.iterrows():
        key = str(row["participant"]).strip() or f"__row_{index}"
        if counts.get(key, 0) >= limit:
            continue
        selected.append(index)
        counts[key] = counts.get(key, 0) + 1
    out.loc[candidates, "Kelly_Bet_Size"] = 0.0
    if selected:
        out.loc[selected, "Kelly_Bet_Size"] = stakes.loc[selected]
    out["participant_exposure_guard_applied"] = len(selected) < len(candidates)
    out["participant_funded_before"] = int(len(candidates))
    out["participant_funded_after"] = int(len(selected))
    out["participant_max_funded"] = limit
    return out


def apply_production_prop_gate(
    card,
    min_expected_value: float = PROP_PRODUCTION_MIN_EXPECTED_VALUE,
    min_win_probability: float = PROP_PRODUCTION_MIN_WIN_PROBABILITY,
    extended_min_win_probability: float = PROP_EXTENDED_MIN_WIN_PROBABILITY,
    min_projection_cushion: float = PROP_PRODUCTION_MIN_PROJECTION_CUSHION,
    min_american_odds: int = PROP_PRODUCTION_MIN_AMERICAN_ODDS,
    blocked_markets: frozenset[str] = PROP_PRODUCTION_BLOCKED_MARKETS,
):
    """Fund only validated, proven props with a meaningful projected edge.

    Model-qualified rows remain on the research card, but a sportsbook minimum
    may only round a ticket that survives this gate.  Probation markets and the
    empirically weak batter-total-bases family therefore cannot be promoted
    into production or parlays by the $1 minimum rule. Production
    tickets must also clear a probability floor, beat the posted line by a
    meaningful model-projection cushion, and avoid prices shorter than -150.
    """
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    index = out.index

    identity_valid = pd.Series(True, index=index)
    for field in PROP_REQUIRED_IDENTITY_FIELDS:
        values = out.get(field, pd.Series("", index=index)).fillna("").astype(str).str.strip()
        identity_valid &= values.ne("") & ~values.str.lower().isin({"nan", "none"})
    identity_valid &= pd.to_numeric(
        out.get("line", pd.Series(float("nan"), index=index)), errors="coerce"
    ).notna()
    odds = pd.to_numeric(
        out.get("odds_american", pd.Series(float("nan"), index=index)), errors="coerce"
    )
    identity_valid &= odds.notna() & odds.ne(0)
    price_allowed = odds.ge(float(min_american_odds))

    market = out.get("market_type", pd.Series("", index=index)).fillna("").astype(str)
    market_allowed = ~market.isin(set(blocked_markets))
    probation = out.get(
        "Market_Probation", pd.Series(False, index=index)
    ).fillna(False).astype(bool)
    expected_value = pd.to_numeric(
        out.get("expected_value", pd.Series(float("nan"), index=index)), errors="coerce"
    )
    win_probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(float("nan"), index=index)), errors="coerce"
    )
    line = pd.to_numeric(
        out.get("line", pd.Series(float("nan"), index=index)), errors="coerce"
    )
    expected_count = pd.to_numeric(
        out.get("expected_count", pd.Series(float("nan"), index=index)), errors="coerce"
    )
    is_under = market.str.lower().str.endswith("_under")
    is_over = market.str.lower().str.endswith("_over")
    projection_cushion = pd.Series(float("nan"), index=index, dtype="float64")
    projection_cushion.loc[is_under] = (
        line.loc[is_under] - expected_count.loc[is_under]
    )
    projection_cushion.loc[is_over] = (
        expected_count.loc[is_over] - line.loc[is_over]
    )
    qualified = out.get(
        "Pick_Status", pd.Series("Actionable", index=index)
    ).astype(str).str.strip().eq("Actionable")
    if "CalibrationSource" in out.columns:
        calibration_source = out["CalibrationSource"].fillna("").astype(str)
        calibration_sample = pd.to_numeric(
            out.get("CalibrationSampleSize", pd.Series(0, index=index)),
            errors="coerce",
        ).fillna(0)
        calibration_ready = (
            calibration_source.eq("directional") & calibration_sample.ge(20)
        )
    else:
        # Keep the pure gate backward-compatible for callers that supply an
        # already-calibrated card without the new audit columns.
        calibration_source = pd.Series("external", index=index)
        calibration_sample = pd.Series(float("nan"), index=index)
        calibration_ready = pd.Series(True, index=index)
    common_eligible = (
        qualified & identity_valid & market_allowed & price_allowed
        & ~probation & calibration_ready
        & projection_cushion.ge(float(min_projection_cushion))
        & expected_value.ge(float(min_expected_value))
    )
    core_eligible = common_eligible & win_probability.ge(float(min_win_probability))
    extended_eligible = (
        common_eligible
        & win_probability.ge(float(extended_min_win_probability))
        & ~core_eligible
    )
    production_eligible = core_eligible | extended_eligible

    reason = pd.Series("Research only", index=index, dtype="object")
    reason.loc[core_eligible] = "Core production qualified"
    reason.loc[extended_eligible] = "Extended production qualified"
    reason.loc[~identity_valid] = "Rejected: missing or invalid prop identity, line, or odds"
    reason.loc[identity_valid & ~market_allowed] = (
        "Research only: batter total bases are production-disabled pending recalibration"
    )
    reason.loc[identity_valid & market_allowed & probation] = (
        "Research only: market is still on probation"
    )
    reason.loc[
        identity_valid & market_allowed & ~probation
        & ~win_probability.ge(float(extended_min_win_probability))
    ] = (
        "Research only: win probability is below "
        f"{float(extended_min_win_probability):.0%}"
    )
    reason.loc[
        identity_valid & market_allowed & ~probation
        & win_probability.ge(float(extended_min_win_probability))
        & ~projection_cushion.ge(float(min_projection_cushion))
    ] = (
        "Research only: model projection advantage is below "
        f"{float(min_projection_cushion):.2f}"
    )
    reason.loc[
        identity_valid & market_allowed & ~probation
        & win_probability.ge(float(extended_min_win_probability))
        & projection_cushion.ge(float(min_projection_cushion))
        & ~expected_value.ge(float(min_expected_value))
    ] = f"Research only: expected value is below {float(min_expected_value):.1%}"
    reason.loc[
        identity_valid & market_allowed & ~probation
        & win_probability.ge(float(extended_min_win_probability))
        & projection_cushion.ge(float(min_projection_cushion))
        & expected_value.ge(float(min_expected_value))
        & ~price_allowed
    ] = (
        "Research only: sportsbook price is shorter than "
        f"{int(min_american_odds):+d}"
    )
    reason.loc[identity_valid & market_allowed & ~calibration_ready] = (
        "Research only: directional prop calibration needs at least 20 graded results"
    )

    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    out["production_identity_valid"] = identity_valid
    out["production_market_allowed"] = market_allowed
    out["production_price_allowed"] = price_allowed
    out["production_min_american_odds"] = int(min_american_odds)
    out["production_calibration_ready"] = calibration_ready
    out["production_calibration_source"] = calibration_source
    out["production_calibration_sample"] = calibration_sample
    out["production_min_expected_value"] = float(min_expected_value)
    out["production_min_win_probability"] = float(min_win_probability)
    out["extended_min_win_probability"] = float(extended_min_win_probability)
    out["production_projection_cushion"] = projection_cushion
    out["production_min_projection_cushion"] = float(min_projection_cushion)
    out["Prop_Tier"] = "Research"
    out.loc[core_eligible, "Prop_Tier"] = "Core"
    out.loc[extended_eligible, "Prop_Tier"] = "Extended"
    out["production_eligible"] = production_eligible
    out["production_gate_reason"] = reason
    out.loc[~production_eligible, "Kelly_Bet_Size"] = 0.0
    out.loc[~identity_valid, "Pick_Status"] = "Rejected"
    if "Status_Reason" not in out.columns:
        out["Status_Reason"] = ""
    out.loc[~production_eligible, "Status_Reason"] = reason.loc[~production_eligible]
    return out


def apply_controlled_prop_rollout(
    card,
    results_history=None,
    *,
    as_of_date: str | None = None,
    pilot_markets: frozenset[str] = PROP_CONTROLLED_PILOT_MARKETS,
    pilot_start_date: str = PROP_CONTROLLED_PILOT_START_DATE,
    min_post_fix_graded: int = PROP_CONTROLLED_PILOT_MIN_POST_FIX_GRADED,
    min_rate: float = PROP_CONTROLLED_PILOT_MIN_RATE,
    pilot_stake: float = PROP_CONTROLLED_PILOT_STAKE,
):
    """Restrict live props to a small, evidence-gated batter-hit pilot.

    The ordinary production gate still validates probability, EV, price,
    projection cushion, identity, calibration, and market probation. This
    rollout layer then keeps every non-hit family research-only and caps fully
    approved batter-hit tickets at the sportsbook minimum. A hit direction may
    return to normal sizing only after it records a fresh post-rollout sample
    with at least ``min_post_fix_graded`` settled picks, a 55% hit rate, and
    positive flat-risk ROI at the exported prices.

    Results on the active slate are excluded through ``as_of_date`` so same-day
    outcomes can never promote or resize a pregame ticket.
    """
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    index = out.index
    market = out.get(
        "market_type", pd.Series("", index=index)
    ).fillna("").astype(str).str.strip().str.lower()
    pilot_markets = frozenset(str(value).strip().lower() for value in pilot_markets)
    allowed = market.isin(pilot_markets)
    eligible = out.get(
        "production_eligible", pd.Series(False, index=index)
    ).fillna(False).astype(bool)
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)

    # Build a leak-safe ledger containing only results generated after this
    # rollout began. Missing or unparseable dates do not count toward graduation.
    records: dict[str, dict[str, float]] = {}
    try:
        clean = normalize_prop_results(results_history, as_of_date=as_of_date)
        if not clean.empty and "result_date" in clean.columns:
            start = pd.to_datetime(pilot_start_date, errors="coerce", utc=True)
            if pd.notna(start):
                clean = clean[
                    clean["result_date"].notna()
                    & clean["result_date"].ge(start)
                ].copy()
            else:
                clean = clean.iloc[0:0].copy()
        else:
            clean = clean.iloc[0:0].copy()
        if not clean.empty:
            records = market_records_from_log(clean, detailed=True)
    except (KeyError, TypeError, ValueError):
        records = {}

    out["controlled_pilot_market_allowed"] = allowed
    out["controlled_pilot_start_date"] = str(pilot_start_date)
    out["controlled_pilot_min_post_fix_graded"] = max(
        0, int(min_post_fix_graded)
    )
    out["controlled_pilot_min_rate"] = float(min_rate)
    out["controlled_pilot_stake"] = round(max(0.0, float(pilot_stake)), 2)
    out["controlled_pilot_post_fix_graded"] = 0
    out["controlled_pilot_post_fix_hit_rate"] = float("nan")
    out["controlled_pilot_post_fix_roi"] = float("nan")
    out["controlled_pilot_graduated"] = False
    out["controlled_pilot_mode"] = False
    out["controlled_pilot_stake_cap_applied"] = False

    minimum = max(0, int(min_post_fix_graded))
    required_rate = float(min_rate)
    graduated = pd.Series(False, index=index, dtype=bool)
    for key in pilot_markets:
        record = records.get(key, {})
        wins = int(record.get("wins", 0) or 0)
        losses = int(record.get("losses", 0) or 0)
        graded = wins + losses
        rate = (wins / graded) if graded else None
        roi = record.get("roi")
        market_rows = market.eq(key)
        out.loc[market_rows, "controlled_pilot_post_fix_graded"] = graded
        if rate is not None:
            out.loc[market_rows, "controlled_pilot_post_fix_hit_rate"] = float(rate)
        if roi is not None:
            out.loc[market_rows, "controlled_pilot_post_fix_roi"] = float(roi)
        direction_graduated = bool(
            graded >= minimum
            and rate is not None
            and rate >= required_rate
            and roi is not None
            and float(roi) > 0.0
        )
        graduated.loc[market_rows] = direction_graduated

    out["controlled_pilot_graduated"] = allowed & graduated
    out["controlled_pilot_mode"] = allowed & ~graduated

    # Even a fully production-qualified pitcher or other prop remains visible
    # for grading but cannot receive a stake during the controlled rollout.
    rejected = out.get(
        "Pick_Status", pd.Series("", index=index)
    ).fillna("").astype(str).str.strip().eq("Rejected")
    calibration_ready = out.get(
        "production_calibration_ready", eligible
    )
    calibration_ready = pd.Series(calibration_ready, index=index).fillna(False).astype(bool)
    outside_scope = ~allowed & ~rejected & calibration_ready
    if outside_scope.any():
        reason = (
            "Research only: controlled rollout currently funds batter-hit "
            "props only"
        )
        out.loc[outside_scope, "production_eligible"] = False
        out.loc[outside_scope, "Kelly_Bet_Size"] = 0.0
        out.loc[outside_scope, "Prop_Tier"] = "Research"
        out.loc[outside_scope, "production_gate_reason"] = reason
        if "Status_Reason" not in out.columns:
            out["Status_Reason"] = ""
        out.loc[outside_scope, "Status_Reason"] = reason

    pilot_eligible = eligible & allowed & ~graduated
    cap = round(max(0.0, float(pilot_stake)), 2)
    cap_applied = pilot_eligible & stakes.gt(cap)
    out.loc[pilot_eligible, "Kelly_Bet_Size"] = stakes.loc[
        pilot_eligible
    ].clip(upper=cap)
    out.loc[cap_applied, "controlled_pilot_stake_cap_applied"] = True
    if pilot_eligible.any():
        out.loc[pilot_eligible, "production_gate_reason"] = (
            "Controlled batter-hit pilot qualified; stake capped at "
            f"${cap:.2f} until {minimum} post-fix graded results have at "
            f"least {required_rate:.0%} accuracy and positive ROI"
        )
    return out


def apply_prop_game_guard(
    card,
    max_funded_per_game: int = PROP_MAX_FUNDED_PER_GAME,
):
    """Keep only the strongest production prop from each matchup.

    Props within one game share lineup, weather, bullpen, and game-state risk.
    Rows displaced by this guard stay visible on the research card and cannot
    be promoted again by NoVig's sportsbook-minimum rule.
    """
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    index = out.index
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    eligible = out.get(
        "production_eligible", pd.Series(False, index=index)
    ).fillna(False).astype(bool)
    candidates = out.index[eligible & stakes.gt(0)]
    limit = max(0, int(max_funded_per_game))

    out["game_exposure_guard_applied"] = False
    out["game_funded_before"] = int(len(candidates))
    out["game_funded_after"] = int(len(candidates))
    out["game_max_funded"] = limit
    if len(candidates) == 0:
        return out

    matchup = out.get("matchup", pd.Series("", index=index)).fillna("").astype(str)
    matchup = matchup.str.lower().str.split().str.join(" ")
    probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    cushion = pd.to_numeric(
        out.get("production_projection_cushion", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    ev = pd.to_numeric(
        out.get("expected_value", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    ranked = pd.DataFrame({
        "matchup": matchup.loc[candidates],
        "probability": probability.loc[candidates],
        "cushion": cushion.loc[candidates],
        "ev": ev.loc[candidates],
        "edge": edge.loc[candidates],
        "stake": stakes.loc[candidates],
    }).sort_values(
        ["probability", "cushion", "ev", "edge", "stake"],
        ascending=[False, False, False, False, False],
    )

    selected = []
    counts: dict[str, int] = {}
    for row_index, row in ranked.iterrows():
        key = str(row["matchup"]).strip() or f"__row_{row_index}"
        if counts.get(key, 0) >= limit:
            continue
        selected.append(row_index)
        counts[key] = counts.get(key, 0) + 1

    dropped = candidates.difference(pd.Index(selected))
    if len(dropped):
        research_reason = (
            "Research only: another production prop from this game ranked higher"
        )
        out.loc[dropped, "Kelly_Bet_Size"] = 0.0
        out.loc[dropped, "production_eligible"] = False
        out.loc[dropped, "production_gate_reason"] = research_reason
        if "Status_Reason" not in out.columns:
            out["Status_Reason"] = ""
        out.loc[dropped, "Status_Reason"] = research_reason
    out["game_exposure_guard_applied"] = bool(len(dropped))
    out["game_funded_after"] = int(len(selected))
    return out


def apply_prop_slate_guard(
    card,
    max_funded_per_slate: int = PROP_MAX_FUNDED_PER_SLATE,
):
    """Fund at most five props, ranking Core ahead of Extended."""
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    index = out.index
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    eligible = out.get(
        "production_eligible", pd.Series(False, index=index)
    ).fillna(False).astype(bool)
    candidates = out.index[eligible & stakes.gt(0)]
    limit = max(0, int(max_funded_per_slate))
    out["slate_guard_applied"] = False
    out["slate_funded_before"] = int(len(candidates))
    out["slate_funded_after"] = int(len(candidates))
    out["slate_max_funded"] = limit
    if len(candidates) <= limit:
        return out

    tier = out.get("Prop_Tier", pd.Series("Research", index=index)).astype(str)
    tier_rank = tier.map({"Core": 2, "Extended": 1}).fillna(0)
    probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    cushion = pd.to_numeric(
        out.get("production_projection_cushion", pd.Series(0.0, index=index)),
        errors="coerce",
    ).fillna(0.0)
    ev = pd.to_numeric(
        out.get("expected_value", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(0.0, index=index)), errors="coerce"
    ).fillna(0.0)
    ranked = pd.DataFrame({
        "tier": tier_rank.loc[candidates],
        "probability": probability.loc[candidates],
        "cushion": cushion.loc[candidates],
        "ev": ev.loc[candidates],
        "edge": edge.loc[candidates],
        "stake": stakes.loc[candidates],
    }).sort_values(
        ["tier", "probability", "cushion", "ev", "edge", "stake"],
        ascending=[False, False, False, False, False, False],
    )
    selected = ranked.head(limit).index
    dropped = candidates.difference(selected)
    research_reason = "Research only: outside the top five funded props on this slate"
    out.loc[dropped, "Kelly_Bet_Size"] = 0.0
    out.loc[dropped, "production_eligible"] = False
    out.loc[dropped, "production_gate_reason"] = research_reason
    out.loc[dropped, "Prop_Tier"] = "Research"
    if "Status_Reason" not in out.columns:
        out["Status_Reason"] = ""
    out.loc[dropped, "Status_Reason"] = research_reason
    out["slate_guard_applied"] = bool(len(dropped))
    out["slate_funded_after"] = int(len(selected))
    return out


def apply_prop_precision_shortlist(
    card,
    max_rows: int = PROP_PRECISION_SHORTLIST_MAX,
):
    """Mark an accuracy-first research shortlist without changing stake gates.

    The available realized-history calibration is used for presentation rank.
    Conservative probability remains the probability used for Kelly sizing and
    every production decision.
    """
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    index = out.index
    calibrated = pd.to_numeric(
        out.get("CalibratedProbability", pd.Series(float("nan"), index=index)),
        errors="coerce",
    )
    conservative = pd.to_numeric(
        out.get("ConservativeWinProbability", out.get(
            "WinProbability", pd.Series(float("nan"), index=index)
        )),
        errors="coerce",
    )
    win_probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(float("nan"), index=index)),
        errors="coerce",
    )
    ev = pd.to_numeric(
        out.get("expected_value", pd.Series(float("nan"), index=index)),
        errors="coerce",
    )
    edge = pd.to_numeric(
        out.get("edge", pd.Series(float("nan"), index=index)),
        errors="coerce",
    )
    ranking_probability = calibrated.where(calibrated.notna(), conservative)
    ranked = pd.DataFrame({
        "ranking_probability": ranking_probability,
        "conservative_probability": conservative,
        "win_probability": win_probability,
        "ev": ev,
        "edge": edge,
    }).dropna(subset=["ranking_probability"]).sort_values(
        [
            "ranking_probability",
            "conservative_probability",
            "win_probability",
            "ev",
            "edge",
        ],
        ascending=[False, False, False, False, False],
        kind="mergesort",
    )
    selected = ranked.head(max(0, int(max_rows))).index
    out["Prop_Precision_Shortlist"] = False
    out["Prop_Precision_Rank"] = pd.Series(pd.NA, index=index, dtype="Int64")
    out["Prop_Ranking_Probability"] = ranking_probability.round(4)
    out["Prop_Ranking_Source"] = "ConservativeWinProbability"
    out.loc[calibrated.notna(), "Prop_Ranking_Source"] = "CalibratedProbability"
    out.loc[selected, "Prop_Precision_Shortlist"] = True
    if len(selected):
        out.loc[selected, "Prop_Precision_Rank"] = pd.array(
            range(1, len(selected) + 1), dtype="Int64"
        )
    return out


def apply_extended_prop_stake_cap(
    card,
    flat_stake: float = PROP_EXTENDED_FLAT_STAKE,
):
    """Use a flat $1 stake for the lower-confidence Extended tier."""
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    tier = out.get(
        "Prop_Tier", pd.Series("Research", index=out.index)
    ).fillna("Research").astype(str)
    eligible = out.get(
        "production_eligible", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    extended = tier.eq("Extended") & eligible & stakes.gt(0)
    out.loc[extended, "Kelly_Bet_Size"] = round(max(0.0, float(flat_stake)), 2)
    if "novig_exposure_after" in out.columns:
        exposure_after = pd.to_numeric(
            out["Kelly_Bet_Size"], errors="coerce"
        ).fillna(0.0).sum()
        out["novig_exposure_after"] = round(float(exposure_after), 2)
    out["extended_flat_stake"] = round(max(0.0, float(flat_stake)), 2)
    out["extended_stake_cap_applied"] = bool(extended.any())
    return out


def apply_probation_share_guard(
    card,
    max_share: float = PROBATION_MAX_PORTFOLIO_SHARE,
    max_funded_tickets: int = PROBATION_MAX_FUNDED_TICKETS,
):
    """Cap probation exposure at 25% of the final funded prop portfolio.

    The share equation is solved against proven exposure, so removing excess
    research tickets cannot leave probation above the requested final share.
    Sportsbook minimums are already applied before this guard runs.
    """
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    probation = out.get(
        "Market_Probation", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    candidates = out.index[probation & stakes.gt(0)]
    proven_exposure = float(stakes[~probation & stakes.gt(0)].sum())
    share = min(max(float(max_share), 0.0), 0.999999)
    share_cap = proven_exposure * share / (1.0 - share) if share > 0 else 0.0
    probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    ev = pd.to_numeric(
        out.get("expected_value", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    ranked = pd.DataFrame({
        "probability": probability.loc[candidates],
        "ev": ev.loc[candidates],
        "edge": edge.loc[candidates],
        "stake": stakes.loc[candidates],
    }).sort_values(
        ["probability", "ev", "edge", "stake"],
        ascending=[False, False, False, False],
    )

    selected = []
    selected_exposure = 0.0
    for index, row in ranked.iterrows():
        if len(selected) >= max(0, int(max_funded_tickets)):
            break
        candidate_stake = float(row["stake"])
        if selected_exposure + candidate_stake <= share_cap + 1e-9:
            selected.append(index)
            selected_exposure += candidate_stake
    out.loc[candidates, "Kelly_Bet_Size"] = 0.0
    if selected:
        out.loc[selected, "Kelly_Bet_Size"] = stakes.loc[selected]
    final_exposure = proven_exposure + selected_exposure
    out["probation_share_guard_applied"] = len(selected) < len(candidates)
    out["probation_max_portfolio_share"] = round(share, 4)
    out["probation_share_exposure_cap"] = round(share_cap, 2)
    out["probation_share_exposure_after"] = round(selected_exposure, 2)
    out["probation_share_after"] = round(
        selected_exposure / final_exposure if final_exposure > 0 else 0.0, 4
    )
    out["probation_share_selected_count"] = int(len(selected))
    return out


def apply_novig_minimum_selection(
    card,
    bankroll: float,
    total_cap_pct: float,
    minimum_bet: float = NOVIG_MINIMUM_BET,
):
    """Make every positive prop stake executable without breaking total exposure."""
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    production_eligible = out.get(
        "production_eligible", pd.Series(True, index=out.index)
    ).fillna(False).astype(bool)
    candidates = out.index[stakes.gt(0) & production_eligible]
    out.loc[stakes.gt(0) & ~production_eligible, "Kelly_Bet_Size"] = 0.0
    exposure_before = float(stakes.sum())
    exposure_cap = max(0.0, float(bankroll or 0.0) * float(total_cap_pct))
    minimum_bet = max(0.0, float(minimum_bet))
    probability = pd.to_numeric(
        out.get("WinProbability", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    edge = pd.to_numeric(
        out.get("edge", pd.Series(0.0, index=out.index)), errors="coerce"
    ).fillna(0.0)
    probation = out.get(
        "Market_Probation", pd.Series(False, index=out.index)
    ).fillna(False).astype(bool)
    ranked = pd.DataFrame({
        "proven": (~probation.loc[candidates]).astype(int),
        "probability": probability.loc[candidates],
        "edge": edge.loc[candidates],
        "original_stake": stakes.loc[candidates],
    }).sort_values(
        ["proven", "probability", "edge", "original_stake"],
        ascending=[False, False, False, False],
    )
    out.loc[candidates, "Kelly_Bet_Size"] = 0.0
    remaining = exposure_cap
    selected = []
    for index, row in ranked.iterrows():
        executable_stake = max(float(row["original_stake"]), minimum_bet)
        if executable_stake <= remaining + 1e-9:
            out.loc[index, "Kelly_Bet_Size"] = round(executable_stake, 2)
            remaining -= executable_stake
            selected.append(index)
    exposure_after = float(pd.to_numeric(
        out.get("Kelly_Bet_Size"), errors="coerce"
    ).fillna(0.0).sum())
    changed = bool(
        (pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0) - stakes)
        .abs().gt(1e-9).any()
    )
    out["novig_minimum_applied"] = changed
    out["novig_minimum_bet"] = round(minimum_bet, 2)
    out["novig_total_exposure_cap"] = round(exposure_cap, 2)
    out["novig_exposure_before"] = round(exposure_before, 2)
    out["novig_exposure_after"] = round(exposure_after, 2)
    out["novig_selected_count"] = int(len(selected))
    return out


def apply_prop_stake_status(card):
    """Separate research rows, qualified reserves, and funded tickets.

    A model signal that fails the production gate remains visible and gradeable,
    but it is explicitly research-only. Only production-eligible rows may be
    called qualified, funded, or admitted to the parlay pipeline.
    """
    import pandas as pd

    if card is None or card.empty:
        return card
    out = card.copy()
    status = out.get(
        "Pick_Status", pd.Series("", index=out.index)
    ).astype(str).str.strip()
    stakes = pd.to_numeric(
        out.get("Kelly_Bet_Size", pd.Series(0.0, index=out.index)),
        errors="coerce",
    ).fillna(0.0)
    model_qualified = status.eq("Actionable")
    production_eligible = out.get(
        "production_eligible", model_qualified
    )
    production_eligible = pd.Series(
        production_eligible, index=out.index
    ).fillna(False).astype(bool)
    funded = model_qualified & production_eligible & stakes.gt(0)
    qualified_unfunded = model_qualified & production_eligible & ~funded
    research = model_qualified & ~production_eligible

    if "Status_Reason" not in out.columns:
        out["Status_Reason"] = ""
    out["Stake_Status"] = "Not Qualified"
    out.loc[funded, "Stake_Status"] = "Funded"
    out.loc[qualified_unfunded, "Stake_Status"] = "Qualified / No Stake"
    out.loc[research, "Stake_Status"] = "Research / No Stake"
    out.loc[funded, "Status_Reason"] = (
        "Qualified and funded after production and portfolio guards"
    )
    out.loc[qualified_unfunded, "Pick_Status"] = "Qualified / No Stake"
    out.loc[research, "Pick_Status"] = "Research / No Stake"

    generic_unfunded_reason = (
        "Cleared production gates but was not selected after exposure, "
        "diversification, and sportsbook-minimum guards"
    )
    gate_reason = out.get(
        "production_gate_reason", pd.Series("", index=out.index)
    ).fillna("").astype(str).str.strip()
    out.loc[qualified_unfunded, "Status_Reason"] = gate_reason.loc[
        qualified_unfunded
    ].where(
        gate_reason.loc[qualified_unfunded].ne(""), generic_unfunded_reason
    )
    out.loc[research, "Status_Reason"] = gate_reason.loc[research].where(
        gate_reason.loc[research].ne(""),
        "Research only: did not clear the production gate",
    )
    return out
