    return {
        "total": simple_select(totals, "total"),
        "spread": simple_select(spreads, "spread"),
        "winner": winner_result,
    }, candidate_debug


def match_kalshi_market(
    game: Dict[str, Any],
    kalshi_markets: List[Dict[str, Any]],
    winner_reason_override: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """Delegates to _match_kalshi_market_impl for fuzzy matching."""
    logger.info(f"🔵 match_kalshi_market called with {len(kalshi_markets)} markets")
    result = _match_kalshi_market_impl(game, kalshi_markets, winner_reason_override)
    logger.info(f"🟢 _match_kalshi_market_impl returned {len(result)} matches")
    return result


# -----------------
# Session defaults
# -----------------

if "last_exception" not in st.session_state:
    st.session_state["last_exception"] = None
if "last_rows_out" not in st.session_state:
    st.session_state["last_rows_out"] = 0
if "games" not in st.session_state:
    st.session_state["games"] = []
if "league" not in st.session_state:
    st.session_state["league"] = "NBA"
if "selected_sports" not in st.session_state:
    st.session_state["selected_sports"] = [ALL_SPORTS_LABEL]
if "commence_stats" not in st.session_state:
    st.session_state["commence_stats"] = {"parsed": 0, "failed": 0, "timezone": get_local_tz()}
if "market_counts" not in st.session_state:
    st.session_state["market_counts"] = {
        "moneyline_available_count": 0,
        "spreads_available_count": 0,
        "totals_available_count": 0,
    }
if "run_id" not in st.session_state:
    st.session_state["run_id"] = None
if "gemini_disabled_reason" not in st.session_state:
    st.session_state["gemini_disabled_reason"] = None


# -----------------
# Data loading helpers
# -----------------

def load_games(selected_leagues: Union[str, List[str]], run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load games from TheOddsAPI for selected leagues.

    Sets st.session_state["games_fetch_status"] to "success", "empty", or "error"
    to enable atomic ingest + UI gating.
    """
    leagues = [selected_leagues] if isinstance(selected_leagues, str) else list(selected_leagues or [])
    all_games_with_times: List[Dict[str, Any]] = []
    commence_stats_total = {"parsed": 0, "failed": 0, "timezone": get_local_tz()}
    moneyline_count = 0
    spreads_count = 0
    totals_count = 0

    # Track whether any league succeeded
    any_games_loaded = False
    all_leagues_failed = True

    # 1. Guard: If no leagues selected or just "All Sports" placeholder (if applicable), skip
    if not leagues:
        logger.info("load_games: No leagues selected, skipping fetch.")
        return []

    for selected_league in leagues:
        sport_key = SPORT_KEYS.get(selected_league)
        if not sport_key:
            st.session_state["last_exception"] = f"Unknown league: {selected_league}"
            continue
        try:
            games_raw = fetch_odds_games(sport_key, run_id=run_id)
            if games_raw and len(games_raw) > 0:
                any_games_loaded = True
                all_leagues_failed = False
        except Exception:
            st.session_state["last_exception"] = traceback.format_exc()
            continue
        normalized = [normalize_game({**g, "sport_key": sport_key}) for g in games_raw]
        with_times, commence_stats = normalize_commence_times(normalized)
        commence_stats_total["parsed"] += commence_stats.get("parsed", 0)
        commence_stats_total["failed"] += commence_stats.get("failed", 0)

        for g in with_times:
            # INITIALIZATION BLOCK
            total_pick_side = None
            total_line = None
            total_pick_odds = None
            spread_engine_used = "missing"
            # 3. Initialization Block (Fix Fatal Loop NameError - Extended)
            total_engine_used = "missing"
            spread_prob_final = 0.5
            total_prob_final = 0.5
            spread_prob_market = 0.5
            total_prob_market = 0.5
            total_pick = None
            spread_pick = None
            kalshi_prob_spread = 0.5
            kalshi_prob_total = 0.5
            model_spread_prob = 0.5
            model_total_prob = 0.5
            game_row = {} # Initialize empty game_row to be safe
            total_engine_used = "missing"
            spread_prob_final = 0.5
            total_prob_final = 0.5
            spread_prob_market = 0.5
            total_prob_market = 0.5
            kalshi_prob_spread = 0.5
            kalshi_prob_total = 0.5
            model_spread_prob = 0.5
            model_total_prob = 0.5
            total_pick = None
            spread_pick = None
            try:
                best = extract_best_market(g)
                warnings = list(best.pop("warnings", []))
                merged_warnings = list(dict.fromkeys((g.get("warnings") or []) + warnings))
                g.update(best)
                g["warnings"] = merged_warnings
                if g.get("best_ml_book") is not None:
                    moneyline_count += 1
                if g.get("best_spread_book") is not None:
                    spreads_count += 1
                if g.get("best_total_book") is not None:
                    totals_count += 1
            except Exception:
                g["warnings"] = list(g.get("warnings") or []) + ["odds_extract_error"]
                st.session_state["last_exception"] = traceback.format_exc()

        all_games_with_times.extend(with_times)

    deduped: Dict[Tuple[Any, Any, Any, Any], Dict[str, Any]] = {}
    for g in all_games_with_times:
        key = (
            g.get("sport_key"),
            g.get("home_team"),
            g.get("away_team"),
            g.get("commence_time_iso_utc"),
        )
        if key not in deduped:
            deduped[key] = g

    games_final = list(deduped.values())

    # Restrict to current local day (post-conversion)
    tz_name = get_local_tz()
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = timezone.utc
    today_local = datetime.now(local_tz).date().isoformat()
