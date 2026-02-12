    if "master_df" in st.session_state:
            del st.session_state["master_df"]
    st.session_state["_last_selected_sports"] = selected_sports
st.session_state["league"] = league
kalshi_required_toggle = st.sidebar.checkbox(
    "Kalshi required", value=st.session_state.get("kalshi_required", True)
)
st.session_state["kalshi_required"] = kalshi_required_toggle
if kalshi_integrator:
    kalshi_integrator.required = kalshi_required_toggle
enable_sentiment = st.sidebar.checkbox(
    "Enable Sentiment", value=st.session_state.get("enable_sentiment", True)
)
st.session_state["enable_sentiment"] = enable_sentiment
if st.sidebar.button("Load Games", width="stretch"):
    # Invalidate master_df when loading new games
    if "master_df" in st.session_state:
        del st.session_state["master_df"]
    load_games(selected_sports or [league])

api_sports_present = (
    get_secret_any("APISPORTS_API_KEY", "API_SPORTS_KEY", "API_SPORTS_API_KEY") is not None
    or any_secret_prefix("APISPORTS_")
)
sportsdata_present = (
    get_secret_any("SPORTSDATA_API_KEY", "SPORTSDATA_KEY") is not None
    or any_secret_prefix("SPORTSDATA_")
)
api_sports_status = "OK" if api_sports_present or any(v for v in api_sports_clients.values() if v) else "MISSING"
sportsdata_status = "OK" if sportsdata_present or any(v for v in sportsdata_clients.values() if v) else "MISSING"
vertex_ready = bool(vertex_endpoint_id) and bool(vertex_info.get("ok"))
gemini_ready = bool(get_secret_any("GEMINI_API_KEY"))
st.sidebar.markdown("---")
st.sidebar.subheader("Status")
sentiment_meta_sidebar = st.session_state.get("sentiment_meta") or init_sentiment_meta()
sentiment_available = int(sentiment_meta_sidebar.get("sentiment_available_count") or 0) > 0
sentiment_cached = bool(sentiment_meta_sidebar.get("sentiment_used_cached"))
sentiment_cooldown = bool(sentiment_meta_sidebar.get("sentiment_rate_limited")) or str(sentiment_meta_sidebar.get("sentiment_status", "")).upper() == "COOLDOWN"
if not enable_sentiment:
    sentiment_status_text = "Disabled"
    sentiment_status_color = "red"
elif sentiment_cooldown:
    sentiment_status_text = "Rate Limited"
    sentiment_status_color = "orange"
elif sentiment_available or sentiment_cached:
    sentiment_status_text = "OK"
    sentiment_status_color = "green"
else:
    sentiment_status_text = "No Data"
    sentiment_status_color = "red"
badges = {
    "OddsAPI": bool(odds_api_key),
    "Vertex": vertex_ready,
    "Gemini": gemini_ready,
    "News": bool(news_api_key),
    "API-Sports": api_sports_status == "OK",
    "SportsData": sportsdata_status == "OK",
    "Kalshi": bool(kalshi_api_key and kalshi_api_secret),
}
for name, ok in badges.items():
    color = "green" if ok else "red"
    st.sidebar.markdown(f"**{name}:** :{color}[{'OK' if ok else 'Missing'}]")
st.sidebar.markdown(f"**Sentiment:** :{sentiment_status_color}[{sentiment_status_text}]")
with st.sidebar.expander("Key sources (API-Sports/SportsData)"):
    st.caption("Lookups: API_SPORTS_KEY, APISPORTS_API_KEY, NBA/NFL specific; SPORTSData: SPORTSDATA_API_KEY/KEY variants")
if not vertex_ready:
    st.sidebar.warning(f"Vertex not ready: {vertex_info.get('error') or 'not configured'}")
with st.sidebar.expander("Vertex / Gemini Status", expanded=False):
    st.json(vertex_info)

render_pipeline_banner()


# -----------------
# Tabs
# -----------------

# --- 6. Tab UI Implementation ---
# Ensure 4-space indentation and exact variable matching
tab_shotgun, tab_master, tab_games, tab_kalshi, tab_sentiment, tab_debug = st.tabs(
    ["🚀 Shotgun Mode", "📊 Master Analysis", "🎮 Games & Odds", "📉 Kalshi", "🧠 Sentiment", "Debug"]
)

with tab_shotgun:
    st.header("🚀 Shotgun Allocation")
    if "shotgun_data" in st.session_state:
        shotgun = st.session_state["shotgun_data"]
        col1, col2, col3 = st.columns(3)

        with col1:
            st.info("🎯 $3 'Snipers' (High Prob)")
            if not shotgun['snipers'].empty:
                st.dataframe(shotgun['snipers'][['Pick', 'AI_Prob', 'AI_Edge']])
            else:
                st.write("No snipers found.")

        with col2:
            st.success("📈 $2 'Strategy' (High EV)")
            if not shotgun['strategy'].empty:
                st.dataframe(shotgun['strategy'][['Pick', 'AI_Prob', 'AI_Edge']])
            else:
                st.write("No strategy picks found.")

        with col3:
            st.warning("🎲 $1 'Longshots' (Lottos)")
            if not shotgun['longshots'].empty:
                st.dataframe(shotgun['longshots'][['Pick', 'AI_Prob', 'AI_Edge']])
            else:
                st.write("No longshots found.")
    else:
        st.info("Run Master Analysis to generate Shotgun picks.")

with tab_games:
    st.header("Games & Odds")
    games = st.session_state.get("games", [])
    sent_map = st.session_state.get("sentiment_map") or {}
    match_lookup: Dict[Tuple[Any, Any, Any, Any], Dict[str, Any]] = {}
    for entry in st.session_state.get("kalshi_match_results") or []:
        game = entry.get("game") or {}
        matches = entry.get("matches") or {}
        winner = matches.get("winner") or {}
        key = (
            game.get("league"),
            game.get("home_team"),
            game.get("away_team"),
            game.get("commence_time_iso_utc") or game.get("commence_time"),
        )
        match_lookup[key] = winner

    if not games:
        st.info("Load games from the sidebar to begin.")
    else:
        rows = []
        for g in games:
            markets = set()
            for bm in g.get("bookmakers") or []:
                for m in bm.get("markets") or []:
                    if m.get("key"):
                        markets.add(m.get("key"))
            home_sent = sent_map.get(g.get("home_team"))
            away_sent = sent_map.get(g.get("away_team"))
            sent_diff = None
            if home_sent is not None and away_sent is not None:
                sent_diff = home_sent - away_sent
            rows.append(
                {
                    "League": g.get("league"),
                    "Home": g.get("home_team"),
                    "Away": g.get("away_team"),
                    "Commence (UTC)": g.get("commence_time_iso_utc")
                    or safe_iso(g.get("commence_time_iso")),
                    "Commence (Local)": fmt_local_time(g.get("commence_time_local")),
                    "Local Date": g.get("commence_date_local") or "",
                    "Books": len(g.get("bookmakers") or []),
                    "MarketsAvailable": ", ".join(sorted(markets)),
                    "home_ml_price": g.get("home_ml_price"),
                    "away_ml_price": g.get("away_ml_price"),
                    "implied_prob_home": g.get("implied_prob_home"),
                    "implied_prob_away": g.get("implied_prob_away"),
                    "home_spread_point": g.get("home_spread_point"),
                    "home_spread_price": g.get("home_spread_price"),
                    "away_spread_point": g.get("away_spread_point"),
                    "away_spread_price": g.get("away_spread_price"),
                    "total_point": g.get("total_point"),
                    "over_price": g.get("over_price"),
                    "under_price": g.get("under_price"),
                    "Home_Sentiment": home_sent,
                    "Away_Sentiment": away_sent,
                    "Sentiment_Diff": sent_diff,
                    "warnings": ",".join(g.get("warnings") or []),
                }
            )
        # Add quick pick/prob columns for ML/Spread/Total plus any known Kalshi match info.
        enriched_rows = []
        for r in rows:
            implied_home = r.get("implied_prob_home")
            implied_away = r.get("implied_prob_away")
            ml_pick = None
            ml_pick_prob = None
            if implied_home is not None or implied_away is not None:
                if (implied_home or 0) >= (implied_away or 0):
                    ml_pick = r["Home"]
                    ml_pick_prob = implied_home
                else:
                    ml_pick = r["Away"]
                    ml_pick_prob = implied_away

            # Spread pick
            spread_pick = None
            spread_pick_prob = None
            spread_line = None
            home_spread_prob = american_to_implied_prob(r.get("home_spread_price"))
            away_spread_prob = american_to_implied_prob(r.get("away_spread_price"))
            if r.get("home_spread_point") is not None:
                spread_pick = r["Home"]
                spread_pick_prob = home_spread_prob
                spread_line = r.get("home_spread_point")
                if away_spread_prob is not None and (away_spread_prob >= (home_spread_prob or 0)):
                    spread_pick = r["Away"]
                    spread_pick_prob = away_spread_prob
                    spread_line = r.get("away_spread_point")

            # Total pick
            total_pick = None
            total_pick_prob = None
            if r.get("total_point") is not None:
                over_prob = american_to_implied_prob(r.get("over_price"))
                under_prob = american_to_implied_prob(r.get("under_price"))
                total_pick = "Over"
                total_pick_prob = over_prob
                if under_prob is not None and (under_prob >= (over_prob or 0)):
                    total_pick = "Under"
                    total_pick_prob = under_prob

            key = (r["League"], r["Home"], r["Away"], r["Commence (UTC)"])
            kalshi_info = match_lookup.get(key, {})

            enriched_rows.append(
                {
                    **r,
                    "ML Pick": ml_pick,
                    "ML Pick Prob": ml_pick_prob,
                    "Spread Pick": f"{spread_pick} {spread_line}" if spread_pick is not None else None,
                    "Spread Pick Prob": spread_pick_prob,
                    "Total Pick": f"{total_pick} {r.get('total_point')}" if total_pick else None,
                    "Total Pick Prob": total_pick_prob,
                    "kalshi_available": kalshi_info.get("kalshi_available"),
                    "kalshi_matched": kalshi_info.get("kalshi_matched"),
                    "kalshi_prob": kalshi_info.get("kalshi_prob"),
                    "kalshi_event_ticker": kalshi_info.get("kalshi_event_ticker"),
                }
            )

        st.dataframe(pd.DataFrame(enriched_rows))

with tab_master:
    st.header("Master Analysis")
    kalshi_status = kalshi_health_check(league)
    if not kalshi_status.get("configured"):
        error_detail = kalshi_status.get("error") or "Kalshi is required and missing keys."
        if kalshi_status.get("status_code"):
            error_detail = f"{error_detail} (status {kalshi_status.get('status_code')}: {kalshi_status.get('response_text_snippet')})"
        st.error(error_detail)
        st.info("Master Analysis is disabled until Kalshi is available.")
    else:
        if kalshi_status.get("error") and not kalshi_status.get("ok"):
            warn_detail = kalshi_status.get("error") or "Kalshi reachable but returned no markets; proceeding without Kalshi data."
            st.warning(warn_detail)
        if kalshi_status.get("warning"):
            st.warning(kalshi_status.get("warning"))
    st.session_state.setdefault("kalshi_match_only", False)
    kalshi_match_only = st.checkbox(
        "Show only games with a Kalshi match",
        value=st.session_state.get("kalshi_match_only", False),
    )
    st.session_state["kalshi_match_only"] = kalshi_match_only
    use_gemini_explanations = st.checkbox(
        "Use Gemini Confidence + Explanation",
        value=st.session_state.get("use_gemini_explanations", True),
        key="use_gemini_explanations",
    )
    use_vertex_numeric_probs = st.checkbox(
        "Use Vertex Numeric Probabilities (debug/optional)",
        value=st.session_state.get("use_vertex_numeric_probs", False),
        key="use_vertex_numeric_probs",
    )
    run_master = st.button(
        "Run Master Analysis",
        key="run_master",
        disabled=(not kalshi_status.get("configured")) and st.session_state.get("kalshi_required", True),
        help="Requires Kalshi availability",
    )
    games = st.session_state.get("games", [])
    if run_master and (not kalshi_status.get("configured")):
        st.error("Kalshi is required but unavailable. Fix Kalshi first.")
        st.stop()

    # Determine if we need to run (user clicked button) or just display (cached df exists)
    df_existing = st.session_state.get("master_df")
    should_run = run_master

    # If we have existing data and didn't request a re-run, use it to skip the heavy lifting
    # We still need to define the helper functions because they are called during the DataFrame construction block
    # Actually, the entire block below constructs the DataFrame. We need to restructure this.

    if should_run:
        st.session_state["DECISION_TRACE_SAMPLES"] = {}

        def store_decision_trace_sample(
            league_code: Optional[str],
            home_team: Optional[str],
            away_team: Optional[str],
            market: str,
            pick: Optional[str],
            final_probability: Optional[float],
            trace_json_raw: Any,
        ) -> None:
            league_code_norm = (league_code or "").upper()
            if league_code_norm not in DECISION_TRACE_SAMPLE_LEAGUES:
                return
            try:
                samples = dict(st.session_state.get("DECISION_TRACE_SAMPLES", {}))
            except Exception:
                samples = {}
            if league_code_norm in samples:
                return
            try:
                parsed_trace = json.loads(trace_json_raw) if isinstance(trace_json_raw, str) else trace_json_raw
            except Exception:
                parsed_trace = trace_json_raw
            samples[league_code_norm] = {
                "league": league_code_norm,
                "home": home_team,
                "away": away_team,
                "market": market,
                "pick": pick,
                "final_probability": final_probability,
                "decision_trace_json": parsed_trace,
            }
            st.session_state["DECISION_TRACE_SAMPLES"] = samples
        api_sports_status_run = api_sports_status
        sportsdata_status_run = sportsdata_status
        df_master = pd.DataFrame(games or [])

        # Inject real-time stats for Vertex
        api_sports_clients, _ = init_data_clients()
        df_master = enrich_with_vertex_features(df_master, api_sports_clients)

        unique_teams = sorted(
            set(df_master.get("home_team", pd.Series([], dtype=str)).dropna().astype(str))
            | set(df_master.get("away_team", pd.Series([], dtype=str)).dropna().astype(str))
        )
        enable_sentiment_master = st.checkbox(
            "Enable sentiment (NewsAPI)",
            value=True,
            key="enable_sentiment_master",
        )
        st.session_state["enable_sentiment"] = enable_sentiment_master
        slate_sentiment = get_slate_sentiment(enable_sentiment_master, unique_teams, "MIXED", news_api_key)
        st.session_state["sentiment_map"] = slate_sentiment.get("map") or {}
        st.session_state["sentiment_meta_map"] = slate_sentiment.get("meta_map") or {}
        st.session_state["sentiment_meta"] = slate_sentiment.get("meta") or init_sentiment_meta()
        st.session_state["sentiment_debug"] = slate_sentiment.get("debug") or {}
        with st.expander("Sentiment Debug", expanded=False):
            meta_view = slate_sentiment.get("meta") or {}
            meta_map_view = slate_sentiment.get("meta_map") or {}
            source_counts: Dict[str, int] = {}
            for mv in meta_map_view.values():
                src_val = str(mv.get("sentiment_source") or "none")
                source_counts[src_val] = source_counts.get(src_val, 0) + 1
            st.write("Sentiment source:", meta_view.get("sentiment_source"))
            st.write("Status counts:", meta_view.get("sentiment_status_counts"))
            st.write("Teams by source:", source_counts)
            st.write(
                "Reddit posts/comments used:",
                meta_view.get("reddit_posts_used", 0),
                meta_view.get("reddit_comments_used", 0),
            )
            st.write("Unique teams:", len(unique_teams))
            st.json(meta_view)
        sentiment_pack_meta = st.session_state.get("sentiment_meta") or init_sentiment_meta()
        sentiment_map: Dict[str, Optional[float]] = st.session_state.get("sentiment_map") or {}
        sentiment_meta_map: Dict[str, Dict[str, Any]] = st.session_state.get("sentiment_meta_map") or {}
        sentiment_status_counts_global = sentiment_pack_meta.get("sentiment_status_counts") or {"NO_CALL": 1}
        if st.session_state.get("kalshi_required", True) and kalshi_integrator:
            try:
                kalshi_integrator.assert_available()
            except Exception as exc:
                st.error(str(exc))
                st.stop()
        commence_times_by_league: Dict[str, List[str]] = {}
        for g in games:
            lg = g.get("league")
            commence_val = g.get("commence_time_iso_utc") or g.get("commence_time") or g.get("commence_time_iso")
            if not commence_val:
                continue
            commence_times_by_league.setdefault(lg, []).append(commence_val)
        sentiment_meta_global: Dict[str, Any] = {**init_sentiment_meta(), **(st.session_state.get("sentiment_meta") or {})}
        sentiment_status_counts_global = sentiment_meta_global.get("sentiment_status_counts") or {"NO_CALL": 1}
        leagues_for_fetch = list({k for k in commence_times_by_league.keys() if k}) or (selected_sports or [league])
        try:
            kalshi_markets_by_league = fetch_kalshi_markets_for_leagues(
                leagues_for_fetch, commence_times_by_league
            )
        except RuntimeError as exc:
            msg = str(exc)
            if "429" in msg or "rate limit" in msg.lower():
                st.error("Kalshi rate-limited. Please retry in ~X seconds.")
            else:
                st.error(msg)
            st.stop()
        except Exception as exc:
            st.error(str(exc))
            st.stop()
        if not kalshi_markets_by_league:
            st.warning(
                "Kalshi markets could not be fetched; proceeding with cached/empty set."
            )
            kalshi_markets_by_league = {}
        all_markets_flat: List[Dict[str, Any]] = []
        for mkts in kalshi_markets_by_league.values():
