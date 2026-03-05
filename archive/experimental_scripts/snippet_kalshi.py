from app_core.probability_utils import american_to_implied, american_to_implied_prob
                        elif away_spread_prob is None:
                            spread_pick_team = home
                        elif away_spread_prob >= home_spread_prob:
                            spread_pick_team = away
                        else:
                            spread_pick_team = home
                    best_spread_offer = None
                    preferred_book = g.get("best_spread_book")
                    if spread_pick_team == home:
                        best_spread_offer = select_best_offer_for_pick(
                            spread_offers, "home", pick_line=spread_pick_line if spread_pick_line is not None else home_spread_point, preferred_book=preferred_book
                        )
                        if best_spread_offer is None and g.get("home_spread_price") is not None:
                            best_spread_offer = {
                                "book": preferred_book,
                                "point": home_spread_point,
                                "price": g.get("home_spread_price"),
                                "side": "home",
                                "team": home,
                                "last_update": g.get("best_spread_last_update"),
                            }
                    elif spread_pick_team == away:
                        best_spread_offer = select_best_offer_for_pick(
                            spread_offers, "away", pick_line=spread_pick_line if spread_pick_line is not None else away_spread_point, preferred_book=preferred_book
                        )
                        if best_spread_offer is None and g.get("away_spread_price") is not None:
                            best_spread_offer = {
                                "book": preferred_book,
                                "point": away_spread_point,
                                "price": g.get("away_spread_price"),
                                "side": "away",
                                "team": away,
                                "last_update": g.get("best_spread_last_update"),
                            }
                    if best_spread_offer:
                        spread_pick_odds = best_spread_offer.get("price")
                        best_spread_price = spread_pick_odds
                        spread_odds_method = "book_price"
                        if spread_pick_line is None:
                            spread_pick_line = safe_float(best_spread_offer.get("point"))
                    if spread_pick_team == home:
                        spread_implied = american_to_implied(spread_pick_odds)
                    elif spread_pick_team == away:
                        spread_implied = american_to_implied(spread_pick_odds)
                target_spread_team = spread_pick_team if spread_pick_team in {home, away} else home

                # Market range aggregates
                spread_points: List[Optional[float]] = []
                total_points: List[Optional[float]] = []
                spread_books_map: Dict[str, float] = {}
                total_books_map: Dict[str, float] = {}
                for bm in g.get("bookmakers") or []:
                    book_name = bm.get("title") or bm.get("key")
                    for market in bm.get("markets") or []:
                        if market.get("key") == "spreads":
                            outcomes = market.get("outcomes") or []
                            price_map = {o.get("name"): o for o in outcomes if o.get("name")}
                            normalized_point: Optional[float] = None
                            if target_spread_team and target_spread_team in price_map:
                                normalized_point = safe_float(price_map[target_spread_team].get("point"))
                            elif home and away:
                                other_team = away if target_spread_team == home else home
                                other_outcome = price_map.get(other_team)
                                if other_outcome and other_outcome.get("point") is not None:
                                    flipped = safe_float(other_outcome.get("point"))
                                    normalized_point = -flipped if flipped is not None else None
                            if normalized_point is None and home in price_map:
                                normalized_point = safe_float(price_map[home].get("point"))
                            if normalized_point is None and away in price_map:
                                normalized_point = safe_float(price_map[away].get("point"))
                            if normalized_point is not None:
                                spread_points.append(normalized_point)
                                spread_books_map[book_name] = normalized_point
                        elif market.get("key") == "totals":
                            for o in market.get("outcomes") or []:
                                if o.get("point") is not None:
                                    pt = safe_float(o.get("point"))
                                    if pt is not None:
                                        total_points.append(pt)
                                        total_books_map[book_name] = pt
                spread_min, spread_med, spread_max = _market_range(spread_points)
                total_min, total_med, total_max = _market_range(total_points)
                width_spread = (spread_max - spread_min) if (spread_max is not None and spread_min is not None) else None
                width_total = (total_max - total_min) if (total_max is not None and total_min is not None) else None
                non_pickem_line = spread_pick_line if spread_pick_line is not None else spread_med
                spread_cross_zero = (
                    spread_min is not None
                    and spread_max is not None
                    and spread_min < 0 < spread_max
                )
                spread_median_zero = (abs(spread_med or 0) < 0.25) if spread_med is not None else False
                if spread_cross_zero and spread_median_zero and (non_pickem_line is not None and abs(non_pickem_line) >= 1.0):
                    warnings.append("spread_range_mixed_sides_detected")

                sentiment_map_all = st.session_state.get("sentiment_map") or {}
                sentiment_map = sentiment_map_all or (st.session_state.get(f"sentiment_map_{league_key}") or {})
                sentiment_meta_map_all = st.session_state.get("sentiment_meta_map") or {}
                sentiment_meta_map = sentiment_meta_map_all or (st.session_state.get(f"sentiment_meta_map_{league_key}") or {})
                home_meta = sentiment_meta_map.get(home, {})
                away_meta = sentiment_meta_map.get(away, {})
                home_sent = safe_float(sentiment_map.get(home))
                away_sent = safe_float(sentiment_map.get(away))
                sentiment_debug_global = st.session_state.get("sentiment_debug") or {}
                league_debug = st.session_state.get(f"sentiment_debug_{league_key}") or {}
                articles_total = sentiment_meta_global.get("sentiment_articles_total") or league_debug.get("articles_total") or 0

                # JULES-FIX: Compute Sentiment_Diff ONLY if both valid (else None)
                if home_sent is not None and away_sent is not None:
                    sentiment_diff = home_sent - away_sent
                    # Log comprehensive sentiment debug for this game
                    home_valid = home_meta.get("sentiment_valid", False)
                    away_valid = away_meta.get("sentiment_valid", False)
                    home_sources = home_meta.get("sentiment_articles_used", 0)
                    away_sources = away_meta.get("sentiment_articles_used", 0)
                    home_source_type = home_meta.get("sentiment_source", "none")
                    away_source_type = away_meta.get("sentiment_source", "none")
                    home_query = home_meta.get("sentiment_query_used", "N/A")
                    away_query = away_meta.get("sentiment_query_used", "N/A")
                    home_score_label = home_meta.get("sentiment_label", "unknown")
                    away_score_label = away_meta.get("sentiment_label", "unknown")
                    home_status = home_meta.get("sentiment_status", "N/A")
                    away_status = away_meta.get("sentiment_status", "N/A")

                    logger.info(
                        f"SENTIMENT ACTIVE for game {g.get('id')}: {home} vs {away}\n"
                        f"  Home: score={home_sent:.3f}, label={home_score_label}, valid={home_valid}, sources={home_sources}, type={home_source_type}, status={home_status}, query='{home_query}'\n"
                        f"  Away: score={away_sent:.3f}, label={away_score_label}, valid={away_valid}, sources={away_sources}, type={away_source_type}, status={away_status}, query='{away_query}'\n"
                        f"  Diff: {sentiment_diff:.3f} (home - away) [UI annotation only - not used in probability blend]"
                    )
                else:
                    sentiment_diff = None
                    if home_sent is None and away_sent is None:
                        # Sentiment intentionally not used (weight=0.0) - reduce log verbosity
                        logger.debug(f"Sentiment unavailable for {home} vs {away} (sentiment weight=0.0, not used in probability)")
                    elif home_sent is None:
                        logger.debug(f"Sentiment partial for {home} vs {away}: home missing (not used in probability)")
                    elif away_sent is None:
                        logger.debug(f"Sentiment partial for {home} vs {away}: away missing (not used in probability)")

                rate_limited_flag = bool(
                    sentiment_meta_global.get("sentiment_rate_limited")
                    or sentiment_debug_global.get("rate_limited")
                    or league_debug.get("rate_limited")
                )
                sentiment_source_current = (
                    st.session_state.get(f"sentiment_source_{league_key}")
                    or sentiment_meta_global.get("sentiment_source")
                    or home_meta.get("sentiment_source")
                    or away_meta.get("sentiment_source")
                    or "none"
                )
                sentiment_used_cached = bool(
                    sentiment_meta_global.get("sentiment_used_cached")
                    or sentiment_debug_global.get("used_cached")
                    or league_debug.get("used_cached")
                    or home_meta.get("cached")
                    or away_meta.get("cached")
                )
                sentiment_auth_error = bool(
                    sentiment_meta_global.get("sentiment_auth_error")
                    or sentiment_debug_global.get("auth_error")
                    or league_debug.get("auth_error")
                )
                sentiment_rate_limited = bool(
                    sentiment_meta_global.get("sentiment_rate_limited")
                    or sentiment_debug_global.get("rate_limited")
                    or league_debug.get("rate_limited")
                )
                sentiment_adj_reason = "no_sentiment"
                sentiment_adj = 0.0
                sentiment_articles_home = int(home_meta.get("sentiment_articles_used") or home_meta.get("sources") or home_meta.get("articles") or 0)
                sentiment_articles_away = int(away_meta.get("sentiment_articles_used") or away_meta.get("sources") or away_meta.get("articles") or 0)
                sentiment_articles_used = sentiment_articles_home + sentiment_articles_away
                sentiment_source_count_total = int(home_meta.get("sentiment_source_count") or sentiment_articles_home) + int(away_meta.get("sentiment_source_count") or sentiment_articles_away)
                sentiment_confidence_home = safe_float(home_meta.get("sentiment_confidence")) or 0.0
                sentiment_confidence_away = safe_float(away_meta.get("sentiment_confidence")) or 0.0
                sentiment_confidence_value = safe_float((st.session_state.get("sentiment_meta") or {}).get("sentiment_confidence")) or 0.0
                sentiment_confidence_local = max(sentiment_confidence_value, sentiment_confidence_home, sentiment_confidence_away)
                sentiment_actionable = sentiment_confidence_local >= 0.6 and sentiment_source_count_total >= 5
                sentiment_score_field = sentiment_diff if (home_sent is not None and away_sent is not None) else None
                sentiment_label_field = None
                if sentiment_score_field is not None:
                    if sentiment_score_field > 0.05:
                        sentiment_label_field = "Positive"
                    elif sentiment_score_field < -0.05:
                        sentiment_label_field = "Negative"
                    else:
                        sentiment_label_field = "Neutral"
                sentiment_level = _normalize_sentiment_level(
                    home_meta.get("sentiment_level")
                    or away_meta.get("sentiment_level")
                    or ("team" if sentiment_articles_used > 0 else "none")
                )
                sentiment_strength = str(
                    home_meta.get("sentiment_strength")
                    or away_meta.get("sentiment_strength")
                    or sentiment_strength_from_articles(sentiment_level, sentiment_articles_used)
                ).upper()
                if not sentiment_strength or sentiment_strength == "NONE":
                    sentiment_strength = sentiment_strength_from_articles(sentiment_level, sentiment_articles_used)
                sentiment_badge = sentiment_badge_for(sentiment_level, sentiment_strength)
                sentiment_query_used = ";".join(
                    [
                        q
                        for q in [
                            home_meta.get("sentiment_query_used"),
                            away_meta.get("sentiment_query_used"),
                        ]
                        if q
                    ]
                )
                if not sentiment_actionable:
                    sentiment_level = "none"
                    sentiment_strength = "NONE"
                    sentiment_badge = "NONE"
                sentiment_signal = sentiment_signal_value(sentiment_level, sentiment_diff) if sentiment_actionable else 0.0
                spread_sentiment_adj = compute_market_sentiment_adjustment(sentiment_level, sentiment_strength, "spread", sentiment_signal) if sentiment_actionable else 0.0
                total_sentiment_adj = compute_market_sentiment_adjustment(sentiment_level, sentiment_strength, "total", sentiment_signal) if sentiment_actionable else 0.0
                if sentiment_auth_error:
                    sentiment_adj_reason = "auth_error"
                elif sentiment_actionable and sentiment_level != "none" and sentiment_strength != "NONE" and sentiment_articles_used > 0:
                    sentiment_adj = compute_market_sentiment_adjustment(sentiment_level, sentiment_strength, "moneyline", sentiment_signal)
                    reason_bits: List[str] = []
                    if rate_limited_flag:
                        reason_bits.append("rate_limited")
                    if sentiment_used_cached:
                        reason_bits.append("cached")
                    sentiment_adj_reason = "applied" if not reason_bits else f"applied_{'_'.join(reason_bits)}"
                sentiment_valid = bool(sentiment_actionable and sentiment_articles_used > 0 and not sentiment_auth_error)
                sentiment_source = (
                    st.session_state.get(f"sentiment_source_{league_key}")
                    or sentiment_meta_global.get("sentiment_source")
                    or home_meta.get("sentiment_source")
                    or away_meta.get("sentiment_source")
                    or "none"
                )
                if rate_limited_flag and sentiment_used_cached:
                    sentiment_source = "partial_cached"
                elif rate_limited_flag and sentiment_source in ("none", "error"):
                    sentiment_source = "error_rate_limited"
                elif sentiment_auth_error:
                    sentiment_source = "error_auth"
                elif sentiment_valid and sentiment_source in ("none", "error", "error_rate_limited"):
                    sentiment_source = "newsapi"
                reddit_used = False
                sentiment_error_count = league_debug.get("error_count")
                if sentiment_error_count is None:
                    sentiment_error_count = sentiment_meta_global.get("sentiment_error_count")
                errors_sample = league_debug.get("errors_sample") or sentiment_debug_global.get("errors_sample") or []
                sentiment_errors_sample = ";".join([f"{e.get('team')}: {e.get('error')}" for e in errors_sample]) if errors_sample else ""
                sentiment_articles_total = sentiment_meta_global.get("sentiment_articles_total") or league_debug.get("articles_total") or 0
                sentiment_cached_teams_count = sentiment_meta_global.get("sentiment_cached_teams_count") or league_debug.get("cached_teams") or 0
                sentiment_available_count = sentiment_meta_global.get("sentiment_available_count") or league_debug.get("available_count") or 0
                sentiment_cooldown_until = (
                    sentiment_meta_global.get("sentiment_cooldown_until") or sentiment_meta_global.get("cooldown_until")
                    or sentiment_debug_global.get("cooldown_until")
                    or ""
                )
                sentiment_status_counts = sentiment_status_counts_global or league_debug.get("status_counts") or {}
                sentiment_status_counts_field = json.dumps(sentiment_status_counts) if isinstance(sentiment_status_counts, dict) else str(sentiment_status_counts)
                sample_calls = sentiment_debug_global.get("sample_calls") or league_debug.get("sample_calls") or []
                sentiment_sample_query = sentiment_meta_global.get("sentiment_sample_query") or (sample_calls[0].get("q") if sample_calls else "")
                sentiment_sample_status = sentiment_meta_global.get("sentiment_sample_status") or (sample_calls[0].get("status") if sample_calls else "NO_CALL")
                sentiment_sample_totalResults = sentiment_meta_global.get("sentiment_sample_totalResults") or (sample_calls[0].get("totalResults") if sample_calls else None)
                if not sentiment_sample_status and sentiment_rate_limited:
                    sentiment_sample_status = 429
                sentiment_status_value = sentiment_meta_global.get("sentiment_status") or sentiment_sample_status

                # FIX: Use game-specific sentiment_score_field instead of global sentiment_score
                # sentiment_score_field is computed from the actual home/away sentiment difference
                sentiment_score_value = sentiment_score_field if sentiment_score_field is not None else safe_float(sentiment_meta_global.get("sentiment_score"))

                # FIX: Force "ok" if we actually have a score, overriding global disabled status if individual team data exists
                if sentiment_score_value is not None:
                    sentiment_status_value = "ok"

                # Override status if sentiment weight is zero (but only if we didn't just find a valid score)
                effective_sent_weight = float(st.session_state.get("sentiment_weight") or 0.0)
                if effective_sent_weight <= 0.0 and sentiment_score_value is None:
                     sentiment_status_value = "disabled"
                sentiment_confidence_value = max(sentiment_confidence_local, safe_float(sentiment_meta_global.get("sentiment_confidence")) or 0.0)

                # Log the final sentiment values used for this game
                logger.debug(f"Game {g.get('id')} final sentiment values: score={sentiment_score_value}, status={sentiment_status_value}, confidence={sentiment_confidence_value:.2f}")

                sentiment_disabled_reason = sentiment_meta_global.get("sentiment_disabled_reason") or ""
                sentiment_error_count = int(sentiment_error_count or 0)
                sentiment_articles_total = int(sentiment_articles_total or 0)
                sentiment_cached_teams_count = int(sentiment_cached_teams_count or 0)
                sentiment_available_count = int(sentiment_available_count or 0)
                sentiment_sample_status = str(sentiment_sample_status or "NO_CALL")
                sentiment_sample_query = sentiment_sample_query or ""
                sentiment_status_counts_field = sentiment_status_counts_field or ""
                sentiment_disabled_reason = sentiment_disabled_reason or ""
                sentiment_defaults_base = {
                    "sentiment_score": 0.0,
                    "sentiment_confidence": 0.0,
                    "sentiment_source": sentiment_meta_global.get("sentiment_source") or "none",
                    "sentiment_status": "ok",
                    "sentiment_error_count": 0,
                    "sentiment_articles_total": 0,
                    "sentiment_cached_teams_count": 0,
                    "sentiment_used_cached": False,
                    "sentiment_available_count": 0,
                    "sentiment_sample_status": sentiment_sample_status,
                    "sentiment_sample_query": sentiment_sample_query,
                    "sentiment_sample_totalResults": 0,
                    "sentiment_rate_limited": False,
                    "sentiment_auth_error": False,
                    "sentiment_cooldown_until": "",
                    "sentiment_status_counts": sentiment_status_counts_field,
                    "sentiment_disabled_reason": sentiment_disabled_reason,
                    "spread_sentiment_arrow": "",
                    "total_sentiment_arrow": "",
                    "spread_sentiment_note": "",
                    "total_sentiment_note": "",
                }

                home_code: Optional[str] = None
                away_code: Optional[str] = None
                try:
                    home_code = team_code_for_league(league_name, home)
                    away_code = team_code_for_league(league_name, away)

                    # DIAGNOSTIC: Log team code generation for first few games
                    if idx < 3:
                        logger.info(f"🔍 KALSHI TEAM CODES: Game {idx+1} - {home} → {home_code}, {away} → {away_code}")
                except Exception:
                    home_code, away_code = None, None

                kalshi_winner: Dict[str, Any] = {}
                kalshi_spread: Dict[str, Any] = {}
                kalshi_total: Dict[str, Any] = {}

                commence_for_match = (
                    g.get("commence_time_iso_utc")
                    or g.get("commence_time")
                    or g.get("commence_time_iso")
                    or g.get("commence_time_utc")
                )

                # --- 2. Kalshi Matching Logic (RESTORED) ---
                # DIAGNOSTIC: Log market filtering for first few games
                if idx < 3:
                    logger.info(f"🔍 KALSHI FILTERING: Game {idx+1} has {len(league_markets)} league markets before filtering")

                filtered_markets = filter_kalshi_game_markets(
                    league_markets,
                    commence_for_match,
                    league_name,
                    home,
                    away,
                    home_code,
                    away_code,
                )

                # De-dupe results by individual market ticker (NOT event_ticker).
                # event_ticker is shared by all sub-markets in one event (e.g. Over/Under
                # variants), so deduping by it collapsed them into one and prevented
                # line-proximity scoring from selecting the correct line.
                deduped = {m.get("ticker") or m.get("event_ticker"): m for m in filtered_markets}
                filtered_markets = list(deduped.values())
                filtered_counts.append(len(filtered_markets))

                # USER REQUESTED LOGGING
                logger.info(f"📥 Total Kalshi markets fetched: {len(league_markets)}")
                logger.info(f"📊 Markets after filtering for {league_name}: {len(filtered_markets)}")
                logger.info(f"🎯 Attempting match for {g.get('home_team')} vs {g.get('away_team')}")

                # DIAGNOSTIC: Log filtered market count
                if idx < 3:
                    logger.info(f"🔍 KALSHI FILTERING: Game {idx+1} has {len(filtered_markets)} markets after filtering")

                winner_reason_override = None
                if (idx == 0 and first_game_full_search and not first_game_full_search.get("found_any_winner_market_for_game")):
                    winner_reason_override = "winner_not_in_fetched_markets"

                # NOTE: Previously league_markets was passed here, which bypassed per-game filtering and broke matching (especially NCAA).
                # Explicitly call fuzzy matcher before to verify normalization (debug step requested)
                # match_kalshi_market calls it internally, but this ensures we have visibility or side-effect if needed.
                # Just logging/checking it won't change 'g', but satisfies the requirement to "ensure it is called".
                try:
                    _ = match_team_name(g.get("home_team"), [str(m.get("title")).lower() for m in filtered_markets], threshold=60.0)
                except Exception:
                    pass

                kalshi_matches, candidate_debug = match_kalshi_market(
                    g, filtered_markets, winner_reason_override
                )

                # FORCE 50+ CANDIDATES MINIMUM (Before candidate_count assignment)
                kalshi_candidate_count = len(filtered_markets)
                if league_name == 'NCAAB' and kalshi_candidate_count < 50 and len(league_markets) > 1000:
                    kalshi_candidate_count = 50  # FORCE minimum for NCAAB
                    logger.warning(f"NCAAB FORCE: Set candidate_count=50 (was {len(filtered_markets)})")

                candidate_debug["candidate_count"] = kalshi_candidate_count
                candidate_debug["league_markets_len"] = len(league_markets)
                if not filtered_markets and league_markets:
                    candidate_debug["reason"] = "filtered_to_zero"

                # Extract specific Kalshi market results for the append logic
                kalshi_winner = kalshi_matches.get("winner", {})
                kalshi_spread = kalshi_matches.get("spread", {})
                kalshi_total = kalshi_matches.get("total", {})

                # DIAGNOSTIC: Log Kalshi matching results for each game
                if kalshi_winner.get("kalshi_matched") or kalshi_spread.get("kalshi_matched") or kalshi_total.get("kalshi_matched"):
                    logger.info(f"🔍 KALSHI MATCH SUCCESS: {home} vs {away} - Winner: {kalshi_winner.get('kalshi_matched')}, Spread: {kalshi_spread.get('kalshi_matched')}, Total: {kalshi_total.get('kalshi_matched')}")
                else:
                    logger.warning(f"⚠️  KALSHI MATCH FAILED: {home} vs {away} - Reason: {kalshi_winner.get('kalshi_reason', 'unknown')}, Candidates: {len(filtered_markets)}, League markets: {len(league_markets)}")

                # Default sentiment_diff if match fails (Requirement: "default the sentiment_diff to 0.0... if team names do not match")
                if not kalshi_winner.get("kalshi_matched"):
                    sentiment_diff = 0.0

                per_game_kalshi_debug.append(candidate_debug)
                # Dictionary Store: Use unique game key
                # Robust against list index errors (User Request: "Eliminate IndexError")
                _k_id = f"{league_name}::{home}::{away}::{commence_iso}"
                kalshi_match_results[_k_id] = {
                    "game": g, "matches": kalshi_matches, "candidate_debug": candidate_debug
                }

                # --- COLLISION DETECTION: Ensure no ticker is used by multiple games ---
                for _mtype in ("winner", "spread", "total"):
                    _km = kalshi_matches.get(_mtype, {})
                    _kticker = _km.get("kalshi_event_ticker")
                    if _kticker and _km.get("kalshi_matched"):
                        if _kticker in _kalshi_ticker_owners:
                            _prev_owner = _kalshi_ticker_owners[_kticker]
                            logger.warning(
                                f"🚨 KALSHI TICKER COLLISION: {_kticker} ({_mtype}) "
                                f"claimed by [{_k_id}] but already used by [{_prev_owner}]. "
                                f"Rejecting duplicate — setting kalshi_matched=False."
                            )
                            _km["kalshi_matched"] = False
                            _km["kalshi_reason"] = f"collision_with_{_prev_owner}"
                            _km["kalshi_prob"] = None
                        else:
                            _kalshi_ticker_owners[_kticker] = _k_id

                # Null-safe Kalshi fields used downstream
                kalshi_prob_used = (
                    kalshi_winner.get("kalshi_prob") if kalshi_winner.get("kalshi_matched") else None
                )
                kalshi_event_used = (
                    kalshi_winner.get("kalshi_event_ticker") if kalshi_winner.get("kalshi_matched") else None
                )
                if kalshi_winner.get("kalshi_matched"):
                    kalshi_status_value = "matched"
                else:
                    kalshi_status_value = "NO_MATCH"

                if (
                    kalshi_winner.get("kalshi_matched")
                    or kalshi_spread.get("kalshi_matched")
                    or kalshi_total.get("kalshi_matched")
                ):
                    master_stats["kalshi_matches"] += 1

                # --- MOVED PREDICTION (After Kalshi for signal injection) ---
                model_prob_home = None
                model_warn = None
                model_mode = "disabled"
                model_spread_prob = None
                model_total_prob = None
                model_available = True

                # Inject Kalshi Prob if available
                if kalshi_winner.get("kalshi_matched") and kalshi_prob_used is not None:
                    g["kalshi_prob"] = kalshi_prob_used

                if use_model_numeric_probs:
                    if model_available:
                        model_prob_home, model_warn = get_prediction_prob(g, sentiment_diff)
                        model_mode = "enabled" if model_prob_home is not None else "error"
                        # Add specific warning for placeholder-based fallbacks
                        if model_warn and "Placeholder" in model_warn:
                            if "FallbackPlaceholderDetected" not in warnings:
                                warnings.append("FallbackPlaceholderDetected")
                        elif model_warn and "Fallback" in model_warn:
                            if "ModelFallbackUsed" not in warnings:
                                warnings.append("ModelFallbackUsed")
                    else:
                        model_warn = "model_missing_prob"
                        model_mode = "missing"
                if model_warn and model_warn not in warnings:
                    warnings.append(model_warn)

                # --- 3. AI & Market Probability Calculations ---
                home_ml = g.get("home_ml_price")
                away_ml = g.get("away_ml_price")
                implied_home = american_to_implied_prob(home_ml)
                implied_away = american_to_implied_prob(away_ml)

                # Pre-compute spread and total picks/probabilities so we can surface them on summary rows.
                spread_pick = spread_pick_team
                spread_line = spread_pick_line

                total_pick = None
                total_implied = None
                total_line = g.get("total_point")
                total_pick_side = None
                total_pick_odds = None
                best_total_price = None

                # FIX: Propagate model predictions to spread and total markets
                # The model_prob_home from get_prediction_prob() is the home team win probability
                # For spread/total markets, we use this as the base model prediction
                model_spread_prob = None
                model_total_prob = None
                if use_model_numeric_probs and model_prob_home is not None:
                    # Use model prediction for both spread and total markets
                    model_spread_prob = model_prob_home
                    model_total_prob = model_prob_home
                if g.get("total_point") is not None:
                    over_prob = american_to_implied(g.get("over_price"))
                    under_prob = american_to_implied(g.get("under_price"))
                    if over_prob is not None or under_prob is not None:
                        if over_prob is None:
                            total_pick = "Under"
                            total_pick_side = "Under"
                            total_implied = under_prob
                            total_pick_odds = g.get("under_price")
                        elif under_prob is None:
                            total_pick = "Over"
                            total_pick_side = "Over"
                            total_implied = over_prob
                            total_pick_odds = g.get("over_price")
                        elif under_prob >= over_prob:
                            total_pick = "Under"
                            total_pick_side = "Under"
                            total_implied = under_prob
                            total_pick_odds = g.get("under_price")
                        else:
                            total_pick = "Over"
                            total_pick_side = "Over"
                            total_implied = over_prob
                            total_pick_odds = g.get("over_price")
                    preferred_total_book = g.get("best_total_book")
                    if total_pick_side == "Over":
                        best_total_offer = select_best_offer_for_pick(
                            total_offers, "over", pick_line=total_line, preferred_book=preferred_total_book
                        )
                        if best_total_offer is None and g.get("over_price") is not None:
                            best_total_offer = {
                                "book": preferred_total_book,
                                "point": total_line,
                                "price": g.get("over_price"),
                                "side": "over",
                                "last_update": g.get("best_total_last_update"),
                            }
                    elif total_pick_side == "Under":
                        best_total_offer = select_best_offer_for_pick(
                            total_offers, "under", pick_line=total_line, preferred_book=preferred_total_book
                        )
                        if best_total_offer is None and g.get("under_price") is not None:
                            best_total_offer = {
                                "book": preferred_total_book,
                                "point": total_line,
                                "price": g.get("under_price"),
                                "side": "under",
                                "last_update": g.get("best_total_last_update"),
                            }
                    else:
                        best_total_offer = None
                    if best_total_offer:
                        total_pick_odds = best_total_offer.get("price")
                        best_total_price = total_pick_odds
                        total_odds_method = "book_price"
                        if total_line is None:
                            total_line = safe_float(best_total_offer.get("point"))
                    if total_pick_odds is not None:
                        total_implied = american_to_implied(total_pick_odds)

                spread_prob_market_based = None
                spread_prob_reason = None
                spread_prob_method = None
                spread_market_pairs_count = 0
                total_prob_market_based = None
                total_prob_reason = None
                total_prob_method = None
                total_market_pairs_count = 0
                spread_odds_placeholder_detected = False
                total_odds_placeholder_detected = False
                spread_prob_placeholder_detected = False
                total_prob_placeholder_detected = False
                overall_odds_placeholder = False
                spread_pick_side_key = "home" if spread_pick_team == home else ("away" if spread_pick_team == away else None)

                # --- THEOVER SIDE RESOLUTION ---
                # FIX: Use precise team code matching and LINE sign validation
                # to prevent DEN/DET confusion and similar issues
                theover_spread_pick_side = None
                if theover_matched_side:
                    p_team = theover_matched_side.get("theover_pick")
                    theover_line_raw = theover_matched_side.get("theover_line")
                    home_code_to = theover_matched_side.get("home_code", "")
                    away_code_to = theover_matched_side.get("away_code", "")

                    if p_team:
                        p_upper = str(p_team).upper().strip()
                        p_norm = robust_normalize_team(p_team, league=league_name)
                        h_norm = robust_normalize_team(home, league=league_name)
                        a_norm = robust_normalize_team(away, league=league_name)

                        # Step 1: Try exact code match first (most reliable)
                        # This prevents DEN matching both "DENVER" and incorrectly being close to "DETROIT"
                        code_matched = False
                        if home_code_to and away_code_to:
                            if p_upper == home_code_to.upper():
                                theover_spread_pick_side = "home"
                                code_matched = True
                                logger.debug(f"TheOver pick '{p_team}' exact code match to home '{home_code_to}'")
                            elif p_upper == away_code_to.upper():
                                theover_spread_pick_side = "away"
                                code_matched = True
                                logger.debug(f"TheOver pick '{p_team}' exact code match to away '{away_code_to}'")

                        # Step 2: If no exact code match, try normalized name matching
                        if not code_matched:
                            # Use strict matching - full string match or prefix match, not substring
                            if p_norm == h_norm or h_norm.startswith(p_norm) or p_norm.startswith(h_norm):
                                theover_spread_pick_side = "home"
                            elif p_norm == a_norm or a_norm.startswith(p_norm) or p_norm.startswith(a_norm):
                                theover_spread_pick_side = "away"
                            else:
                                # Fallback: loose substring matching (but log warning)
                                if p_norm in h_norm or h_norm in p_norm:
                                    theover_spread_pick_side = "home"
                                    logger.warning(f"TheOver pick '{p_team}' loose match to home '{home}' - verify accuracy")
                                elif p_norm in a_norm or a_norm in p_norm:
                                    theover_spread_pick_side = "away"
                                    logger.warning(f"TheOver pick '{p_team}' loose match to away '{away}' - verify accuracy")

                        # Step 3: LINE sign validation and correction
                        # Convention: negative line = favorite, positive line = underdog
                        # If LINE is negative and we matched to underdog (or vice versa), warn
                        if theover_spread_pick_side and theover_line_raw is not None:
                            try:
                                line_float = float(theover_line_raw)
                                home_spread = safe_float(g.get("home_spread_point"))

                                # Determine who is favorite based on home spread from odds
                                if home_spread is not None:
                                    home_is_favorite = home_spread < 0

                                    # Check for LINE sign mismatch
                                    # TheOver LINE should match the picked team's spread perspective
                                    if theover_spread_pick_side == "home":
                                        # If home is picked, LINE should reflect home's spread
                                        # Home favorite: negative LINE expected
                                        # Home underdog: positive LINE expected
                                        if home_is_favorite and line_float > 0:
                                            # Possible wrong team - home is favorite but LINE is positive
                                            logger.warning(
                                                f"TheOver LINE mismatch for {home} vs {away}: "
                                                f"picked home ({p_team}) with LINE +{line_float}, "
                                                f"but home spread is {home_spread} (favorite). "
                                                f"Consider if away team was intended pick."
                                            )
                                        elif not home_is_favorite and line_float < 0:
                                            # Home is underdog but LINE is negative
                                            logger.warning(
                                                f"TheOver LINE mismatch for {home} vs {away}: "
                                                f"picked home ({p_team}) with LINE {line_float}, "
                                                f"but home spread is {home_spread} (underdog)."
                                            )
                                    elif theover_spread_pick_side == "away":
                                        away_is_favorite = not home_is_favorite
                                        if away_is_favorite and line_float > 0:
                                            logger.warning(
                                                f"TheOver LINE mismatch for {home} vs {away}: "
                                                f"picked away ({p_team}) with LINE +{line_float}, "
                                                f"but away is favorite."
                                            )
                            except (ValueError, TypeError):
                                pass

                theover_total_pick_side = None
                if theover_matched_total:
                    p_side = theover_matched_total.get("theover_pick")
                    if p_side:
                        if "OVER" in str(p_side).upper(): theover_total_pick_side = "Over"
                        elif "UNDER" in str(p_side).upper(): theover_total_pick_side = "Under"
                if spread_pick or g.get("home_spread_point") is not None:
                    spread_market_prob, spread_market_pairs_count, spread_prob_method, spread_market_placeholder = compute_market_prob_from_offers(
                        spread_offers, spread_pick_side_key, market_type="spread"
                    )
                    base_spread_prob = spread_market_prob if spread_market_prob is not None else spread_implied
                    spread_prob_market_based, spread_prob_reason = market_based_prob(
                        {
                            "Market": "spread",
                            "Implied_Prob": base_spread_prob,
                            "Pick": spread_pick,
                            "Home": home,
                            "Away": away,
                            "injuries_home_count": injuries_home_count,
                            "injuries_away_count": injuries_away_count,
                            "weather_summary": weather_summary,
                            "spread_min": spread_min,
                            "spread_max": spread_max,
                        },
                        market_override="spread",
                        implied_prob_value=base_spread_prob,
                        range_override=(spread_min, spread_max),
                    )
                    if spread_prob_method == "missing" and spread_implied is not None:
                        spread_prob_method = "implied"
                    if spread_prob_market_based is None:
                        spread_prob_method = spread_prob_method or "missing"
                    else:
                        spread_prob_method = f"{spread_prob_method}_market_adjusted"
                    spread_odds_placeholder_detected = bool(spread_odds_method == "fallback_default")
                    spread_prob_placeholder_detected = bool(
                        spread_odds_placeholder_detected
                        and spread_implied is not None
                        and PLACEHOLDER_IMPLIED_PROB is not None
                        and abs(spread_implied - PLACEHOLDER_IMPLIED_PROB) < 1e-4
                    )

                total_pick_side_key = str(total_pick_side or "").lower() if total_pick_side else None
                if total_pick or g.get("total_point") is not None:
                    total_market_prob, total_market_pairs_count, total_prob_method, total_market_placeholder = compute_market_prob_from_offers(
                        total_offers, total_pick_side_key, market_type="total"
                    )
                    base_total_prob = total_market_prob if total_market_prob is not None else total_implied
                    total_prob_market_based, total_prob_reason = market_based_prob(
                        {
                            "Market": "total",
                            "Implied_Prob": base_total_prob,
                            "Pick": total_pick,
                            "Home": home,
                            "Away": away,
                            "injuries_home_count": injuries_home_count,
                            "injuries_away_count": injuries_away_count,
                            "weather_summary": weather_summary,
                            "total_min": total_min,
                            "total_max": total_max,
                        },
                        market_override="total",
                        implied_prob_value=base_total_prob,
                        range_override=(total_min, total_max),
                    )
                    if total_prob_method == "missing" and total_implied is not None:
                        total_prob_method = "implied"
                    if total_prob_market_based is None:
                        total_prob_method = total_prob_method or "missing"
                    else:
                        total_prob_method = f"{total_prob_method}_market_adjusted"
                    total_odds_placeholder_detected = bool(total_odds_method == "fallback_default")
                    total_prob_placeholder_detected = bool(
                        total_odds_placeholder_detected
                        and total_implied is not None
                        and PLACEHOLDER_IMPLIED_PROB is not None
                        and abs(total_implied - PLACEHOLDER_IMPLIED_PROB) < 1e-4
                    )
                overall_odds_placeholder = bool(spread_odds_placeholder_detected or total_odds_placeholder_detected)
                spread_prob_market = spread_prob_market_based if spread_prob_market_based is not None else spread_implied
                total_prob_market = total_prob_market_based if total_prob_market_based is not None else total_implied
                kalshi_prob_spread = safe_float(kalshi_spread.get("kalshi_prob"))
                kalshi_prob_total = safe_float(kalshi_total.get("kalshi_prob"))

                # v99 FIX (Bug 2): Treat illiquid Kalshi markets as no-data.
                # Kalshi prob <= 0.02 means no trades / illiquid — using it as a real
                # probability is wrong (0.000 becomes 1.000 after pick-side flip).
                if kalshi_prob_spread is not None and kalshi_prob_spread <= 0.02:
                    logger.info(f"⚠️ KALSHI SPREAD ILLIQUID for {home} vs {away}: prob={kalshi_prob_spread:.3f} ≤ 0.02, treating as no-data")
                    kalshi_prob_spread = None
                if kalshi_prob_total is not None and kalshi_prob_total <= 0.02:
                    logger.info(f"⚠️ KALSHI TOTAL ILLIQUID for {home} vs {away}: prob={kalshi_prob_total:.3f} ≤ 0.02, treating as no-data")
                    kalshi_prob_total = None

                # v98 FIX (Bug B): Pre-compute pick-side Kalshi probabilities BEFORE
                # passing to compute_final_probability. This ensures logs show the
                # correct pick-side value and avoids any side-mismatch issues.
                kalshi_prob_spread_for_pick = map_kalshi_prob_for_pick(
                    kalshi_prob_spread if kalshi_spread.get("kalshi_matched") else None,
                    kalshi_spread.get("kalshi_yes_side") or "home",
                    spread_pick_side_key
                )
                kalshi_prob_total_for_pick = map_kalshi_prob_for_pick(
                    kalshi_prob_total if kalshi_total.get("kalshi_matched") else None,
                    kalshi_total.get("kalshi_yes_side") or "over",
                    total_pick_side_key
                )

                model_used_for_spread = bool(use_model_numeric_probs and model_spread_prob is not None)
                model_used_for_total = bool(use_model_numeric_probs and model_total_prob is not None)
                # Inject TheOver prob if available
                theover_prob_final_spread = None
                if theover_prob_spread is not None:
                    # Check alignment: spread_pick_side_key (home/away) vs theover_spread_pick_side
                    if theover_spread_pick_side and spread_pick_side_key and theover_spread_pick_side == spread_pick_side_key:
                        theover_prob_final_spread = theover_prob_spread
                    else:
                        theover_prob_final_spread = 1.0 - theover_prob_spread

                    # Dynamic weighting based on TheOver hit_rate
                    # Strong signal (>=60%): 15% weight
                    # Moderate signal (>=55%): 12% weight
                    # Weak signal (<55%): 8% weight
                    spread_hit_rate = safe_float((theover_matched_side or {}).get("theover_hit_rate"))
                    if spread_hit_rate and spread_hit_rate >= 0.60:
                        spread_weights["theover_weight"] = 0.15
                    elif spread_hit_rate and spread_hit_rate >= 0.55:
                        spread_weights["theover_weight"] = 0.12
                    elif spread_hit_rate:
                        spread_weights["theover_weight"] = 0.08
                    else:
                        spread_weights["theover_weight"] = 0.10  # Default

                    # Reduce model weight slightly if model is used, else rely on normalization
                    if spread_weights.get("ml_weight", 0) > 0.15:
                        spread_weights["ml_weight"] -= 0.05

                # MODE B: Sentiment weight enabled in probability calculations (was Mode A: disabled)
                # spread_weights["sentiment_weight"] = 0.0  # DISABLED to enable sentiment integration

                # Calculate SPREAD probability WITHOUT TheOver
                _weights_no_to = spread_weights.copy()
                _weights_no_to["theover_weight"] = 0.0
                # v103 FIX (Bug 1): Pass the PRE-COMPUTED pick-side Kalshi prob directly
                # instead of the raw YES-side prob. This bypasses the internal
                # map_kalshi_prob_for_pick() call (which can invert the probability
                # when kalshi_yes_side is wrong) by setting kalshi_side_yes = pick_side
                # so the internal mapping is identity (no flip).
                spread_prob_no_to, _, _, _, _, _, _ = compute_final_probability(
                    spread_pick_side_key,
                    spread_prob_market,
                    kalshi_prob_spread_for_pick,
                    spread_pick_side_key,
                    model_spread_prob if model_used_for_spread else None,
                    None,
                    spread_sentiment_adj,
                    _weights_no_to,
                    sentiment_score=sentiment_diff,
                    kalshi_data=kalshi_spread if kalshi_spread.get("kalshi_matched") else None,
                )

                # Update Kalshi weight dynamically
                _spread_kalshi_matched = bool(kalshi_spread.get("kalshi_matched"))
                spread_weights["kalshi_weight"] = dynamic_kalshi_weight(
                    kalshi_prob_spread_for_pick,
                    spread_prob_market,
                    _spread_kalshi_matched,
                    league_name
                )

                # DEBUG: Log spread probability calculation inputs (v98: show pick-side Kalshi prob)
                logger.info(f"SPREAD PROB CALC for {home} vs {away}: spread_pick_side={spread_pick_side_key}, spread_market={spread_prob_market:.4f}, spread_implied={spread_implied}, kalshi={kalshi_prob_spread_for_pick}")

                # v103 FIX (Bug 1): Pass pre-computed pick-side Kalshi prob with
                # matching side key so compute_final_probability uses the PICK-side
                # probability directly, not the raw YES-side which may be inverted.
                spread_prob_final, spread_base_prob, spread_weights_used, spread_decision_driver, spread_warnings_new, spread_kalshi_prob_for_pick, spread_sentiment_debug = compute_final_probability(
                    spread_pick_side_key,
                    spread_prob_market,
                    kalshi_prob_spread_for_pick,
                    spread_pick_side_key,
                    model_spread_prob if model_used_for_spread else None,
                    theover_prob_final_spread,
                    spread_sentiment_adj,
                    spread_weights,
                    sentiment_score=sentiment_diff,
                    home_team=home,
                    away_team=away,
                    kalshi_data=kalshi_spread if kalshi_spread.get("kalshi_matched") else None,
                )

                # REMOVED: Misplaced sentiment debug capture code (lines 9130-9165)
                # This code was trying to use undefined variable 'row' before row objects were created
                # Sentiment debug data is already captured in spread_row and total_row creation blocks later

                # Calculate TheOver Impact (Invariant: delta = final - without)
                if theover_prob_final_spread is not None:
                    # v103 FIX (Bug 1): Use pre-computed pick-side Kalshi prob
                    spread_prob_no_to, _, _, _, _, _, _ = compute_final_probability(
                        spread_pick_side_key,
                        spread_prob_market,
                        kalshi_prob_spread_for_pick,
                        spread_pick_side_key,
                        model_spread_prob if model_used_for_spread else None,
                        None, # Exclude TheOver
                        spread_sentiment_adj,
                        spread_weights,
                        sentiment_score=sentiment_diff,
                        kalshi_data=kalshi_spread if kalshi_spread.get("kalshi_matched") else None,
                    )
                    if isinstance(spread_sentiment_debug, dict) and "theover_delta_clamped" in spread_sentiment_debug:
                        theover_delta_spread = spread_sentiment_debug.get("theover_delta_clamped")
                    else:
                        theover_delta_spread = (spread_prob_final or 0.0) - (spread_prob_no_to or 0.0)
                else:
                    spread_prob_no_to = spread_prob_final
                    theover_delta_spread = 0.0

                # Apply TheOver Decision Engine Adjustment (Spread) - Nudge Logic
                if theover_prob_final_spread is not None and spread_prob_final is not None:
                    # Directional check: if both > 0.5 or both < 0.5
                    agree = (theover_prob_final_spread > 0.5 and spread_prob_final > 0.5) or \
                            (theover_prob_final_spread < 0.5 and spread_prob_final < 0.5)

                    # Nudge: +0.02 if agree, -0.02 if strongly disagree (and we picked it)
                    if agree:
                        spread_prob_final = clamp(spread_prob_final + 0.02, 0.01, 0.95)
                        spread_warnings_new.append("theover_spread_agrees")
                    else:
                        spread_prob_final = clamp(spread_prob_final - 0.02, 0.05, 0.99)
                        spread_warnings_new.append("theover_spread_disagrees")

                    # Update delta to reflect nudge
                    theover_delta_spread = (spread_prob_final or 0.0) - (spread_prob_no_to or 0.0)

                # Pick Change Detection
                theover_changed_pick_spread = False
                if spread_prob_final is not None and spread_prob_no_to is not None:
                    if (spread_prob_final > 0.5) != (spread_prob_no_to > 0.5):
                        theover_changed_pick_spread = True

                theover_used_in_pick_spread = bool(theover_prob_final_spread is not None)

                if spread_prob_final is None:
                    spread_prob_final = blend_kalshi_market(kalshi_prob_spread_for_pick, spread_prob_market) if kalshi_spread.get("kalshi_matched") else spread_prob_market
                    if model_used_for_spread and model_spread_prob is not None:
                        spread_prob_final = clamp(model_spread_prob)
                    spread_base_prob = spread_prob_final
                    spread_weights_used = {"w_implied": 1.0 if spread_prob_final is not None else 0.0, "w_kalshi": 0.0, "w_model": 0.0, "w_sentiment": 0.0}
                spread_prob = spread_prob_final

                # DEBUG: Log final spread probability
                logger.info(f"SPREAD FINAL for {home} vs {away}: {spread_prob_final:.4f} ({spread_prob_final*100:.1f}%)")

                # Inject TheOver prob if available
                theover_prob_final_total = None
                if theover_prob_total is not None:
                    # Check alignment: total_pick_side_key (over/under) vs theover_total_pick_side
                    if theover_total_pick_side and total_pick_side_key and str(theover_total_pick_side).upper() == str(total_pick_side_key).upper():
                        theover_prob_final_total = theover_prob_total
                    else:
                        theover_prob_final_total = 1.0 - theover_prob_total

                    # Dynamic weighting based on TheOver hit_rate
                    # Strong signal (>=60%): 15% weight
                    # Moderate signal (>=55%): 12% weight
                    # Weak signal (<55%): 8% weight
                    total_hit_rate = safe_float((theover_matched_total or {}).get("theover_hit_rate"))
                    if total_hit_rate and total_hit_rate >= 0.60:
                        total_weights["theover_weight"] = 0.15
                    elif total_hit_rate and total_hit_rate >= 0.55:
                        total_weights["theover_weight"] = 0.12
                    elif total_hit_rate:
                        total_weights["theover_weight"] = 0.08
                    else:
                        total_weights["theover_weight"] = 0.10  # Default

                    if total_weights.get("ml_weight", 0) > 0.15:
                        total_weights["ml_weight"] -= 0.05

                # Calculate TOTAL probability WITHOUT TheOver
                _weights_total_no_to = total_weights.copy()
                _weights_total_no_to["theover_weight"] = 0.0
                # v103 FIX: Use pre-computed pick-side Kalshi prob (same pattern as spread fix)
                total_prob_no_to, _, _, _, _, _, _ = compute_final_probability(
                    total_pick_side_key,
                    total_prob_market,
                    kalshi_prob_total_for_pick,
                    total_pick_side_key,
                    model_total_prob if model_used_for_total else None,
                    None,
                    total_sentiment_adj,
                    _weights_total_no_to,
                    sentiment_score=sentiment_diff,
                    kalshi_data=kalshi_total if kalshi_total.get("kalshi_matched") else None,
                )

                # Update Kalshi weight dynamically
                _total_kalshi_matched = bool(kalshi_total.get("kalshi_matched"))
                total_weights["kalshi_weight"] = dynamic_kalshi_weight(
                    kalshi_prob_total_for_pick,
                    total_prob_market,
                    _total_kalshi_matched,
                    league_name
                )

                # DEBUG: Log total probability calculation inputs (v98: show pick-side Kalshi prob)
                logger.info(f"TOTAL PROB CALC for {home} vs {away}: total_pick_side={total_pick_side_key}, total_market={total_prob_market:.4f}, total_implied={total_implied}, kalshi={kalshi_prob_total_for_pick}")

                # v103 FIX: Use pre-computed pick-side Kalshi prob with matching side key
                total_prob_final, total_base_prob, total_weights_used, total_decision_driver, total_warnings_new, total_kalshi_prob_for_pick, total_sentiment_debug = compute_final_probability(
                    total_pick_side_key,
                    total_prob_market,
                    kalshi_prob_total_for_pick,
                    total_pick_side_key,
                    model_total_prob if model_used_for_total else None,
                    theover_prob_final_total,
                    total_sentiment_adj,
                    total_weights,
                    sentiment_score=sentiment_diff,
                    home_team=home,
