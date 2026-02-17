    logger.info(f"   - Spread markets: {len(spreads)}" + (f" (sample: {spreads[0].get('ticker', 'N/A')})" if spreads else ""))
    logger.info(f"   - Unknown markets: {len(unknown)}" + (f" (sample: {unknown[0].get('ticker', 'N/A')})" if unknown else ""))

    # DEBUG: Log pricing field availability on first market to verify API response format
    if kalshi_markets:
        _sample = kalshi_markets[0]
        logger.info(f"   💲 PRICING FIELDS (sample {_sample.get('ticker', 'N/A')}): "
                     f"yes_bid_dollars={_sample.get('yes_bid_dollars')}, "
                     f"yes_ask_dollars={_sample.get('yes_ask_dollars')}, "
                     f"yes_bid={_sample.get('yes_bid')}, "
                     f"yes_ask={_sample.get('yes_ask')}, "
                     f"last_price_dollars={_sample.get('last_price_dollars')}, "
                     f"last_price={_sample.get('last_price')}")

    # FIX: Enhanced debug logging for spread/total matching
    if totals:
        logger.info(f"   📊 TOTAL MARKET DETAILS: {len(totals)} markets available")
        for t in totals[:3]:
            logger.info(f"      - {t.get('ticker')} | title: {t.get('title', '')[:40]} | last_price: {t.get('last_price')}")
    if spreads:
        logger.info(f"   📊 SPREAD MARKET DETAILS: {len(spreads)} markets available")
        for s in spreads[:3]:
            logger.info(f"      - {s.get('ticker')} | title: {s.get('title', '')[:40]} | last_price: {s.get('last_price')}")

    winner_candidate_debug: List[Dict[str, Any]] = []
    best_winner: Optional[Dict[str, Any]] = None
    best_score: Optional[float] = None
    best_reason = "no_candidates"
    candidate_count = 0
    strict_candidates: List[Tuple[float, Dict[str, Any]]] = []
    fallback_with_date: List[Tuple[float, Dict[str, Any]]] = []
    fallback_no_date: List[Tuple[float, Dict[str, Any]]] = []

    def infer_yes_side(market: Dict[str, Any]) -> Optional[str]:
        codes = kalshi_ticker_team_codes(market)
        if codes:
            first, second = codes
            if first and first == home_code_expected:
                return "home"
            if first and first == away_code_expected:
                return "away"
            if second and second == home_code_expected:
                return "home"
            if second and second == away_code_expected:
                return "away"
        return None

    home_code_candidates = [c.upper() for c in team_code_candidates(league_name, game.get("home_team"))]
    away_code_candidates = [c.upper() for c in team_code_candidates(league_name, game.get("away_team"))]

    for m in kalshi_markets or []:
        ticker_upper = str(m.get("event_ticker") or m.get("ticker") or "").upper()
        title_lower = str(m.get("title") or "").lower()
        if not (
            ticker_upper.startswith(winner_prefix)
            or "GAME-" in ticker_upper
            or "GAME" in ticker_upper
            or "winner" in title_lower
        ):
            continue
        tokens = market_tokens(m)
        date_match = bool(allowed_date_tokens and any(tok in ticker_upper for tok in allowed_date_tokens))
        home_hit = bool(home_tokens.intersection(tokens))
        away_hit = bool(away_tokens.intersection(tokens))
        code_home_hit = home_code_candidates and any(code in ticker_upper for code in home_code_candidates if len(code) >= 3)
        code_away_hit = away_code_candidates and any(code in ticker_upper for code in away_code_candidates if len(code) >= 3)
        code_hit = bool(code_home_hit and code_away_hit)
        team_hit = bool(home_hit and away_hit)

        # Enhanced Fuzzy Matching using match_team_name from prediction_engine
        if not (team_hit or code_hit):
            # 1. Try fuzzy match on market title vs home/away team names (Legacy)
            fuzzy_home = match_team_name(game.get("home_team"), [title_lower], threshold=70.0)
            fuzzy_away = match_team_name(game.get("away_team"), [title_lower], threshold=70.0)

            if fuzzy_home and fuzzy_away:
                team_hit = True

            # 2. Try RapidFuzz direct token set match (New Fallback)
            # Helps with "Lions" vs "Detroit Lions" where 'Lions' is subset
            if not team_hit and fuzz:
                home_raw = str(game.get("home_team") or "").lower()
                away_raw = str(game.get("away_team") or "").lower()

                # token_set_ratio handles subset matching well (e.g. "Lions" in "Detroit Lions")
                score_h = fuzz_scorer(home_raw, title_lower)
                score_a = fuzz_scorer(away_raw, title_lower)

                # Find MATCHTHRESHOLD
                MATCHTHRESHOLD = 60 if league_name == 'NCAAB' else 85  # Ultra-low for NCAAB
                if score_h >= MATCHTHRESHOLD and score_a >= MATCHTHRESHOLD:
                    team_hit = True

        if not (team_hit or code_hit):
            continue
        candidate_count += 1
        score = (2 if team_hit else 0) + (2 if code_hit else 0) + (1 if date_match else 0)
        debug_row = {
            "title": m.get("title"),
            "ticker": m.get("event_ticker") or m.get("ticker"),
            "liquidity": m.get("liquidity"),
            "volume": m.get("volume"),
            "open_interest": m.get("open_interest"),
            "last_price": m.get("last_price"),
            "yes_bid_dollars": m.get("yes_bid_dollars"),
            "yes_ask_dollars": m.get("yes_ask_dollars"),
            "last_price_dollars": m.get("last_price_dollars"),
            "score": score,
            "date_match": date_match,
            "home_hit": team_hit or code_home_hit,
            "away_hit": team_hit or code_away_hit,
        }
        winner_candidate_debug.append(debug_row)
        if date_match and code_hit:
            strict_candidates.append((score, m))
        elif date_match:
            fallback_with_date.append((score, m))
        else:
            fallback_no_date.append((score, m))

    if strict_candidates:
        best_score, best_winner = max(strict_candidates, key=lambda kv: kv[0])
        best_reason = "strict_match"
    elif fallback_with_date:
        best_score, best_winner = max(fallback_with_date, key=lambda kv: kv[0])
        best_reason = "fallback_title_match"
    elif fallback_no_date:
        best_score, best_winner = max(fallback_no_date, key=lambda kv: kv[0])
        best_reason = "fallback_no_date_token"

    # FORCE NCAAB MATCH if ANY spread/total found (User Request 2)
    if not best_winner and (spreads or totals) and league_name == 'NCAAB':
        if spreads:
            best_winner = sorted(spreads, key=lambda m: m.get('volume', 0) or 0, reverse=True)[0]
        elif totals:
            best_winner = sorted(totals, key=lambda m: m.get('volume', 0) or 0, reverse=True)[0]
        else:
            best_winner = kalshi_markets[0] # Should not happen given check above

        # Log with requested format
        away_tm = game.get('away_team', 'Away')
        home_tm = game.get('home_team', 'Home')
        logger.info(f"NCAAB FORCE MARKET: {best_winner.get('ticker')} {away_tm}@{home_tm}")

        best_reason = "forced_spread_total_fallback"
        best_score = 60.0 # User requested MATCHTHRESHOLD=60, so we give it 60 to pass checks

    if best_winner:
        prob = winner_prob(best_winner)
        winner_result = {
            "kalshi_available": True,
            "kalshi_label": "matched_winner",
            "kalshi_event_ticker": best_winner.get("event_ticker") or best_winner.get("ticker"),
            "kalshi_reason": best_reason,
            "kalshi_matched": True,
            "kalshi_prob": prob,
            "kalshi_market_type": "winner",
            "kalshi_match_score": best_score,
            "kalshi_ticker": best_winner.get("event_ticker") or best_winner.get("ticker"),
            "kalshi_line": None,
            "kalshi_title": best_winner.get("title"),
            "kalshi_yes_side": infer_yes_side(best_winner),
        }
    else:
        # LOG MATCH FAILURES (~line 8700 before return None)
        if not best_winner:
            logger.warning(f"❌ NO MATCH for {game.get('home_team')} vs {game.get('away_team')}")
            logger.warning(f"   Available market team names: {[m.get('ticker', m.get('title', '')) for m in kalshi_markets[:5]]}")
            logger.warning(f"   Game teams: home='{normalize_team_name(game.get('home_team'))}' away='{normalize_team_name(game.get('away_team'))}'")
            logger.warning(f"   Original failure log: {game.get('away_team')}@{game.get('home_team')} | bestscore={best_score if 'best_score' in locals() else 0} | markets={len(kalshi_markets)}")

        no_reason = winner_reason_override or best_reason or "no_winner_market_for_game"
        winner_result = base_result(no_reason, "winner")

    def simple_select(markets: List[Dict[str, Any]], market_type: str) -> Dict[str, Any]:
        if not markets:
            return base_result(f"no_{market_type}_market", market_type)

        # Task 1.2: Improved Kalshi Contract Selection for Spreads/Totals
        # Instead of blindly taking markets[0], score candidates and infer YES side intelligently

        # Get sportsbook consensus line for line-proximity scoring
        sportsbook_total_line = safe_float(game.get("total_point"))
        sportsbook_spread_line = safe_float(game.get("home_spread_point"))

        scored_candidates = []
        for market in markets:
            score = 0.0
            ticker_upper = str(market.get("event_ticker") or market.get("ticker") or "").upper()

            # Scoring: Prefer markets that match date token
            if allowed_date_tokens and any(tok in ticker_upper for tok in allowed_date_tokens):
                score += 100.0

            # Prefer markets with team code matches (require 3+ char codes to prevent false positives)
            if home_code_candidates and away_code_candidates:
                code_home_hit = any(code in ticker_upper for code in home_code_candidates if len(code) >= 3)
                code_away_hit = any(code in ticker_upper for code in away_code_candidates if len(code) >= 3)
                if code_home_hit and code_away_hit:
                    score += 50.0

            # Line-proximity scoring: strongly prefer markets whose line matches the sportsbook
            # Without this, the system picks an arbitrary line (e.g., Over 250 instead of Over 215)
            # which produces extreme probabilities (5% for Over 250 is correct but irrelevant)
            market_line = safe_float(market.get("floor_strike") or market.get("cap_strike"))
            if market_line is not None and market_type in ("total", "spread"):
                ref_line = sportsbook_total_line if market_type == "total" else sportsbook_spread_line
                if ref_line is not None:
                    line_diff = abs(market_line - ref_line)
                    if line_diff <= 0.5:
                        score += 300.0  # Exact match (within rounding)
                    elif line_diff <= 2.0:
                        score += 250.0  # Very close
                    elif line_diff <= 5.0:
                        score += 200.0 - (line_diff * 10)  # Close, decreasing bonus
                    elif line_diff <= 10.0:
                        score += 100.0  # Moderate distance
                    # Lines > 10 points away get no line bonus

            scored_candidates.append((score, market))

        # Select best-scoring candidate
        if scored_candidates:
            best_score, chosen = max(scored_candidates, key=lambda x: x[0])
        else:
            chosen = None
            best_score = 0.0

        # MINIMUM SCORE THRESHOLD: Prevent phantom Kalshi matches from cross-game contamination.
        # Score breakdown: date match=100, both team codes=50, line proximity=up to 300.
        # Without a date match (score < 100), the market is likely for a different game/date.
        # A date-only match (100) without team codes is still weak evidence, but date+line
        # (400) is acceptable since line proximity validates the match.
        MINIMUM_MATCH_SCORE = 100.0
        if chosen is None or best_score < MINIMUM_MATCH_SCORE:
            logger.info(f"⚠️ KALSHI {market_type.upper()} REJECTED: score={best_score:.0f} below threshold {MINIMUM_MATCH_SCORE} "
                        f"(candidates={len(scored_candidates)})")
            return base_result(f"no_{market_type}_market_below_threshold", market_type)

        prob, line = extract_prob_and_line(chosen, market_type)

        # Task 1.2: Intelligent YES side inference
        if market_type == "total":
            # For total markets, YES side is "over" or "under" — determine from ticker/title
            # DO NOT use infer_yes_side() which returns "home"/"away" from team codes
            # FIX: Use individual market ticker first (contains "OVER"/"UNDER" suffix),
            # then fall back to event_ticker (which is just the event-level identifier
            # like KXNBATOTAL-26FEB08BOSSEA and never contains Over/Under direction)
            chosen_ticker = str(chosen.get("ticker") or chosen.get("event_ticker") or "").upper()
            chosen_title = str(chosen.get("title") or "").lower()
            if "UNDER" in chosen_ticker or "under" in chosen_title:
                yes_side_inferred = "under"
            else:
                yes_side_inferred = "over"  # Default for totals (YES = Over is standard)
        else:
            # For winner/spread markets, parse team code from individual contract ticker suffix.
            # Individual tickers look like: KXNBASPREAD-26FEB10INDNYK-NYK27
            # The suffix after the last dash (NYK27) contains the team code for the YES side.
            # Strip trailing digits/dots to get the team code (NYK).
            individual_ticker = str(chosen.get("ticker") or "").upper()
            suffix = individual_ticker.rsplit("-", 1)[-1] if "-" in individual_ticker else ""
            suffix_team = re.sub(r'[\d.]+$', '', suffix).strip()

            if suffix_team and len(suffix_team) >= 2:
                home_codes_upper = {c.upper() for c in (home_code_candidates or [])}
                away_codes_upper = {c.upper() for c in (away_code_candidates or [])}
                if suffix_team in home_codes_upper:
                    yes_side_inferred = "home"
                    logger.info(f"  → yes_side=home from ticker suffix '{suffix_team}' matching home_codes={home_codes_upper}")
                elif suffix_team in away_codes_upper:
                    yes_side_inferred = "away"
                    logger.info(f"  → yes_side=away from ticker suffix '{suffix_team}' matching away_codes={away_codes_upper}")
                else:
                    # Suffix didn't match known codes, fall back to legacy inference
                    yes_side_inferred = infer_yes_side(chosen)
                    if not yes_side_inferred:
                        yes_side_inferred = "home"
                    logger.info(f"  → yes_side={yes_side_inferred} (suffix '{suffix_team}' unmatched, legacy fallback)")
            else:
                # No parseable suffix, fall back to legacy inference
                yes_side_inferred = infer_yes_side(chosen)
                if not yes_side_inferred:
                    yes_side_inferred = "home"
                logger.info(f"  → yes_side={yes_side_inferred} (no suffix parsed from '{individual_ticker}', legacy fallback)")

        # Add debug logging - use info level for visibility
        prob_str = f"{prob:.3f}" if prob else "N/A"
        logger.info(f"✅ KALSHI {market_type.upper()} MATCH: ticker={chosen.get('ticker') or chosen.get('event_ticker')}, "
                    f"prob={prob_str}, line={line}, yes_side={yes_side_inferred}, score={best_score}")

        # Warn if selected Kalshi line is far from sportsbook line
        if market_type == "total" and line is not None and sportsbook_total_line is not None:
            line_gap = abs(line - sportsbook_total_line)
            if line_gap > 5.0:
                logger.warning(f"⚠️ KALSHI LINE MISMATCH: Kalshi line={line} vs Sportsbook line={sportsbook_total_line} (gap={line_gap:.1f})")
        elif market_type == "spread" and line is not None and sportsbook_spread_line is not None:
            line_gap = abs(line - sportsbook_spread_line)
            if line_gap > 3.0:
                logger.warning(f"⚠️ KALSHI LINE MISMATCH: Kalshi line={line} vs Sportsbook line={sportsbook_spread_line} (gap={line_gap:.1f})")

        return {
