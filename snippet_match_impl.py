

def _match_kalshi_market_impl(
    game: Dict[str, Any],
    kalshi_markets: List[Dict[str, Any]],
    winner_reason_override: Optional[str] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    logger.info(f"🟢 _match_kalshi_market_impl: received {len(kalshi_markets)} markets")
    # Use fuzzy matching for team names
    if rapidfuzz is not None:
        fuzz_scorer = fuzz.token_set_ratio
    else:
        # Fallback if rapidfuzz missing
        fuzz_scorer = lambda s1, s2: 100 if s1 in s2 or s2 in s1 else 0

    def base_result(reason: str, market_type: str) -> Dict[str, Any]:
        return {
            "kalshi_available": bool(kalshi_integrator),
            "kalshi_label": None,
            "kalshi_event_ticker": None,
            "kalshi_reason": reason,
            "kalshi_matched": False,
            "kalshi_prob": None,
            "kalshi_market_type": market_type,
            "kalshi_match_score": None,
            "kalshi_ticker": None,
            "kalshi_line": None,
            "kalshi_title": None,
        }

    def norm_team(name: Any) -> str:
        return re.sub(r"[^a-z0-9 ]", "", str(name or "").lower()).strip()

    def league_from_game(g: Dict[str, Any]) -> str:
        skey = (g.get("sport_key") or g.get("league") or g.get("League") or "").lower()
        mapping = {
            "basketball_nba": "NBA",
            "nba": "NBA",
            "basketball_ncaab": "NCAAB",
            "ncaab": "NCAAB",
            "americanfootball_nfl": "NFL",
            "nfl": "NFL",
            "americanfootball_ncaaf": "NCAAF",
            "ncaaf": "NCAAF",
            "icehockey_nhl": "NHL",
            "nhl": "NHL",
            "baseball_mlb": "MLB",
            "mlb": "MLB",
        }
        return mapping.get(skey, skey.upper())

    def _kalshi_prices(market: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Read Kalshi bid/ask/last prices, returning (yes_bid, yes_ask, no_bid, last_price) all normalized to 0-1.

        Prefers the current *_dollars string fields (already in 0-1 dollar range).
        Falls back to deprecated integer-cent fields (divided by 100).
        """
        def _read(dollars_key: str, cents_key: str) -> Optional[float]:
            # Prefer _dollars field (string like "0.5600", already 0-1 range)
            d = safe_float(market.get(dollars_key))
            if d is not None and d > 0:
                return d
            # Fallback to deprecated integer-cent field
            c = safe_float(market.get(cents_key))
            if c is not None and c > 0:
                return c / 100.0
            return None

        return (
            _read("yes_bid_dollars", "yes_bid"),
            _read("yes_ask_dollars", "yes_ask"),
            _read("no_bid_dollars", "no_bid"),
            _read("last_price_dollars", "last_price"),
        )

    def extract_prob_and_line(
        market: Dict[str, Any], market_type: str
    ) -> Tuple[Optional[float], Optional[float]]:
        # 1. Line detection
        line = safe_float(market.get("floor_strike"))
        if line is None:
            line = safe_float(market.get("cap_strike"))
        if line is not None:
            try:
                line = float(line)
            except Exception:
                line = None

        # 2. Probability (midpoint of yes_bid and yes_ask)
        # Uses _dollars fields (current API) with fallback to cent fields (deprecated)
        yes_bid, yes_ask, no_bid, last_price = _kalshi_prices(market)

        prob = None

        if yes_bid is not None and yes_ask is not None:
            # Direct midpoint (values already normalized to 0-1)
            prob = (yes_bid + yes_ask) / 2.0

        elif yes_bid is not None and no_bid is not None:
            # Implied ask = 1.0 - no_bid (values already 0-1)
            implied_yes_ask = 1.0 - no_bid
            prob = (yes_bid + implied_yes_ask) / 2.0

        elif yes_bid is not None:
            # Prefer last_price (actual trade) over yes_bid (lowest buy offer)
            # On thin markets, yes_bid is heavily biased low (e.g. 0.05) while
            # last_price reflects the most recent agreed-upon fair value
            if last_price is not None and last_price > 0:
                prob = last_price
            else:
                prob = yes_bid

        elif no_bid is not None:
            # Fallback if YES bid missing (Prob YES = 1 - Prob NO)
            prob = 1.0 - no_bid

        # Final fallback to last_price if all bids empty
        if prob is None and last_price is not None and last_price > 0:
            prob = last_price

        return clamp(prob, 0.0, 1.0), line

    def winner_score(market: Dict[str, Any]) -> float:
        for key in ["liquidity_dollars", "liquidity", "volume", "open_interest", "last_price_dollars", "last_price"]:
            try:
                val = float(market.get(key))
                if val is not None:
                    return val
            except Exception:
                continue
        return 0.0

    def winner_prob(market: Dict[str, Any]) -> Optional[float]:
        # Use midpoint of yes_bid and yes_ask (same logic as extract_prob_and_line)
        # Reads _dollars fields (current API) with fallback to cent fields (deprecated)
        yes_bid, yes_ask, no_bid, last_price = _kalshi_prices(market)

        prob = None

        if yes_bid is not None and yes_ask is not None:
            # Direct midpoint (values already normalized to 0-1)
            prob = (yes_bid + yes_ask) / 2.0
        elif yes_bid is not None and no_bid is not None:
            # Implied ask = 1.0 - no_bid (values already 0-1)
            implied_yes_ask = 1.0 - no_bid
            prob = (yes_bid + implied_yes_ask) / 2.0
        elif yes_bid is not None:
            # Prefer last_price (actual trade) over yes_bid (lowest buy offer)
            if last_price is not None and last_price > 0:
                prob = last_price
            else:
                prob = yes_bid
        elif no_bid is not None:
            # Fallback if YES bid missing (Prob YES = 1 - Prob NO)
            prob = 1.0 - no_bid

        # Final fallback to last_price if all bids empty
        if prob is None and last_price is not None:
            prob = clamp(last_price, 0.0, 1.0)

        return clamp(prob, 0.0, 1.0) if prob is not None else None

    # Compute diagnostic fields BEFORE early returns so they're always available
    league_name = league_from_game(game)

    commence_raw = (
        game.get("commence_time_iso_utc")
        or game.get("commence_time")
        or game.get("commence_time_iso")
        or game.get("commence_time_utc")
    )
    game_dt_utc = parse_commence_to_utc(commence_raw)
    if isinstance(game_dt_utc, datetime) and game_dt_utc.tzinfo is None:
        game_dt_utc = game_dt_utc.replace(tzinfo=timezone.utc)

    tz_name = get_local_tz()
    local_tz = None
    try:
        local_tz = ZoneInfo(tz_name)
    except Exception:
        local_tz = None
    game_local = game_dt_utc.astimezone(local_tz) if (game_dt_utc and local_tz) else game_dt_utc
    base_date = game_local.date() if game_local else None
    allowed_date_tokens: List[str] = []
    if base_date:
        for delta in (-1, 0, 1):
            allowed_date_tokens.append((base_date + timedelta(days=delta)).strftime("%y%b%d").upper())
    if not allowed_date_tokens:
        token_from_local = kalshi_date_token_from_local(game.get("commence_date_local"))
        if token_from_local:
            allowed_date_tokens.append(token_from_local)
    date_token = allowed_date_tokens[1] if len(allowed_date_tokens) > 1 else (allowed_date_tokens[0] if allowed_date_tokens else None)
    winner_prefix = league_game_prefix(league_name)

    away_code_expected = team_code_for_league(league_name, game.get("away_team"))
    home_code_expected = team_code_for_league(league_name, game.get("home_team"))

    def _early_debug(reason: str) -> Dict[str, Any]:
        """Build a debug dict for early-return paths that still includes diagnostics."""
        return {
            "total": [], "spread": [], "winner": [],
            "winner_meta": {
                "expected_date_token": date_token,
                "expected_codes": {"away": away_code_expected, "home": home_code_expected},
                "winner_match_status": "early_return",
                "winner_no_match_reason": reason,
                "matched_event_ticker": None,
                "matched_ticker": None,
                "kalshi_date_token_used": date_token,
                "winner_prefix": winner_prefix,
                "strict_candidate_count": 0,
                "allowed_date_tokens": allowed_date_tokens,
            },
            "kalshi_game_prefix_used": winner_prefix,
            "kalshi_wanted_tokens": allowed_date_tokens,
        }

    if not kalshi_integrator:
        base = {t: base_result("kalshi_not_configured", t) for t in ["total", "spread", "winner"]}
        return base, _early_debug("kalshi_not_configured")
    if not kalshi_markets:
        base = {t: base_result("no_game_like_markets_in_window", t) for t in ["total", "spread", "winner"]}
        logger.warning(
            f"⚠️ KALSHI IMPL: No markets passed to matcher for "
            f"{game.get('away_team')} @ {game.get('home_team')} | "
            f"league={league_name} | prefix={winner_prefix} | dates={allowed_date_tokens} | "
            f"codes=({away_code_expected}@{home_code_expected})"
        )
        return base, _early_debug("no_game_like_markets_in_window")

    def market_tokens(market: Dict[str, Any]) -> set:
        blob = " ".join(
            [
                str(market.get("event_ticker") or market.get("ticker") or ""),
                str(market.get("title") or ""),
                str(market.get("rules") or market.get("rules_primary") or ""),
            ]
        )
        cleaned = re.sub(r"[^a-z0-9 ]", " ", blob.lower())
        return {t for t in cleaned.split() if t}

    def team_token_set(team_name: Any) -> set:
        base_tokens = team_tokens(team_name)
        codes = {c.lower() for c in team_code_candidates(league_name, team_name) or []}
        return set(base_tokens).union(codes)

    home_tokens = team_token_set(game.get("home_team"))
    away_tokens = team_token_set(game.get("away_team"))

    # Classify markets and add debug logging
    totals = [m for m in kalshi_markets if classify_kalshi_market(m) == "total"]
    spreads = [m for m in kalshi_markets if classify_kalshi_market(m) == "spread"]
    winners = [m for m in kalshi_markets if classify_kalshi_market(m) == "winner"]
    unknown = [m for m in kalshi_markets if classify_kalshi_market(m) == "unknown"]

    # DEBUG: Log market type counts and samples
    logger.info(f"📊 KALSHI MARKET CLASSIFICATION for {game.get('away_team')} @ {game.get('home_team')}:")
    logger.info(f"   Total markets received: {len(kalshi_markets)}")
    logger.info(f"   - Winner markets: {len(winners)}" + (f" (sample: {winners[0].get('ticker', 'N/A')})" if winners else ""))
    logger.info(f"   - Total markets: {len(totals)}" + (f" (sample: {totals[0].get('ticker', 'N/A')})" if totals else ""))
    logger.info(f"   - Spread markets: {len(spreads)}" + (f" (sample: {spreads[0].get('ticker', 'N/A')})" if spreads else ""))
