# IMPROVED KALSHI MATCHING LOGIC
# This fixes the issue where Kalshi markets exist but aren't being matched correctly

def validate_with_kalshi(kalshi_integrator, home_team: str, away_team: str, 
                        side: str, sportsbook_prob: float, sport: str) -> Dict:
    """
    IMPROVED: Validate sportsbook odds with Kalshi prediction market
    
    Args:
        kalshi_integrator: KalshiIntegrator instance
        home_team: Home team name
        away_team: Away team name
        bet_type: 'spread', 'moneyline', 'total_over', 'total_under'
        bet_team: Which team you're betting on (for spread/ML)
        sportsbook_prob: Sportsbook implied probability
        sport: 'NFL', 'NBA', etc.
    
    Returns:
        Dict with kalshi_prob, validation, etc.
    """
    try:
        markets = kalshi_integrator.get_sports_markets()
        
        # IMPROVED MATCHING LOGIC
        for market in markets:
            title = market.get('title', '').upper()
            ticker = market.get('ticker', '').upper()
            subtitle = market.get('subtitle', '').upper()
            
            # Check if this market is about the right game
            # Need BOTH teams mentioned to ensure it's the right game
            has_home = home_team.upper() in title or home_team.upper() in ticker
            has_away = away_team.upper() in title or away_team.upper() in ticker
            
            if not (has_home and has_away):
                continue  # Not the right game
            
            # Now determine which side of the Kalshi market to use
            # Kalshi markets are typically binary: "Will [Team] win?" or "Will [Team] cover?"
            
            # Get orderbook
            orderbook = kalshi_integrator.get_orderbook(market.get('ticker', ''))
            if not orderbook:
                continue
            
            yes_bids = orderbook.get('yes', [])
            no_bids = orderbook.get('no', [])
            
            if not yes_bids:
                continue
            
            # Determine which probability to use based on bet type and team
            kalshi_prob = None
            
            # CASE 1: Moneyline bet
            if bet_type == 'moneyline':
                # Kalshi market like "Will [Team] win this game?"
                # Need to check which team the YES side represents
                
                if bet_team.upper() in title:
                    # If betting on the team in the market title, use YES probability
                    kalshi_prob = yes_bids[0].get('price', 0) / 100
                else:
                    # If betting on the other team, use NO probability (or 1 - YES)
                    if no_bids:
                        kalshi_prob = no_bids[0].get('price', 0) / 100
                    else:
                        kalshi_prob = 1.0 - (yes_bids[0].get('price', 0) / 100)
            
            # CASE 2: Spread bet
            elif bet_type == 'spread':
                # Kalshi market might be "Will [Team] cover the spread?"
                # Or "Will [Team] win by more than X points?"
                
                if 'COVER' in title or 'SPREAD' in title:
                    # Market is about covering the spread
                    if bet_team.upper() in title:
                        kalshi_prob = yes_bids[0].get('price', 0) / 100
                    else:
                        if no_bids:
                            kalshi_prob = no_bids[0].get('price', 0) / 100
                        else:
                            kalshi_prob = 1.0 - (yes_bids[0].get('price', 0) / 100)
                elif 'WIN' in title:
                    # Market is about winning outright - less precise for spread bets
                    # Use it anyway but with lower confidence
                    if bet_team.upper() in title:
                        kalshi_prob = yes_bids[0].get('price', 0) / 100
                    else:
                        if no_bids:
                            kalshi_prob = no_bids[0].get('price', 0) / 100
                        else:
                            kalshi_prob = 1.0 - (yes_bids[0].get('price', 0) / 100)
            
            # CASE 3: Totals (over/under)
            elif bet_type in ['total_over', 'total_under']:
                # Kalshi market like "Will total points be over X?"
                if 'OVER' in title or 'TOTAL' in title:
                    if bet_type == 'total_over':
                        kalshi_prob = yes_bids[0].get('price', 0) / 100
                    else:  # total_under
                        if no_bids:
                            kalshi_prob = no_bids[0].get('price', 0) / 100
                        else:
                            kalshi_prob = 1.0 - (yes_bids[0].get('price', 0) / 100)
            
            # If we found a matching probability, validate it
            if kalshi_prob is not None:
                discrepancy = abs(kalshi_prob - sportsbook_prob)
                
                # Determine validation status
                if discrepancy < 0.05:  # Within 5%
                    validation = 'confirms'
                    confidence_boost = 0.10
                    edge = 0
                elif discrepancy < 0.10:  # 5-10% difference
                    if kalshi_prob > sportsbook_prob:
                        validation = 'kalshi_higher'
                        confidence_boost = 0.05
                        edge = kalshi_prob - sportsbook_prob
                    else:
                        validation = 'kalshi_lower'
                        confidence_boost = -0.05
                        edge = sportsbook_prob - kalshi_prob
                else:  # >10% difference
                    if kalshi_prob > sportsbook_prob:
                        validation = 'strong_kalshi_higher'
                        confidence_boost = 0.15
                        edge = kalshi_prob - sportsbook_prob
                    else:
                        validation = 'strong_contradiction'
                        confidence_boost = -0.10
                        edge = 0
                
                return {
                    'kalshi_prob': kalshi_prob,
                    'kalshi_available': True,
                    'discrepancy': discrepancy,
                    'validation': validation,
                    'edge': edge,
                    'confidence_boost': confidence_boost,
                    'market_ticker': market.get('ticker', ''),
                    'market_title': market.get('title', ''),
                    'matched': True  # NEW: Flag that we found a match
                }
        
        # No matching market found
        return {
            'kalshi_prob': None,
            'kalshi_available': False,
            'discrepancy': 0,
            'validation': 'unavailable',
            'edge': 0,
            'confidence_boost': 0,
            'market_ticker': None,
            'market_title': None,
            'matched': False  # NEW: Flag that we didn't find a match
        }
    
    except Exception as e:
        return {
            'kalshi_prob': None,
            'kalshi_available': False,
            'discrepancy': 0,
            'validation': 'error',
            'edge': 0,
            'confidence_boost': 0,
            'market_ticker': None,
            'market_title': None,
            'matched': False,
            'error': str(e)  # NEW: Include error message
        }


# USAGE EXAMPLE:
# Instead of:
#   kalshi_data = validate_with_kalshi(kalshi, home, away, 'home', base_prob, skey)
#
# Use:
#   bet_type = 'spread'  # or 'moneyline', 'total_over', 'total_under'
#   bet_team = away  # The team you're actually betting on (Washington for +13.5)
#   kalshi_data = validate_with_kalshi_improved(kalshi, home, away, bet_type, bet_team, base_prob, skey)
