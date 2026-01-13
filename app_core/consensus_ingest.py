import random
import logging
from typing import Dict, Any, List

logger = logging.getLogger("consensus_ingest")

class ConsensusIngest:
    """
    Simulates fetching 'Consensus' data (Ticket % vs Money %) for games.
    In a real scenario, this would scrape Action Network, VegasInsider, or Covers.
    For now, it generates mock data to demonstrate the 'Sharp' money logic.
    """

    @staticmethod
    def fetch_consensus_data(games: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Returns a dictionary keyed by a unique game identifier (e.g. "{Home} vs {Away}")
        containing 'ticket_pct' and 'money_pct' for the Spread market (primary).
        """
        consensus_map = {}

        if not games:
            return consensus_map

        for game in games:
            home = game.get("home_team")
            away = game.get("away_team")

            if not home or not away:
                continue

            # Mock Logic:
            # Generate random splits.
            # Occasionally generate a "Sharp" signal (Money >> Ticket)

            # Key: "{Home} vs {Away}" - Needs to match how we look it up in streamlit_app
            key = f"{home} vs {away}"

            # Default: Balanced
            ticket_pct = random.randint(30, 70) / 100.0
            # Money often tracks tickets, but sometimes diverges
            money_pct = ticket_pct + (random.randint(-15, 15) / 100.0)

            # Force a "Sharp" Play occasionally (20% chance)
            if random.random() < 0.20:
                # Sharp action usually implies lower ticket count but high money
                ticket_pct = random.randint(20, 45) / 100.0
                money_pct = ticket_pct + (random.randint(15, 30) / 100.0)

            # Clamp
            ticket_pct = max(0.01, min(0.99, ticket_pct))
            money_pct = max(0.01, min(0.99, money_pct))

            consensus_map[key] = {
                "ticket_pct": ticket_pct,
                "money_pct": money_pct,
                "sharpness": money_pct - ticket_pct
            }

        return consensus_map
