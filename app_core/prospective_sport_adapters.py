"""Six-sport prospective research registry.

Legacy capture algorithms stay intact.  Registry dispatch only supplies a
common restore/cycle/backup/audit boundary; no adapter grants wager authority.
"""

from app_core import mlb_prospective_store as mlb_store
from app_core import ncaaf_prospective_store as ncaaf_store
from app_core import nfl_market_store as nfl_store
from app_core.odds_research_adapter import OddsResearchAdapter


class LegacySportAdapter:
    def __init__(self, sport, supported_markets, store, path_name):
        self.sport = sport
        self.supported_markets = supported_markets
        self.store = store
        self.path_name = path_name

    def restore(self, path, client, folder, session):
        return self.store.sync(path, client=client, folder=folder, session=session)

    def backup(self, path, client, folder, session):
        return self.store.sync(path, client=client, folder=folder, session=session)

    def upcoming_events(self, *args, **kwargs):
        # Legacy runners perform discovery and capture atomically with their
        # existing provider/model contracts; no second discovery call is made.
        return None

    def capture_pregame(self, path, state, cfbd_key, odds_key, backup, budget):
        from app_core import research_scheduler as scheduler
        if self.sport == "MLB":
            return scheduler.run_mlb(path, state, backup)
        if self.sport == "NCAAF":
            return scheduler.run_ncaaf(path, state, cfbd_key, odds_key, backup, budget.request)
        from app_core import nfl_market
        return nfl_market.run(path, odds_key, backup, budget.request)

    def capture_closes(self, *args, **kwargs):
        # Preserve specialized existing close capture paths. No generic close
        # is fabricated from a market snapshot.
        return {"verified_closes": 0, "close_status": "LEGACY_SPECIALIZED"}

    def grade(self, cycle_result):
        # Grading remains integrated in each established legacy runner.
        return cycle_result.get("graded", 0)

    def run_cycle(self, path, state, cfbd_key, odds_key, backup, budget):
        return self.capture_pregame(path, state, cfbd_key, odds_key, backup, budget)

    def audit(self, path):
        records = self.store.records(path)
        captured = set()
        graded = set()
        for record in records:
            data = record.get("data", {})
            if record["kind"] == "capture":
                for event in data.get("events", []):
                    game_id = event.get("game_id", event.get("cfbd_id", event.get("event_id")))
                    if game_id is not None:
                        captured.add(str(game_id))
            elif record["kind"] == "scores":
                game_id = data.get("game_id", data.get("cfbd_id", data.get("event_id")))
                if game_id is not None:
                    graded.add(str(game_id))
        return {"captured_events": len(captured), "graded_events": len(captured & graded),
                "production_eligible": False}

    def health(self, path):
        return self.audit(path)


ADAPTERS = {
    "NFL": LegacySportAdapter("NFL", ("SPREAD", "TOTAL"), nfl_store, "nfl-market.sqlite3"),
    "NCAAF": LegacySportAdapter("NCAAF", ("SPREAD", "TOTAL"), ncaaf_store, "ncaaf-prospective.sqlite3"),
    "NBA": OddsResearchAdapter("NBA"),
    "NCAAB": OddsResearchAdapter("NCAAB"),
    "MLB": LegacySportAdapter("MLB", ("RUN_LINE", "TOTAL"), mlb_store, "mlb-prospective.sqlite3"),
    "NHL": OddsResearchAdapter("NHL"),
}
DEFAULT_SPORTS = tuple(ADAPTERS)


def get_adapter(sport):
    try:
        return ADAPTERS[sport]
    except KeyError:
        raise ValueError("unsupported_research_sport") from None


def parse_sports(value):
    if isinstance(value, str):
        sports = [part.strip().upper() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        sports = list(value)
    else:
        raise ValueError("Invalid sports")
    if not sports or any(s not in ADAPTERS for s in sports) or len(set(sports)) != len(sports):
        raise ValueError("Invalid sports")
    return sports
