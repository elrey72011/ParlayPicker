"""Durable, conservative scheduler-only API budgets. No secrets in state."""
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
import requests
from app_core.research_schedule import is_open

LIMITS = {"CFBD": {"daily": 25, "rolling_31_days": 500},
          "ODDS": {"daily": 200, "rolling_31_days": 7500}}
NEW_SPORT_KEYS = {"basketball_nba": "NBA", "basketball_ncaab": "NCAAB",
                  "icehockey_nhl": "NHL"}
NEW_SPORT_CREDIT_LIMITS = {"daily": 40, "rolling_31_days": 1000}
NEW_SPORT_CALL_LIMITS = {"daily": 45, "rolling_31_days": 1200}
# Per-sport overrides of the two limits above. Owner 2026-11-07: NCAAB (55-65 games a
# day) spent its 40 credits by mid-afternoon, pausing closes all day and capture from
# about 4 PM ET. NCAAB gets 120 credits / 135 requests a day; NBA and NHL keep the
# defaults. Every credit still counts against the plan-wide ODDS cap above (200/day),
# so total spend is unchanged; NCAAB can use more of it.
SPORT_CREDIT_LIMITS = {"NCAAB": {"daily": 120, "rolling_31_days": 3700}}
SPORT_CALL_LIMITS = {"NCAAB": {"daily": 135, "rolling_31_days": 4200}}


def sport_credit_limits(sport):
    return SPORT_CREDIT_LIMITS.get(sport, NEW_SPORT_CREDIT_LIMITS)


def sport_call_limits(sport):
    return SPORT_CALL_LIMITS.get(sport, NEW_SPORT_CALL_LIMITS)
# Certified closes (Odds API historical snapshots, ~10 credits per game per market) have
# their own allowance, so they never crowd out capture. The plan quota is 20,000
# credits/month. Owner decisions, 2026-11-05:
#   D13 - NBA spread only, about 80 credits/day.
#   D14 - NCAAB spread prioritized. About 400 closed games (200 validation + 200
#         holdout) is roughly 4,000 credits in total, so the rolling cap bounds it.
CERTIFIED_CLOSE_LIMITS = {"NBA": {"daily": 100, "rolling_31_days": 2500}}  # D13
NCAAB_CERTIFIED_CLOSE_LIMITS = {"NCAAB": {"daily": 400, "rolling_31_days": 4500}}  # D14
ALL_CERTIFIED_CLOSE_LIMITS = {**CERTIFIED_CLOSE_LIMITS, **NCAAB_CERTIFIED_CLOSE_LIMITS}
# D4 option A (owner-chosen 2026-11-06): Pinnacle as the fair-price reference for NBA and
# NCAAB spreads. The capture request names the US-region books plus Pinnacle instead of
# ``regions=us``, so one raw response holds every book and each quote still replays
# against it. 11 US books + Pinnacle = 12 bookmakers = 2 region equivalents, so a
# request costs 4 credits instead of 2; the extra 2 are charged to their own allowance.
PINNACLE_ENV = "PARLAYPICKER_PINNACLE_SPORTS"
PINNACLE_SPORTS = ("NBA", "NCAAB", "NCAAF")  # NCAAF added 2026-11-06 (owner option B)
US_BOOKMAKERS = ("betonlineag", "betmgm", "betrivers", "betus", "bovada", "williamhill_us", "draftkings",
                 "fanatics", "fanduel", "lowvig", "mybookieag")
PINNACLE_BOOKMAKERS = ",".join(US_BOOKMAKERS + ("pinnacle",))
PINNACLE_EXTRA_UNITS = 2
PINNACLE_LIMITS = {"daily": 60, "rolling_31_days": 1500}
# Reference-only books: captured for the fair price, never an entry price, a plan's
# reference quote, a research pick or a certified-close request (Pinnacle is not
# bettable for US users, and naming it would double certified-close credits).
REFERENCE_BOOKS = frozenset({"pinnacle"})


def pinnacle_sports(setting=None):
    """Sports with Pinnacle enabled (``PARLAYPICKER_PINNACLE_SPORTS``, e.g. "NBA,NCAAB"); off by default."""
    import os
    raw = (setting or os.environ.get)(PINNACLE_ENV, "") or ""
    return {s.strip().upper() for s in raw.split(",") if s.strip().upper() in PINNACLE_SPORTS}


class BudgetLimit(ValueError):
    pass


class Budget:
    def __init__(self, state, persist, *, clock=None, get=None, limits=None):
        self.state=state
        self.persist=persist
        self.clock=clock or (lambda:datetime.now(timezone.utc))
        self.get=get or requests.get
        self.limits=limits or LIMITS
        self.blocked=[]
        self.ledger=state.setdefault("api_budget_v1",{})
        if not isinstance(self.ledger,dict):raise ValueError("invalid_api_budget_state")
        self.sport_ledger=state.setdefault("api_budget_by_sport_v1",{})
        self.call_ledger=state.setdefault("api_calls_by_sport_v1",{})
        if not isinstance(self.sport_ledger,dict) or not isinstance(self.call_ledger,dict):
            raise ValueError("invalid_api_budget_state")
        self.close_ledger=state.setdefault("api_budget_certified_close_v1",{})
        if not isinstance(self.close_ledger,dict):
            raise ValueError("invalid_api_budget_state")
        self.pinnacle_ledger=state.setdefault("api_budget_pinnacle_v1",{})
        if not isinstance(self.pinnacle_ledger,dict):
            raise ValueError("invalid_api_budget_state")

    def usage(self, provider, ledger=None):
        today=self.clock().astimezone(timezone.utc).date()
        days=(self.ledger if ledger is None else ledger).get(provider,{})
        if not isinstance(days,dict):raise ValueError("invalid_api_budget_state")
        daily=rolling=0
        for day,units in days.items():
            date=datetime.strptime(day,"%Y-%m-%d").date()
            if isinstance(units,bool) or not isinstance(units,int) or units<0 or date>today:
                raise ValueError("invalid_api_budget_state")
            if date==today:daily+=units
            if today-timedelta(days=30)<=date<=today:rolling+=units
        return {"daily":daily,"rolling_31_days":rolling}

    def reserve(self, provider, units, sport=None):
        usage=self.usage(provider)
        for window,limit in self.limits[provider].items():
            if usage[window]+units>limit:
                code=provider+":"+window
                self.blocked.append(code)
                raise BudgetLimit("api_budget:"+code)
        if sport in NEW_SPORT_KEYS.values():
            for ledger, limits, unit_count, prefix in (
                    (self.sport_ledger, sport_credit_limits(sport), units, "credits"),
                    (self.call_ledger, sport_call_limits(sport), 1, "calls")):
                sport_usage = self.usage(sport, ledger)
                for window, limit in limits.items():
                    if sport_usage[window] + unit_count > limit:
                        code = f"{sport}:{prefix}:{window}"
                        self.blocked.append(code)
                        raise BudgetLimit("api_budget:" + code)
        today=self.clock().astimezone(timezone.utc).date().isoformat()
        days=self.ledger.setdefault(provider,{})
        days[today]=days.get(today,0)+units
        if sport is not None:
            for ledger, count in ((self.sport_ledger, units), (self.call_ledger, 1)):
                sport_days = ledger.setdefault(sport, {})
                sport_days[today] = sport_days.get(today, 0) + count
        # Persist/read-back before network: crashes and timeouts cannot refund requests.
        self.persist()
        return today

    def reserve_pinnacle(self, units):
        """Charge the extra Pinnacle credits to their own allowance and to the plan-wide ODDS total."""
        usage = self.usage("PINNACLE", self.pinnacle_ledger)
        overall = self.usage("ODDS")
        for window, limit in PINNACLE_LIMITS.items():
            if usage[window] + units > limit:
                code = f"PINNACLE:{window}"
                self.blocked.append(code)
                raise BudgetLimit("api_budget:" + code)
        for window, limit in self.limits["ODDS"].items():
            if overall[window] + units > limit:
                code = f"ODDS:{window}"
                self.blocked.append(code)
                raise BudgetLimit("api_budget:" + code)
        today = self.clock().astimezone(timezone.utc).date().isoformat()
        for ledger, key in ((self.pinnacle_ledger, "PINNACLE"), (self.ledger, "ODDS")):
            days = ledger.setdefault(key, {})
            days[today] = days.get(today, 0) + units

    def reserve_certified_close(self, sport, units, *, already_spent=False):
        """Reserve Odds API credits for one certified-close request against its own allowance.

        ``already_spent=True`` records credits the provider has already charged (a cost
        above the reservation): it is always recorded, never refused.
        """
        limits = ALL_CERTIFIED_CLOSE_LIMITS.get(sport)
        if limits is None and not already_spent:
            self.blocked.append(f"{sport}:certified_close:not_budgeted")
            raise BudgetLimit(f"api_budget:{sport}:certified_close:not_budgeted")
        if isinstance(units, bool) or not isinstance(units, int) or units < 0:
            raise ValueError("invalid_certified_close_units")
        usage = self.usage(sport, self.close_ledger)
        for window, limit in ({} if already_spent else limits).items():
            if usage[window] + units > limit:
                code = f"{sport}:certified_close:{window}"
                self.blocked.append(code)
                raise BudgetLimit("api_budget:" + code)
        today = self.clock().astimezone(timezone.utc).date().isoformat()
        days = self.close_ledger.setdefault(sport, {})
        days[today] = days.get(today, 0) + units
        # Persist/read-back before network: crashes and timeouts cannot refund requests.
        self.persist()
        return today

    def request(self, url, **kwargs):
        if not is_open(self.clock()):
            self.blocked.append("outside_operating_window")
            raise BudgetLimit("api_budget:outside_operating_window")
        parsed=urlparse(url)
        params=kwargs.get("params",{})
        if parsed.scheme!="https" or parsed.query:
            raise ValueError("unbudgeted_provider_request")
        sport = None
        extra = 0
        if parsed.netloc=="api.collegefootballdata.com" and parsed.path in ("/games","/games/teams"):
            provider,units,sport="CFBD",1,"NCAAF"
        elif parsed.netloc=="api.the-odds-api.com" and parsed.path in ("/v4/sports/americanfootball_ncaaf/odds", "/v4/sports/americanfootball_nfl/odds"):
            sport = "NCAAF" if "americanfootball_ncaaf" in parsed.path else "NFL"
            if params.get("bookmakers") is not None:
                # D4 for NCAAF: the 11 US books plus Pinnacle (2 region equivalents x 3 markets).
                if (sport != "NCAAF" or params.get("bookmakers") != PINNACLE_BOOKMAKERS or "regions" in params
                        or params.get("markets") != "h2h,spreads,totals"):
                    raise ValueError("unbudgeted_odds_markets")
                provider, units, extra = "ODDS", 3, 3
            else:
                if params.get("regions")!="us" or params.get("markets")!="h2h,spreads,totals":
                    raise ValueError("unbudgeted_odds_markets")
                provider,units="ODDS",3
        elif parsed.netloc=="api.the-odds-api.com" and parsed.path=="/v4/sports/americanfootball_nfl/events":
            if set(params)-{"apiKey", "dateFormat"}:
                raise ValueError("unbudgeted_nfl_parameters")
            provider,units="ODDS",0
            sport = "NFL"
        elif parsed.netloc=="api.the-odds-api.com" and parsed.path=="/v4/sports/americanfootball_nfl/scores":
            if params.get("daysFrom")!=3 or set(params)-{"apiKey", "dateFormat", "daysFrom", "eventIds"}:
                raise ValueError("unbudgeted_nfl_parameters")
            provider,units="ODDS",2
            sport = "NFL"
        elif parsed.netloc == "api.the-odds-api.com" and parsed.path.startswith("/v4/sports/"):
            parts = parsed.path.strip("/").split("/")
            if len(parts) != 4 or parts[:2] != ["v4", "sports"] or parts[2] not in NEW_SPORT_KEYS:
                raise ValueError("unbudgeted_provider_request")
            sport = NEW_SPORT_KEYS[parts[2]]
            endpoint = parts[3]
            if endpoint == "events":
                if set(params) - {"apiKey", "dateFormat"}:
                    raise ValueError("unbudgeted_odds_parameters")
                provider, units = "ODDS", 0
            elif endpoint == "participants":
                if set(params) - {"apiKey"}:
                    raise ValueError("unbudgeted_odds_parameters")
                provider, units = "ODDS", 1
            elif endpoint == "odds" and "bookmakers" in params:
                if (params.get("bookmakers") != PINNACLE_BOOKMAKERS or sport not in PINNACLE_SPORTS or
                        "regions" in params or params.get("markets") != "spreads,totals" or
                        params.get("oddsFormat") != "american" or
                        set(params) - {"apiKey", "dateFormat", "bookmakers", "markets", "oddsFormat", "eventIds"}):
                    raise ValueError("unbudgeted_odds_markets")
                provider, units, extra = "ODDS", 2, PINNACLE_EXTRA_UNITS
            elif endpoint == "odds":
                if (params.get("regions") != "us" or params.get("markets") != "spreads,totals" or
                        params.get("oddsFormat") != "american" or
                        set(params) - {"apiKey", "dateFormat", "regions", "markets", "oddsFormat", "eventIds"}):
                    raise ValueError("unbudgeted_odds_markets")
                provider, units = "ODDS", 2
            elif endpoint == "scores":
                if params.get("daysFrom") != 3 or set(params) - {"apiKey", "dateFormat", "daysFrom", "eventIds"}:
                    raise ValueError("unbudgeted_odds_parameters")
                provider, units = "ODDS", 2
            else:
                raise ValueError("unbudgeted_provider_request")
            ids = params.get("eventIds")
            if ids is not None and (not isinstance(ids, str) or len(ids.split(",")) > 24 or
                                    any(not ident or len(ident) > 128 or not ident.isalnum() for ident in ids.split(","))):
                raise ValueError("unbudgeted_event_ids")
        else:
            raise ValueError("unbudgeted_provider_request")
        if self.state.get("api_budget_cost_mismatch"):
            raise BudgetLimit("api_budget:provider_cost_review_required")
        if extra:
            # Refuse before any reservation, so a spent Pinnacle allowance costs nothing.
            usage = self.usage("PINNACLE", self.pinnacle_ledger)
            if any(usage[w] + extra > limit for w, limit in PINNACLE_LIMITS.items()):
                code = "PINNACLE:" + next(w for w, limit in PINNACLE_LIMITS.items() if usage[w] + extra > limit)
                self.blocked.append(code)
                raise BudgetLimit("api_budget:" + code)
        reserved_day=self.reserve(provider,units,sport)
        if extra:
            self.reserve_pinnacle(extra)
            self.persist()
        response=self.get(url,**kwargs)
        if provider=="ODDS":
            raw=getattr(response,"headers",{}).get("x-requests-last")
            if raw is not None:
                try:actual=int(raw)
                except (ValueError,TypeError):actual=-1
                units += extra
                if actual<0 or actual>units:
                    self.state["api_budget_cost_mismatch"]=True
                    if actual>units:
                        self.ledger[provider][reserved_day]+=actual-units
                        if sport is not None:
                            self.sport_ledger[sport][reserved_day] += actual-units
                    self.persist()
                    raise BudgetLimit("api_budget:provider_cost_review_required")
            # No refunds for empty/failed responses: fixed upper-bound accounting.
        return response

    def report(self):
        return {"units":"CFBD requests; ODDS credits: odds 3, NFL scores 2, NFL events 0",
                "usage":{p:self.usage(p) for p in self.limits},"limits":self.limits,
                "by_sport":{s:{"provider_units":self.usage(s,self.sport_ledger),
                                "calls":self.usage(s,self.call_ledger)}
                            for s in ("NFL","NCAAF",*sorted(NEW_SPORT_KEYS.values()))},
                "new_sport_credit_limits":NEW_SPORT_CREDIT_LIMITS,
                "certified_close_limits":ALL_CERTIFIED_CLOSE_LIMITS,
                "certified_close_usage":{s:self.usage(s,self.close_ledger) for s in ALL_CERTIFIED_CLOSE_LIMITS},
                "new_sport_call_limits":NEW_SPORT_CALL_LIMITS,
                "sport_credit_limits":{s:sport_credit_limits(s) for s in sorted(NEW_SPORT_KEYS.values())},
                "sport_call_limits":{s:sport_call_limits(s) for s in sorted(NEW_SPORT_KEYS.values())},
                "pinnacle_limits":PINNACLE_LIMITS,
                "pinnacle_usage":self.usage("PINNACLE",self.pinnacle_ledger),
                "paused":sorted(set(self.blocked)),
                "cost_review_required":bool(self.state.get("api_budget_cost_mismatch")),
                "scope":"Scheduler only, starting when installed; prior and manual usage are not counted."}
