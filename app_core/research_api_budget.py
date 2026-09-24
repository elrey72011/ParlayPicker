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
                    (self.sport_ledger, NEW_SPORT_CREDIT_LIMITS, units, "credits"),
                    (self.call_ledger, NEW_SPORT_CALL_LIMITS, 1, "calls")):
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

    def request(self, url, **kwargs):
        if not is_open(self.clock()):
            self.blocked.append("outside_operating_window")
            raise BudgetLimit("api_budget:outside_operating_window")
        parsed=urlparse(url)
        params=kwargs.get("params",{})
        if parsed.scheme!="https" or parsed.query:
            raise ValueError("unbudgeted_provider_request")
        sport = None
        if parsed.netloc=="api.collegefootballdata.com" and parsed.path in ("/games","/games/teams"):
            provider,units,sport="CFBD",1,"NCAAF"
        elif parsed.netloc=="api.the-odds-api.com" and parsed.path in ("/v4/sports/americanfootball_ncaaf/odds", "/v4/sports/americanfootball_nfl/odds"):
            if params.get("regions")!="us" or params.get("markets")!="h2h,spreads,totals" or params.get("bookmakers"):
                raise ValueError("unbudgeted_odds_markets")
            provider,units="ODDS",3
            sport = "NCAAF" if "americanfootball_ncaaf" in parsed.path else "NFL"
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
        reserved_day=self.reserve(provider,units,sport)
        response=self.get(url,**kwargs)
        if provider=="ODDS":
            raw=getattr(response,"headers",{}).get("x-requests-last")
            if raw is not None:
                try:actual=int(raw)
                except (ValueError,TypeError):actual=-1
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
                "new_sport_call_limits":NEW_SPORT_CALL_LIMITS,
                "paused":sorted(set(self.blocked)),
                "cost_review_required":bool(self.state.get("api_budget_cost_mismatch")),
                "scope":"Scheduler only, starting when installed; prior and manual usage are not counted."}
