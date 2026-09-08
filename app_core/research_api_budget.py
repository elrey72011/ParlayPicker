"""Durable, conservative scheduler-only API budgets. No secrets in state."""
from datetime import datetime, timezone, timedelta
from urllib.parse import urlparse
import requests
from app_core.research_schedule import is_open

LIMITS = {"CFBD": {"daily": 25, "rolling_31_days": 500},
          "ODDS": {"daily": 200, "rolling_31_days": 5000}}


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

    def usage(self, provider):
        today=self.clock().astimezone(timezone.utc).date()
        days=self.ledger.get(provider,{})
        if not isinstance(days,dict):raise ValueError("invalid_api_budget_state")
        daily=rolling=0
        for day,units in days.items():
            date=datetime.strptime(day,"%Y-%m-%d").date()
            if isinstance(units,bool) or not isinstance(units,int) or units<0 or date>today:
                raise ValueError("invalid_api_budget_state")
            if date==today:daily+=units
            if today-timedelta(days=30)<=date<=today:rolling+=units
        return {"daily":daily,"rolling_31_days":rolling}

    def reserve(self, provider, units):
        usage=self.usage(provider)
        for window,limit in self.limits[provider].items():
            if usage[window]+units>limit:
                code=provider+":"+window
                self.blocked.append(code)
                raise BudgetLimit("api_budget:"+code)
        today=self.clock().astimezone(timezone.utc).date().isoformat()
        days=self.ledger.setdefault(provider,{})
        days[today]=days.get(today,0)+units
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
        if parsed.netloc=="api.collegefootballdata.com" and parsed.path in ("/games","/games/teams"):
            provider,units="CFBD",1
        elif parsed.netloc=="api.the-odds-api.com" and parsed.path=="/v4/sports/americanfootball_ncaaf/odds":
            if params.get("regions")!="us" or params.get("markets")!="h2h,spreads,totals" or params.get("bookmakers"):
                raise ValueError("unbudgeted_odds_markets")
            provider,units="ODDS",3
        else:
            raise ValueError("unbudgeted_provider_request")
        if self.state.get("api_budget_cost_mismatch"):
            raise BudgetLimit("api_budget:provider_cost_review_required")
        reserved_day=self.reserve(provider,units)
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
                    self.persist()
                    raise BudgetLimit("api_budget:provider_cost_review_required")
            # No refunds for empty/failed responses: fixed upper-bound accounting.
        return response

    def report(self):
        return {"units":"CFBD requests; ODDS credits conservatively reserved at 3 per call",
                "usage":{p:self.usage(p) for p in self.limits},"limits":self.limits,
                "paused":sorted(set(self.blocked)),
                "cost_review_required":bool(self.state.get("api_budget_cost_mismatch")),
                "scope":"Scheduler only, starting when installed; prior and manual usage are not counted."}
