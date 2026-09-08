from datetime import datetime, timezone, timedelta
import pytest
from app_core.research_api_budget import Budget, BudgetLimit
from app_core.research_schedule import is_open

NOW=datetime(2026,9,8,16,tzinfo=timezone.utc)
CFBD="https://api.collegefootballdata.com/games"
ODDS="https://api.the-odds-api.com/v4/sports/americanfootball_ncaaf/odds"
PARAMS={"regions":"us","markets":"h2h,spreads,totals"}


class Response:
    headers={"x-requests-last":"3"}


def test_reserve_persist_before_request_and_restart_cap():
    state={};events=[]
    limits={"CFBD":{"daily":2,"rolling_31_days":2}}
    b=Budget(state,lambda:events.append("save"),clock=lambda:NOW,get=lambda *a,**kw:events.append("request"),limits=limits)
    b.request(CFBD);b.request(CFBD)
    assert events==["save","request","save","request"]
    restored=Budget(state,lambda:None,clock=lambda:NOW,limits=limits)
    with pytest.raises(BudgetLimit):restored.request(CFBD)
    assert state["api_budget_v1"]["CFBD"]["2026-09-08"]==2


def test_persist_failure_never_calls_provider():
    calls=[]
    def bad():raise RuntimeError("storage failed")
    b=Budget({},bad,clock=lambda:NOW,get=lambda *a,**kw:calls.append(1))
    with pytest.raises(RuntimeError):b.request(CFBD)
    assert not calls


def test_timeout_keeps_reservation_and_odds_costs_credits():
    def fail(*a,**kw):raise TimeoutError()
    state={};b=Budget(state,lambda:None,clock=lambda:NOW,get=fail)
    with pytest.raises(TimeoutError):b.request(CFBD)
    assert b.usage("CFBD")["daily"]==1
    b.get=lambda *a,**kw:Response()
    b.request(ODDS,params=PARAMS)
    assert b.usage("ODDS")["daily"]==3
    with pytest.raises(ValueError):b.request(ODDS,params={"regions":"us,uk","markets":"h2h,spreads,totals"})


def test_daily_and_rolling_limits_do_not_reset_at_month_boundary():
    current=[datetime(2026,9,30,17,tzinfo=timezone.utc)]
    b=Budget({},lambda:None,clock=lambda:current[0],get=lambda *a,**kw:Response(),limits={"CFBD":{"daily":1,"rolling_31_days":2}})
    b.request(CFBD)
    with pytest.raises(BudgetLimit):b.request(CFBD)
    current[0]+=timedelta(days=1);b.request(CFBD)
    current[0]+=timedelta(days=1)
    with pytest.raises(BudgetLimit):b.request(CFBD)
    current[0]+=timedelta(days=30);b.request(CFBD)


def test_overnight_and_dst():
    for month,offset in ((1,5),(7,4)):
        assert not is_open(datetime(2026,month,8,11+offset,44,tzinfo=timezone.utc))
        assert is_open(datetime(2026,month,8,11+offset,45,tzinfo=timezone.utc))
        assert is_open(datetime(2026,month,9,2+offset,30,tzinfo=timezone.utc))
        assert not is_open(datetime(2026,month,9,2+offset,31,tzinfo=timezone.utc))
    calls=[]
    b=Budget({},lambda:None,clock=lambda:NOW.replace(hour=12),get=lambda *a,**kw:calls.append(1))
    with pytest.raises(BudgetLimit):b.request(CFBD)
    assert not calls


def test_unexpected_provider_price_stops_further_spend():
    class Changed:headers={"x-requests-last":"4"}
    b=Budget({},lambda:None,clock=lambda:NOW,get=lambda *a,**kw:Changed())
    with pytest.raises(BudgetLimit):b.request(ODDS,params=PARAMS)
    assert b.usage("ODDS")["daily"]==4
    with pytest.raises(BudgetLimit):b.request(CFBD)


def test_invalid_ledger_fails_closed():
    b=Budget({"api_budget_v1":{"CFBD":{"2026-09-08":-1}}},lambda:None,clock=lambda:NOW)
    with pytest.raises(ValueError):b.request(CFBD)


def test_cron_yields_30_local_runs_in_winter_and_summer():
    from zoneinfo import ZoneInfo
    for month in (1,7):
        start=datetime(2026,month,8,tzinfo=timezone.utc)
        runs=[]
        for minute in range(48*60):
            at=start+timedelta(minutes=minute)
            local=at.astimezone(ZoneInfo("America/New_York"))
            if ((local.day==8 and local.hour>=11) or (local.day==9 and local.hour<3)) and at.hour in list(range(8))+list(range(15,24)) and at.minute in (15,45) and is_open(at):
                runs.append(local.strftime("%H:%M"))
        assert len(runs)==30 and runs[0]=="11:45" and runs[-1]=="02:15"
