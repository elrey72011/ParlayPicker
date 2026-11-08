"""Decision capture enters only at the plan's frozen entry books (go-live, owner 2026-11-08). SYNTHETIC."""
from datetime import timedelta

from app_core import decision_capture as dc
from app_core import prospective_research_models as models
from tests.test_decision_capture import _Ledger, _Reply, _response, _store, clock  # noqa: F401  (fixture)


def test_decision_quotes_keep_only_the_plan_entry_books(tmp_path, clock):  # noqa: F811
    path, _, start = _store(tmp_path, clock)
    later = start - timedelta(minutes=40)
    clock[0] = later + timedelta(minutes=1)
    ledger = _Ledger(clock)
    dc.capture(path, "NBA", "SPREAD", key="k", budget=ledger.budget, clock=lambda: clock[0],
               request_get=lambda url, **kw: _Reply([_response(start, later, home_price=-105, away_price=-115,
                                                               books=("book", "novig"))]))
    events, quotes, _ = models._read_scope(path, "NBA", "SPREAD")
    at = later + timedelta(minutes=5)
    assert {q["sportsbook"] for q in dc.decision_quotes(events, quotes, at).values()} <= {"book", "novig"}
    only = dc.decision_quotes(events, quotes, at, books=frozenset({"Novig"}))
    assert len(only) == 2 and {q["sportsbook"] for q in only.values()} == {"novig"}   # both sides, Novig only
    assert dc.decision_quotes(events, quotes, at, books=frozenset({"DraftKings"})) == {}
