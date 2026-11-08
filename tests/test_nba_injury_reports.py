"""NBA official injury reports: slots, parsing, changes and the write-once day record. SYNTHETIC."""
from datetime import date, datetime, timezone

import pytest

from app_core import nba_injury_reports as nir

HEADER = [(23.1, 115.2, "Game"), (53.0, 115.2, "Date"), (119.6, 115.2, "Game"), (149.4, 115.2, "Time"),
          (200.0, 115.2, "Matchup"), (264.2, 115.2, "Team"), (425.0, 115.2, "Player"), (456.9, 115.2, "Name"),
          (585.7, 115.2, "Current"), (624.1, 115.2, "Status"), (666.1, 115.2, "Reason")]
TITLE = [(289.1, 62.0, "Injury"), (355.6, 62.0, "Report:"), (438.3, 62.0, "11/07/26"), (398.5, 545.4, "Page")]


def _row(y, player, status, reason, *, date_=None, time_=None, matchup=None, team=None):
    words = [(426.0, y, player), (586.7, y, status), (667.1, y, reason)]
    for x, value in ((24.1, date_), (120.6, time_), (201.0, matchup), (265.2, team)):
        if value:
            words.append((x, y, value))
    return words


def _page(*rows, header=True):
    return (TITLE + (HEADER if header else [])) + [w for row in rows for w in row]


def test_slots_cover_the_et_day_every_15_minutes_and_name_the_league_url():
    day = date(2026, 11, 7)
    slots = nir.slots(day)
    assert len(slots) == 96 and slots[0].hour == 0 and slots[-1].minute == 45
    assert nir.slot_url(slots[0]).endswith("Injury-Report_2026-11-07_12_00AM.pdf")
    assert nir.slot_url(slots[68]).endswith("Injury-Report_2026-11-07_05_00PM.pdf")


def test_rows_carry_game_and_team_forward_across_lines_and_pages_and_join_wrapped_reasons():
    page1 = _page(
        _row(136.3, "DeRozan, DeMar", "Available", "Injury/Illness - Low Back Spasms", date_="11/07/2026",
             time_="05:00 (ET)", matchup="DEN@IND", team="Denver Nuggets"),
        _row(158.3, "Strawther, Julian", "Out", "Injury/Illness - Left Knee Sprain"),
        [(667.1, 180.0, "Injury/Illness - Left Knee")],                            # wrapped reason: above...
        _row(188.4, "Furphy, Johnny", "Out", "Inflammation/Non-Surgical", team="Indiana Pacers"),
        [(667.1, 197.0, "Procedure")],                                             # ...and below its row
        [(265.2, 220.0, "Atlanta Hawks"), (667.1, 220.0, "NOT YET SUBMITTED")])
    page2 = _page(_row(136.3, "Clowney, Noah", "Questionable", "Injury/Illness - Illness"), header=False)
    rows = nir.rows_from_words([page1, page2])
    assert [(r["team"], r["player"], r["status"]) for r in rows] == [
        ("Denver Nuggets", "DeRozan, DeMar", "Available"), ("Denver Nuggets", "Strawther, Julian", "Out"),
        ("Indiana Pacers", "Furphy, Johnny", "Out"), ("Atlanta Hawks", None, "NOT_YET_SUBMITTED"),
        ("Atlanta Hawks", "Clowney, Noah", "Questionable")]
    assert rows[2]["reason"] == "Injury/Illness - Left Knee Inflammation/Non-Surgical Procedure"
    assert all(r["game_date"] == "11/07/2026" and r["matchup"] == "DEN@IND" for r in rows)
    assert rows[0]["game_time"] == "05:00 (ET)"


def _r(player, status, team="Atlanta Hawks"):
    return {"game_date": "11/07/2026", "matchup": "BKN@ATL", "team": team, "player": player, "status": status}


def test_changes_mark_moves_into_and_out_of_unavailable_as_material():
    before = [_r("A", "Questionable"), _r("B", "Out"), _r("C", "Probable"), _r(None, "NOT_YET_SUBMITTED")]
    after = [_r("A", "Out"), _r("B", "Available"), _r("C", "Questionable"), _r("D", "Out")]
    found = {c["player"]: (c["previous"], c["status"], c["material"]) for c in nir.changes(before, after)}
    assert found == {"A": ("Questionable", "Out", True), "B": ("Out", "Available", True),
                     "C": ("Probable", "Questionable", False), "D": (None, "Out", True)}
    assert nir.changes(after, after) == []


class Client:
    def __init__(self):
        self.objects = {}

    def get_object(self, *, Key):
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": self.objects[Key]}

    def put_object(self, *, Key, Body, IfNoneMatch):
        assert IfNoneMatch == "*" and Key not in self.objects
        self.objects[Key] = Body


def test_a_day_is_recorded_once_with_changed_reports_kept_and_repeats_pointing_back(monkeypatch):
    day = date(2026, 11, 7)
    statuses = {"04_00PM": "Questionable", "04_15PM": "Questionable", "04_30PM": "Out"}
    real_slots = nir.slots
    monkeypatch.setattr(nir, "slots", lambda d: [s for s in real_slots(d) if s.hour == 16 and s.minute < 45])
    monkeypatch.setattr(nir, "parse", lambda body: [_r("A", body.decode().split(":")[1])])

    def fetch(url):
        stamp = url.rsplit("_", 2)[-2] + "_" + url.rsplit("_", 1)[-1][:-4]
        return 200, f"pdf:{statuses[stamp]}:{stamp}".encode()
    client = Client()
    early = datetime(2026, 11, 8, 3, tzinfo=timezone.utc)                     # 10 PM ET: the day is not over
    assert nir.record_day(client, day, fetch, now=early)["status"] == "DAY_NOT_OVER"
    now = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)
    result = nir.record_day(client, day, fetch, now=now)
    assert result == {"day": "2026-11-07", "status": "RECORDED", "reports": 3, "changes": 1, "material_changes": 1}
    import json
    manifest = json.loads(client.objects[nir.day_key(day)])
    kept = [e for e in manifest["reports"] if e.get("stored")]
    assert len(kept) == 2 and manifest["reports"][1]["same_rows_as"] == manifest["reports"][0]["slot"]
    assert len([k for k in client.objects if "/pdf/" in k]) == 2
    assert nir.record_day(client, day, fetch, now=now)["status"] == "ALREADY_RECORDED"


def test_recorder_script_reports_each_day(monkeypatch):
    import importlib
    script = importlib.import_module("scripts.record_nba_injury_reports")
    monkeypatch.setattr(nir, "record_day", lambda client, day, fetch, now: (
        {"day": day.isoformat(), "status": "RECORDED", "reports": 90, "changes": 40, "material_changes": 30}
        if day.day == 6 else (_ for _ in ()).throw(RuntimeError("drive"))))
    results = script.run(object(), start=date(2026, 11, 6), end=date(2026, 11, 7),
                         now=datetime(2026, 11, 8, 12, tzinfo=timezone.utc))
    assert [r["status"] for r in results] == ["RECORDED", "ERROR"]
    assert "| 2026-11-06 | RECORDED | 90 | 40 | 30 |" in script.markdown(results)


@pytest.mark.parametrize("hour, half", [(0, "12_00AM"), (12, "12_00PM"), (13, "01_00PM")])
def test_slot_url_twelve_hour_clock(hour, half):
    slot = datetime(2026, 11, 7, hour, tzinfo=nir.ET)
    assert nir.slot_url(slot).endswith(f"2026-11-07_{half}.pdf")
