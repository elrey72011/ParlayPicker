"""One key for the spellings of a college basketball team, and uploaded names set to the feed's (owner 2026-11-17).

TheOver's file and the odds feed spell some schools differently: "Tennessee State" and "Tennessee St",
"Mount St. Mary's" and "Mt. St. Mary's", "Seattle U" and "Seattle". The shared name table
(``core.team_mapper``) knows most schools but not every one, so such a game was matched only by
similarity, and the app rejects a line it cannot tie to an exact game: its pick reads "Total line
unresolved". That cost one of TheOver's picks on 16 November 2026 and two on the 17th.

``team_key`` folds the standard variants together and nothing else:

* ``St`` at the start of a name, or after ``Mount``, is ``Saint``; anywhere else it is ``State``;
* ``Mt`` is ``Mount``;
* a closing ``U``, ``Univ`` or ``University`` is dropped, and so is ``and``;
* case, spaces, punctuation and accents do not count.

``reconcile`` gives an uploaded game the feed's spelling of its two teams when that game has no
exact counterpart in the feed and exactly one feed game of the same day has the same two keys, with
no other uploaded game claiming it. Anything less certain is left as it was, so a merely similar name
is still a fuzzy match and its line is still rejected. A typo is not a variant: "Tennesee State"
stays unmatched.

College basketball only (``LEAGUES``): the pros are matched by the shared table's exact maps, and
college football has its own event identity. Nothing here changes a line, a price or a probability.
"""
from __future__ import annotations

import re
import unicodedata

import pandas as pd

LEAGUES = frozenset({"NCAAB"})
NEEDED = ("league", "home_team", "away_team", "game_date", "matchup_id")
_CLOSING = frozenset({"u", "univ", "university"})
_APOSTROPHES = re.compile(r"['’`.]")
_NOT_A_WORD = re.compile(r"[^a-z0-9]+")


def team_key(name) -> str:
    """The name with its standard variants folded together; '' for a missing name."""
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return ""
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")     # "José" reads as "Jose"
    text = _NOT_A_WORD.sub(" ", _APOSTROPHES.sub("", plain.lower())).strip()
    words = []
    for word in text.split():
        if word == "and":
            continue
        if word == "mt":
            word = "mount"
        elif word == "st":
            word = "saint" if not words or words[-1] == "mount" else "state"
        words.append(word)
    while len(words) > 1 and words[-1] in _CLOSING:
        words.pop()
    return "".join(words)


def pair_key(home, away) -> str:
    """The two teams' keys in a fixed order, so a game reads the same whichever side is listed as home."""
    return "|".join(sorted((team_key(home), team_key(away))))


def _day(frame, day_of):
    """The day key of each row as text: ``day_of(frame)`` when given, else ``game_date`` as it stands."""
    days = frame["game_date"] if day_of is None else day_of(frame)
    return pd.Series(days, index=frame.index).astype("string").fillna("").astype(str)


def _games(frame, day):
    """One entry a game for the leagues covered: (league, day, home, away, game id). By position: an index may repeat."""
    league = frame["league"].astype(str).str.upper().to_numpy()
    days, ids = day.to_numpy(), frame["matchup_id"].astype(str).to_numpy()
    homes, aways = frame["home_team"].to_numpy(), frame["away_team"].to_numpy()
    seen, out = set(), []
    for i in range(len(frame)):
        game = (league[i], days[i], homes[i], aways[i], ids[i])
        if league[i] in LEAGUES and game[:4] not in seen:
            seen.add(game[:4])
            out.append(game)
    return out


def _by_key(games):
    """{(league, day, the two team keys): games}; a game with a nameless side has no key."""
    out = {}
    for game in games:
        if team_key(game[2]) and team_key(game[3]):
            out.setdefault((game[0], game[1], pair_key(game[2], game[3])), []).append(game)
    return out


def reconcile(upload, live, day_of=None):
    """(the uploaded rows, with the feed's names and id where a game is certain; what was done).

    ``upload`` and ``live`` each need ``league``, ``home_team``, ``away_team``, ``game_date`` and
    ``matchup_id``. ``day_of(frame)`` gives each row's day in one form for both (the pipeline passes
    its Eastern-day helper, since the two frames keep ``game_date`` differently). The report lists
    the games given the feed's names (``reconciled``) and those left alone because more than one
    game shared a key (``ambiguous``).
    """
    report = {"reconciled": [], "ambiguous": []}
    for frame in (upload, live):
        if frame is None or len(frame) == 0 or not set(NEEDED) <= set(frame.columns):
            return upload, report
    upload_day, live_day = _day(upload, day_of), _day(live, day_of)
    feed_games, uploaded_games = _games(live, live_day), _games(upload, upload_day)
    feed_ids, uploaded_ids = {game[4] for game in feed_games}, {game[4] for game in uploaded_games}
    # A game already the same on both sides is settled; only the others are looked at.
    open_feed = _by_key(game for game in feed_games if game[4] not in uploaded_ids)
    open_upload = _by_key(game for game in uploaded_games if game[4] not in feed_ids)
    out = None
    for key, mine in open_upload.items():
        theirs = open_feed.get(key) or []
        if not theirs:
            continue
        if len(mine) != 1 or len(theirs) != 1:
            report["ambiguous"].append({"league": key[0], "game_date": key[1],
                                        "uploaded": [f"{away} at {home}" for _, _, home, away, _ in mine],
                                        "feed": [f"{away} at {home}" for _, _, home, away, _ in theirs]})
            continue
        (league, day, home, away, _), (_, _, feed_home, feed_away, feed_id) = mine[0], theirs[0]
        spelling = {team_key(feed_home): feed_home, team_key(feed_away): feed_away}
        new_home, new_away = spelling.get(team_key(home)), spelling.get(team_key(away))
        if len(spelling) != 2 or not new_home or not new_away or new_home == new_away:
            continue                                  # two teams of one game never share a key
        changed = [f"{old} -> {new}" for old, new in ((home, new_home), (away, new_away)) if old != new]
        if not changed:
            continue                                  # the same names already: nothing for this step to do
        if out is None:
            out = upload.copy()
        rows = (out["league"].astype(str).str.upper().eq(league) & upload_day.eq(day)
                & out["home_team"].eq(home) & out["away_team"].eq(away))
        out.loc[rows, "home_team"] = new_home
        out.loc[rows, "away_team"] = new_away
        out.loc[rows, "matchup_id"] = feed_id
        if "game_key" in out.columns:
            out.loc[rows, "game_key"] = f"{league}|{str(new_home).upper()}|{str(new_away).upper()}"
        report["reconciled"].append({"league": league, "game_date": day, "matchup_id": feed_id,
                                     "uploaded": f"{away} at {home}", "feed": f"{feed_away} at {feed_home}",
                                     "changed": "; ".join(changed), "rows": int(rows.sum())})
    return (upload if out is None else out), report


def notes(report) -> dict:
    """{feed game id: "Tennessee State -> Tennessee St"} for the games ``reconcile`` gave the feed's names."""
    return {entry["matchup_id"]: entry["changed"] for entry in (report or {}).get("reconciled", []) if entry.get("changed")}
