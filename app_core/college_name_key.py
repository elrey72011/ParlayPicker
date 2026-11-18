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
* case, spaces, punctuation and accents do not count;
* the feed's short forms of a school everyone else writes out (owner 2026-11-18): ``CSU`` or ``CS``
  before Fullerton, Northridge or Bakersfield is ``Cal State``; after Loyola, ``Chi`` is ``Chicago``
  and ``Md`` is ``Maryland``; ``Oh`` after Miami is ``Ohio``; the ``Mn`` after St Thomas is dropped;
  ``Intl`` is ``International``; ``Purdue Fort Wayne`` is ``Fort Wayne``.

``school_key`` is the same key with a college nickname dropped from the end of the name (owner
2026-11-18). The shared table strips the commonest nicknames only, so the feed's "San Diego St
Aztecs" and "Boston Univ Terriers" kept theirs, and it reads a plain "Creighton" as "Creighton
Bluejays" while it reads the feed's "Creighton Bluejays" as "Creighton". Such a game matched
nothing: on 18 November 2026 Texas Southern at San Diego State was rated on a pick attached by
name similarity alone. ``MASCOTS`` lists the nicknames of Division I. No school's own name ends
in one of them, which is what makes dropping one from the end safe: "Illinois State" never loses
its "State", and a name that is a nickname and nothing else keeps it. ``HALF_NICKNAMES`` does the
same for the first word of a two-word nickname whose second word the shared table has already
taken ("Florida Intl Golden", from "Golden Panthers").

``reconcile`` gives an uploaded game the feed's spelling of its two teams when that game has no
exact counterpart in the feed and exactly one feed game of the same day has the same two keys, with
no other uploaded game claiming it. It looks by ``team_key`` first and, for the games still open,
by ``school_key``. Anything less certain is left as it was, so a merely similar name is still a
fuzzy match and its line is still rejected. A typo is not a variant: "Tennesee State" stays
unmatched.

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
_CAL_STATE = frozenset({"fullerton", "northridge", "bakersfield"})
_WRITTEN_OUT = {"chi": ("loyola", "chicago"), "md": ("loyola", "maryland"), "oh": ("miami", "ohio")}    # tag: (after, reads as)
_APOSTROPHES = re.compile(r"['’`.]")
_NOT_A_WORD = re.compile(r"[^a-z0-9]+")

# The nicknames of Division I men's basketball, as words read the way ``team_key`` reads them (no
# apostrophes). A longer nickname is tried before a shorter one: "Red Raiders" before "Raiders".
MASCOTS = frozenset(tuple(nickname.split()) for nickname in (
    "49ers", "aggies", "anteaters", "antelopes", "aztecs", "badgers", "beacons", "bearcats", "bearkats", "bears",
    "beavers", "bengals", "billikens", "bison", "bisons", "blazers", "bluejays", "bobcats", "boilermakers", "bonnies",
    "braves", "broncos", "broncs", "bruins", "buccaneers", "buckeyes", "buffaloes", "bulldogs", "bulls", "camels",
    "cardinal", "cardinals", "catamounts", "cavaliers", "chanticleers", "chargers", "chippewas", "colonels",
    "colonials", "commodores", "cornhuskers", "cougars", "cowboys", "coyotes", "crimson", "crusaders", "cyclones",
    "demons", "dolphins", "dons", "dragons", "ducks", "dukes", "eagles", "explorers", "falcons", "flames", "flyers",
    "friars", "gaels", "gamecocks", "gators", "gauchos", "governors", "greyhounds", "grizzlies", "hatters",
    "hawkeyes", "hawks", "highlanders", "hilltoppers", "hokies", "hoosiers", "hornets", "hoyas", "hurricanes",
    "huskies", "islanders", "jackrabbits", "jaguars", "jaspers", "jayhawks", "kangaroos", "keydets", "knights",
    "lakers", "lancers", "leathernecks", "leopards", "lions", "lobos", "longhorns", "lopes", "lumberjacks",
    "mastodons", "matadors", "mavericks", "midshipmen", "miners", "minutemen", "mocs", "monarchs", "mountaineers",
    "musketeers", "mustangs", "norse", "orange", "ospreys", "owls", "paladins", "panthers", "patriots", "peacocks",
    "penguins", "phoenix", "pilots", "pioneers", "pirates", "pride", "privateers", "quakers", "racers", "raiders",
    "ramblers", "rams", "rattlers", "razorbacks", "rebels", "redbirds", "redhawks", "retrievers", "revolutionaries",
    "roadrunners", "rockets", "roos", "royals", "saints", "salukis", "seahawks", "seawolves", "seminoles", "sharks",
    "shockers", "skyhawks", "sooners", "spartans", "spiders", "stags", "sycamores", "terrapins", "terriers", "texans",
    "thunderbirds", "tigers", "titans", "tommies", "toreros", "trailblazers", "tribe", "tritons", "trojans", "utes",
    "vandals", "vaqueros", "vikings", "volunteers", "warhawks", "warriors", "waves", "wildcats", "wolfpack",
    "wolverines", "wolves", "zips",
    "big green", "big red", "black bears", "black knights", "blue demons", "blue devils", "blue hens", "blue hose",
    "blue raiders", "crimson tide", "delta devils", "demon deacons", "fightin blue hens", "fighting camels",
    "fighting hawks", "fighting illini", "fighting irish", "golden bears", "golden eagles", "golden flashes",
    "golden gophers", "golden griffins", "golden grizzlies", "golden hurricane", "golden lions", "great danes",
    "green wave", "horned frogs", "mean green", "mountain hawks", "nittany lions", "purple aces", "purple eagles",
    "ragin cajuns", "rainbow warriors", "red flash", "red foxes", "red raiders", "red storm", "red wolves",
    "river hawks", "runnin bulldogs", "runnin rebels", "scarlet knights", "screaming eagles", "sun devils",
    "tar heels", "thundering herd", "wolf pack", "yellow jackets"))
_LONGEST_MASCOT = max(len(nickname) for nickname in MASCOTS)
# The first word of a two-word nickname, left behind when the shared table strips the second: it knows "Eagles",
# "Panthers" and "Hawks", so "Southern Indiana Screaming Eagles" reaches here as "Southern Indiana Screaming" and
# "Florida Int'l Golden Panthers" as "Florida Intl Golden". No school's own name ends in one of these either.
HALF_NICKNAMES = frozenset({"black", "blue", "delta", "fighting", "golden", "mountain", "purple", "rainbow", "red", "river",
                            "runnin", "screaming"})


def _words(name) -> list:
    """The name's words with the standard variants folded together; [] for a missing name."""
    if name is None or (not isinstance(name, str) and pd.isna(name)):
        return []
    plain = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")     # "José" reads as "Jose"
    raw = _NOT_A_WORD.sub(" ", _APOSTROPHES.sub("", plain.lower())).split()
    words = []
    for position, word in enumerate(raw):
        if word == "and":
            continue
        if word == "mt":
            word = "mount"
        elif word == "st":
            word = "saint" if not words or words[-1] == "mount" else "state"
        elif word in ("csu", "cs") and raw[position + 1:position + 2] and raw[position + 1] in _CAL_STATE:
            words.append("cal")
            word = "state"
        elif word in _WRITTEN_OUT and words[-1:] == [_WRITTEN_OUT[word][0]]:
            word = _WRITTEN_OUT[word][1]
        elif word == "mn" and words[-1:] == ["thomas"]:
            continue
        elif word == "intl":
            word = "international"
        words.append(word)
    return words[1:] if words[:3] == ["purdue", "fort", "wayne"] else words


def _key(words) -> str:
    """The words run together, without a closing ``U``, ``Univ`` or ``University``."""
    words = list(words)
    while len(words) > 1 and words[-1] in _CLOSING:
        words.pop()
    return "".join(words)


def team_key(name) -> str:
    """The name with its standard variants folded together; '' for a missing name."""
    return _key(_words(name))


def school_key(name) -> str:
    """``team_key`` of the name without a college nickname at its end: "San Diego St Aztecs" reads as "San Diego State"."""
    words = _words(name)
    if tuple(words) not in MASCOTS:                                          # a nickname and nothing else stays whole
        for size in range(min(_LONGEST_MASCOT, len(words) - 1), 0, -1):       # longest first; at least one word stays
            if tuple(words[-size:]) in MASCOTS:
                return _key(words[:-size])
        if len(words) > 1 and words[-1] in HALF_NICKNAMES:
            return _key(words[:-1])
    return _key(words)


def pair_key(home, away) -> str:
    """The two teams' keys in a fixed order, so a game reads the same whichever side is listed as home."""
    return "|".join(sorted((team_key(home), team_key(away))))


# How ``reconcile`` looks for an uploaded game's counterpart, in this order.
RULES = (("spelling", team_key), ("mascot", school_key))


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


def _by_key(games, key_of=team_key):
    """{(league, day, the two team keys): games}; a game with a nameless side has no key."""
    out = {}
    for game in games:
        home, away = key_of(game[2]), key_of(game[3])
        if home and away:
            out.setdefault((game[0], game[1], "|".join(sorted((home, away)))), []).append(game)
    return out


def _without(games, gone):
    """``games`` less the entries of ``gone``, told apart by identity: a name may be missing, and that cannot be compared."""
    return [game for game in games if not any(game is other for other in gone)]


def reconcile(upload, live, day_of=None):
    """(the uploaded rows, with the feed's names and id where a game is certain; what was done).

    ``upload`` and ``live`` each need ``league``, ``home_team``, ``away_team``, ``game_date`` and
    ``matchup_id``. ``day_of(frame)`` gives each row's day in one form for both (the pipeline passes
    its Eastern-day helper, since the two frames keep ``game_date`` differently). The report lists
    the games given the feed's names (``reconciled``, each with the ``rule`` that found it:
    ``spelling`` or ``mascot``) and those left alone because more than one game shared a key
    (``ambiguous``).
    """
    report = {"reconciled": [], "ambiguous": []}
    for frame in (upload, live):
        if frame is None or len(frame) == 0 or not set(NEEDED) <= set(frame.columns):
            return upload, report
    upload_day, live_day = _day(upload, day_of), _day(live, day_of)
    feed_games, uploaded_games = _games(live, live_day), _games(upload, upload_day)
    feed_ids, uploaded_ids = {game[4] for game in feed_games}, {game[4] for game in uploaded_games}
    # A game already the same on both sides is settled; only the others are looked at.
    open_feed = [game for game in feed_games if game[4] not in uploaded_ids]
    open_upload = [game for game in uploaded_games if game[4] not in feed_ids]
    out = None
    for rule, key_of in RULES:
        feed_by_key = _by_key(open_feed, key_of)
        for key, mine in _by_key(open_upload, key_of).items():
            theirs = feed_by_key.get(key) or []
            if not theirs:
                continue
            if len(mine) != 1 or len(theirs) != 1:
                report["ambiguous"].append({"league": key[0], "game_date": key[1],
                                            "uploaded": [f"{away} at {home}" for _, _, home, away, _ in mine],
                                            "feed": [f"{away} at {home}" for _, _, home, away, _ in theirs]})
                open_upload, open_feed = _without(open_upload, mine), _without(open_feed, theirs)    # said once
                continue
            (league, day, home, away, _), (_, _, feed_home, feed_away, feed_id) = mine[0], theirs[0]
            spelling = {key_of(feed_home): feed_home, key_of(feed_away): feed_away}
            new_home, new_away = spelling.get(key_of(home)), spelling.get(key_of(away))
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
                                         "changed": "; ".join(changed), "rows": int(rows.sum()), "rule": rule})
            open_upload, open_feed = _without(open_upload, mine), _without(open_feed, theirs)
    return (upload if out is None else out), report


def notes(report) -> dict:
    """{feed game id: "Tennessee State -> Tennessee St"} for the games ``reconcile`` gave the feed's names."""
    return {entry["matchup_id"]: entry["changed"] for entry in (report or {}).get("reconciled", []) if entry.get("changed")}
