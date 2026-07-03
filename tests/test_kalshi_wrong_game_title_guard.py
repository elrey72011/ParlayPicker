"""Wrong-game Kalshi title guard (3 Jul).

The event matcher paired a Cubs/Cardinals Under 10.5 with a market titled
"Boston vs Los Angeles A Total Runs?" — a same-strike (10.5) contract from a
different game's ladder. Its bogus 73% P(Under) flipped consensus to Agrees,
inflated the blend to a fake +29% EV, and the empty-card recovery staked it $40
at S-Tier. Spreads reject such titles implicitly (subject orientation misses);
totals took the price on faith. The guard rejects a totals title that names
NEITHER team of the row's game, while tolerating Kalshi's truncated team names.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_core.kalshi_integrator import kalshi_title_references_matchup


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def _check(title: str, home: str, away: str) -> bool:
    return kalshi_title_references_matchup(f" {_norm(title)} ", _norm(home), _norm(away))


def test_wrong_game_title_is_rejected():
    # The exact 3 Jul failure: Cubs/StL row, Boston/LAA market.
    assert _check("Boston vs Los Angeles A Total Runs?", "chicago cubs", "saint louis") is False


def test_every_truncated_title_format_from_the_3jul_slate_passes():
    # Real (title, home, away) pairs from the 3 Jul export — all legitimate
    # matches, several with Kalshi's truncated team names. None may be rejected.
    cases = [
        ("Tampa Bay vs Houston Total Runs?", "houston", "tampa bay"),
        ("Miami vs A's Total Runs?", "athletics", "miami"),
        ("New York M vs Atlanta Total Runs?", "atlanta", "new york mets"),
        ("Toronto vs Seattle Total Runs?", "seattle", "toronto"),
        ("Pittsburgh vs Washington Total Runs?", "washington", "pittsburgh"),
        ("San Diego vs Los Angeles D Total Runs?", "los angeles dodgers", "san diego"),
        ("Minnesota vs New York Y Total Runs?", "new york yankees", "minnesota"),
        ("Boston vs Los Angeles A Total Runs?", "los angeles angels", "boston"),
        ("San Francisco vs Colorado Total Runs?", "colorado", "san francisco"),
        ("Milwaukee vs Arizona Total Runs?", "arizona", "milwaukee"),
        ("Baltimore vs Cincinnati Total Runs?", "cincinnati", "baltimore"),
        ("Chicago WS vs Cleveland Total Runs?", "cleveland", "chicago white sox"),
    ]
    for title, home, away in cases:
        assert _check(title, home, away) is True, f"legit match rejected: {title!r} for {home}/{away}"


def test_truncated_fragment_does_not_cross_match_similar_team():
    # "New York Y" (Yankees) must not pass for a Mets game.
    assert _check("Minnesota vs New York Y Total Runs?", "new york mets", "philadelphia") is False


def test_empty_or_unparseable_title_is_trusted():
    # The guard fires only on positive evidence of a wrong game.
    assert _check("", "chicago cubs", "saint louis") is True
    assert kalshi_title_references_matchup("  ", "chicago cubs", "saint louis") is True
    assert kalshi_title_references_matchup(" some market ", "", "") is True


def test_teamless_title_is_trusted():
    # A title that names no teams at all cannot confirm OR deny the game;
    # identity is left to the other guards rather than rejected on no evidence.
    assert _check("Total Runs over 8.5?", "chicago cubs", "saint louis") is True


def test_tiny_fragments_cannot_match_everything():
    # A sub-4-char fragment ("LA") must not satisfy the prefix rule.
    assert _check("LA vs NY Total Runs?", "los angeles dodgers", "new york mets") is False
