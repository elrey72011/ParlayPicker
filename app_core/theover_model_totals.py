"""TheOver's model-backed totals pick as one of the blend's inputs (owner 2026-11-16).

Until now TheOver's signals were kept on each row and never used (owner 2026-11-07), and TheOver's
uploads carry no win probability, so the blend's TheOver input was always empty. Nine nights of
graded candidates (7 to 15 November 2026) changed that for one signal: TheOver's totals pick for
the games where it lists the recent hit rate of the model behind it. Taken before the start these
went 147-92, ahead on eight of nine nights, against 113-119 for its totals picks with no hit rate
listed. Found on six nights, it held on the three earlier ones (52-29). The app's own pick, which
for 451 of 525 games was the book's price and nothing else, went 275-249.

The owner asked that the pipeline use TheOver together with the rest of the data, not as a
separate adjustment. So ``blend_input`` fills the blend's TheOver input where the upload gave
none: on the side of a total TheOver's model picked, the hit rate TheOver lists for that model,
counted as at most ``CAP``; on the other side, one minus that. The blend then weighs it against
the market, Kalshi and the app's own model with the weight it already gives TheOver, so a Kalshi
price or a model that disagrees pulls the estimate back.

The blend is told where the input came from (``compute_blended_probability``'s ``theover_source``,
the second value ``blend_input`` returns). For the pick, an NBA total takes the standard weight and
not the NBA-totals boost: the first live board (16 November 2026) had its NBA totals at 60% to 61%
under that boost, above every college total, and the pick had gone 9-11 in the NBA.

``CAP`` is 70% because higher listed rates did no better (80% or more: 30-23). Replayed over the
nine nights with the blend's own weights the changed prediction went 290-235; the blended estimate
for TheOver's side averaged 58.5% where those sides won 61.5%.

Not every league's pick is used (owner 2026-11-17). Taken before the start over ten graded nights (7 to
16 November) the pick went 81-45 in college basketball and 10-15 in the NBA, so the NBA's picks are kept
for the record and no longer enter the blend (``RESEARCH_ONLY_LEAGUES``). The upload reader marks each pick's row when it
reads the file (``mark``, into ``IN_BLEND_COLUMN``: 1 may feed the blend, 0 for the record only), the
production pipeline marks it again once each game's league is settled, and ``derived`` leaves a
marked-off game empty. The mark travels with the pick's hit rate through every merge; a row with no
mark is treated as before, so an analysis saved earlier is unchanged until it is refreshed. The NBA's
picks stay on the homepage list and in the exports, which is what lets their record be kept.

Nor is a pick used when its game is not certainly the uploaded one (owner 2026-11-18). For a game the
upload did not match (its ``orientation_source`` starts with ``default``) the pipeline still attaches
TheOver's columns from the uploaded game with the most similar team names, at a similarity of 65 in
100. On 18 November 2026 that moved the rating of Texas Southern at San Diego State from 48.5% to
56.3%. ``mark`` gives such a pick 0 as well: listed and exported for the record, never blended. A
game matched exactly, or by its teams under another id, is unaffected.

This is the blended ``calibrated_probability``, so the expected value the wager checks read moves
with it and a pick can be labelled a qualified lean. It cannot fund a stake: every wager still has
to pass the terminal market authority (``core.sport_market_gate``), which candidate fields cannot
satisfy. Nine nights is not proof: if these picks stop winning, set ``CAP`` to zero and the blend
is what it was.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAP = 0.70
SOURCE = "theover_model_pick"
UPLOAD = "theover_upload"
SOURCE_COLUMN = "blend_in_theover_source"
OTHER_SIDE = {"total_over": "total_under", "total_under": "total_over"}
# Leagues whose model pick is kept for the record and not blended (owner 2026-11-17: the NBA's went
# 10-15 over ten nights). Empty this set to blend every league's pick again.
RESEARCH_ONLY_LEAGUES = frozenset({"NBA"})
IN_BLEND_COLUMN = "theover_model_in_blend"      # set by the upload reader: 1.0 may feed the blend, 0.0 record only
MATCH_COLUMN = "orientation_source"             # how the pipeline matched the game to the upload
UNMATCHED = "default"                           # its value, or the start of it, for a game the upload did not match


def _market(frame) -> pd.Series:
    return frame["market_type"].astype(str).str.strip().str.lower()


def by_similarity_only(frame) -> pd.Series:
    """True on each row of a game the upload did not match: any TheOver column it carries came from the
    uploaded game with the most similar names, which is not certainly this one. False where the frame
    does not say how its games were matched (the upload itself, an analysis saved before the column)."""
    if frame is None or MATCH_COLUMN not in getattr(frame, "columns", ()):
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    return frame[MATCH_COLUMN].fillna("").astype(str).str.strip().str.lower().str.startswith(UNMATCHED).astype(bool)


def backed(frame) -> pd.Series:
    """True on each totals row TheOver's model picked: its pick, with a hit rate listed for the model."""
    if frame is None or not {"market_type", "theover_pick_agrees", "theover_model_hit_rate"} <= set(frame.columns):
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    picked = pd.to_numeric(frame["theover_pick_agrees"], errors="coerce")
    hit = pd.to_numeric(frame["theover_model_hit_rate"], errors="coerce")
    return (_market(frame).isin(list(OTHER_SIDE)) & picked.eq(1) & hit.gt(0) & hit.lt(1)).fillna(False).astype(bool)


def in_blend_mark(league, index=None) -> pd.Series:
    """The upload reader's mark for a pick's row: 0.0 where its league is research only, else 1.0."""
    if league is None:
        return pd.Series(1.0, index=index, dtype="float64")
    names = pd.Series(league, index=index).fillna("").astype(str).str.strip().str.upper()
    return (~names.isin(RESEARCH_ONLY_LEAGUES)).astype("float64")


def mark(frame) -> pd.Series:
    """``IN_BLEND_COLUMN`` for ``frame``: on both sides of each total TheOver's model picked, 1.0 or 0.0
    from the row's league, and 0.0 where the game was matched by name similarity alone
    (``by_similarity_only``); empty everywhere else. The upload reader stamps it, and the production
    pipeline stamps it again once each game's league and match are settled."""
    empty = pd.Series(np.nan, index=getattr(frame, "index", None), dtype="float64")
    if frame is None or not {"market_type", "theover_pick_agrees", "theover_model_hit_rate"} <= set(frame.columns):
        return empty
    hit = pd.to_numeric(frame["theover_model_hit_rate"], errors="coerce")
    listed = (_market(frame).isin(list(OTHER_SIDE)) & pd.to_numeric(frame["theover_pick_agrees"], errors="coerce").notna()
              & hit.gt(0) & hit.lt(1)).fillna(False)
    league = frame["league"] if "league" in frame.columns else None
    return in_blend_mark(league, index=frame.index).where(~by_similarity_only(frame), 0.0).where(listed)


def in_blend(frame) -> pd.Series:
    """True on each row whose pick may feed the blend: every row, except one the upload reader marked 0."""
    if frame is None or IN_BLEND_COLUMN not in getattr(frame, "columns", ()):
        return pd.Series(True, index=getattr(frame, "index", None), dtype=bool)
    return pd.to_numeric(frame[IN_BLEND_COLUMN], errors="coerce").ne(0).astype(bool)


def derived(frame, cap=CAP) -> pd.Series:
    """TheOver's probability for each totals row from its model pick, on ``frame``'s index; NaN where it has none.

    The picked side gets the listed hit rate, at most ``cap``; the other side of the same game gets
    one minus that. A game is left empty when there is no such pick, the upload names both sides, or
    the pick's row is marked research only (``in_blend``).
    """
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    flag = backed(frame) & in_blend(frame)
    if not cap or not flag.any() or "matchup_id" not in frame.columns:
        return out
    market = _market(frame)
    hit = pd.to_numeric(frame["theover_model_hit_rate"], errors="coerce")
    for _, index in frame.groupby("matchup_id", sort=False).groups.items():
        index = list(index)
        picked = [i for i in index if flag.loc[i]]
        sides = {market.loc[i] for i in picked}
        if len(sides) != 1:
            continue
        side = next(iter(sides))
        value = min(float(hit.loc[picked[0]]), float(cap))
        out.loc[[i for i in index if market.loc[i] == side]] = value
        out.loc[[i for i in index if market.loc[i] == OTHER_SIDE[side]]] = 1.0 - value
    return out


def blend_input(frame, current=None, cap=CAP):
    """``(TheOver input for the blend, where it came from)`` for every row of ``frame``, on its index.

    ``current`` is the input the pipeline already has (a probability from the upload, faded where
    the pipeline fades it). It is kept wherever it has a value; the model pick fills the rest.
    The source is ``UPLOAD``, ``SOURCE`` or ''.
    """
    if current is None:
        current = pd.Series(np.nan, index=frame.index, dtype="float64")
    current = pd.to_numeric(pd.Series(current, index=frame.index), errors="coerce").astype("float64")
    fill = derived(frame, cap)
    source = pd.Series("", index=frame.index, dtype=object)
    source[current.notna()] = UPLOAD
    source[current.isna() & fill.notna()] = SOURCE
    return current.where(current.notna(), fill), source
