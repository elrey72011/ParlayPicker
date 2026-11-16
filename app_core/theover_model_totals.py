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

``CAP`` is 70% because higher listed rates did no better (80% or more: 30-23). Replayed over the
nine nights with the blend's own weights the changed prediction went 290-235; the blended estimate
for TheOver's side averaged 58.5% where those sides won 61.5%.

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


def _market(frame) -> pd.Series:
    return frame["market_type"].astype(str).str.strip().str.lower()


def backed(frame) -> pd.Series:
    """True on each totals row TheOver's model picked: its pick, with a hit rate listed for the model."""
    if frame is None or not {"market_type", "theover_pick_agrees", "theover_model_hit_rate"} <= set(frame.columns):
        return pd.Series(False, index=getattr(frame, "index", None), dtype=bool)
    picked = pd.to_numeric(frame["theover_pick_agrees"], errors="coerce")
    hit = pd.to_numeric(frame["theover_model_hit_rate"], errors="coerce")
    return (_market(frame).isin(list(OTHER_SIDE)) & picked.eq(1) & hit.gt(0) & hit.lt(1)).fillna(False).astype(bool)


def derived(frame, cap=CAP) -> pd.Series:
    """TheOver's probability for each totals row from its model pick, on ``frame``'s index; NaN where it has none.

    The picked side gets the listed hit rate, at most ``cap``; the other side of the same game gets
    one minus that. A game is left empty when there is no such pick or the upload names both sides.
    """
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    flag = backed(frame)
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
