"""TheOver's model-backed totals pick moves the app's win estimate (owner 2026-11-16).

Until now TheOver's signals were kept on each row and never used (owner 2026-11-07). Nine nights
of graded candidates (7 to 15 November 2026) changed that for one of them: TheOver's totals pick
for the games where it lists the recent hit rate of the model behind it. Taken before the start
these went 147-92, ahead on eight of nine nights, against 113-119 for its totals picks with no
hit rate listed. Found on six nights, it held on the three earlier ones (52-29). The app's own
pick, which for 451 of 525 games was the book's price and nothing else, went 275-249.

``apply`` adds ``SHIFT`` of win probability to the side TheOver's model picked and takes it from
the other side, so the pair sums to what it did before. Replayed over the same nine nights the
changed prediction went 290-235, and 289-236 with a shift of 3 points or of 8, so the size is not
tuned. ``SHIFT`` is about half of what those nights showed.

It moves the selection probability only (``best_available_probability``): the number that chooses
the game's pick and that the board shows as the win estimate. The blended
``calibrated_probability`` and the ``expected_value`` that the wager checks read are not touched,
so no stake or approval can come from it. Nine nights is not proof: if these picks stop winning,
set ``SHIFT`` to zero and the prediction is what it was.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SHIFT = 0.05
FLOOR, CEILING = 0.01, 0.99
COLUMN = "best_available_theover_model_shift"
SOURCE_SUFFIX = "+theover_model_total"
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


def apply(frame, probability="best_available_probability", shift=SHIFT):
    """``(probability, shift applied)`` for every row of ``frame``, on its index.

    In each game where TheOver's model picked one side of the total, that side gains ``shift`` and
    the other side loses it. The shift is cut back when either side would leave ``FLOOR`` to
    ``CEILING``. A game is left alone when there is no such pick, when the upload names both sides,
    or when the picked side has no probability. Rows that are not totals never move.
    """
    values = pd.to_numeric(frame.get(probability, pd.Series(np.nan, index=frame.index)), errors="coerce").astype("float64").copy()
    moved = pd.Series(0.0, index=frame.index, dtype="float64")
    flag = backed(frame)
    if not shift or not flag.any() or "matchup_id" not in frame.columns:
        return values, moved
    market = _market(frame)
    for _, index in frame.groupby("matchup_id", sort=False).groups.items():
        index = list(index)
        sides = set(market.loc[index][flag.loc[index]])
        if len(sides) != 1:
            continue
        side = next(iter(sides))
        up = [i for i in index if market.loc[i] == side and pd.notna(values.loc[i])]
        down = [i for i in index if market.loc[i] == OTHER_SIDE[side] and pd.notna(values.loc[i])]
        if not up:
            continue
        room = min([float(shift)] + [CEILING - values.loc[i] for i in up] + [values.loc[i] - FLOOR for i in down])
        if room <= 0:
            continue
        values.loc[up] += room
        moved.loc[up] = room
        values.loc[down] -= room
        moved.loc[down] = -room
    return values, moved
