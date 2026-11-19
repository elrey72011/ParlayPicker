"""What each win estimate was built from: the sportsbook's price alone, or the price and something else.

Owner 2026-11-17. Over nine graded nights (7 to 15 November 2026) the app's win estimate was the
book's own no-vig price and nothing else for 451 of 525 games, and those picks went 233-217: a coin
flip less the book's margin. An estimate that is only the price holds no information the price
does not, so the board and the exports now say which picks those are.

``basis`` reads the inputs the blend recorded for each row (``blend_in_market``, ``blend_in_kalshi``,
``blend_in_ml``, ``blend_in_theover`` and where that last one came from) and applies the same rules
``core.streamlit_pipeline.compute_blended_probability`` uses to decide whether an input counts:

* Kalshi counts only at 55% or more (the blend's first tier); below that the blend leaves it out.
* TheOver counts unless it is missing or exactly 50%, its "no prediction" value.
* The model counts when it gave a probability.
* Sentiment counts only when it moves more than two points from neutral.

It names what it finds and judges nothing: Kalshi is another market's price, and only TheOver's
model-backed totals pick has a graded record behind it so far (``app_core.theover_model_totals``).
Display and audit only. It changes no probability, pick, expected value, gate or stake, and a
column it cannot read is treated as an input that was not there.

**No lean (owner 2026-11-19).** A pick whose estimate is the price alone is no longer shown as a pick:
the boards, the pick details and the compact workbook read ``NO_LEAN`` in its place (``withhold``).
Price-only picks were 279-261 over the graded nights to 18 November. The selection is unchanged, and
the candidate audit, the full best-picks export and the saved evidence still hold every candidate
with its label, so those games keep being graded as the baseline they are.
"""
from __future__ import annotations

import pandas as pd

COLUMN = "rating_basis"
PRICE_ONLY = "Price only"
NO_PRICE = "No price"
KALSHI_MIN = 0.55                  # the blend's first tier: Kalshi is used only at this or more
SENTIMENT_DEAD_BAND = 0.02         # the blend ignores sentiment within this of neutral (0.5)
KALSHI, MODEL, SENTIMENT = "Kalshi", "model", "sentiment"
THEOVER_MODEL_PICK, THEOVER_PROBABILITY = "TheOver model pick", "TheOver probability"
NO_LEAN = "No lean (price only)"
NO_LEAN_CAPTION = ("A game whose pick would rest on the price alone shows \"" + NO_LEAN + "\" in place of a pick. "
                   "Every candidate is still in the candidate audit.")
CAPTION = ("Built from says what the win estimate rests on. \"Price only\" means it is the sportsbook's own "
           "price with the book's margin removed and nothing else, so it holds no view the price does not. "
           "Anything after \"Price +\" is an input that also moved the estimate; naming it does not mean it "
           "has been shown to win.")


def _numbers(frame, column):
    if column not in frame.columns:
        return pd.Series(float("nan"), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").astype("float64")


def inputs(frame) -> pd.Series:
    """Per row, the tuple of inputs besides the price that counted in the blend, in a fixed order."""
    if frame is None or len(frame) == 0:
        return pd.Series([], index=getattr(frame, "index", None), dtype=object)
    from app_core.theover_model_totals import SOURCE, SOURCE_COLUMN
    kalshi = _numbers(frame, "blend_in_kalshi")
    model = _numbers(frame, "blend_in_ml")
    theover = _numbers(frame, "blend_in_theover")
    theover = theover.where((theover - 0.5).abs() >= 1e-9)
    source = (frame[SOURCE_COLUMN].fillna("").astype(str) if SOURCE_COLUMN in frame.columns
              else pd.Series("", index=frame.index))
    sentiment = 0.5 + _numbers(frame, "sentiment_diff") * 0.5        # as the production blend converts it
    found = []
    # By position, not label: a candidate frame's index can repeat.
    for k, m, t, where, s in zip(kalshi.to_numpy(), model.to_numpy(), theover.to_numpy(), source.to_numpy(),
                                 sentiment.to_numpy()):
        row = []
        if pd.notna(k) and k >= KALSHI_MIN:
            row.append(KALSHI)
        if pd.notna(m):
            row.append(MODEL)
        if pd.notna(t):
            row.append(THEOVER_MODEL_PICK if where == SOURCE else THEOVER_PROBABILITY)
        if pd.notna(s) and abs(s - 0.5) > SENTIMENT_DEAD_BAND:
            row.append(SENTIMENT)
        found.append(tuple(row))
    return pd.Series(found, index=frame.index, dtype=object)


def basis(frame) -> pd.Series:
    """Per row: ``Price only``, ``Price + <inputs>``, or what stands in when the row has no price.

    A frame on which the blend's inputs were never recorded (no ``blend_in_market`` column) gets
    empty labels: nothing is claimed about an estimate whose inputs cannot be read.
    """
    others = inputs(frame)
    if len(others) == 0:
        return pd.Series([], index=others.index, dtype=object)
    if "blend_in_market" not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    priced = _numbers(frame, "blend_in_market").notna()

    def label(extra, has_price):
        if has_price:
            return PRICE_ONLY if not extra else "Price + " + ", ".join(extra)
        return NO_PRICE if not extra else "No price: " + ", ".join(extra)
    return pd.Series([label(extra, bool(has_price)) for extra, has_price in zip(others, priced)],
                     index=frame.index, dtype=object)


def price_only(labels) -> pd.Series:
    """True where a ``basis`` label says the estimate is the price alone."""
    return pd.Series(labels).fillna("").astype(str).eq(PRICE_ONLY)


def withhold(frame, labels, pick, describing=()) -> pd.DataFrame:
    """A copy of ``frame`` that shows no pick where the estimate is the price alone.

    On those rows column ``pick`` reads ``NO_LEAN`` and the ``describing`` columns, which tell of the
    pick withheld (its price, its estimate), are emptied. ``labels`` are the rows' ``basis`` labels, by
    position; a row with no label, or any other label, is left as it is.
    """
    out = frame.copy()
    hide = price_only(labels).set_axis(out.index)
    if not hide.any():
        return out
    for column in [c for c in describing if c in out.columns and c != pick]:
        out[column] = out[column].where(~hide, float("nan") if pd.api.types.is_numeric_dtype(out[column]) else "")
    out[pick] = out[pick].where(~hide, NO_LEAN)
    return out


def counts(labels) -> dict:
    """{"picks", "price_only"} over the rows that carry a label."""
    known = pd.Series(labels).dropna().astype(str)
    known = known[known.ne("")]
    return {"picks": int(len(known)), "price_only": int(known.eq(PRICE_ONLY).sum())}
