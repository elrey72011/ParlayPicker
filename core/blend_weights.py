"""Market-anchored blend weights, fitted from settled picks (owner request 2026-11-06).

The app's blend (``core.streamlit_pipeline.compute_blended_probability``) averages
probabilities with hand-set weights and gives Kalshi 40% only when it says >= 55%. The
principled form starts from the market and moves only by what each input shows it knows:

    logit(p) = logit(p_market) + sum_i w_i * (logit(p_i) - logit(p_market))

Each ``w_i`` is fitted by L2-regularised logistic regression with the market as a fixed
offset, so a weight shrinks to 0 ("trust the market") unless the results support it. A
missing input contributes 0. This study:

* uses the verified, settled saved best picks (``core.research_performance.accepted_rows``)
  and the probabilities recorded at pick time, oriented to the pick side;
* splits them in time (earliest ``TRAIN_SHARE`` to fit, the rest to score once);
* reports fitted weights with bootstrap 90% intervals, next to the current hand-set ones;
* scores log loss and Brier on the later picks for: the market alone, the current blend
  (``calibrated_probability``), and the fitted blend.

Only inputs saved with each pick can be fitted: Kalshi, the ML model and, from picks
saved on or after 2026-11-06, TheOver (``theover_probability``, pick side; a percentage is
read as a fraction, and TheOver's exactly-0.5 "no prediction" value counts as missing, as
in the app's blend). Pinnacle's no-vig price at the pick's line (``pinnacle_fair_probability``,
saved from 2026-11-06) is fitted the same way, so one fit gives the Kalshi and Pinnacle weights
together, each relative to the book's own no-vig price. Sentiment is not saved, so it is not fitted.
Descriptive only; it changes no weight.
"""
from __future__ import annotations

import math

import numpy as np

INPUTS = {"kalshi": "kalshi_probability", "ml": "ml_probability", "theover": "theover_probability",
          "pinnacle": "pinnacle_fair_probability"}
CURRENT = {"kalshi": "0.40 when Kalshi >= 0.55, else 0", "ml": "0.15 (tier 1) / 0.35 (tier 2)",
           "theover": "0.20 standard (NBA totals 0.30, NHL 0.09)",
           "pinnacle": "0 (reference only; not in the blend)"}
L2 = 1.0
TRAIN_SHARE = 0.7
BOOTSTRAP = 200
MIN_ROWS = 100
CLIP = 1e-4


def _prob(value):
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    return p if 0.0 < p < 1.0 else None


def _theover(value):
    """TheOver as a fraction (it can arrive as a percentage); exactly 0.5 means no prediction."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return None
    p = p / 100.0 if 1.0 < p <= 100.0 else p
    return None if abs(p - 0.5) < 1e-9 else _prob(p)


def _logit(p):
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p / (1 - p))


def rows_from(accepted):
    """[(start, market, {input: p or None}, current blend p or None, y)] in time order."""
    out = []
    for row in accepted:
        m = _prob(row.get("fair_market_probability")) or _prob(row.get("market_probability"))
        if m is None or row.get("candidate_outcome") not in ("WIN", "LOSS"):
            continue
        inputs = {name: (_theover if name == "theover" else _prob)(row.get(field)) for name, field in INPUTS.items()}
        out.append((str(row.get("game_start_utc") or ""), m, inputs, _prob(row.get("calibrated_probability")),
                    1 if row["candidate_outcome"] == "WIN" else 0))
    return sorted(out, key=lambda r: r[0])


def _design(rows):
    offset = np.array([_logit(m) for _, m, _, _, _ in rows])
    x = np.array([[(_logit(p) - _logit(m)) if p is not None else 0.0 for p in (inp[n] for n in INPUTS)]
                  for _, m, inp, _, _ in rows])
    y = np.array([r[4] for r in rows], dtype=float)
    return offset, x, y


def fit(offset, x, y, l2=L2, iterations=50):
    """Newton-Raphson for logistic regression with a fixed offset and an L2 penalty; no intercept."""
    w = np.zeros(x.shape[1])
    for _ in range(iterations):
        p = 1 / (1 + np.exp(-(offset + x @ w)))
        grad = x.T @ (p - y) + l2 * w
        hess = (x * (p * (1 - p))[:, None]).T @ x + l2 * np.eye(len(w))
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return w


def _scores(p, y):
    p = np.clip(p, CLIP, 1 - CLIP)
    return {"log_loss": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "brier": float(np.mean((p - y) ** 2))}


def anchor_comparison(accepted):
    """Book price vs Pinnacle as the anchor, on settled picks that recorded both."""
    pairs = []
    for row in accepted:
        m = _prob(row.get("fair_market_probability")) or _prob(row.get("market_probability"))
        pin = _prob(row.get("pinnacle_fair_probability"))
        if m is not None and pin is not None and row.get("candidate_outcome") in ("WIN", "LOSS"):
            pairs.append((m, pin, 1 if row["candidate_outcome"] == "WIN" else 0))
    out = {"rows": len(pairs)}
    if len(pairs) >= MIN_ROWS:
        y = np.array([p[2] for p in pairs], dtype=float)
        out.update(book=_scores(np.array([p[0] for p in pairs]), y), pinnacle=_scores(np.array([p[1] for p in pairs]), y))
    else:
        out["status"] = f"TOO_FEW_PICKS_WITH_PINNACLE (need {MIN_ROWS})"
    return out


def study(accepted, *, seed=7):
    rows = rows_from(accepted)
    report = {"schema": "blend-weights-v1", "rows": len(rows), "inputs": list(INPUTS),
              "current_weights": CURRENT, "weight_authority": False,
              "not_fitted": ["sentiment"],
              "present": {n: sum(1 for r in rows if r[2][n] is not None) for n in INPUTS},
              "anchor_comparison": anchor_comparison(accepted)}
    if len(rows) < MIN_ROWS:
        report["status"] = f"TOO_FEW_SETTLED_PICKS (need {MIN_ROWS})"
        return report
    cut = int(len(rows) * TRAIN_SHARE)
    train, test = rows[:cut], rows[cut:]
    o_tr, x_tr, y_tr = _design(train)
    w = fit(o_tr, x_tr, y_tr)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(BOOTSTRAP):
        idx = rng.integers(0, len(train), len(train))
        draws.append(fit(o_tr[idx], x_tr[idx], y_tr[idx]))
    draws = np.array(draws)
    o_te, x_te, y_te = _design(test)
    fitted = 1 / (1 + np.exp(-(o_te + x_te @ w)))
    market = 1 / (1 + np.exp(-o_te))
    current_rows = [(r[3], r[4]) for r in test if r[3] is not None]
    report.update(status="FITTED", train_rows=len(train), test_rows=len(test),
                  weights={n: {"fitted": float(w[i]), "low_90": float(np.percentile(draws[:, i], 5)),
                               "high_90": float(np.percentile(draws[:, i], 95))}
                           for i, n in enumerate(INPUTS)},
                  test_scores={"market_only": _scores(market, y_te), "fitted_blend": _scores(fitted, y_te),
                               "current_blend": (_scores(np.array([p for p, _ in current_rows]),
                                                         np.array([y for _, y in current_rows], dtype=float))
                                                 if current_rows else None),
                               "current_blend_rows": len(current_rows)})
    return report


def build(database):
    from core.research_performance import accepted_rows, saved_rows
    accepted, _ = accepted_rows(saved_rows(database))
    return study(accepted)


def markdown(report):
    lines = ["## Blend weights fitted from settled picks (market-anchored)", "",
             "logit(p) = logit(market) + sum of w x (logit(input) - logit(market)). A weight near 0 means "
             "\"trust the market\". Descriptive only; it changes no weight. TheOver and sentiment: TheOver is "
             "saved with picks from 2026-11-06 and fitted once enough settle; sentiment is not saved.", ""]
    anchor = report.get("anchor_comparison") or {}
    if "pinnacle" in anchor:
        anchor_line = (f"- Anchor test on {anchor['rows']} picks with a Pinnacle price: log loss book "
                       f"{anchor['book']['log_loss']:.4f} vs Pinnacle {anchor['pinnacle']['log_loss']:.4f} (lower is better).")
    else:
        anchor_line = f"- Anchor test (book vs Pinnacle): {anchor.get('status', 'no data')} ({anchor.get('rows', 0)} picks)."
    if report.get("status") != "FITTED":
        return "\n".join(lines + [f"- {report['rows']} settled picks: {report['status']}.", anchor_line, ""]) + "\n"
    lines.append(anchor_line)
    lines.append("")
    lines += [f"Fitted on the earliest {report['train_rows']} picks, scored once on the next {report['test_rows']}.", "",
              "| input | picks with it | fitted weight (90% interval) | current hand-set weight |", "|---|---|---|---|"]
    for name, w in report["weights"].items():
        lines.append(f"| {name} | {report['present'][name]} | {w['fitted']:+.2f} ({w['low_90']:+.2f} to {w['high_90']:+.2f}) "
                     f"| {report['current_weights'][name]} |")
    lines += ["", "| blend (later picks) | log loss | Brier |", "|---|---|---|"]
    for name, s in report["test_scores"].items():
        if isinstance(s, dict):
            lines.append(f"| {name.replace('_', ' ')} | {s['log_loss']:.4f} | {s['brier']:.4f} |")
    lines += ["", "Lower is better. A blend that does not beat \"market only\" on the later picks adds no information."]
    return "\n".join(lines) + "\n"
