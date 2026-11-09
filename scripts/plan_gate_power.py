"""Pass probability of a market-skill plan's gates, by how right the model is (a stylized simulation).

``docs/roadmap/VALIDATION-ROADMAP-2026-11-09.md`` section 3. It reads no evidence and grants nothing.

Each game has a stake-book fair probability m near 0.5 and a model probability p = m + d,
d ~ N(0, sd). The truth is m + share*d: share 1 means the model is right, 0 means the market is.
A bet is a side whose EV at the vigged price is at least the minimum edge. Per-bet CLV is
N(share * relative edge, 2%). Run: ``python3 scripts/plan_gate_power.py``.
"""
import numpy as np
rng = np.random.default_rng(7)

def window(n_games, sd_d, truth_share, min_edge, vig=0.0238, min_bets=50, sims=4000, clv_only=False, roi_floor=0.0):
    out = np.zeros(sims, bool); nb = []
    for s in range(sims):
        m = np.clip(rng.normal(0.5, 0.03, n_games), 0.3, 0.7)
        d = rng.normal(0, sd_d, n_games)
        p = np.clip(m + d, 0.02, 0.98)
        q = np.clip(m + truth_share * d, 0.02, 0.98)
        y = rng.random(n_games) < q
        brier_ok = np.mean((p - y) ** 2) <= np.mean((m - y) ** 2)
        ll = lambda a: -np.mean(np.where(y, np.log(a), np.log(1 - a)))
        ll_ok = ll(p) <= ll(m)
        # price for each side: fair odds shaded by half the vig
        side = p >= m                      # bet the side the model likes
        ps = np.where(side, p, 1 - p); ms = np.where(side, m, 1 - m); ys = np.where(side, y, ~y)
        odds = 1 / (ms * (1 + vig))
        ev = ps * odds - 1
        bet = ev >= min_edge
        k = bet.sum(); nb.append(k)
        if k < min_bets: continue
        roi = np.mean(np.where(ys[bet], odds[bet] - 1, -1.0))
        clv = np.mean(rng.normal(truth_share * (ps[bet] - ms[bet]) / ms[bet], 0.02))
        if clv_only:
            out[s] = clv >= 0 and roi >= roi_floor
        else:
            out[s] = brier_ok and ll_ok and roi >= roi_floor and clv >= 0
    return out.mean(), np.mean(nb)



def clv_t_gate(bets, mean_clv, sims=20000, sd=0.02):
    x = rng.normal(mean_clv, sd, (sims, bets))
    return float(((x.mean(1) / (x.std(1, ddof=1) / np.sqrt(bets))) >= 2).mean())


def v2_window(bets, mean_clv, *, clv_sd=0.02, bet_share=0.25, sd_d=0.03, sims=4000, min_t=2.0, roi_floor=-0.25,
              max_brier=0.26, max_log_loss=0.715, max_ece=0.15, seed=11):
    """P(one ``market-skill-v2`` window passes) with exactly ``bets`` bets (the frozen minimum).

    Every blocking gate that has sampling noise is simulated: the CLV t gate, the ROI loss floor
    and the absolute probability bounds (Brier, log loss, calibration error) on ``bets / bet_share``
    games. Bets are priced at a low-vig book, so a bet's true EV is its CLV. Each bet's CLV is
    N(``mean_clv``, ``clv_sd``), independent of its result.
    """
    local = np.random.default_rng(seed)
    games = int(round(bets / bet_share))
    passed = 0
    for _ in range(sims):
        clv = local.normal(mean_clv, clv_sd, bets)
        t_ok = clv.mean() / (clv.std(ddof=1) / np.sqrt(bets)) >= min_t
        odds = local.uniform(1.87, 2.0, bets)
        won = local.random(bets) < np.clip((1 + mean_clv) / odds, 0.02, 0.98)
        roi_ok = np.mean(np.where(won, odds - 1, -1.0)) >= roi_floor
        m = np.clip(local.normal(0.5, 0.03, games), 0.3, 0.7)
        d = local.normal(0, sd_d, games)
        p = np.clip(m + d, 0.02, 0.98)
        share = 1.0 if mean_clv > 0 else 0.0           # with no edge the market is right
        y = local.random(games) < np.clip(m + share * d, 0.02, 0.98)
        brier = np.mean((p - y) ** 2)
        log_loss = -np.mean(np.where(y, np.log(p), np.log(1 - p)))
        bins = np.minimum((p * 10).astype(int), 9)
        ece = sum((bins == b).mean() * abs(p[bins == b].mean() - y[bins == b].mean()) for b in np.unique(bins))
        passed += bool(t_ok and roi_ok and brier <= max_brier and log_loss <= max_log_loss and ece <= max_ece)
    return passed / sims


def v2_plan(bets, mean_clv, **kwargs):
    """P(both windows pass): the validation and holdout windows are separate games."""
    return v2_window(bets, mean_clv, **kwargs) * v2_window(bets, mean_clv, **dict(kwargs, seed=kwargs.get("seed", 11) + 1))


if __name__ == "__main__":
    print("market-skill-v2, both windows, every noisy blocking gate (acceptance: 1% CLV >= 0.90, no edge <= 0.05)")
    print("bets per window  clv sd  true mean CLV  P(plan passes)")
    for k in (50, 60, 75, 100):
        for sd in (0.02, 0.03):
            for mu in (0.0, 0.005, 0.01, 0.02):
                print(f"{k:4d}  {sd:.2f}  {mu:.3f}  {v2_plan(k, mu, clv_sd=sd):.2f}")
    print("Current gates (Brier and log loss <= market, >= 50 bets, ROI >= 0, mean CLV >= 0), edge 2%")
    print("games  sd   share | bets  P(window)  P(both windows)")
    for n in (200, 400, 800):
        for sd in (0.02, 0.03):
            for share in (1.0, 0.5, 0.0):
                pw, k = window(n, sd, share, 0.02)
                print(f"{n:5d} {sd:.2f} {share:.1f} | {k:4.0f}  {pw:.2f}  {pw * pw:.2f}")
    print("CLV-led gate alone (mean bet CLV at least 2 standard errors above 0)")
    print("bets  true mean CLV  P(pass)")
    for k in (50, 100, 200):
        for mu in (0.0, 0.005, 0.01, 0.02):
            print(f"{k:4d}  {mu:.3f}  {clv_t_gate(k, mu):.2f}")
