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


if __name__ == "__main__":
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
