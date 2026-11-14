"""Wild cluster bootstrap for the DiD coefficient with few clusters (Reviewer 5, round 2, #1).

Jilin has 16 grid cells and the cities 4-13, so cluster-robust t-tests can over-reject. We use the
restricted wild cluster bootstrap (WCR; Cameron, Gelbach & Miller 2008) with Webb six-point
weights (Webb 2023), which performs well with few clusters (MacKinnon & Webb 2018):

  1. impose H0: beta3 = b0 and fit the restricted model;
  2. y* = restricted fit + restricted residual x w_g, with one weight w_g per cluster;
  3. t* = (beta3* - b0) / se_CRV1*;  p = share of |t*| >= |t|  (symmetric);
  4. 95% CI by test inversion: the set of b0 with p(b0) >= 0.05 (bisection on each side).

One-way clustering, by grid cell (the dimension with few clusters) or by calendar date. The
design (cell fixed effects, Post, Treat, PostTreat, optional ERA5 covariates) is identical to the
main DiD (did.did_frame / did.within_design). Bootstrap t-statistics are computed in closed form
from the G x G cluster-score matrix (Roodman et al. 2019), so B = 9,999 is cheap.

Out: outputs/analysis/wild_bootstrap.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import did as DID

B_DEFAULT = 9999
WEBB = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])


def webb_weights(G, B, rng):
    return WEBB[rng.integers(0, 6, size=(G, B))]


class WCR:
    """Restricted wild cluster bootstrap for coefficient j of y = X b + u (X already demeaned)."""

    def __init__(self, y, X, groups, j, B=B_DEFAULT, seed=12345):
        self.y, self.X, self.j = np.asarray(y, float), np.asarray(X, float), j
        self.gid = pd.factorize(pd.Series(groups))[0]
        self.G = int(self.gid.max()) + 1
        N, K = self.X.shape
        self.c = self.G / (self.G - 1) * (N - 1) / (N - K)          # CRV1, as statsmodels
        self.XtXi = np.linalg.inv(self.X.T @ self.X)
        a = self.XtXi[j]                                             # row j of (X'X)^-1
        q = self.X @ a                                               # beta_j = sum q_i y_i
        self.q = q
        # Q_g = sum_{i in g} q_i x_i  (G x K)
        self.Q = self._csum(q[:, None] * self.X)
        self.W = webb_weights(self.G, B, np.random.default_rng(seed))
        self.Xr = np.delete(self.X, j, axis=1)
        self.XrtXri = np.linalg.inv(self.Xr.T @ self.Xr)
        self.beta = float(q @ self.y)
        u = self.y - self.X @ (self.XtXi @ (self.X.T @ self.y))
        self.se = float(np.sqrt(self.c * np.sum(self._csum(q * u) ** 2)))

    def _csum(self, M):
        M = np.asarray(M, float)
        out = np.zeros((self.G,) + M.shape[1:])
        np.add.at(out, self.gid, M)
        return out

    def pvalue(self, b0=0.0):
        """Symmetric WCR bootstrap p-value for H0: beta_j = b0."""
        xj = self.X[:, self.j]
        yr = self.y - b0 * xj
        ut = yr - self.Xr @ (self.XrtXri @ (self.Xr.T @ yr))       # restricted residuals
        S = self._csum(ut[:, None] * self.X)                         # G x K: X_h' u_h
        v = S @ self.XtXi[self.j]                                    # beta*_j - b0 = v'w
        d = self._csum(self.q * ut)                                  # diag: sum_{i in g} q_i u_i
        C = np.diag(d) - self.Q @ self.XtXi @ S.T                    # score_g(u*) = (C w)_g
        num = v @ self.W
        se = np.sqrt(self.c * np.sum((C @ self.W) ** 2, axis=0))
        tstar = num / se
        t = (self.beta - b0) / self.se
        return float(np.mean(np.abs(tstar) >= abs(t)))

    def ci(self, level=0.95, tol=1e-3):
        alpha = 1 - level
        half = 10 * self.se

        def root(lo, hi):            # p(lo) < alpha <= p(hi) or reverse; bisection
            plo = self.pvalue(lo) >= alpha
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if (self.pvalue(mid) >= alpha) == plo:
                    lo = mid
                else:
                    hi = mid
                if abs(hi - lo) < tol * max(self.se, 1e-9):
                    break
            return 0.5 * (lo + hi)

        lo_out = self.beta - half
        while self.pvalue(lo_out) >= alpha:
            lo_out -= half
        hi_out = self.beta + half
        while self.pvalue(hi_out) >= alpha:
            hi_out += half
        return root(lo_out, self.beta), root(self.beta, hi_out)


def did_bootstrap(panel, value, treat_year=2020, control_year=2019, meteo_cols=None,
                  cluster="cell", B=B_DEFAULT, seed=12345, ci=True):
    d, terms = DID.did_frame(panel, value, treat_year, control_year, meteo_cols)
    y, X, groups = DID.within_design(d, value, terms, cluster=cluster)
    j = terms.index("PostTreat")
    w = WCR(y, X, groups, j, B=B, seed=seed)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    out = {"beta": w.beta, "n_clusters": w.G, "p_analytic": float(res.pvalues[j]),
           "se_crv1": w.se, "p_wcr": w.pvalue(0.0)}
    assert abs(out["beta"] - res.params[j]) < 1e-8 * max(1, abs(res.params[j]))
    assert abs(out["se_crv1"] - res.bse[j]) < 1e-6 * max(1, res.bse[j])
    if ci:
        out["ci_lo_wcr"], out["ci_hi_wcr"] = w.ci()
    return out


def run(cfg, B=B_DEFAULT):
    import significance as S
    panel = S.build_panel(cfg, pre_gap=2)
    meteo = [m for m in S.METEO if m in panel.columns]
    rows = []
    for region, idx in S.regions_of(panel).items():
        sub = panel.loc[idx]
        for pol in S.POLLUTANTS:
            if sub.loc[sub.year == 2019, pol].notna().sum() < 50:
                continue
            for adj, cols in (("nometeo", None), ("meteo", meteo)):
                for cl in ("cell", "date"):
                    r = did_bootstrap(sub, pol, meteo_cols=cols, cluster=cl, B=B,
                                      ci=(pol == "no2"))
                    rows.append({"region": region, "pollutant": pol, "adj": adj, "cluster": cl, **r})
                    print(f"  {region:9s} {pol:5s} {adj:7s} {cl:4s} G={r['n_clusters']:3d} "
                          f"b={r['beta']:8.3f} p_an={r['p_analytic']:.4f} p_wcr={r['p_wcr']:.4f}",
                          flush=True)
    tab = pd.DataFrame(rows)
    out = os.path.join(S.ROOT, "outputs", "analysis", "wild_bootstrap.csv")
    tab.to_csv(out, index=False)
    print(f"wrote {out}")
    return tab


def _selftest(reps=300, B=399):
    """Size check: 16 clusters with strongly correlated within-cluster errors and beta3 = 0.
    The WCR test should reject close to 5%; the analytic CRV1 test may over-reject."""
    rng = np.random.default_rng(1)
    rej_wcr = rej_an = 0
    for r in range(reps):
        rows = []
        cells = 16
        ce = rng.normal(0, 2, size=(cells, 2))                 # cell x period shocks
        for c in range(cells):
            for year in (2019, 2020):
                for period in ("pre", "peri"):
                    pi = period == "peri"
                    for t in range(10):
                        rows.append({"grid_id": c, "year": year, "period": period,
                                     "date": f"{year}-{period}-{t}",
                                     "v": 50 + c - 5 * pi + ce[c, int(pi)] * (year == 2020)
                                          + rng.normal(0, 1)})
        p = pd.DataFrame(rows)
        o = did_bootstrap(p, "v", cluster="cell", B=B, seed=r, ci=False)
        rej_wcr += o["p_wcr"] < 0.05
        rej_an += o["p_analytic"] < 0.05
    print(f"size at 5% over {reps} sims (16 clusters): WCR {rej_wcr / reps:.3f}, "
          f"analytic CRV1 {rej_an / reps:.3f}")
    # power / agreement with a large-G design: the did self-test panel
    rng = np.random.RandomState(0)
    rows = []
    for year in (2019, 2020):
        for period in ("pre", "peri"):
            for c in range(40):
                for t in range(21):
                    rows.append({"grid_id": c, "year": year, "period": period, "date": f"{year}-{period}-{t}",
                                 "v": c + (period == "peri") * (-8 - 2 * (year == 2020)) + rng.randn() * 3})
    o = did_bootstrap(pd.DataFrame(rows), "v", cluster="cell", B=999)
    print(f"known effect -2: beta={o['beta']:.2f}, WCR CI=({o['ci_lo_wcr']:.2f}, {o['ci_hi_wcr']:.2f}), "
          f"p_wcr={o['p_wcr']:.4f}")
    assert o["ci_lo_wcr"] < -2 < o["ci_hi_wcr"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("-B", type=int, default=B_DEFAULT)
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        import significance as S
        run(S._load_cfg(), B=a.B)
