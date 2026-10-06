"""Stage 4 — Difference-in-Differences: lockdown effect net of the Spring-Festival holiday.

The confound: in 2020 the Lunar New Year (holiday) and the COVID lockdown coincide.
2019 gives a clean holiday-only control. The DiD differences out the common holiday
effect, isolating the lockdown-attributable change:

    DiD = (peri_2020 - pre_2020) - (peri_2019 - pre_2019)

Estimated by OLS with cell fixed effects and cluster-robust SE (by cell):

    C_it = b0 + b1*Post + b2*Treat + b3*(Post*Treat) + [meteo] + alpha_cell + e
            Post = 1 if peri (during),  Treat = 1 if treated year (2020)
            b3   = lockdown effect net of holiday   <-- the estimand

Unit of analysis: cell x day. Meteo covariates (ERA5) optional; without them the DiD is
still valid (the 2019 control removes the holiday effect) but weather-year confounds are
uncontrolled -> include meteo when available for robustness.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.api as sm


def _within_ols(d: pd.DataFrame, value: str, terms: list[str], entity="grid_id"):
    """Entity (cell) fixed-effects estimator via within-transformation (cell-demeaning),
    with cluster-robust SE by cell. Fast equivalent of OLS with C(cell) dummies.
    Returns (result, term_index) where params are aligned to `terms`.
    """
    d = d.dropna(subset=[value] + terms).copy()
    g = d.groupby(entity)
    yd = d[value].to_numpy() - g[value].transform("mean").to_numpy()
    X = np.column_stack([d[t].to_numpy() - g[t].transform("mean").to_numpy() for t in terms])
    res = sm.OLS(yd, X).fit(cov_type="cluster", cov_kwds={"groups": d[entity].to_numpy()})
    return res, {t: i for i, t in enumerate(terms)}


def cohens_d(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp) if sp > 0 else np.nan


def _within(panel, year, value):
    s = panel[(panel.year == year)]
    pre = s.loc[s.period == "pre", value]; peri = s.loc[s.period == "peri", value]
    return {"pre_mean": pre.mean(), "peri_mean": peri.mean(),
            "abs": peri.mean() - pre.mean(),
            "pct": 100 * (peri.mean() - pre.mean()) / pre.mean() if pre.mean() else np.nan,
            "d": cohens_d(peri, pre)}


def estimate_did(panel: pd.DataFrame, value: str, treat_year: int,
                 control_year: int = 2019, meteo_cols=None) -> dict:
    """DiD for one pollutant/region panel. Returns the interaction estimate + within diffs."""
    d = panel[panel.year.isin([control_year, treat_year]) &
              panel.period.isin(["pre", "peri"])].copy()
    d = d.dropna(subset=[value]).copy()
    d["Post"] = (d.period == "peri").astype(float)
    d["Treat"] = (d.year == treat_year).astype(float)
    d["PostTreat"] = d["Post"] * d["Treat"]
    terms = ["Post", "Treat", "PostTreat"]
    if meteo_cols:
        terms += [c for c in meteo_cols if c in d.columns and d[c].notna().any()]
    res, ix = _within_ols(d, value, terms)
    j = ix["PostTreat"]
    ci = res.conf_int()[j]
    return {
        "did_abs": float(res.params[j]),             # lockdown net of holiday (ug/m3)
        "did_se": float(res.bse[j]),
        "did_p": float(res.pvalues[j]),
        "did_ci_lo": float(ci[0]), "did_ci_hi": float(ci[1]),
        "holiday_2019": _within(d, control_year, value),   # H1
        "treat_within": _within(d, treat_year, value),     # H2
        "n_obs": int(res.nobs),
    }


def estimate_reversion(panel, value, ref_year=2019, late_year=2021) -> dict:
    """H4: did the peri (festival) level in late_year revert to the ref_year level?"""
    d = panel[panel.year.isin([ref_year, late_year]) & (panel.period == "peri")].copy()
    d = d.dropna(subset=[value]).copy()
    d["Late"] = (d.year == late_year).astype(float)
    res, ix = _within_ols(d, value, ["Late"])
    ci = res.conf_int()[0]
    ref = d.loc[d.year == ref_year, value].mean()
    coef = float(res.params[0])
    return {"reversion_abs": coef,
            "reversion_pct": 100 * coef / ref if ref else np.nan,
            "reversion_p": float(res.pvalues[0]),
            "reversion_ci_lo": float(ci[0]), "reversion_ci_hi": float(ci[1])}


def parallel_trends_placebo(panel, value, treat_year, control_year=2019) -> dict:
    """Placebo DiD on (placebo vs pre) pre-periods: should be ~0 if trends are parallel."""
    d = panel[panel.year.isin([control_year, treat_year]) &
              panel.period.isin(["placebo", "pre"])].copy()
    d = d.dropna(subset=[value]).copy()
    d["Post"] = (d.period == "pre").astype(float)
    d["Treat"] = (d.year == treat_year).astype(float)
    d["PostTreat"] = d["Post"] * d["Treat"]
    res, ix = _within_ols(d, value, ["Post", "Treat", "PostTreat"])
    j = ix["PostTreat"]
    return {"placebo_did": float(res.params[j]), "placebo_p": float(res.pvalues[j])}


# --------------------------------------------------------------------------
def _selftest():
    """Synthesize cell x day panels with a KNOWN lockdown effect; DiD must recover it."""
    rng = np.random.RandomState(0)
    cells = np.arange(40); days = 21
    base = 50 + rng.rand(40) * 20          # per-cell baseline
    HOLIDAY = -8.0                          # festival drop (both years)
    LOCKDOWN = -15.0                        # extra 2020 drop (the truth)
    rows = []
    for year in (2019, 2020):
        for period in ("pre", "peri"):
            for c in cells:
                for _ in range(days):
                    val = base[c]
                    if period == "peri":
                        val += HOLIDAY + (LOCKDOWN if year == 2020 else 0)
                    val += rng.randn() * 3
                    rows.append({"grid_id": c, "year": year, "period": period, "no2": val})
    panel = pd.DataFrame(rows)
    r = estimate_did(panel, "no2", treat_year=2020)
    assert abs(r["did_abs"] - LOCKDOWN) < 1.5, r["did_abs"]
    assert r["did_p"] < 1e-6
    assert abs(r["holiday_2019"]["abs"] - HOLIDAY) < 1.5
    print(f"did self-test OK: recovered lockdown={r['did_abs']:.2f} (truth {LOCKDOWN}), "
          f"holiday={r['holiday_2019']['abs']:.2f} (truth {HOLIDAY}), p={r['did_p']:.1e}")


if __name__ == "__main__":
    _selftest()
