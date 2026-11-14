"""Diagnostics for the ERA5 adjustment of the DiD (Reviewer 5, round 2, #3).

Several small-region estimates change sharply when ERA5 covariates are added (e.g. Wuhan NO2).
Two diagnostics, for the regions the conclusions rest on and for the most unstable estimates:

A. Collinearity. In the main (adjusted) regression, how much of the identifying variation of
   PostTreat is absorbed by the weather covariates? After cell demeaning, PostTreat is first
   residualised on Post and Treat; we report the R^2 of that residual on the covariates
   (share of identifying variation explained by weather; VIF = 1/(1-R^2)) and the covariate
   coefficients with two-way clustered SEs.

B. Weather relationship estimated outside 2020. The pollutant-weather coefficients are estimated
   from all January-March days of 2019 and 2021 (no lockdown), with cell fixed effects and
   year x window-segment fixed effects (placebo/pre/peri/post/other), within the region. The
   2019 and 2020 window data are then adjusted with these fixed coefficients and the DiD is
   estimated WITHOUT covariates, so the 2020 treatment term cannot load on the weather slopes.
   Uncertainty of both stages: date-block bootstrap (dates resampled within each stage).

Out: outputs/analysis/meteo_diagnostics_coefs.csv, outputs/analysis/meteo_diagnostics.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import did as DID
import significance as S
from windows import lny_windows

ROOT = S.ROOT
TARGETS = [("National", "no2"), ("BTH", "no2"), ("Jilin", "no2"), ("Xinjiang", "no2"),
           ("Jiangsu", "no2"), ("Guangdong", "no2"), ("Wuhan", "no2"), ("Nanjing", "no2"),
           ("Beijing", "no2"), ("Shanghai", "no2"), ("Guangzhou", "no2"),
           ("Nanjing", "pm2_5"), ("National", "o3")]
B_BOOT = 199


def _region_mask(df, region):
    if region == "National":
        return np.ones(len(df), bool)
    return ((df["region"] == region) | (df["study_city"] == region)).to_numpy()


def collinearity(sub, pol, meteo):
    d, terms = DID.did_frame(sub, pol, 2020, 2019, meteo)
    met = [t for t in terms if t not in ("Post", "Treat", "PostTreat")]
    y, X, groups = DID.within_design(d, pol, terms, cluster="twoway")
    ix = {t: i for i, t in enumerate(terms)}
    base = X[:, [ix["Post"], ix["Treat"]]]
    pt = X[:, ix["PostTreat"]]
    r0 = pt - base @ np.linalg.lstsq(base, pt, rcond=None)[0]
    M = X[:, [ix[m] for m in met]]
    M = M - base @ np.linalg.lstsq(base, M, rcond=None)[0]
    fit = M @ np.linalg.lstsq(M, r0, rcond=None)[0]
    r2 = 1 - np.sum((r0 - fit) ** 2) / np.sum(r0 ** 2)
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    coefs = [{"term": t, "coef": res.params[ix[t]], "se": res.bse[ix[t]], "p": res.pvalues[ix[t]]}
             for t in met + ["PostTreat"]]
    return r2, coefs


def fit_sample(cfg, chap, met):
    """Jan-Mar 2019 and 2021, every day, with year x segment labels."""
    anchors = {y: cfg["lny_anchors"][y] for y in (2019, 2021)}
    win = lny_windows(anchors)[["date", "year", "period"]]
    d = chap[chap.date.str[:4].isin(["2019", "2021"]) & chap.date.str[5:7].isin(["01", "02", "03"])]
    d = d.merge(win, on="date", how="left")
    d["year"] = d["date"].str[:4].astype(int)
    d["period"] = d["period"].fillna("other")
    d["seg"] = d["year"].astype(str) + "_" + d["period"]
    return d.merge(met, on=["grid_id", "date"], how="inner")


def weather_slopes(fit, pol, meteo):
    """Pollutant ~ meteo + cell FE + year x segment FE on the fit sample."""
    d = fit.dropna(subset=[pol] + meteo)
    return pd.Series(_demean_fit(d, pol, meteo), index=meteo)


def adjusted_did(sub, pol, slopes):
    """DiD without covariates on concentrations adjusted with fixed (out-of-window) slopes."""
    d = sub[sub.year.isin([2019, 2020]) & sub.period.isin(["pre", "peri"])].dropna(
        subset=[pol] + list(slopes.index)).copy()
    d["adj"] = d[pol] - d[list(slopes.index)].to_numpy() @ slopes.to_numpy()
    return DID.estimate_did(d, "adj", 2020, 2019, cluster="twoway")


def run(cfg, B=B_BOOT):
    panel = S.build_panel(cfg, pre_gap=2)
    meteo = [m for m in S.METEO if m in panel.columns]
    chap = S._chap_all(os.path.join(ROOT, "outputs", "targets"))
    reg = pd.read_csv(os.path.join(ROOT, "outputs", "grids", "cell_regions.csv"))
    chap = chap.merge(reg[["grid_id", "region", "study_city"]], on="grid_id", how="left")
    met = pd.read_parquet(os.path.join(ROOT, "outputs", "features", "era5_windows.parquet"))
    met["wind_dir_sin"] = np.sin(np.radians(met["wind_dir"]))
    met["wind_dir_cos"] = np.cos(np.radians(met["wind_dir"]))
    met = met[["grid_id", "date"] + meteo]
    rng = np.random.default_rng(7)
    rows, coefrows = [], []
    for region, pol in TARGETS:
        sub = panel[_region_mask(panel, region)]
        r2, coefs = collinearity(sub, pol, meteo)
        for c in coefs:
            coefrows.append({"region": region, "pollutant": pol, **c})
        main_nm = DID.estimate_did(sub, pol, 2020, cluster="twoway")
        main_m = DID.estimate_did(sub, pol, 2020, meteo_cols=meteo, cluster="twoway")
        fit = fit_sample(cfg, chap[_region_mask(chap, region)], met)
        slopes = weather_slopes(fit, pol, meteo)
        oow = adjusted_did(sub, pol, slopes)
        # date-block bootstrap of both stages
        fdates = fit["date"].unique()
        ddates = sub.loc[sub.year.isin([2019, 2020]) & sub.period.isin(["pre", "peri"]), "date"].unique()
        dyear = pd.Series(ddates).str[:4].to_numpy()
        boots = []
        for b in range(B):
            fd = rng.choice(fdates, size=len(fdates), replace=True)
            dd = np.concatenate([rng.choice(ddates[dyear == y], size=(dyear == y).sum(), replace=True)
                                 for y in ("2019", "2020")])
            sl = _slopes_resampled(fit, pol, meteo, fd)
            boots.append(_did_point(sub, pol, sl, dd))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rows.append({"region": region, "pollutant": pol, "n_cells": sub.grid_id.nunique(),
                     "DiD_nometeo": main_nm["did_abs"], "DiD_meteo": main_m["did_abs"],
                     "r2_PostTreat_on_meteo": r2, "vif_PostTreat": 1 / (1 - r2),
                     "DiD_outside_window_weather": oow["did_abs"],
                     "oow_ci_lo_twoway": oow["did_ci_lo"], "oow_ci_hi_twoway": oow["did_ci_hi"],
                     "oow_p_twoway": oow["did_p"],
                     "oow_ci_lo_boot": lo, "oow_ci_hi_boot": hi,
                     "oow_boot_share_same_sign": float(np.mean(np.sign(boots) == np.sign(oow["did_abs"]))),
                     "fit_days": len(fdates)})
        print(f"  {region:9s} {pol:5s} R2={r2:.3f}  nometeo={main_nm['did_abs']:7.2f}  "
              f"meteo={main_m['did_abs']:7.2f}  outside-window={oow['did_abs']:7.2f} "
              f"[{lo:.2f}, {hi:.2f}]", flush=True)
    adir = os.path.join(ROOT, "outputs", "analysis")
    pd.DataFrame(coefrows).to_csv(os.path.join(adir, "meteo_diagnostics_coefs.csv"), index=False)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(adir, "meteo_diagnostics.csv"), index=False)
    print(f"wrote {adir}/meteo_diagnostics.csv")
    return tab


def _demean_fit(d, pol, meteo):
    seg = pd.get_dummies(d["seg"], drop_first=True, dtype=float)
    Z = np.column_stack([d[meteo].to_numpy(), seg.to_numpy()])
    g = d["grid_id"].to_numpy()
    yd = d[pol].to_numpy() - pd.Series(d[pol].to_numpy()).groupby(g).transform("mean").to_numpy()
    Zd = Z - pd.DataFrame(Z).groupby(g).transform("mean").to_numpy()
    return np.linalg.lstsq(Zd, yd, rcond=None)[0][:len(meteo)]


def _slopes_resampled(fit, pol, meteo, dates):
    d = fit.dropna(subset=[pol] + meteo)
    idx = d.groupby("date").indices
    rows = np.concatenate([idx[x] for x in dates if x in idx])
    return pd.Series(_demean_fit(d.iloc[rows], pol, meteo), index=meteo)


def _did_point(sub, pol, slopes, dates):
    """Point DiD (within estimator) on the resampled dates."""
    d = sub[sub.year.isin([2019, 2020]) & sub.period.isin(["pre", "peri"])].dropna(
        subset=[pol] + list(slopes.index))
    d = d.assign(adj=d[pol] - d[list(slopes.index)].to_numpy() @ slopes.to_numpy())
    idx = d.groupby("date").indices
    rows = np.concatenate([idx[x] for x in dates if x in idx])
    e = d.iloc[rows]
    dd, terms = DID.did_frame(e, "adj", 2020, 2019)
    y, X, _ = DID.within_design(dd, "adj", terms, cluster="cell")
    return float(np.linalg.lstsq(X, y, rcond=None)[0][terms.index("PostTreat")])


if __name__ == "__main__":
    run(S._load_cfg())
