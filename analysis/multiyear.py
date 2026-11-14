"""Multiple pre-pandemic control years (Reviewer 3, round 2, #4 and #6; Reviewer 5, round 2, #2).

The main DiD uses 2019 as the single holiday-only control. Here the 2020 change (peri - pre) is
compared with the holiday change of several pre-pandemic years, from one consistent source:

  station : CNEMC station history 2015-2021 (pipeline/cnemc_history.py), all six pollutants,
            station-balanced (same stations every year), gaseous values on the 25 degC basis;
  chap    : CHAP 1 km PM2.5, PM10 and O3 (the only CHAP products with a consistent 1 km record
            before 2019), 2015-2020.

Estimates (windows and inference as in the main specification: pre = LNY-21..LNY-3,
peri = LNY..LNY+20, two-way clustering by grid cell and date, with and without ERA5):
  pair_<c>  : 2020 vs control year c (c = 2015..2019), the main two-year DiD;
  pooled    : cell x year fixed effects, Post, Post x 1[2020] -> 2020 change minus the mean
              holiday change of the control years;
  trend     : pooled + Post x (year - 2020): the holiday change may drift linearly across years
              (e.g. with declining emissions); 2020 is compared with the extrapolated holiday change;
  pretrend  : the same pooled design on the placebo window (LNY-42..-22) vs the pre window, and
              the two-year 2020-vs-2019 version, as in the main pre-trend test;
  placebos  : the pooled estimator applied to each control year in turn, treated against the
              other control years (leave-one-out) -> rank of the 2020 estimate; and every pairwise
              DiD among control years (10 pairs) for the pairwise 2020 vs 2019 estimate.
The pooled estimate also gets a wild cluster bootstrap p-value (cluster = cell; WCR, Webb).

Out: outputs/analysis/multiyear_<source>.csv, multiyear_placebo_<source>.csv,
     outputs/analysis/cnemc_consistency.csv (station history vs the 2019-21 CNEMC model table)
"""
from __future__ import annotations
import os, sys
from itertools import combinations
import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import did as DID
import significance as S
from windows import attach_windows
from wild_bootstrap import WCR

ROOT = S.ROOT
CONTROLS = [2015, 2016, 2017, 2018, 2019]
TREAT = 2020
PRE_GAP = 2
CHAP_POLS = ["pm2_5", "pm10", "o3"]
REF_FACTOR = 273.15 / 298.15        # same as pipeline/cnemc_history.py


def _anchors(cfg):
    a = {int(k): v for k, v in cfg["lny_anchors"].items()}
    a.update({int(k): v for k, v in cfg["placebo_lny_anchors"].items()})
    return a


def _attach(df, cfg, years):
    a = _anchors(cfg)
    d = attach_windows(df, {y: a[y] for y in years}, pre_gap=PRE_GAP)
    reg = pd.read_csv(os.path.join(ROOT, "outputs", "grids", "cell_regions.csv"))
    d = d.merge(reg[["grid_id", "region", "study_city"]], on="grid_id", how="left")
    met = pd.read_parquet(os.path.join(ROOT, "outputs", "features", "era5_windows.parquet"))
    met["wind_dir_sin"] = np.sin(np.radians(met["wind_dir"]))
    met["wind_dir_cos"] = np.cos(np.radians(met["wind_dir"]))
    return d.merge(met, on=["grid_id", "date"], how="left").reset_index(drop=True)


def load_station(cfg):
    d = pd.read_parquet(os.path.join(ROOT, "outputs", "cnemc", "cnemc_cells.parquet"))
    return _attach(d, cfg, CONTROLS + [TREAT, 2021]), S.POLLUTANTS


def load_chap(cfg):
    tdir = os.path.join(ROOT, "outputs", "targets")
    merged = None
    for p in CHAP_POLS:
        parts = [pd.read_parquet(os.path.join(tdir, f"chap_{p}_{y}_D1K.parquet"))
                 for y in CONTROLS + [TREAT]]
        d = pd.concat(parts, ignore_index=True)
        merged = d if merged is None else merged.merge(d, on=["grid_id", "date"], how="outer")
    return _attach(merged, cfg, CONTROLS + [TREAT]), CHAP_POLS


def _meteo_terms(d, meteo):
    return [c for c in (meteo or []) if c in d.columns and d[c].notna().any()]


def pooled(sub, pol, treat, controls, meteo=None, trend=False, bootstrap=False, B=9999,
           periods=("pre", "peri")):
    """2020 (or a placebo year) change minus the mean control-year holiday change.
    periods=("placebo", "pre") gives the placebo-window pre-trend test of the same design."""
    years = [treat] + list(controls)
    d = sub[sub.year.isin(years) & sub.period.isin(list(periods))].dropna(subset=[pol]).copy()
    d["Post"] = (d.period == periods[1]).astype(float)
    d["PostTreat"] = d["Post"] * (d.year == treat)
    terms = ["Post", "PostTreat"]
    if trend:
        d["PostTrend"] = d["Post"] * (d.year - treat)
        terms.append("PostTrend")
    terms += _meteo_terms(d, meteo)
    d = d.dropna(subset=terms)
    d["cy"] = d["grid_id"].astype(str) + "_" + d["year"].astype(str)   # cell x year FE
    y, X, _ = DID.within_design(d, pol, terms, entity="cy", cluster="cell")
    groups = np.column_stack([pd.factorize(d["grid_id"])[0], pd.factorize(d["date"])[0]])
    res = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": groups})
    j = terms.index("PostTreat")
    ci = res.conf_int()[j]
    out = {"DiD": float(res.params[j]), "ci_lo": float(ci[0]), "ci_hi": float(ci[1]),
           "p": float(res.pvalues[j]), "n_obs": int(res.nobs), "n_cells": d.grid_id.nunique()}
    if trend:
        out["holiday_trend_per_year"] = float(res.params[terms.index("PostTrend")])
        out["holiday_trend_p"] = float(res.pvalues[terms.index("PostTrend")])
    if bootstrap:
        w = WCR(y, X, d["grid_id"].to_numpy(), j, B=B)
        out["p_wcr_cell"] = w.pvalue(0.0)
    return out


def holiday_changes(sub, pol, years):
    """Raw peri - pre change per year (cell-balanced means), for the descriptive table."""
    d = sub[sub.year.isin(years) & sub.period.isin(["pre", "peri"])]
    m = d.groupby(["year", "period"])[pol].mean().unstack("period")
    return (m["peri"] - m["pre"]).to_dict(), (100 * (m["peri"] - m["pre"]) / m["pre"]).to_dict()


def run_source(cfg, source):
    panel, pols = load_station(cfg) if source == "station" else load_chap(cfg)
    meteo = [m for m in S.METEO if m in panel.columns]
    raw = None
    if source == "station":
        # sensitivity: undo the 25 degC reference-state conversion of 2015-2018 gaseous values,
        # in case the archive already reports them on the post-2018 basis
        raw = panel.copy()
        old = raw.year <= 2018
        for g in ("so2", "no2", "co", "o3"):
            raw.loc[old, g] = raw.loc[old, g] / REF_FACTOR
    rows, plac = [], []
    for region, idx in S.regions_of(panel).items():
        sub = panel.loc[idx]
        for pol in pols:
            have = [y for y in CONTROLS + [TREAT]
                    if sub.loc[(sub.year == y) & sub.period.isin(["pre", "peri"]), pol].notna().sum() >= 50]
            if TREAT not in have or len(have) < len(CONTROLS) + 1:
                continue
            ch_abs, ch_pct = holiday_changes(sub, pol, CONTROLS + [TREAT])
            for adj, cols in (("nometeo", None), ("meteo", meteo)):
                rec = {"source": source, "region": region, "pollutant": pol, "adj": adj,
                       **{f"change_{y}": ch_abs[y] for y in CONTROLS + [TREAT]},
                       **{f"change_pct_{y}": ch_pct[y] for y in CONTROLS + [TREAT]}}
                for c in CONTROLS:
                    r = DID.estimate_did(sub, pol, TREAT, c, meteo_cols=cols, cluster="twoway")
                    rec.update({f"pair_{c}": r["did_abs"], f"pair_{c}_p": r["did_p"],
                                f"pair_{c}_ci_lo": r["did_ci_lo"], f"pair_{c}_ci_hi": r["did_ci_hi"]})
                po = pooled(sub, pol, TREAT, CONTROLS, cols, bootstrap=(pol == "no2" or source == "chap"))
                tr = pooled(sub, pol, TREAT, CONTROLS, cols, trend=True)
                rec.update({f"pooled_{k}": v for k, v in po.items()})
                if raw is not None and pol in ("so2", "no2", "co", "o3"):
                    pu = pooled(raw.loc[idx], pol, TREAT, CONTROLS, cols)
                    rec.update({"pooled_unconverted_DiD": pu["DiD"], "pooled_unconverted_p": pu["p"]})
                rec.update({f"trend_{k}": v for k, v in tr.items()})
                pt = pooled(sub, pol, TREAT, CONTROLS, cols, periods=("placebo", "pre"))
                p19 = DID.parallel_trends_placebo(sub, pol, TREAT, 2019, meteo_cols=cols, cluster="twoway")
                rec.update({"pretrend_pooled_DiD": pt["DiD"], "pretrend_pooled_p": pt["p"],
                            "pretrend_2019_DiD": p19["placebo_did"], "pretrend_2019_p": p19["placebo_p"]})
                # leave-one-out placebos: each control year against the remaining controls
                loo = []
                for y in CONTROLS:
                    rest = [c for c in CONTROLS if c != y]
                    pl = pooled(sub, pol, y, rest, cols)
                    loo.append(pl["DiD"])
                    plac.append({"source": source, "region": region, "pollutant": pol, "adj": adj,
                                 "kind": "loo_pooled", "treated": y, "control": "+".join(map(str, rest)),
                                 "DiD": pl["DiD"], "p": pl["p"]})
                pairs = []
                for a, b in combinations(CONTROLS, 2):
                    r = DID.estimate_did(sub, pol, b, a, meteo_cols=cols, cluster="twoway")
                    pairs.append(r["did_abs"])
                    plac.append({"source": source, "region": region, "pollutant": pol, "adj": adj,
                                 "kind": "pair", "treated": b, "control": str(a),
                                 "DiD": r["did_abs"], "p": r["did_p"]})
                loo, pairs = np.array(loo), np.array(pairs)
                # rank p: share of {2020} U placebos at least as extreme as 2020 (two-sided)
                rec["pooled_rank_p"] = (1 + np.sum(np.abs(loo) >= abs(po["DiD"]))) / (len(loo) + 1)
                rec["pooled_placebo_min"], rec["pooled_placebo_max"] = loo.min(), loo.max()
                rec["pair2019_rank_p"] = (1 + np.sum(np.abs(pairs) >= abs(rec["pair_2019"]))) / (len(pairs) + 1)
                rec["pair_placebo_min"], rec["pair_placebo_max"] = pairs.min(), pairs.max()
                rec["pair_placebo_sd"] = pairs.std(ddof=1)
                rows.append(rec)
                print(f"  {source} {region:9s} {pol:5s} {adj:7s} pair2019={rec['pair_2019']:7.2f} "
                      f"pooled={po['DiD']:7.2f} (p={po['p']:.3f}) trend={tr['DiD']:7.2f} "
                      f"(p={tr['p']:.3f}) loo=[{loo.min():.2f},{loo.max():.2f}]", flush=True)
    adir = os.path.join(ROOT, "outputs", "analysis")
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(adir, f"multiyear_{source}.csv"), index=False)
    pd.DataFrame(plac).to_csv(os.path.join(adir, f"multiyear_placebo_{source}.csv"), index=False)
    print(f"wrote {adir}/multiyear_{source}.csv")
    return tab


def consistency(cfg):
    """Station history (this script's source) vs the CNEMC table used for the ML-LUR model,
    at identical cell-days of the 2020 and 2021 windows."""
    st = pd.read_parquet(os.path.join(ROOT, "outputs", "cnemc", "cnemc_cells.parquet"))
    st = st.drop(columns=["o3"]).rename(columns={"o3_mean": "o3"})   # model table: O3 daily mean
    old = S._cnemc_existing()
    m = st.merge(old, on=["grid_id", "date"], suffixes=("_hist", "_model"))
    m = m[m.date.str[:4].isin(["2020", "2021"])]
    rows = []
    for p in S.POLLUTANTS:
        s = m.dropna(subset=[f"{p}_hist", f"{p}_model"])
        rows.append({"pollutant": p, "n_cell_days": len(s),
                     "r": s[[f"{p}_hist", f"{p}_model"]].corr().iloc[0, 1],
                     "mean_hist": s[f"{p}_hist"].mean(), "mean_model": s[f"{p}_model"].mean()})
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(ROOT, "outputs", "analysis", "cnemc_consistency.csv"), index=False)
    print(tab.round(3).to_string(index=False))
    return tab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["station", "chap"], required=True)
    ap.add_argument("--check", action="store_true", help="station history vs model-table CNEMC")
    a = ap.parse_args()
    cfg = S._load_cfg()
    if a.check:
        consistency(cfg)
    else:
        run_source(cfg, a.source)
