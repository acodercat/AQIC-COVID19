"""Stage 4 (revision) — year-pair placebo DiD: the same holiday-netted DiD, but between two
consecutive NO-lockdown years (Reviewer 5, A2). If the 2020-vs-2019 lockdown estimate reflects
the lockdown rather than ordinary inter-annual variation, it should stand clearly above these.

Pairs (each DiD is computed within a single CHAP product version):
  PM2.5, PM10, O3 : 2019 vs 2018   (1 km D1K, same product as the main analysis)
  NO2, SO2, CO    : 2018 vs 2017   (10 km D10K V1 -- the 1 km V2 product starts in 2019, so a
                                    2019-vs-2018 pair would mix product versions)

Specification = the revised main specification: pre window ends LNY-3d (pre_gap=2), cell fixed
effects, two-way (cell + date) clustered SE, with and without ERA5 covariates.

Inputs : outputs/targets/chap_<pol>_<year>_<D1K|D10K>.parquet
         outputs/analysis/significance_table_rev_gap2_twoway_{meteo,nometeo}.csv (2020 estimates)
Output : outputs/analysis/placebo_years.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import did as DID
import significance as S
from windows import attach_windows

ROOT = S.ROOT
PAIRS = (  # (pollutant, control_year, placebo "treated" year, CHAP product)
    [(p, 2018, 2019, "D1K") for p in ("pm2_5", "pm10", "o3")]
    # gaseous: every consecutive pair within the 10 km V1 product -> a placebo distribution
    + [(p, y, y + 1, "D10K") for p in ("no2", "so2", "co") for y in range(2013, 2018)]
)
PRE_GAP = 2


def load_pair(cfg, pol, years, res, meteo):
    tdir = os.path.join(ROOT, "outputs", "targets")
    parts = []
    for y in years:
        f = os.path.join(tdir, f"chap_{pol}_{y}_{res}.parquet")
        if not os.path.exists(f):
            print(f"  [missing] {f}"); return None
        parts.append(pd.read_parquet(f))
    d = pd.concat(parts, ignore_index=True)
    anchors = {**{int(k): v for k, v in cfg["lny_anchors"].items()},
               **{int(k): v for k, v in cfg["placebo_lny_anchors"].items()}}
    d = attach_windows(d, {y: anchors[y] for y in years}, pre_gap=PRE_GAP)
    reg = pd.read_csv(os.path.join(ROOT, "outputs", "grids", "cell_regions.csv"))
    d = d.merge(reg[["grid_id", "region", "study_city"]], on="grid_id", how="left")
    if meteo is not None:
        d = d.merge(meteo, on=["grid_id", "date"], how="left")
    return d.reset_index(drop=True)


def run(cfg):
    mfile = os.path.join(ROOT, "outputs", "features", "era5_windows.parquet")
    meteo = None
    if os.path.exists(mfile):
        meteo = pd.read_parquet(mfile)
        meteo["wind_dir_sin"] = np.sin(np.radians(meteo["wind_dir"]))
        meteo["wind_dir_cos"] = np.cos(np.radians(meteo["wind_dir"]))
    rows = []
    for pol, c, t, res in PAIRS:
        panel = load_pair(cfg, pol, (c, t), res, meteo)
        if panel is None:
            continue
        mcols = [m for m in S.METEO if m in panel.columns
                 and panel.loc[panel.period.isin(["pre", "peri"]), m].notna().all()]
        for region, idx in S.regions_of(panel).items():
            sub = panel.loc[idx]
            if sub[pol].notna().sum() < 50:
                continue
            for adj, cols in (("nometeo", None), ("meteo", mcols or None)):
                if adj == "meteo" and not cols:
                    continue
                r = DID.estimate_did(sub, pol, treat_year=t, control_year=c,
                                     meteo_cols=cols, cluster="twoway")
                rows.append({"region": region, "pollutant": pol, "adj": adj,
                             "pair": f"{t}v{c}", "product": res,
                             "placebo_year_DiD": r["did_abs"], "ci_lo": r["did_ci_lo"],
                             "ci_hi": r["did_ci_hi"], "p": r["did_p"],
                             "n_cells": sub.loc[sub[pol].notna(), "grid_id"].nunique(),
                             "n_obs": r["n_obs"]})
    tab = pd.DataFrame(rows)
    if tab.empty:
        print("no placebo pairs available yet"); return tab
    for _, g in tab.groupby(["pollutant", "adj", "pair"]):   # BH-FDR, same families as main
        tab.loc[g.index, "q"] = multipletests(g["p"], method="fdr_bh")[1]
    # side-by-side with the 2020-vs-2019 estimate from the same specification
    adir = os.path.join(ROOT, "outputs", "analysis")
    main = pd.concat([pd.read_csv(os.path.join(adir, f"significance_table_rev_gap2_twoway_{adj}.csv"))
                      .assign(adj=adj) for adj in ("meteo", "nometeo")], ignore_index=True)
    main = main[["region", "pollutant", "adj", "H3_lockdown_DiD", "H3_ci_lo", "H3_ci_hi", "H3_qval"]]
    tab = tab.merge(main, on=["region", "pollutant", "adj"], how="left").rename(columns={
        "H3_lockdown_DiD": "lockdown_2020_DiD", "H3_ci_lo": "lockdown_ci_lo",
        "H3_ci_hi": "lockdown_ci_hi", "H3_qval": "lockdown_q"})
    tab["ratio_2020_to_placebo"] = (tab["lockdown_2020_DiD"].abs()
                                    / tab["placebo_year_DiD"].abs())
    out = os.path.join(adir, "placebo_years.csv")
    tab.to_csv(out, index=False)
    print(f"wrote {out} ({len(tab)} rows)")
    # Where does the 2020 lockdown estimate sit within the no-lockdown placebo distribution?
    summ = (tab.groupby(["region", "pollutant", "adj"])
            .agg(n_pairs=("pair", "size"), placebo_mean=("placebo_year_DiD", "mean"),
                 placebo_sd=("placebo_year_DiD", "std"),
                 placebo_min=("placebo_year_DiD", "min"), placebo_max=("placebo_year_DiD", "max"),
                 n_placebo_sig=("q", lambda q: int((q < 0.05).sum())),
                 lockdown_2020_DiD=("lockdown_2020_DiD", "first"),
                 lockdown_q=("lockdown_q", "first")).reset_index())
    # more extreme than every placebo in the direction of the 2020 effect (needs >= 3 pairs)
    beyond = np.where(summ.lockdown_2020_DiD < 0, summ.lockdown_2020_DiD < summ.placebo_min,
                      summ.lockdown_2020_DiD > summ.placebo_max)
    summ["lockdown_beyond_all_placebos"] = pd.Series(beyond, dtype="boolean").where(summ.n_pairs >= 3)
    summ["z_vs_placebo"] = ((summ.lockdown_2020_DiD - summ.placebo_mean) / summ.placebo_sd
                            ).where(summ.n_pairs >= 3)
    out2 = os.path.join(adir, "placebo_years_summary.csv")
    summ.to_csv(out2, index=False)
    print(f"wrote {out2}")
    print(summ[summ.region == "National"].round(2).to_string(index=False))
    return tab


if __name__ == "__main__":
    run(S._load_cfg())
