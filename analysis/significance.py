"""Stage 4 — build the Spring-Festival vs lockdown significance table (the deliverable).

For each pollutant x region:
  H1 holiday (2019)        : peri_2019 - pre_2019
  H2 holiday+lockdown (2020): peri_2020 - pre_2020
  H3 lockdown net of holiday: DiD = H2 - H1   (+ SE, 95% CI, p)   <-- headline
  H4 reversion (2021 vs 2019): peri_2021 - peri_2019
  parallel-trends placebo   : should be ~0
Effect sizes: absolute (ug/m3; CO mg/m3), % change, Cohen's d.
Multiplicity: Benjamini-Hochberg FDR (q-values) on the DiD p-values, per pollutant.

Inputs: outputs/targets/chap_<pol>_<year>_D1K.parquet (per pollutant-year, 747 cells).
        outputs/grids/cell_regions.csv
        [optional] outputs/features/era5_windows.parquet (meteo covariates)
Output: outputs/analysis/significance_table.csv  (+ marginal_means.csv)
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import did as DID
from windows import attach_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLLUTANTS = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
METEO = ["t2m", "rh", "wind_speed", "wind_dir_sin", "wind_dir_cos", "pwat", "sp"]


def _load_cfg():
    import yaml
    return yaml.safe_load(open(os.path.join(ROOT, "pipeline", "config.yaml")))


def _chap_all(tdir) -> pd.DataFrame | None:
    """Wide CHAP daily panel across all downloaded years/pollutants (grid_id, date, <pol>)."""
    merged = None
    for p in POLLUTANTS:
        parts = sorted(glob.glob(os.path.join(tdir, f"chap_{p}_*_D1K.parquet")))
        if not parts:
            continue
        d = pd.concat([pd.read_parquet(x) for x in parts], ignore_index=True)
        merged = d if merged is None else merged.merge(d, on=["grid_id", "date"], how="outer")
    if merged is not None:
        merged["source"] = "CHAP"
    return merged


def _cnemc_existing() -> pd.DataFrame:
    """Existing CNEMC daily panel (2019-12 .. 2021-11) from the original train/test CSVs."""
    cols = ["grid_id", "date"] + POLLUTANTS
    d = pd.concat([pd.read_csv(os.path.join(ROOT, "dataset/train_set.csv"), usecols=cols),
                   pd.read_csv(os.path.join(ROOT, "dataset/test_set.csv"), usecols=cols)],
                  ignore_index=True)
    d["source"] = "CNEMC"
    return d


def build_panel(cfg, **win_kw) -> pd.DataFrame:
    """Combine CHAP-2019 (gap) + existing CNEMC 2020/2021, sourcing each LNY-window
    year from the right product; attach regions + windows + optional meteo.
    The DiD differences out a constant per-year source bias, so this is valid as long
    as bias is stable within each year's pre/peri window (documented assumption).
    `win_kw` (pre/peri/post) overrides the default 21-day window length.
    """
    tdir = os.path.join(ROOT, "outputs", "targets")
    chap = _chap_all(tdir)
    cnemc = _cnemc_existing()
    anchors = {int(k): v for k, v in cfg["lny_anchors"].items()}

    # Prefer CHAP for any year it covers; CNEMC fills only the years CHAP lacks.
    tagged = []
    chap_years = set()
    if chap is not None:
        cw = attach_windows(chap, anchors, **win_kw)
        chap_years = set(cw["year"].unique())
        tagged.append(cw)
    cn = attach_windows(cnemc, anchors, **win_kw)
    cn = cn[~cn["year"].isin(chap_years)]
    if len(cn):
        tagged.append(cn)
    merged = pd.concat(tagged, ignore_index=True)
    print(f"  sources: CHAP years={sorted(chap_years)}, "
          f"CNEMC years={sorted(set(cn['year'].unique()))}")

    reg = pd.read_csv(os.path.join(ROOT, "outputs", "grids", "cell_regions.csv"))
    merged = merged.merge(reg[["grid_id", "region", "study_city"]], on="grid_id", how="left")

    mfile = os.path.join(ROOT, "outputs", "features", "era5_windows.parquet")
    if os.path.exists(mfile):
        met = pd.read_parquet(mfile)
        if {"wind_dir"}.issubset(met.columns):
            met["wind_dir_sin"] = np.sin(np.radians(met["wind_dir"]))
            met["wind_dir_cos"] = np.cos(np.radians(met["wind_dir"]))
        merged = merged.merge(met, on=["grid_id", "date"], how="left")
        print(f"  meteo covariates merged ({met.shape[0]} cell-days)")
    else:
        print("  [note] no ERA5 window meteo -> DiD without weather covariates (still valid)")
    return merged


def regions_of(panel) -> dict:
    """Return {region_name: cell-mask} for 5 provinces + 5 cities + National."""
    out = {"National": panel.index}
    for r in panel["region"].dropna().unique():
        out[r] = panel.index[panel["region"] == r]
    for c in panel["study_city"].dropna().unique():
        out[c] = panel.index[panel["study_city"] == c]
    return out


def run(cfg):
    panel = build_panel(cfg)
    have = [p for p in POLLUTANTS if p in panel.columns]
    meteo = [m for m in METEO if m in panel.columns]
    rows = []
    for region, idx in regions_of(panel).items():
        sub = panel.loc[idx]
        for p in have:
            # need a real 2019 (CHAP) control and 2020 (CNEMC) treatment for the DiD
            if sub.loc[sub.year == 2019, p].notna().sum() < 50:
                continue
            if sub.loc[sub.year == 2020, p].notna().sum() < 50:
                continue
            try:
                did = DID.estimate_did(sub, p, treat_year=2020, meteo_cols=meteo)
                rev = DID.estimate_reversion(sub, p)
                plac = DID.parallel_trends_placebo(sub, p, treat_year=2020)
            except Exception as e:
                print(f"  [skip] {region}/{p}: {e}"); continue
            rows.append({"region": region, "pollutant": p,
                         "H1_holiday2019_abs": did["holiday_2019"]["abs"],
                         "H1_holiday2019_pct": did["holiday_2019"]["pct"],
                         "H2_2020_abs": did["treat_within"]["abs"],
                         "H2_2020_pct": did["treat_within"]["pct"],
                         "H3_lockdown_DiD": did["did_abs"], "H3_se": did["did_se"],
                         "H3_ci_lo": did["did_ci_lo"], "H3_ci_hi": did["did_ci_hi"],
                         "H3_p": did["did_p"], "H3_cohend": did["treat_within"]["d"],
                         "H4_reversion_abs": rev["reversion_abs"],
                         "H4_reversion_pct": rev["reversion_pct"], "H4_p": rev["reversion_p"],
                         "placebo_DiD": plac["placebo_did"], "placebo_p": plac["placebo_p"],
                         "n_obs": did["n_obs"]})
    tab = pd.DataFrame(rows)
    if len(tab):
        tab["H3_qval"] = np.nan
        for p in have:                       # BH-FDR within each pollutant family
            m = tab["pollutant"] == p
            if m.sum():
                tab.loc[m, "H3_qval"] = multipletests(tab.loc[m, "H3_p"], method="fdr_bh")[1]
    outdir = os.path.join(ROOT, "outputs", "analysis"); os.makedirs(outdir, exist_ok=True)
    tab.to_csv(os.path.join(outdir, "significance_table.csv"), index=False)
    print(f"\nwrote {outdir}/significance_table.csv ({len(tab)} region x pollutant rows)")
    return tab


if __name__ == "__main__":
    run(_load_cfg())
