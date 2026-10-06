"""Revision (R2 #5, R5 2.7 / C4) — does CHAP reproduce the station-observed lockdown signal?

The DiD uses CHAP, so its reliability must be shown for CHAP itself, during the lockdown, and
specifically for the CHANGE the DiD estimates (a smoothing ML product could attenuate an abrupt,
out-of-distribution signal). CNEMC station data (aggregated to the same 0.25 deg cells) exist for
Dec 2019 - Nov 2021, so the 2020 and 2021 LNY windows can be compared at identical cells:

  A. daily agreement CHAP vs CNEMC in the 2020 pre+peri window (r, bias, OLS slope);
  B. within-2020 change (peri - pre, = H2) per cell: CNEMC vs CHAP -- national/regional means,
     cross-cell correlation, and attenuation ratio CHAP/CNEMC;
  C. the same DiD (2020 treated, 2021 control; pre_gap=2, two-way clustered) estimated on each
     source at identical cell-days.

Caveat: CHAP was trained with CNEMC observations, so agreement at monitored cells is an upper
bound on accuracy, not independent validation. The comparison is still informative about
attenuation of the abrupt lockdown signal. CHAP O3 is MDA8 whereas the CNEMC series is a daily
mean, so O3 levels (not changes) differ systematically.

Out: outputs/verify/chap_lockdown_validation.csv (+ _daily.csv)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import did as DID
import significance as S
from windows import attach_windows

PRE_GAP = 2
REGIONS = ["National", "BTH", "Jilin", "Jiangsu", "Guangdong", "Xinjiang"]


def paired_panel(cfg):
    anchors = {y: cfg["lny_anchors"][y] for y in (2020, 2021)}
    cn = attach_windows(S._cnemc_existing(), anchors, pre_gap=PRE_GAP)
    ch = attach_windows(S._chap_all(os.path.join(ROOT, "outputs", "targets")), anchors,
                        pre_gap=PRE_GAP)
    keys = ["grid_id", "date", "year", "period", "event_day"]
    m = cn[keys + S.POLLUTANTS].merge(ch[keys + S.POLLUTANTS], on=keys,
                                      suffixes=("_cnemc", "_chap"))
    m = m[m.period.isin(["pre", "peri"])]
    reg = pd.read_csv(os.path.join(ROOT, "outputs", "grids", "cell_regions.csv"))
    return m.merge(reg[["grid_id", "region", "study_city"]], on="grid_id", how="left")


def run(cfg):
    m = paired_panel(cfg)
    rows, daily = [], []
    for region in REGIONS:
        sub = m if region == "National" else m[m.region == region]
        for p in S.POLLUTANTS:
            a, b = f"{p}_cnemc", f"{p}_chap"
            s = sub.dropna(subset=[a, b])
            # cells observed in both periods of BOTH years -> identical sample for both sources
            full = s.groupby("grid_id")[["year", "period"]].apply(
                lambda g: len(set(zip(g.year, g.period)))) == 4
            s = s[s.grid_id.isin(full[full].index)]
            if s.grid_id.nunique() < 3:
                continue
            # A. daily agreement, 2020 window
            s20 = s[s.year == 2020]
            slope = np.polyfit(s20[a], s20[b], 1)[0]
            # B. within-2020 change per cell
            cm = s20.groupby(["grid_id", "period"])[[a, b]].mean().unstack("period")
            dcn = cm[(a, "peri")] - cm[(a, "pre")]
            dch = cm[(b, "peri")] - cm[(b, "pre")]
            # C. DiD 2020 vs 2021 on each source, identical cell-days
            dd = {}
            for src, col in (("cnemc", a), ("chap", b)):
                r = DID.estimate_did(s.rename(columns={col: "v"}), "v", treat_year=2020,
                                     control_year=2021, cluster="twoway")
                dd[src] = r
            rows.append({
                "region": region, "pollutant": p, "n_cells": s.grid_id.nunique(),
                "A_r_daily_2020": s20[[a, b]].corr().iloc[0, 1],
                "A_bias_chap_minus_cnemc": (s20[b] - s20[a]).mean(),
                "A_slope_chap_on_cnemc": slope,
                "B_change2020_cnemc": dcn.mean(), "B_change2020_chap": dch.mean(),
                "B_attenuation_chap_over_cnemc": dch.mean() / dcn.mean() if dcn.mean() else np.nan,
                "B_r_change_across_cells": np.corrcoef(dcn, dch)[0, 1],
                "C_DiD_2020v2021_cnemc": dd["cnemc"]["did_abs"],
                "C_ci_cnemc": f"[{dd['cnemc']['did_ci_lo']:.2f}, {dd['cnemc']['did_ci_hi']:.2f}]",
                "C_p_cnemc": dd["cnemc"]["did_p"],
                "C_DiD_2020v2021_chap": dd["chap"]["did_abs"],
                "C_ci_chap": f"[{dd['chap']['did_ci_lo']:.2f}, {dd['chap']['did_ci_hi']:.2f}]",
                "C_p_chap": dd["chap"]["did_p"],
            })
            if region == "National":   # national daily means for plotting / inspection
                g = s.groupby(["year", "event_day"])[[a, b]].mean().reset_index()
                daily.append(g.rename(columns={a: "cnemc", b: "chap"}).assign(pollutant=p))
    tab = pd.DataFrame(rows)
    odir = os.path.join(ROOT, "outputs", "verify"); os.makedirs(odir, exist_ok=True)
    tab.to_csv(os.path.join(odir, "chap_lockdown_validation.csv"), index=False)
    pd.concat(daily).to_csv(os.path.join(odir, "chap_lockdown_validation_daily.csv"), index=False)
    pd.set_option("display.width", 250)
    print(tab.round(3).to_string(index=False))
    return tab


if __name__ == "__main__":
    run(S._load_cfg())
