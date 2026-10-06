"""Stage B (offline) — robustness of the DiD lockdown estimate to the event-window length.

The headline DiD uses 21-day pre/peri/post windows aligned to the Lunar New Year, with 2019 as
the holiday-only control. As a clean robustness check (no collinearity with the shifting LNY
calendar dates, unlike day-of-year detrending), we re-estimate the lockdown-net-of-holiday effect
(H3) using 14-, 21- and 28-day windows. A stable sign and magnitude across window lengths, together
with the parallel-trends placebo, indicates the estimate is not an artefact of the window choice.

If outputs/features/era5_windows.parquet is present (ERA5 download succeeded), a meteorology-adjusted
column is also produced.
Out: outputs/analysis/robustness_table.csv
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import significance as S
import did as DID

ROOT = S.ROOT
LENGTHS = [14, 21, 28]


def run(cfg, pre_gap=0, cluster="cell", tag=""):
    """Original call (no args) reproduces robustness_table.csv. Revised specification:
    run(cfg, pre_gap=2, cluster="twoway", tag="rev") -> robustness_table_rev.csv, which reports
    each window length with and without ERA5 covariates, with 95% CIs."""
    # Build a panel per window length (each re-aligns the pre/peri windows).
    panels = {L: S.build_panel(cfg, pre=L, peri=L, post=L, pre_gap=pre_gap) for L in LENGTHS}
    base = panels[21]
    meteo = [m for m in S.METEO if m in base.columns]
    rows = []
    for region, idx in S.regions_of(base).items():
        for p in [p for p in S.POLLUTANTS if p in base.columns]:
            sub21 = base.loc[idx]
            if sub21.loc[sub21.year == 2019, p].notna().sum() < 50:
                continue
            rec = {"region": region, "pollutant": p}
            try:
                for L in LENGTHS:
                    pl = panels[L]
                    sub = pl[(pl["region"] == region) | (pl["study_city"] == region)] \
                        if region != "National" else pl
                    for adj, cols in ((None, None), ("meteo", meteo if tag else None)):
                        if adj and not cols:
                            continue
                        d = DID.estimate_did(sub, p, 2020, meteo_cols=cols, cluster=cluster)
                        sfx = f"{L}d" + (f"_{adj}" if adj else "")
                        rec[f"H3_{sfx}"] = d["did_abs"]
                        if tag:
                            rec[f"ci_lo_{sfx}"], rec[f"ci_hi_{sfx}"] = d["did_ci_lo"], d["did_ci_hi"]
                            rec[f"p_{sfx}"] = d["did_p"]
                    if L == 21 and not tag:
                        rec["H3_p"] = d["did_p"]
                        rec["placebo_p"] = DID.parallel_trends_placebo(sub, p, 2020)["placebo_p"]
                if meteo and not tag:
                    rec["H3_21d_meteo"] = DID.estimate_did(base.loc[idx], p, 2020, meteo_cols=meteo)["did_abs"]
            except Exception as e:
                rec["error"] = str(e)[:60]
            rows.append(rec)
    tab = pd.DataFrame(rows)
    outdir = os.path.join(ROOT, "outputs", "analysis"); os.makedirs(outdir, exist_ok=True)
    fname = f"robustness_table{'_' + tag if tag else ''}.csv"
    tab.to_csv(os.path.join(outdir, fname), index=False)
    print("National H3 across window lengths (14/21/28 days):")
    cols = [c for c in tab.columns if c.startswith("H3_") and c[3:5].isdigit()]
    print(tab[tab.region == "National"][["pollutant"] + cols].round(3).to_string(index=False))
    print(f"\nwrote {outdir}/{fname}  (meteo: {'YES' if meteo else 'NO'})")
    return tab


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre-gap", type=int, default=0)
    ap.add_argument("--cluster", choices=["cell", "twoway"], default="cell")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    run(S._load_cfg(), pre_gap=a.pre_gap, cluster=a.cluster, tag=a.tag)
