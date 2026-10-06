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


def run(cfg):
    # Build a panel per window length (each re-aligns the pre/peri windows).
    panels = {L: S.build_panel(cfg, pre=L, peri=L, post=L) for L in LENGTHS}
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
                    d = DID.estimate_did(sub, p, 2020, meteo_cols=None)
                    rec[f"H3_{L}d"] = d["did_abs"]
                    if L == 21:
                        rec["H3_p"] = d["did_p"]
                        rec["placebo_p"] = DID.parallel_trends_placebo(sub, p, 2020)["placebo_p"]
                if meteo:
                    rec["H3_21d_meteo"] = DID.estimate_did(base.loc[idx], p, 2020, meteo_cols=meteo)["did_abs"]
            except Exception as e:
                rec["error"] = str(e)[:60]
            rows.append(rec)
    tab = pd.DataFrame(rows)
    outdir = os.path.join(ROOT, "outputs", "analysis"); os.makedirs(outdir, exist_ok=True)
    tab.to_csv(os.path.join(outdir, "robustness_table.csv"), index=False)
    print("National H3 across window lengths (14/21/28 days):")
    print(tab[tab.region == "National"][["pollutant", "H3_14d", "H3_21d", "H3_28d", "placebo_p"]]
          .round(3).to_string(index=False))
    print(f"\nwrote {outdir}/robustness_table.csv  (meteo column: {'YES' if meteo else 'NO'})")
    return tab


if __name__ == "__main__":
    run(S._load_cfg())
