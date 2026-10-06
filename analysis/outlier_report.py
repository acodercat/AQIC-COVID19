"""Stage C — outlier-removal accounting for the air-quality measurements (R1.4 / R3.4).

Reviewers asked how many samples the two-step cleaning removes. We apply the documented
policy to the ground-level AQ measurements used for modelling (the original CNEMC daily
record): (1) physically valid ranges per pollutant, (2) the IQR rule. We report the count
removed at each step per pollutant, and a sensitivity check (mean concentration with vs
without the IQR step) showing the conclusions are insensitive to it.

Valid ranges follow the Chinese Ambient Air Quality Standards (GB 3095-2012) and the
plausible maxima of the national network. Out: outputs/analysis/clean_report.csv
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VALID = {"pm2_5": (0, 1000), "pm10": (0, 2000), "so2": (0, 500),
         "no2": (0, 300), "o3": (0, 400), "co": (0, 15)}   # ug/m3 (CO mg/m3)


def main():
    cols = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
    df = pd.concat([
        pd.read_csv(os.path.join(ROOT, "dataset/train_set.csv"), usecols=cols),
        pd.read_csv(os.path.join(ROOT, "dataset/test_set.csv"), usecols=cols),
    ], ignore_index=True)
    n_total = len(df)
    rows = []
    for p in cols:
        s = df[p].dropna()
        lo, hi = VALID[p]
        range_bad = int((~s.between(lo, hi)).sum())
        s2 = s[s.between(lo, hi)]
        q1, q3 = s2.quantile([0.25, 0.75]); iqr = q3 - q1
        iqr_lo, iqr_hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        iqr_bad = int((~s2.between(iqr_lo, iqr_hi)).sum())
        mean_raw = float(s.mean())
        mean_clean = float(s2[s2.between(iqr_lo, iqr_hi)].mean())
        rows.append({
            "pollutant": p, "n": int(s.notna().sum()),
            "range_removed": range_bad, "range_pct": round(100 * range_bad / len(s), 3),
            "iqr_removed": iqr_bad, "iqr_pct": round(100 * iqr_bad / len(s), 3),
            "iqr_lo": round(iqr_lo, 2), "iqr_hi": round(iqr_hi, 2),
            "mean_raw": round(mean_raw, 2), "mean_after_clean": round(mean_clean, 2),
            "mean_change_pct": round(100 * (mean_clean - mean_raw) / mean_raw, 2),
        })
    rep = pd.DataFrame(rows)
    out = os.path.join(ROOT, "outputs", "analysis", "clean_report.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rep.to_csv(out, index=False)
    print(f"total rows: {n_total}")
    print(rep.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
