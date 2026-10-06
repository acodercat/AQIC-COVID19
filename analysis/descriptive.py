"""Part 1 — recompute ALL in-text descriptive numbers from the single CHAP dataset.

The manuscript's original descriptive numbers came from a different source and conflict with the
CHAP-based DiD by 20-40%. This recomputes, from CHAP at the study cells:
  - per-city pre vs 2020-lockdown means + % change (Beijing/Shanghai/Wuhan/Nanjing/Guangzhou);
  - national pre vs 2020-lockdown means;
  - national Lunar-New-Year peri-window means for 2019/2020/2021 (year comparison).
Windows are LNY-event-aligned (pre = LNY-21..-1, peri = LNY..+20; the 2020 peri ~ the 2020-CLD).
National = mean over the 747 monitoring-grid cells (the modelling grid), consistent with the DiD.

Out: outputs/analysis/descriptive_numbers.csv  (+ printed summary for transcription)
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from windows import lny_windows

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLL = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
CITIES = ["Beijing", "Shanghai", "Wuhan", "Nanjing", "Guangzhou"]


def _load_year(p, year):
    f = os.path.join(ROOT, "outputs", "targets", f"chap_{p}_{year}_D1K.parquet")
    return pd.read_parquet(f) if os.path.exists(f) else None


def _panel(cfg):
    reg = pd.read_csv(os.path.join(ROOT, "outputs/grids/cell_regions.csv"))[
        ["grid_id", "region", "study_city"]]
    anchors = {int(k): v for k, v in cfg["lny_anchors"].items()}
    win = lny_windows(anchors)
    frames = []
    for p in POLL:
        parts = []
        for y in (2019, 2020, 2021):
            d = _load_year(p, y)
            if d is not None:
                parts.append(d)
        if not parts:
            continue
        d = pd.concat(parts, ignore_index=True).merge(win, on="date", how="inner")
        frames.append(d.merge(reg, on="grid_id", how="left"))
    # merge pollutants on (grid_id,date,year,period,...) -> wide
    base = frames[0][["grid_id", "date", "year", "period", "region", "study_city"]]
    out = base.drop_duplicates()
    for fr, p in zip(frames, POLL):
        out = out.merge(fr[["grid_id", "date", p]], on=["grid_id", "date"], how="left")
    return out


def main():
    import yaml
    cfg = yaml.safe_load(open(os.path.join(ROOT, "pipeline", "config.yaml")))
    panel = _panel(cfg)
    rows = []

    def mean_of(df, p):
        return float(df[p].mean())

    # ---- per-city: 2020 pre vs peri (lockdown) ----
    print("=== CITY: 2020 pre-window vs lockdown(peri) ===")
    for city in CITIES:
        sub = panel[(panel.study_city == city) & (panel.year == 2020)]
        for p in POLL:
            pre = mean_of(sub[sub.period == "pre"], p)
            peri = mean_of(sub[sub.period == "peri"], p)
            pct = 100 * (peri - pre) / pre if pre else np.nan
            rows.append({"scope": city, "pollutant": p, "year": 2020,
                         "pre": round(pre, 2), "peri": round(peri, 2), "pct_change": round(pct, 2)})
        r = {x["pollutant"]: x for x in rows if x["scope"] == city}
        print(f"  {city:10s} NO2 {r['no2']['pre']}->{r['no2']['peri']} ({r['no2']['pct_change']:+.1f}%)  "
              f"PM2.5 {r['pm2_5']['pre']}->{r['pm2_5']['peri']} ({r['pm2_5']['pct_change']:+.1f}%)  "
              f"O3 {r['o3']['pre']}->{r['o3']['peri']} ({r['o3']['pct_change']:+.1f}%)")

    # ---- national: 2020 pre vs peri ----
    print("\n=== NATIONAL (747 cells): 2020 pre vs peri ===")
    natn = panel[panel.year == 2020]
    for p in POLL:
        pre = mean_of(natn[natn.period == "pre"], p)
        peri = mean_of(natn[natn.period == "peri"], p)
        pct = 100 * (peri - pre) / pre if pre else np.nan
        rows.append({"scope": "National", "pollutant": p, "year": 2020,
                     "pre": round(pre, 2), "peri": round(peri, 2), "pct_change": round(pct, 2)})
        print(f"  {p:6s}: {pre:.2f} -> {peri:.2f}  ({pct:+.1f}%)")

    # ---- national LNY peri means per year (2019/2020/2021) ----
    print("\n=== NATIONAL LNY peri-window mean by year ===")
    for p in POLL:
        vals = {}
        for y in (2019, 2020, 2021):
            vals[y] = mean_of(panel[(panel.year == y) & (panel.period == "peri")], p)
            rows.append({"scope": "National-LNYperi", "pollutant": p, "year": y,
                         "pre": np.nan, "peri": round(vals[y], 2), "pct_change": np.nan})
        d2021 = 100 * (vals[2021] - vals[2020]) / vals[2020] if vals[2020] else np.nan
        print(f"  {p:6s}: 2019={vals[2019]:.2f}  2020={vals[2020]:.2f}  2021={vals[2021]:.2f}  "
              f"(2021 vs 2020 {d2021:+.1f}%)")

    out = os.path.join(ROOT, "outputs", "analysis", "descriptive_numbers.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
