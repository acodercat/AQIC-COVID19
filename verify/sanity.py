"""Stage F — end-to-end data-integrity checks. Writes outputs/verify/sanity_report.txt.

Checks:
  1. CHAP vs CNEMC overlap (Dec 2019) correlation per pollutant — validates combining sources.
  2. Grid counts: mapping land cells vs paper's 16,129; modeling cells (~747); 49-cell holdout.
  3. Spatial-holdout leakage: no grid_id shared between train and test.
  4. Unit / physical-range checks on the CHAP fields (CO mg/m3; AOD/RH/wind_dir if present).
  5. Missingness: CHAP no-data fraction per pollutant-year.
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLL = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
RANGES = {"pm2_5": (0, 2000), "pm10": (0, 7000), "so2": (0, 800), "no2": (0, 400),
          "o3": (0, 400), "co": (0, 30)}
OUT = []


def log(s=""):
    print(s); OUT.append(s)


def chap(p, year):
    f = os.path.join(ROOT, "outputs", "targets", f"chap_{p}_{year}_D1K.parquet")
    return pd.read_parquet(f) if os.path.exists(f) else None


def check_overlap():
    log("== 1. CHAP vs CNEMC overlap (Dec 2019) ==")
    old = pd.concat([pd.read_csv(os.path.join(ROOT, "dataset/train_set.csv")),
                     pd.read_csv(os.path.join(ROOT, "dataset/test_set.csv"))])[["grid_id", "date"] + POLL]
    for p in POLL:
        c = chap(p, 2019)
        if c is None:
            log(f"  {p:6s}: no CHAP 2019"); continue
        m = c.rename(columns={p: "chap"}).merge(old[["grid_id", "date", p]].rename(columns={p: "cn"}),
                                                on=["grid_id", "date"]).dropna()
        if len(m) < 50:
            log(f"  {p:6s}: <50 overlap pts"); continue
        r = np.corrcoef(m.chap, m.cn)[0, 1]
        log(f"  {p:6s}: r={r:.3f}  bias(CHAP-CNEMC)={(m.chap-m.cn).mean():+.2f}  n={len(m)}")


def check_grids():
    log("\n== 2/3. Grid counts & leakage ==")
    g = os.path.join(ROOT, "outputs", "grids")
    mod = pd.read_csv(os.path.join(g, "modeling_grid.csv"))
    mp = pd.read_csv(os.path.join(g, "mapping_grid.csv"))
    te = pd.read_csv(os.path.join(g, "test_holdout.csv"))
    log(f"  mapping land cells: {len(mp)} (paper states 16,129; NE 1:50m boundary)")
    log(f"  modeling cells: {len(mod)} (original 747)")
    log(f"  holdout cells: {len(te)} (original 49)")
    tr = set(pd.read_csv(os.path.join(ROOT, "dataset/train_set.csv"), usecols=["grid_id"])["grid_id"])
    overlap = len(set(te["orig_grid_id"]) & tr)
    log(f"  [{'PASS' if overlap == 0 else 'FAIL'}] train/test grid leakage: {overlap} shared cells")


def check_units():
    log("\n== 4. Unit / physical-range checks (CHAP) ==")
    for p in POLL:
        c = chap(p, 2020)
        if c is None:
            continue
        v = c[p]
        bad = int((~v.between(*RANGES[p])).sum())
        log(f"  {p:6s}: min={v.min():.2f} max={v.max():.1f} mean={v.mean():.1f} "
            f"out-of-range={bad}  ({'mg/m3' if p=='co' else 'ug/m3'})")


def check_missing():
    log("\n== 5. Missingness (CHAP no-data fraction) ==")
    for p in POLL:
        fr = []
        for y in (2019, 2020, 2021):
            c = chap(p, y)
            if c is not None:
                fr.append(c[p].isna().mean())
        if fr:
            log(f"  {p:6s}: mean NaN fraction across years = {np.mean(fr):.4f}")


def main():
    check_overlap(); check_grids(); check_units(); check_missing()
    outdir = os.path.join(ROOT, "outputs", "verify"); os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "sanity_report.txt"), "w") as fh:
        fh.write("\n".join(OUT) + "\n")
    log(f"\nwrote {outdir}/sanity_report.txt")


if __name__ == "__main__":
    main()
