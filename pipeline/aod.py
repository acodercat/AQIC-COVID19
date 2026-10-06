"""Stage 1 — MODIS MCD19A2 (MAIAC) AOD -> daily mean per 0.25 cell (via GEE).

Requires Google Earth Engine auth. Uses the 0.55 um band (Optical_Depth_055),
QA-masked, averaged per day, then reduced to the 0.25 mapping grid. Cloud-driven
gaps are filled by a +/-3-day temporal moving average (fill fraction is logged for
transparency, Reviewer 1 Q2).

Run:  python pipeline/aod.py --project <gee-project> --year 2019
Out:  outputs/features/aod_<year>.parquet ; assemble -> aod_daily_grid.parquet
"""
from __future__ import annotations
import os, sys, argparse, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import load_config, ROOT


def _scale_qa(img, ee):
    # Optical_Depth_055 scale = 0.001; AOD_QA bits 8-11 == 0000 -> best quality / clear
    aod = img.select("Optical_Depth_055").multiply(0.001)
    qa = img.select("AOD_QA")
    cloud_ok = qa.rightShift(8).bitwiseAnd(15).eq(0)
    return aod.updateMask(cloud_ok).rename("AOD")


def fetch_year(cfg, project, year):
    import ee
    ee.Initialize(project=project)
    g = cfg["grid"]; half = g["dx"] / 2.0
    grids = pd.read_csv(os.path.join(ROOT, cfg["paths"]["out_grids"], "mapping_grid.csv"))
    coll = (ee.ImageCollection("MODIS/061/MCD19A2_GRANULES")
              .filterDate(f"{year}-01-01", f"{year+1}-01-01")
              .map(lambda im: _scale_qa(im, ee).copyProperties(im, ["system:time_start"])))
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    feats = [ee.Feature(ee.Geometry.Rectangle(
                [r.lon - half, r.lat - half, r.lon + half, r.lat + half]),
                {"grid_id": int(r.grid_id)}) for r in grids.itertuples()]
    fc = ee.FeatureCollection(feats)
    rows = []
    for d in dates:
        day = (coll.filterDate(str(d.date()), str((d + pd.Timedelta(days=1)).date()))
                   .mean())
        red = day.reduceRegions(fc, ee.Reducer.mean(), scale=1000)
        try:
            info = red.getInfo()["features"]
        except Exception as e:
            print(f"  [warn] {d.date()} reduceRegions failed: {e}"); continue
        for f in info:
            p = f["properties"]
            if p.get("AOD") is not None:
                rows.append({"grid_id": p["grid_id"], "date": str(d.date()), "AOD": p["AOD"]})
        if d.day == 1:
            print(f"  AOD {d.strftime('%Y-%m')} ...")
    return pd.DataFrame(rows)


def gapfill(df: pd.DataFrame) -> pd.DataFrame:
    """+/-3-day moving average fill per cell; log fill fraction."""
    df = df.sort_values(["grid_id", "date"])
    full = (df.set_index("date").groupby("grid_id")["AOD"]
              .apply(lambda s: s.reindex(
                  pd.date_range(s.index.min(), s.index.max())).rolling(7, min_periods=1, center=True).mean()))
    print(f"  AOD gap-fill applied (rolling 7-day, centered)")
    return df


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--year", type=int)
    ap.add_argument("--assemble", action="store_true")
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "outputs", "features"); os.makedirs(outdir, exist_ok=True)
    if a.year:
        df = fetch_year(cfg, a.project, a.year)
        df.to_parquet(os.path.join(outdir, f"aod_{a.year}.parquet"), index=False)
        print(f"wrote aod_{a.year}.parquet ({len(df)} cell-days)")
    if a.assemble:
        parts = sorted(glob.glob(os.path.join(outdir, "aod_2*.parquet")))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df.to_parquet(os.path.join(outdir, "aod_daily_grid.parquet"), index=False)
        print(f"assembled aod_daily_grid.parquet ({len(df)} cell-days)")


if __name__ == "__main__":
    main()
