"""Stage B — small ERA5 slice over the Lunar-New-Year windows, for DiD weather covariates.

Instead of full 3-year ERA5 (~25 GB), we download only the months spanning the LNY
windows (Dec–Mar) of 2019/2020/2021, at 4 synoptic hours (00/06/12/18 UTC, matching the
original 6-hourly meteorology), over the China bbox, then derive near-surface features
(metcalc) and aggregate to daily means in Beijing time, restricted to the 747 modeling cells.

Output: outputs/features/era5_windows.parquet  (grid_id, date, t2m, rh, spfh, sp,
        wind_speed, wind_dir, pwat[, gust]) — consumed by analysis/significance.py.

Credentials: ~/.cdsapirc (already validated).
Run:  python pipeline/era5_windows.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metcalc
from grid import load_config, ROOT, latlon_to_cell
from era5 import VARS, SHORT

# Window-spanning months per LNY year (covers placebo..post). 2019 has no Dec-2018 in CHAP,
# so we start 2019 in January (the 2019 placebo is truncated, consistent with CHAP coverage).
YEAR_MONTHS = {
    2019: [(2019, [1, 2, 3])],
    2020: [(2019, [12]), (2020, [1, 2, 3])],
    2021: [(2021, [1, 2, 3])],
}
HOURS = ["00:00", "06:00", "12:00", "18:00"]


def download(cfg, cal_year, months) -> str:
    import cdsapi
    g = cfg["grid"]
    outdir = os.path.join(ROOT, "outputs", "features", "_era5win"); os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"era5win_{cal_year}_{months[0]:02d}-{months[-1]:02d}.nc")
    if os.path.exists(out):
        return out
    area = [g["lat_max"], g["lon_min"], g["lat_min"], g["lon_max"]]  # N,W,S,E
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "format": "netcdf", "variable": VARS,
        "year": str(cal_year), "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in range(1, 32)], "time": HOURS,
        "area": area, "grid": [g["dx"], g["dx"]],
    }, out)
    return out


def to_daily_cells(cfg, nc_paths, keep_cells) -> pd.DataFrame:
    import xarray as xr
    g = cfg["grid"]
    frames = []
    for path in nc_paths:
        ds = xr.open_dataset(path).rename({k: v for k, v in SHORT.items() if k in xr.open_dataset(path).variables})
        df = ds[["t2m", "d2m", "u10", "v10", "sp", "pwat", "gust"]].to_dataframe().reset_index()
        latc = "latitude" if "latitude" in df else "lat"
        lonc = "longitude" if "longitude" in df else "lon"
        timec = "time" if "time" in df else "valid_time"
        feats = metcalc.derive_era5_features(df["t2m"], df["d2m"], df["u10"], df["v10"], df["sp"], df["gust"])
        gid = latlon_to_cell(df[latc].values, df[lonc].values, g)["grid_id"]
        long = pd.DataFrame({"time": df[timec], "grid_id": gid, "pwat": df["pwat"].values, **feats})
        long = long[long["grid_id"].isin(keep_cells)]
        daily = metcalc.to_beijing_daily(
            long, "time", ["t2m", "rh", "spfh", "sp", "wind_speed", "wind_dir", "gust", "pwat"],
            cfg["period"]["tz_offset_hours"], how={"gust": "max"})
        frames.append(daily)
    return pd.concat(frames, ignore_index=True)


def main():
    cfg = load_config()
    keep = set(pd.read_csv(os.path.join(ROOT, "outputs/grids/modeling_grid.csv"))["grid_id"])
    paths = []
    for lny_year, reqs in YEAR_MONTHS.items():
        for cal_year, months in reqs:
            print(f"  ERA5 window slice: {cal_year} months {months} ...")
            paths.append(download(cfg, cal_year, months))
    df = to_daily_cells(cfg, paths, keep)
    out = os.path.join(ROOT, "outputs", "features", "era5_windows.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} cell-days, {df['grid_id'].nunique()} cells, "
          f"{df['date'].min()}..{df['date'].max()})")


if __name__ == "__main__":
    main()
