"""Stage 1 — ERA5 near-surface meteorology -> daily 0.25 grid (Beijing time).

Downloads ERA5 single-level hourly fields for the China bbox over the study period
via cdsapi, then derives the model's met features (metcalc) and aggregates to daily
means in UTC+8. ERA5 is already on a 0.25 grid, so cells align to the canonical grid
directly (cell-center match).

Credentials: a Copernicus CDS account + ~/.cdsapirc  (https://cds.climate.copernicus.eu).
Run:  python pipeline/era5.py --year 2019           # one year at a time (large)
Out:  outputs/features/era5_<year>.nc  and  outputs/features/era5_daily_grid.parquet
"""
from __future__ import annotations
import os, sys, argparse, glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metcalc
from grid import load_config, ROOT, latlon_to_cell

VARS = ["2m_temperature", "2m_dewpoint_temperature",
        "10m_u_component_of_wind", "10m_v_component_of_wind",
        "surface_pressure", "total_column_water_vapour",
        "instantaneous_10m_wind_gust"]
SHORT = {"t2m": "t2m", "d2m": "d2m", "u10": "u10", "v10": "v10",
         "sp": "sp", "tcwv": "pwat", "i10fg": "gust"}


def download(cfg, year: int) -> str:
    """Submit a CDS request for one year; returns the netCDF path."""
    try:
        import cdsapi
    except ImportError:
        raise SystemExit("pip install cdsapi, and set up ~/.cdsapirc")
    g = cfg["grid"]
    outdir = os.path.join(ROOT, "outputs", "features"); os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"era5_{year}.nc")
    if os.path.exists(out):
        print(f"  era5_{year}.nc exists, skipping download"); return out
    # CDS area = [North, West, South, East]
    area = [g["lat_max"], g["lon_min"], g["lat_min"], g["lon_max"]]
    c = cdsapi.Client()
    c.retrieve("reanalysis-era5-single-levels", {
        "product_type": "reanalysis", "format": "netcdf", "variable": VARS,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": area, "grid": [g["dx"], g["dx"]],
    }, out)
    return out


def to_daily_grid(cfg, nc_paths: list[str]) -> pd.DataFrame:
    """Load hourly netCDF(s), derive features, aggregate to daily UTC+8 per cell."""
    import xarray as xr
    g = cfg["grid"]
    frames = []
    for path in nc_paths:
        ds = xr.open_dataset(path)
        ds = ds.rename({k: v for k, v in SHORT.items() if k in ds.variables})
        # stack to long (time, lat, lon)
        df = ds[["t2m", "d2m", "u10", "v10", "sp", "pwat", "gust"]].to_dataframe().reset_index()
        latc = "latitude" if "latitude" in df else "lat"
        lonc = "longitude" if "longitude" in df else "lon"
        timec = "time" if "time" in df else "valid_time"
        feats = metcalc.derive_era5_features(df["t2m"], df["d2m"], df["u10"], df["v10"],
                                             df["sp"], df["gust"])
        cell = latlon_to_cell(df[latc].values, df[lonc].values, g)
        long = pd.DataFrame({"time": df[timec], "grid_id": cell["grid_id"], **feats})
        vcols = ["t2m", "rh", "spfh", "sp", "wind_speed", "wind_dir", "gust", "pwat"]
        long["pwat"] = df["pwat"].values
        daily = metcalc.to_beijing_daily(long, "time", vcols,
                                         cfg["period"]["tz_offset_hours"],
                                         how={"gust": "max"})
        frames.append(daily)
    out = pd.concat(frames, ignore_index=True)
    return out


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, help="download+process a single year")
    ap.add_argument("--assemble", action="store_true",
                    help="combine all outputs/features/era5_*.nc into the parquet")
    a = ap.parse_args()
    outdir = os.path.join(ROOT, "outputs", "features")
    if a.year:
        nc = download(cfg, a.year)
        df = to_daily_grid(cfg, [nc])
        df.to_parquet(os.path.join(outdir, f"era5_daily_{a.year}.parquet"), index=False)
        print(f"wrote era5_daily_{a.year}.parquet ({len(df)} cell-days)")
    if a.assemble:
        parts = sorted(glob.glob(os.path.join(outdir, "era5_daily_*.parquet")))
        df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        df.to_parquet(os.path.join(outdir, "era5_daily_grid.parquet"), index=False)
        print(f"assembled era5_daily_grid.parquet ({len(df)} cell-days from {len(parts)} years)")


if __name__ == "__main__":
    main()
