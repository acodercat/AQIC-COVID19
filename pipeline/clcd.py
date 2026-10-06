"""Stage 1 — CLCD 30 m land cover -> per-cell class AREA (m^2) and FRACTION (%).

Downloads the CLCD GeoTIFF (Zenodo, public; covers 2019) and computes, for each
0.25 cell, the area per class (reproduces the original m^2 schema) AND the fraction
(reproduces the manuscript's "%" text). Ocean is derived from the China land mask
(cells/pixels outside the boundary). CLCD is reprojected to EPSG:4326; 30 m pixel
area is taken from the raster's local pixel size.

CLCD class codes -> paper's 7-class schema:
  1 Cropland -> CultivatedLand   2 Forest -> WoodLand    4 Grassland -> GrassLand
  5 Water -> Waters              8 Impervious -> UrbanRural   9 Barren -> UnusedLand
(Shrub/Wetland/Snow are folded: Shrub->WoodLand, Wetland->Waters, Snow->UnusedLand.)

Run:  python pipeline/clcd.py --tif CLCD_v01_2019_albert.tif --year 2019
Out:  outputs/features/landcover_grid.parquet
"""
from __future__ import annotations
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regrid
from grid import load_config, ROOT

CLASS_MAP = {1: "CultivatedLand", 2: "WoodLand", 3: "WoodLand",      # shrub->wood
             4: "GrassLand", 5: "Waters", 6: "Waters",              # wetland->water
             7: "UnusedLand", 8: "UrbanRural", 9: "UnusedLand"}      # snow/barren->unused
OUT_CLASSES = ["CultivatedLand", "WoodLand", "GrassLand", "Waters", "UrbanRural", "UnusedLand"]


def to_grid(cfg, tif: str, year: int) -> pd.DataFrame:
    import rioxarray  # noqa: F401
    import xarray as xr
    g = cfg["grid"]
    grid_kw = dict(lat0=g["lat0"], lon0=g["lon0"], dx=g["dx"], ncol=g["ncol"])
    da = xr.open_dataarray(tif, engine="rasterio").squeeze()
    da = da.rio.reproject("EPSG:4326")
    lat = da["y"].values; lon = da["x"].values
    LON, LAT = np.meshgrid(lon, lat)
    code = np.asarray(da.values)
    # per-pixel area (m^2) ~ (dx_deg*111320)*(dy_deg*111320*cos(lat))
    dy = abs(float(lat[1] - lat[0])); dx = abs(float(lon[1] - lon[0]))
    pix_area = (dy * 111320.0) * (dx * 111320.0 * np.cos(np.radians(LAT)))
    wide = regrid.class_areas(LAT, LON, code, pix_area, CLASS_MAP, **grid_kw)
    for c in OUT_CLASSES:
        if c not in wide: wide[c] = 0.0
    wide["land_m2"] = wide[OUT_CLASSES].sum(axis=1)
    # cell nominal area for Ocean = max(land area across cells at that latitude) proxy:
    # better: cell_area from center latitude
    clat, _ = _cell_centers(wide["grid_id"].values, g)
    cell_area = (g["dx"] * 111320.0) ** 2 * np.cos(np.radians(clat))
    wide["Ocean"] = np.clip(cell_area - wide["land_m2"], 0, None)
    for c in OUT_CLASSES + ["Ocean"]:
        wide[f"{c}_pct"] = 100.0 * wide[c] / cell_area
    wide["year"] = year
    return wide


def _cell_centers(grid_id, g):
    idx = np.asarray(grid_id) - 1
    row = idx // g["ncol"]; col = idx % g["ncol"]
    return g["lat0"] + row * g["dx"], g["lon0"] + col * g["dx"]


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--tif", required=True, help="path to CLCD GeoTIFF")
    ap.add_argument("--year", type=int, required=True)
    a = ap.parse_args()
    df = to_grid(cfg, a.tif, a.year)
    out = os.path.join(ROOT, "outputs", "features", "landcover_grid.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} cells)")


if __name__ == "__main__":
    main()
