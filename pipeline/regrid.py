"""Area-weighted aggregation of a fine raster to the canonical 0.25 grid.

Used by CHAP (1 km), AOD (1 km), CLCD (30 m), SRTM (30 m). For a regular fine
lat/lon raster, every fine pixel is assigned to the 0.25 cell its CENTER falls in
(equivalent to area-weighting for equal-area fine pixels), then reduced by mean
(concentrations/elevation) or summed per class (land cover).

These functions are pure numpy/pandas and self-tested below; they do not download
anything. Each source module produces a fine (lat, lon, value) array/xarray and
calls these to land on the 0.25 grid.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def assign_cells(lat, lon, lat0=18.0, lon0=73.0, dx=0.25, ncol=249):
    """Vectorized point -> (row, col, grid_id) on the canonical grid."""
    row = np.floor((np.asarray(lat) - (lat0 - dx / 2)) / dx).astype(int)
    col = np.floor((np.asarray(lon) - (lon0 - dx / 2)) / dx).astype(int)
    gid = row * ncol + col + 1
    return row, col, gid


def reduce_mean(lat, lon, values, **grid):
    """Mean of fine values within each 0.25 cell. NaNs ignored. -> DataFrame(grid_id,value)."""
    lat = np.asarray(lat, float); lon = np.asarray(lon, float); values = np.asarray(values, float)
    m = np.isfinite(values)
    _, _, gid = assign_cells(lat[m], lon[m], **grid)
    df = pd.DataFrame({"grid_id": gid, "value": values[m]})
    return df.groupby("grid_id", as_index=False)["value"].mean()


def class_areas(lat, lon, classcode, pixel_area_m2, class_map: dict, **grid):
    """Per-cell area (m^2) for each land-cover class.

    classcode: integer class per fine pixel. class_map: {raw_code: out_name}.
    pixel_area_m2: scalar or per-pixel array of fine-pixel area.
    Returns DataFrame: grid_id + one column of m^2 per output class.
    """
    lat = np.asarray(lat, float); lon = np.asarray(lon, float)
    code = np.asarray(classcode)
    area = np.broadcast_to(np.asarray(pixel_area_m2, float), code.shape).ravel()
    _, _, gid = assign_cells(lat.ravel(), lon.ravel(), **grid)
    out_name = pd.Series(code.ravel()).map(class_map)
    df = pd.DataFrame({"grid_id": gid, "cls": out_name.values, "area": area})
    df = df.dropna(subset=["cls"])
    wide = (df.groupby(["grid_id", "cls"])["area"].sum()
              .unstack("cls", fill_value=0.0).reset_index())
    return wide


def _selftest():
    # 2x2 deg of fine pixels at 0.05 spacing around cell (18.25, 73.0)
    fine_lat = np.arange(18.15, 18.36, 0.05)
    fine_lon = np.arange(72.90, 73.11, 0.05)
    LON, LAT = np.meshgrid(fine_lon, fine_lat)
    # cell assignment: center (18.25,73.0) -> row1 col0 -> gid = 1*249+0+1 = 250
    r, c, g = assign_cells([18.25], [73.0])
    assert (r[0], c[0], g[0]) == (1, 0, 250), (r, c, g)
    # reduce_mean of constant field returns that constant for covered cells
    vals = np.full(LAT.shape, 7.0)
    out = reduce_mean(LAT.ravel(), LON.ravel(), vals.ravel())
    assert np.allclose(out["value"], 7.0)
    # class_areas: half pixels class 10 (->A), half class 20 (->B)
    code = np.where(LON < 73.0, 10, 20)
    wide = class_areas(LAT, LON, code, 1.0, {10: "A", 20: "B"})
    tot = wide[["A", "B"]].sum().sum()
    assert tot == code.size, (tot, code.size)
    print("regrid self-test OK")


if __name__ == "__main__":
    _selftest()
