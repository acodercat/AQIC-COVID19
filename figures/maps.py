"""Stage 5 — national pre/peri/post maps (Fig 6/7 style) + 2019-vs-2021 reversion.

Computes 21-day-average pollutant fields over the FULL China 0.25 grid for each
LNY period (pre/peri/post) and year, from the cached CHAP daily files, then plots:
  - per-year pre/peri/post maps (like the manuscript Fig 6),
  - a 2020-CLD vs 2021-reference vs DIFF panel (like Fig 7),
  - a 2019 (pre-COVID) baseline panel and a 2019->2021 reversion DIFF (the new asks).

Window fields are cached to outputs/targets/mapfields_<pol>_<year>.parquet.
Run:  python figures/maps.py            # builds whatever years are downloaded
"""
from __future__ import annotations
import os, sys, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import chap, regrid                      # noqa: E402
from grid import load_config             # noqa: E402
from windows import lny_windows          # noqa: E402

PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$",
          "no2": "NO$_2$", "o3": "O$_3$", "co": "CO"}
PERIODS = ["pre", "peri", "post"]


def compute_fields(cfg, pollutant, year) -> pd.DataFrame | None:
    """21-day-average full-grid field per period for one pollutant-year (cached)."""
    import xarray as xr
    cache = os.path.join(ROOT, "outputs", "targets", f"mapfields_{pollutant}_{year}.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    try:
        paths = chap.fetch(pollutant, year, "D1K")     # cached nc paths
    except SystemExit:
        return None
    date2path = {chap._infer_date(p): p for p in paths}
    g = cfg["grid"]; gk = dict(lat0=g["lat0"], lon0=g["lon0"], dx=g["dx"], ncol=g["ncol"])
    full = pd.read_csv(os.path.join(ROOT, "outputs", "grids", "mapping_grid.csv"))
    keep = set(full["grid_id"])
    win = lny_windows({year: cfg["lny_anchors"][year]})
    rows = []
    for period in PERIODS:
        dates = win.loc[win.period == period, "date"]
        sel = [date2path[d] for d in dates if d in date2path]
        if not sel:
            continue
        acc = cnt = None
        lat = lon = None
        for p in sel:
            ds = xr.open_dataset(p)
            var = list(ds.data_vars)[0]
            if lat is None:
                lat = ds["lat" if "lat" in ds.coords else "latitude"].values
                lon = ds["lon" if "lon" in ds.coords else "longitude"].values
            v = np.asarray(ds[var].squeeze().values, float)
            v = np.where(v < -900, np.nan, v)
            m = np.isfinite(v)
            acc = np.where(m, v, 0.0) if acc is None else acc + np.where(m, v, 0.0)
            cnt = m.astype(float) if cnt is None else cnt + m
        mean_field = np.where(cnt > 0, acc / np.maximum(cnt, 1), np.nan)
        LON, LAT = np.meshgrid(lon, lat)
        cell = regrid.reduce_mean(LAT.ravel(), LON.ravel(), mean_field.ravel(), **gk)
        cell = cell[cell["grid_id"].isin(keep)].merge(full[["grid_id", "lat", "lon"]], on="grid_id")
        cell["period"] = period
        rows.append(cell)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True).rename(columns={"value": pollutant})
    out.to_parquet(cache, index=False)
    return out


def _scatter(ax, df, col, vmin, vmax, title, cmap="RdYlGn_r"):
    sc = ax.scatter(df["lon"], df["lat"], c=df[col], s=4, marker="s",
                    cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9); ax.set_aspect(1.15)
    ax.set_xticks([]); ax.set_yticks([])
    return sc


def panel_pre_peri_post(cfg, pollutant, year, outdir):
    df = compute_fields(cfg, pollutant, year)
    if df is None:
        return None
    vmax = np.nanpercentile(df[pollutant], 98)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, period in zip(axes, PERIODS):
        d = df[df.period == period]
        sc = _scatter(ax, d, pollutant, 0, vmax, f"{period}")
    fig.suptitle(f"{PRETTY.get(pollutant, pollutant)} — {year} pre/peri/post Lunar New Year")
    fig.colorbar(sc, ax=axes, shrink=0.7, label="conc.")
    path = os.path.join(outdir, f"map_{pollutant}_{year}_prepost.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def reversion_panel(cfg, pollutant, outdir):
    """2019 peri vs 2021 peri vs DIFF (did 2021 revert to pre-COVID?)."""
    a = compute_fields(cfg, pollutant, 2019)
    b = compute_fields(cfg, pollutant, 2021)
    if a is None or b is None:
        return None
    a = a[a.period == "peri"][["grid_id", "lat", "lon", pollutant]].rename(columns={pollutant: "y2019"})
    b = b[b.period == "peri"][["grid_id", pollutant]].rename(columns={pollutant: "y2021"})
    m = a.merge(b, on="grid_id"); m["diff"] = m["y2021"] - m["y2019"]
    vmax = np.nanpercentile(m[["y2019", "y2021"]].values, 98)
    dmax = np.nanpercentile(np.abs(m["diff"]), 98)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    _scatter(axes[0], m, "y2019", 0, vmax, "2019 (pre-COVID)")
    _scatter(axes[1], m, "y2021", 0, vmax, "2021")
    sc = _scatter(axes[2], m, "diff", -dmax, dmax, "2021 − 2019 (Δ)", cmap="RdBu_r")
    fig.suptitle(f"{PRETTY.get(pollutant, pollutant)} — reversion to pre-COVID (LNY window)")
    fig.colorbar(sc, ax=axes[2], shrink=0.7, label="Δ conc.")
    path = os.path.join(outdir, f"map_{pollutant}_reversion.png")
    fig.savefig(path, dpi=140, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    cfg = load_config()
    cfg["lny_anchors"] = {int(k): v for k, v in cfg["lny_anchors"].items()}
    outdir = os.path.join(ROOT, "outputs", "figures"); os.makedirs(outdir, exist_ok=True)
    have = sorted({os.path.basename(f).split("_")[1]
                   for f in glob.glob(os.path.join(ROOT, "outputs/targets/chap_*_*_D1K.parquet"))})
    # rebuild pollutant tokens from filenames (chap_<pol>_<year>_D1K)
    pols = sorted({"_".join(os.path.basename(f).split("_")[1:-2])
                   for f in glob.glob(os.path.join(ROOT, "outputs/targets/chap_*_*_D1K.parquet"))})
    for p in pols:
        for y in (2019, 2020, 2021):
            r = panel_pre_peri_post(cfg, p, y, outdir)
            if r: print("  ", r)
        r = reversion_panel(cfg, p, outdir)
        if r: print("  ", r)


if __name__ == "__main__":
    main()
