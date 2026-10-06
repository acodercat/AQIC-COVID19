"""Composite reversion figure: 2021 minus 2019 (LNY peri window) for all 6 pollutants (R3.15).
Reuses the cached mapfields_<pol>_<year>.parquet produced by figures/maps.py.
Out: outputs/figures/reversion_composite.png
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
from grid import load_config
G = load_config()["grid"]
POLL = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$",
          "no2": "NO$_2$", "o3": "O$_3$ (MDA8)", "co": "CO"}


def field(p, y):
    f = os.path.join(ROOT, "outputs", "targets", f"mapfields_{p}_{y}.parquet")
    d = pd.read_parquet(f)
    return d[d.period == "peri"][["grid_id", "lat", "lon", p]].rename(columns={p: "v"})


def raster(d, col):
    lats = np.arange(G["lat_min"], G["lat_max"] + 1e-9, G["dx"])
    lons = np.arange(G["lon_min"], G["lon_max"] + 1e-9, G["dx"])
    arr = np.full((len(lats), len(lons)), np.nan)
    ri = np.round((d["lat"].values - G["lat_min"]) / G["dx"]).astype(int)
    ci = np.round((d["lon"].values - G["lon_min"]) / G["dx"]).astype(int)
    ok = (ri >= 0) & (ri < len(lats)) & (ci >= 0) & (ci < len(lons))
    arr[ri[ok], ci[ok]] = d[col].values[ok]
    return arr, (lons[0], lons[-1], lats[0], lats[-1])


cmap = plt.cm.RdBu_r.copy(); cmap.set_bad("0.92")
fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.5))
for ax, p in zip(axes.ravel(), POLL):
    a = field(p, 2019).merge(field(p, 2021), on=["grid_id", "lat", "lon"], suffixes=("19", "21"))
    a["diff"] = a.v21 - a.v19
    dmax = float(np.nanpercentile(np.abs(a["diff"]), 98))
    arr, ext = raster(a, "diff")
    im = ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=-dmax, vmax=dmax,
                   aspect="auto", interpolation="nearest")
    ax.set_title(f"{PRETTY[p]}: 2021 $-$ 2019", fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)   # full map-height colour bar
    cb.set_label("$\\Delta$ conc. (" + ("mg" if p == "co" else "$\\mu$g") + " m$^{-3}$)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
fig.tight_layout()
out = os.path.join(ROOT, "outputs", "figures", "reversion_composite.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
print("wrote", out)
