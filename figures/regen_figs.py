"""Part 1 — regenerate Figs 5, 6, 7 from the single CHAP dataset (consistency + R3.15).

Fig 6: 2020 pre/peri/post national maps, 3 rows (periods) x 6 cols (pollutants), CHAP.
Fig 7: Lunar-New-Year-window national maps, 3 rows (2019/2020/2021) x 6 cols, CHAP -> adds the
       2019 pre-COVID baseline the reviewer requested.
Fig 5: SO2 event-time curves for the five study provinces (2019/2020/2021).
Reuses figures/maps.py:compute_fields (reads the cached mapfields_*.parquet).
Writes outputs/figures/{fig5,fig6,fig7}_chap.png and copies over Figures/fig{5,6,7}.png.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "analysis")); sys.path.insert(0, os.path.join(ROOT, "pipeline"))
import maps
from grid import load_config
from windows import lny_windows
cfg = load_config(); cfg["lny_anchors"] = {int(k): v for k, v in cfg["lny_anchors"].items()}
POLL = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$", "no2": "NO$_2$", "o3": "O$_3$ (MDA8)", "co": "CO"}
OUT = os.path.join(ROOT, "outputs", "figures")


UNITS = {"pm2_5": "μg m$^{-3}$", "pm10": "μg m$^{-3}$", "so2": "μg m$^{-3}$",
         "no2": "μg m$^{-3}$", "o3": "μg m$^{-3}$", "co": "mg m$^{-3}$"}


def _raster(d, g):
    """Rasterize an (lat, lon, v) frame onto the full 0.25 grid (NaN off-land)."""
    lats = np.arange(g["lat_min"], g["lat_max"] + 1e-9, g["dx"])
    lons = np.arange(g["lon_min"], g["lon_max"] + 1e-9, g["dx"])
    arr = np.full((len(lats), len(lons)), np.nan)
    ri = np.round((d["lat"].values - g["lat_min"]) / g["dx"]).astype(int)
    ci = np.round((d["lon"].values - g["lon_min"]) / g["dx"]).astype(int)
    ok = (ri >= 0) & (ri < len(lats)) & (ci >= 0) & (ci < len(lons))
    arr[ri[ok], ci[ok]] = d["v"].values[ok]
    return arr, (lons[0], lons[-1], lats[0], lats[-1])


def _maps_grid(col_keys, get_field, fname, col_label):
    """Pollutants (rows) x periods/years (cols) raster maps, sized to the journal text width
    (~6.9 in) so the labels print at true size; one vertical colour bar per pollutant row."""
    import matplotlib as mpl
    g = cfg["grid"]
    nr, nc = len(POLL), len(col_keys)
    fig, axes = plt.subplots(nr, nc, figsize=(2.05 * nc + 0.8, 1.55 * nr + 0.35), squeeze=False)
    cmap = plt.cm.YlOrRd.copy(); cmap.set_bad("0.92")
    vmax, cache = {}, {}
    for p in POLL:
        allv = []
        for ck in col_keys:
            d = get_field(p, ck); cache[(p, ck)] = d
            if d is not None and len(d):
                allv.append(d["v"].values)
        vmax[p] = float(np.nanpercentile(np.concatenate(allv), 98)) if allv else 1.0
    for i, p in enumerate(POLL):
        for j, ck in enumerate(col_keys):
            ax = axes[i, j]; d = cache[(p, ck)]
            if d is not None and len(d):
                arr, ext = _raster(d, g)
                ax.imshow(arr, origin="lower", extent=ext, cmap=cmap, vmin=0, vmax=vmax[p],
                          aspect="auto", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4)
            if i == 0:
                ax.set_title(col_label(ck), fontsize=9, fontweight="bold", pad=4)
            if j == 0:
                ax.set_ylabel(PRETTY[p], fontsize=9)
        sm = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(0, vmax[p]), cmap=cmap)
        cb = fig.colorbar(sm, ax=axes[i, :].tolist(), location="right", fraction=0.02, pad=0.015)
        cb.ax.tick_params(labelsize=6)
        cb.set_label(UNITS[p], fontsize=6.5)
    fig.savefig(os.path.join(OUT, fname), dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", fname)


def fig6():
    def gf(p, period):
        d = maps.compute_fields(cfg, p, 2020)
        if d is None: return None
        s = d[d.period == period][["lat", "lon", p]].rename(columns={p: "v"})
        return s
    _maps_grid(["pre", "peri", "post"], gf, "fig6_chap.png",
               lambda k: {"pre": "Pre", "peri": "Peri (lockdown)", "post": "Post"}[k])


def fig7():
    def gf(p, year):
        d = maps.compute_fields(cfg, p, year)
        if d is None: return None
        return d[d.period == "peri"][["lat", "lon", p]].rename(columns={p: "v"})
    _maps_grid([2019, 2020, 2021], gf, "fig7_chap.png", lambda k: str(k))


def fig5():
    reg = pd.read_csv(os.path.join(ROOT, "outputs/grids/cell_regions.csv"))
    win = lny_windows(cfg["lny_anchors"])
    provs = ["BTH", "Jiangsu", "Jilin", "Guangdong", "Xinjiang"]
    so2 = pd.concat([pd.read_parquet(os.path.join(ROOT, "outputs/targets", f"chap_so2_{y}_D1K.parquet"))
                     for y in (2019, 2020, 2021)], ignore_index=True)
    so2 = so2.merge(win, on="date", how="inner").merge(reg[["grid_id", "region"]], on="grid_id")
    fig, axes = plt.subplots(1, 5, figsize=(7.3, 1.9), sharey=True)
    for ax, prov in zip(axes, provs):
        d = so2[so2.region == prov]
        for yr, c in zip([2019, 2020, 2021], ["#1b9e77", "#d95f02", "#7570b3"]):
            g = d[d.year == yr].groupby("event_day")["so2"].mean()
            ax.plot(g.index, g.values, color=c, lw=1.6, label=str(yr))
        ax.axvspan(0, 20, color="grey", alpha=0.12); ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(prov); ax.grid(alpha=0.3); ax.set_xlabel("Days from LNY")
    axes[0].set_ylabel("SO$_2$ ($\\mu$g m$^{-3}$)"); axes[0].legend(title="Year", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_chap.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote fig5_chap.png")


if __name__ == "__main__":
    fig5(); fig6(); fig7()
    for n in (5, 6, 7):
        src = os.path.join(OUT, f"fig{n}_chap.png")
        if os.path.exists(src):
            import shutil; shutil.copy(src, os.path.join(ROOT, "..", "Figures", f"fig{n}.png"))
    print("copied fig5/6/7 into Figures/")
