"""Stage E — Figure 4 restructured (R1.7 + R3.10).

One panel per pollutant (shared scale across cities, per R1.7), each showing the five study
cities' daily concentration through the 2020 COVID phases — outbreak → lockdown (2020-CLD,
23 Jan–13 Feb, shaded) → stabilization → control — so the before/after contrast and the
cross-city comparison are both visible (R3.10).
Out: outputs/figures/fig4_cities.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLL = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$",
          "no2": "NO$_2$", "o3": "O$_3$ (MDA8)", "co": "CO"}
CITY_COLORS = {"Beijing": "#d7191c", "Shanghai": "#fdae61", "Wuhan": "#7b3294",
               "Nanjing": "#2c7bb6", "Guangzhou": "#1a9641"}
CLD = ("2020-01-23", "2020-02-13")     # 2020 lockdown window


def main():
    reg = pd.read_csv(os.path.join(ROOT, "outputs/grids/cell_regions.csv"))
    city_cells = reg.dropna(subset=["study_city"])[["grid_id", "study_city"]]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.0), sharex=True)
    for ax, p in zip(axes.ravel(), POLL):
        f = os.path.join(ROOT, "outputs", "targets", f"chap_{p}_2020_D1K.parquet")
        if not os.path.exists(f):
            ax.set_visible(False); continue
        d = pd.read_parquet(f).merge(city_cells, on="grid_id")
        d = d[pd.to_datetime(d.date).between("2020-01-01", "2020-03-31")]
        g = d.groupby(["date", "study_city"])[p].mean().reset_index()
        for city, c in CITY_COLORS.items():
            s = g[g.study_city == city]
            ax.plot(pd.to_datetime(s.date), s[p], color=c, lw=1.5, label=city)
        ax.axvspan(pd.Timestamp(CLD[0]), pd.Timestamp(CLD[1]), color="grey", alpha=0.15)
        ax.set_title(PRETTY[p]); ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
    axes[0, 0].legend(fontsize=8, title="City")
    fig.supylabel("Concentration (µg m$^{-3}$; CO in mg m$^{-3}$)")
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs", "figures", "fig4_cities.png")
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
