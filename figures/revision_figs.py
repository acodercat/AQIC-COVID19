"""Revision figures (Sci Rep round 1).

(1) did_forest_rev.png  — DiD lockdown effect (net of holiday) per pollutant x region under the
    revised specification, with and without ERA5 covariates side by side (two-way clustered 95% CI;
    filled = BH-FDR q < 0.05). Replaces the single-specification did_forest.png (Fig. 8).
(2) placebo_years_no2.png — year-pair placebo test for NO2: DiD between consecutive no-lockdown
    years (grey) vs the 2020 lockdown estimate (red), per region, with and without meteorology.

Run after analysis/significance.py (rev tags) and analysis/placebo_years.py.
Out: outputs/figures/
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ADIR = os.path.join(ROOT, "outputs", "analysis")
FDIR = os.path.join(ROOT, "outputs", "figures")
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$",
          "no2": "NO$_2$", "o3": "O$_3$ (MDA8)", "co": "CO"}
ORDER = ["National", "BTH", "Jilin", "Jiangsu", "Guangdong", "Xinjiang",
         "Beijing", "Shanghai", "Nanjing", "Wuhan", "Guangzhou"]
NCELLS = {"National": 747, "BTH": 56, "Guangdong": 53, "Jiangsu": 46, "Xinjiang": 26, "Jilin": 16,
          "Beijing": 13, "Shanghai": 10, "Guangzhou": 9, "Nanjing": 5, "Wuhan": 4}
SPEC = {"nometeo": ("o", "#d7191c", "no meteorology"), "meteo": ("s", "#2c7bb6", "ERA5-adjusted")}


def _labels():
    return [f"{r} ({NCELLS[r]})" for r in ORDER]


def did_forest_rev():
    tabs = {a: pd.read_csv(os.path.join(ADIR, f"significance_table_rev_gap2_twoway_{a}.csv"))
            for a in SPEC}
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 6.4), sharey=True)
    y0 = np.arange(len(ORDER))[::-1]
    for ax, p in zip(axes.ravel(), PRETTY):
        for k, (adj, (mk, col, _)) in enumerate(SPEC.items()):
            d = tabs[adj][tabs[adj].pollutant == p].set_index("region").reindex(ORDER)
            y = y0 + (0.18 if k == 0 else -0.18)
            sig = (d["H3_qval"] < 0.05).to_numpy()
            ax.errorbar(d["H3_lockdown_DiD"], y,
                        xerr=[d["H3_lockdown_DiD"] - d["H3_ci_lo"], d["H3_ci_hi"] - d["H3_lockdown_DiD"]],
                        fmt="none", ecolor=col, lw=0.8, alpha=0.7)
            ax.scatter(d["H3_lockdown_DiD"], y, marker=mk, s=18, zorder=3, edgecolor=col,
                       facecolor=np.where(sig, col, "white"), lw=0.9)
        ax.axvline(0, color="k", lw=0.7)
        ax.axhline(y0[ORDER.index("Xinjiang")] - 0.5, color="0.7", lw=0.6, ls=":")
        ax.set_title(PRETTY[p], fontsize=9)
        ax.set_yticks(y0); ax.set_yticklabels(_labels(), fontsize=7)
        ax.tick_params(axis="x", labelsize=7); ax.grid(axis="x", alpha=0.3)
    handles = [plt.Line2D([], [], marker=mk, color=col, ls="", label=lab)
               for mk, col, lab in SPEC.values()]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=8, frameon=False)
    fig.supxlabel("Lockdown effect net of the holiday, 2020 vs 2019 (DiD; $\\mu$g m$^{-3}$, "
                  "CO in mg m$^{-3}$)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FDIR, "did_forest_rev.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


def placebo_years_no2():
    t = pd.read_csv(os.path.join(ADIR, "placebo_years.csv"))
    t = t[t.pollutant == "no2"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.9), sharey=True)
    y0 = np.arange(len(ORDER))[::-1]
    for ax, (adj, (_, _, lab)) in zip(axes, SPEC.items()):
        d = t[t.adj == adj]
        for r, y in zip(ORDER, y0):
            s = d[d.region == r]
            ax.scatter(s["placebo_year_DiD"], np.full(len(s), y), s=14, color="0.55",
                       zorder=2, label="no-lockdown year pairs (2014v13 … 2018v17)" if r == "National" else None)
            if len(s):
                ax.scatter(s["lockdown_2020_DiD"].iloc[0], y, s=40, marker="D", color="#d7191c",
                           zorder=3, label="2020 v 2019 (lockdown)" if r == "National" else None)
        ax.axvline(0, color="k", lw=0.7)
        ax.axhline(y0[ORDER.index("Xinjiang")] - 0.5, color="0.7", lw=0.6, ls=":")
        ax.set_title(f"NO$_2$, {lab}", fontsize=9)
        ax.set_yticks(y0); ax.set_yticklabels(_labels(), fontsize=7)
        ax.tick_params(axis="x", labelsize=7); ax.grid(axis="x", alpha=0.3)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, fontsize=7, frameon=False)
    fig.supxlabel("Holiday-netted DiD ($\\mu$g m$^{-3}$)", fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(FDIR, "placebo_years_no2.png")
    fig.savefig(out, dpi=300, bbox_inches="tight"); plt.close(fig)
    return out


if __name__ == "__main__":
    print(did_forest_rev())
    print(placebo_years_no2())
