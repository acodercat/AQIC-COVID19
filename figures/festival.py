"""Stage 5 — figures for the Spring-Festival vs lockdown analysis.

(1) Event-time curves: for a region, daily pollutant mean vs days-from-LNY, one line
    per year (2019/2020/2021), with the peri (festival/lockdown) window shaded. Makes
    the holiday-vs-lockdown contrast visible (2020 should dip below the 2019 holiday line).
(2) DiD forest plot: H3 (lockdown net of holiday) per pollutant x region with 95% CI;
    filled markers where BH-FDR q<0.05.

Run after analysis/significance.py. Reads the assembled panel + significance_table.csv.
Out: outputs/figures/festival_<region>.png, did_forest.png
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
import significance as S

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$",
          "no2": "NO$_2$", "o3": "O$_3$", "co": "CO"}


def event_curves(panel, region, pollutants, outdir):
    sub = panel if region == "National" else panel[(panel.region == region) |
                                                    (panel.study_city == region)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for ax, p in zip(axes.ravel(), pollutants):
        if p not in sub:
            ax.set_visible(False); continue
        for yr, col in zip([2019, 2020, 2021], ["#1b9e77", "#d95f02", "#7570b3"]):
            g = (sub[sub.year == yr].groupby("event_day")[p].mean())
            ax.plot(g.index, g.values, color=col, lw=1.6, label=str(yr))
        ax.axvspan(0, 20, color="grey", alpha=0.12)           # peri window
        ax.axvline(0, color="k", lw=0.6, ls="--")
        ax.set_title(PRETTY.get(p, p)); ax.grid(alpha=0.3)
    axes[0, 0].legend(title="Year", fontsize=8)
    fig.suptitle(f"Air quality around the Lunar New Year — {region}\n"
                 f"(day 0 = LNY; shaded = 21-day festival/lockdown window)")
    fig.supxlabel("Days from Lunar New Year"); fig.supylabel("Concentration")
    fig.tight_layout()
    path = os.path.join(outdir, f"festival_{region}.png")
    fig.savefig(path, dpi=140); plt.close(fig)
    return path


def did_forest(tab, outdir):
    """One compact forest panel per pollutant (regions on the y-axis)."""
    polls = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    for ax, p in zip(axes.ravel(), polls):
        d = tab[tab.pollutant == p].dropna(subset=["H3_lockdown_DiD"]).copy()
        d = d.sort_values("H3_lockdown_DiD")
        y = np.arange(len(d))
        sig = d["H3_qval"] < 0.05 if "H3_qval" in d else d["H3_p"] < 0.05
        ax.errorbar(d["H3_lockdown_DiD"], y,
                    xerr=[d["H3_lockdown_DiD"] - d["H3_ci_lo"], d["H3_ci_hi"] - d["H3_lockdown_DiD"]],
                    fmt="none", ecolor="0.6", lw=1)
        ax.scatter(d["H3_lockdown_DiD"], y, c=np.where(sig, "#d7191c", "white"),
                   edgecolor="k", zorder=3, s=34)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(y); ax.set_yticklabels(d["region"], fontsize=8)
        ax.set_title(PRETTY.get(p, p), fontsize=12)
        ax.grid(axis="x", alpha=0.3)
        ax.margins(y=0.04)
    fig.supxlabel("Lockdown effect net of the holiday (difference-in-differences; "
                  "$\\mu$g m$^{-3}$, CO in mg m$^{-3}$). Filled markers: FDR $q<0.05$.", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, "did_forest.png")
    fig.savefig(path, dpi=160, bbox_inches="tight"); plt.close(fig)
    return path


def main():
    cfg = S._load_cfg()
    panel = S.build_panel(cfg)
    outdir = os.path.join(ROOT, "outputs", "figures"); os.makedirs(outdir, exist_ok=True)
    have = [p for p in S.POLLUTANTS if p in panel.columns]
    for region in ["National", "BTH", "Jiangsu", "Jilin", "Guangdong", "Xinjiang"]:
        if region == "National" or (panel.region == region).any():
            print("  ", event_curves(panel, region, have, outdir))
    tabpath = os.path.join(ROOT, "outputs", "analysis", "significance_table.csv")
    if os.path.exists(tabpath):
        print("  ", did_forest(pd.read_csv(tabpath), outdir))
    else:
        print("  [note] run analysis/significance.py first for the forest plot")


if __name__ == "__main__":
    main()
