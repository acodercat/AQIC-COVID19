"""Part 2 — event-study (leads/lags) evidence for the parallel-trends assumption (R3 DiD rigor).

For each pollutant we plot the cell-mean year-difference (2020 minus 2019) at each event day
relative to the Lunar New Year, from LNY-21 to LNY+20, with 95% CIs. Because both years are drawn
from the same CHAP product at the same grid cells, the pre-period (event day < 0) difference should
be flat and near a constant offset if trends are parallel; a clear downward (NO2) or upward (O3)
break at event day 0 then identifies the lockdown. This is the standard reviewer-expected check.

Out: outputs/figures/event_study.png  (-> Figures/fig10.png) and outputs/analysis/event_study.csv
(pre-period mean |gap| vs peri-period mean |gap|, per pollutant, quantifying parallel-trends).
"""
from __future__ import annotations
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pipeline"))
POLL = ["pm2_5", "pm10", "so2", "no2", "o3", "co"]
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$", "no2": "NO$_2$", "o3": "O$_3$ (MDA8)", "co": "CO"}
ANCHOR = {2019: "2019-02-05", 2020: "2020-01-25"}


def _eventday_panel(p):
    frames = []
    for y in (2019, 2020):
        d = pd.read_parquet(os.path.join(ROOT, "outputs", "targets", f"chap_{p}_{y}_D1K.parquet"))
        a = pd.Timestamp(ANCHOR[y])
        d["ed"] = (pd.to_datetime(d["date"]) - a).dt.days
        d = d[(d.ed >= -21) & (d.ed <= 20)][["grid_id", "ed", p]].rename(columns={p: "v"})
        d["year"] = y
        frames.append(d)
    df = pd.concat(frames)
    w = df.pivot_table(index=["grid_id", "ed"], columns="year", values="v").reset_index().dropna()
    w["gap"] = w[2020] - w[2019]
    return w


def run():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
    rows = []
    for ax, p in zip(axes.ravel(), POLL):
        w = _eventday_panel(p)
        g = w.groupby("ed")["gap"].agg(["mean", "std", "count"]).reset_index()
        se = g["std"] / np.sqrt(g["count"])
        ax.axhline(0, color="k", lw=0.6)
        ax.axvspan(0, 20, color="grey", alpha=0.12)
        ax.errorbar(g["ed"], g["mean"], yerr=1.96 * se, fmt="o-", ms=3, lw=1, color="#d7191c", ecolor="0.6")
        ax.set_title(PRETTY[p]); ax.grid(alpha=0.3)
        pre = g[g.ed < 0]["mean"]; peri = g[g.ed >= 0]["mean"]
        rows.append({"pollutant": p, "pre_mean_gap": round(pre.mean(), 3),
                     "pre_sd_gap": round(pre.std(), 3), "peri_mean_gap": round(peri.mean(), 3),
                     "ratio_peri_pre": round(abs(peri.mean()) / max(abs(pre.mean()), 1e-6), 2)})
    fig.supxlabel("Days from Lunar New Year"); fig.supylabel("2020 $-$ 2019 concentration difference")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(ROOT, "outputs", "figures", "event_study.png"), dpi=140, bbox_inches="tight")
    plt.close(fig)
    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(ROOT, "outputs", "analysis", "event_study.csv"), index=False)
    print(tab.to_string(index=False))
    import shutil
    shutil.copy(os.path.join(ROOT, "outputs", "figures", "event_study.png"),
                os.path.join(ROOT, "..", "Figures", "fig10.png"))
    print("\nwrote event_study.png -> Figures/fig10.png and event_study.csv")


if __name__ == "__main__":
    run()
