"""Stage D — Fig 3 validation scatter (R3.9/R1.7) + CO/SO2 supplementary (R1.8).

Predicted vs observed on the spatial-holdout test set, with the 1:1 reference (dashed),
the OLS fit (red), and R2/RMSE annotated. Main 4 pollutants -> Fig 3; CO/SO2 -> Fig S7.
"""
import os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$",
          "no2": "NO$_2$", "o3": "O$_3$", "co": "CO"}


def panel(ax, p):
    d = pd.read_csv(os.path.join(ROOT, "outputs", "models", f"{p}_pred.csv"))
    y, yh = d["y_test"].values, d["predictions"].values
    m = np.isfinite(y) & np.isfinite(yh); y, yh = y[m], yh[m]
    r2 = 1 - np.sum((y - yh)**2) / np.sum((y - y.mean())**2)
    rmse = np.sqrt(np.mean((y - yh)**2))
    ax.scatter(y, yh, s=3, alpha=0.15, c="#2c7bb6", edgecolors="none")
    lim = np.nanpercentile(np.concatenate([y, yh]), 99.5)
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    b, a = np.polyfit(y, yh, 1)
    ax.plot([0, lim], [a, a + b * lim], "r-", lw=1.2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title(PRETTY[p], fontsize=11)
    ax.set_xlabel("Observed"); ax.set_ylabel("Predicted")
    ax.text(0.05, 0.92, f"$R^2$={r2:.2f}\nRMSE={rmse:.1f}", transform=ax.transAxes,
            va="top", fontsize=9, bbox=dict(fc="white", ec="0.7", alpha=0.8))


def make(pollutants, ncol, path, title):
    nrow = int(np.ceil(len(pollutants) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 3.05 * nrow))
    for ax, p in zip(np.atleast_1d(axes).ravel(), pollutants):
        panel(ax, p)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    outdir = os.path.join(ROOT, "outputs", "figures")
    make(["pm2_5", "pm10", "o3", "no2"], 2, os.path.join(outdir, "fig3_validation.png"),
         "Spatial-holdout validation: predicted vs observed (dashed = 1:1, red = OLS fit)")
    make(["co", "so2"], 2, os.path.join(ROOT, "outputs", "supplementary", "FigS7_CO_SO2_validation.png"),
         "Validation for CO and SO$_2$ (Supplementary)")
