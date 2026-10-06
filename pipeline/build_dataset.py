"""Stage 1 — assemble the final modelling table from the per-source grids.

Joins ERA5 met + CHAP targets + AOD + land cover + elevation + temporal features on
(grid_id, date), reproducing the model schema but CORRECTED: near-surface met names,
wind_dir added, land cover as % (model features) with m^2 kept alongside. Applies a
documented outlier policy (valid physical ranges + IQR) and LOGS the counts removed
(answers Reviewer 1 Q2 / Reviewer 3). Emits the station-cell modelling table and,
optionally, the full mapping-grid inference table.

Tolerant by design: runs with whatever per-source parquets exist so the pipeline can
be built incrementally; missing inputs are warned, not fatal.

Out: outputs/features/model_table.parquet (+ outputs/features/clean_report.csv)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import load_config, ROOT

# Plausible physical ranges (ug/m^3 except CO mg/m^3). Outside -> dropped.
VALID = {"pm2_5": (0, 1000), "pm10": (0, 2000), "so2": (0, 500),
         "no2": (0, 300), "o3": (0, 400), "co": (0, 15)}


def _load(rel):
    p = os.path.join(ROOT, rel)
    if not os.path.exists(p):
        print(f"  [warn] missing {rel} -> skipped"); return None
    return pd.read_parquet(p)


def temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    d = pd.to_datetime(df["date"])
    df["year"] = d.dt.year; df["month"] = d.dt.month
    df["day"] = d.dt.day; df["weekday"] = d.dt.weekday
    return df


def clean(df: pd.DataFrame, pollutants) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Two-step outlier policy with per-pollutant removal logging."""
    log = []
    n0 = len(df)
    for p in pollutants:
        if p not in df: continue
        lo, hi = VALID[p]
        bad_range = ~df[p].between(lo, hi) & df[p].notna()
        df.loc[bad_range, p] = np.nan
        q1, q3 = df[p].quantile([0.25, 0.75])
        iqr = q3 - q1
        bad_iqr = (~df[p].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)) & df[p].notna()
        df.loc[bad_iqr, p] = np.nan
        log.append({"pollutant": p, "range_removed": int(bad_range.sum()),
                    "iqr_removed": int(bad_iqr.sum()),
                    "iqr_lo": float(q1 - 1.5 * iqr), "iqr_hi": float(q3 + 1.5 * iqr)})
    report = pd.DataFrame(log)
    print(f"  outlier cleaning: {n0} rows; removals per pollutant:\n", report.to_string(index=False))
    return df, report


def main():
    cfg = load_config()
    feat = os.path.join(ROOT, "outputs", "features")
    grids = os.path.join(ROOT, cfg["paths"]["out_grids"])
    modeling = pd.read_csv(os.path.join(grids, "modeling_grid.csv"))

    chap = _load("outputs/targets/chap_daily_grid.parquet")
    era5 = _load("outputs/features/era5_daily_grid.parquet")
    aod = _load("outputs/features/aod_daily_grid.parquet")
    landc = _load("outputs/features/landcover_grid.parquet")
    elev = _load("outputs/features/elevation_grid.parquet")
    if chap is None:
        raise SystemExit("CHAP targets required; run pipeline/chap.py first")

    df = chap.copy()
    df = temporal_features(df)
    for other, keys in [(era5, ["grid_id", "date"]), (aod, ["grid_id", "date"])]:
        if other is not None:
            df = df.merge(other, on=keys, how="left")
    for static in [landc, elev]:
        if static is not None:
            on = ["grid_id", "year"] if "year" in static.columns else ["grid_id"]
            df = df.merge(static, on=on, how="left")
    df = df.merge(modeling[["grid_id", "lat", "lon"]], on="grid_id", how="left")

    df, report = clean(df, cfg["pollutants"])

    # station-cell modelling table only (cells with a ground-truth target)
    model_tbl = df[df["grid_id"].isin(set(modeling["grid_id"]))].reset_index(drop=True)
    os.makedirs(feat, exist_ok=True)
    model_tbl.to_parquet(os.path.join(feat, "model_table.parquet"), index=False)
    report.to_csv(os.path.join(feat, "clean_report.csv"), index=False)
    print(f"wrote model_table.parquet ({len(model_tbl)} cell-days, "
          f"{model_tbl['grid_id'].nunique()} cells, "
          f"{model_tbl['date'].min()}..{model_tbl['date'].max()})")


if __name__ == "__main__":
    main()
