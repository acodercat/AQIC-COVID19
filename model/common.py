"""Shared model utilities: config, data loading, metrics, spatial CV splitter."""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config(path: str | None = None) -> dict:
    import yaml
    path = path or os.path.join(ROOT, "pipeline", "config.yaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


def _read(path: str) -> pd.DataFrame:
    full = os.path.join(ROOT, path)
    if path.endswith(".parquet"):
        return pd.read_parquet(full)
    return pd.read_csv(full)


def load_dataset(cfg) -> tuple[pd.DataFrame, pd.DataFrame, list[str], str]:
    """Return (train_df, test_df, feature_cols, group_col). Test is the spatial holdout."""
    ds = cfg["dataset"]
    train = _read(ds["train"])
    test = _read(ds["test"])
    feats = list(ds["features"])
    missing = [c for c in feats if c not in train.columns]
    if missing:
        raise SystemExit(f"features missing from data: {missing}")
    return train, test, feats, ds["group"]


# ---- metrics ----------------------------------------------------------------
def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float); y_pred = np.asarray(y_pred, float)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom > 0
    return float(np.mean(2 * np.abs(y_pred - y_true)[mask] / denom[mask]) * 100)


def metrics(y_true, y_pred) -> dict:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "SMAPE": smape(y_true, y_pred),
    }


def rmse_scorer():
    """Negative RMSE scorer for sklearn search (greater is better)."""
    from sklearn.metrics import make_scorer
    return make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp)),
                       greater_is_better=True)


def spatial_cv(n_splits: int):
    """GroupKFold so no grid_id appears in both train and validation of a fold."""
    return GroupKFold(n_splits=n_splits)
