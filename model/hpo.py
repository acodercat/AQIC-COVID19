"""Random Search CV hyperparameter tuning for LightGBM, using a SPATIAL splitter.

The manuscript claims RSCV was used but the original code hard-coded the params.
This implements the real thing: RandomizedSearchCV over a genuine LightGBM space,
with GroupKFold(grid_id) so the search respects the spatial holdout structure.
"""
from __future__ import annotations
import numpy as np
from scipy.stats import randint, uniform, loguniform
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

from common import rmse_scorer, spatial_cv

SEARCH_SPACE = {
    "num_leaves": randint(31, 512),
    "max_depth": [-1, 8, 12, 16, 20],
    "learning_rate": loguniform(0.02, 0.15),
    "n_estimators": [200, 400, 600],
    "subsample": uniform(0.6, 0.4),          # bagging_fraction in [0.6, 1.0]
    "subsample_freq": randint(1, 7),
    "colsample_bytree": uniform(0.6, 0.4),   # feature_fraction in [0.6, 1.0]
    "min_child_samples": randint(20, 200),
    "reg_alpha": loguniform(1e-3, 10.0),
    "reg_lambda": loguniform(1e-3, 10.0),
}


def search(X, y, groups, n_iter: int, cv_folds: int, random_state: int, n_jobs: int):
    """Return (best_estimator, best_params, n_configs_evaluated)."""
    base = lgb.LGBMRegressor(
        boosting_type="gbdt", objective="regression",
        random_state=random_state, n_jobs=n_jobs, verbose=-1,
    )
    rs = RandomizedSearchCV(
        base, SEARCH_SPACE, n_iter=n_iter,
        scoring=rmse_scorer(), cv=spatial_cv(cv_folds),
        random_state=random_state, n_jobs=n_jobs, refit=True, error_score="raise",
    )
    rs.fit(X, y, groups=groups)
    n_configs = len(rs.cv_results_["params"])
    return rs.best_estimator_, rs.best_params_, n_configs
