"""Stage 3 — train one ML-LUR LightGBM model per pollutant with real RSCV + spatial CV.

For each of the 6 pollutants:
  1. RandomizedSearchCV over a genuine LightGBM space using GroupKFold(grid_id).
  2. Report spatial cross-validated skill (the honest estimate).
  3. Refit on all train cells; evaluate on the disjoint spatial-holdout test set
     (independent CNEMC station cells) -> RMSE / MAE / R2 / SMAPE.
  4. Temporal-generalization check: train 2019-2020 -> test 2021.
Outputs -> outputs/models/: metrics.csv, best_params.json, <pollutant>_pred.csv

Usage:
  python model/train.py                 # full run, all 6 pollutants
  python model/train.py --quick         # fast smoke test (small n_iter, subsample)
  python model/train.py --pollutants pm2_5 no2
"""
from __future__ import annotations
import os, sys, json, argparse, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_dataset, metrics, spatial_cv, ROOT
import hpo


def _cv_report(model, X, y, groups, folds):
    """Out-of-fold spatial-CV metrics with the chosen estimator settings."""
    oof = np.zeros(len(y))
    for tr, va in spatial_cv(folds).split(X, y, groups):
        m = model.__class__(**model.get_params())
        m.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = m.predict(X.iloc[va])
    return metrics(y, oof)


def run(cfg, pollutants, quick: bool):
    train, test, feats, group = load_dataset(cfg)
    mc = cfg["model"]
    n_iter = 6 if quick else mc["hpo_n_iter"]
    folds = 3 if quick else mc["cv_folds"]
    if quick:
        train = train.groupby(group, group_keys=False).sample(frac=0.15, random_state=1)

    outdir = os.path.join(ROOT, cfg["paths"]["out_models"])
    os.makedirs(outdir, exist_ok=True)
    rows, best_params = [], {}

    sub = None if quick else mc.get("hpo_subsample")
    for p in pollutants:
        t0 = time.time()
        Xtr, ytr, gtr = train[feats], train[p], train[group]
        Xte, yte = test[feats], test[p]
        print(f"\n=== {p} === (train {len(Xtr)}, test {len(Xte)}, n_iter={n_iter}, folds={folds})",
              flush=True)

        # RSCV on a subsample for tractability; final model refit on the full training set.
        if sub and len(Xtr) > sub:
            sidx = ytr.sample(sub, random_state=1).index
            Xs, ys, gs = Xtr.loc[sidx], ytr.loc[sidx], gtr.loc[sidx]
        else:
            Xs, ys, gs = Xtr, ytr, gtr
        est, params, n_cfg = hpo.search(Xs, ys, gs, n_iter, folds,
                                        mc["random_state"], mc["n_jobs"])
        best_params[p] = params
        cv_m = _cv_report(est, Xs, ys, gs, folds)

        est.fit(Xtr, ytr)
        test_m = metrics(yte, est.predict(Xte))

        # temporal generalization: train 2019-2020 -> 2021
        temp_m = {}
        if "year" in train.columns:
            tr_t = train[train["year"] < 2021]; te_t = train[train["year"] >= 2021]
            if len(te_t) and len(tr_t):
                mt = est.__class__(**est.get_params()); mt.fit(tr_t[feats], tr_t[p])
                temp_m = metrics(te_t[p], mt.predict(te_t[feats]))

        pd.DataFrame({"y_test": yte.values,
                      "predictions": est.predict(Xte)}).to_csv(
            os.path.join(outdir, f"{p}_pred.csv"), index=False)

        rows.append({"pollutant": p, "n_configs": n_cfg,
                     **{f"cv_{k}": v for k, v in cv_m.items()},
                     **{f"test_{k}": v for k, v in test_m.items()},
                     **{f"temporal2021_{k}": v for k, v in temp_m.items()},
                     "seconds": round(time.time() - t0, 1)})
        print(f"  spatial-CV R2={cv_m['R2']:.3f} RMSE={cv_m['RMSE']:.2f} | "
              f"holdout R2={test_m['R2']:.3f} RMSE={test_m['RMSE']:.2f} | "
              f"{rows[-1]['seconds']}s")

    pd.DataFrame(rows).to_csv(os.path.join(outdir, "metrics.csv"), index=False)
    with open(os.path.join(outdir, "best_params.json"), "w") as fh:
        json.dump(best_params, fh, indent=2, default=str)
    print(f"\nwrote {outdir}/metrics.csv and best_params.json")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pollutants", nargs="*", default=cfg["pollutants"])
    a = ap.parse_args()
    run(cfg, a.pollutants, a.quick)


if __name__ == "__main__":
    main()
