"""Stage 3 — surrogate-model comparison (Reviewer 3): LightGBM vs XGBoost vs SVR.

The manuscript asserts "LightGBM emerged as the most effective surrogate model"
but provides no evidence. This trains all three under the SAME spatial-holdout
protocol and emits a quantitative comparison table.

SVR is O(n^2) and cannot run on ~450k rows -> trained on a config-sized subsample
(documented in the output). Tree models use the full training set.

Usage:
  python model/compare.py --pollutants pm2_5 no2     # default: all 6
  python model/compare.py --quick
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, load_dataset, metrics, ROOT

import lightgbm as lgb
import xgboost as xgb
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def _models(seed, n_jobs):
    return {
        "LightGBM": lgb.LGBMRegressor(n_estimators=800, num_leaves=255, max_depth=-1,
                                      learning_rate=0.05, subsample=0.8, subsample_freq=5,
                                      colsample_bytree=0.9, random_state=seed,
                                      n_jobs=n_jobs, verbose=-1),
        "XGBoost": xgb.XGBRegressor(n_estimators=800, max_depth=10, learning_rate=0.05,
                                    subsample=0.8, colsample_bytree=0.9, tree_method="hist",
                                    random_state=seed, n_jobs=n_jobs),
        "SVR(RBF)": make_pipeline(StandardScaler(), SVR(C=10.0, gamma="scale")),
    }


def run(cfg, pollutants, quick):
    train, test, feats, group = load_dataset(cfg)
    mc = cfg["model"]
    sub = 5000 if quick else mc["svr_subsample"]
    rows = []
    for p in pollutants:
        Xtr, ytr = train[feats], train[p]
        Xte, yte = test[feats], test[p]
        for name, model in _models(mc["random_state"], mc["n_jobs"]).items():
            if name.startswith("SVR"):
                idx = ytr.sample(min(sub, len(ytr)), random_state=1).index
                xx, yy, note = Xtr.loc[idx], ytr.loc[idx], f"subsample={len(idx)}"
            else:
                xx, yy, note = Xtr, ytr, "full"
            t0 = time.time()
            model.fit(xx, yy)
            m = metrics(yte, model.predict(Xte))
            rows.append({"pollutant": p, "model": name, "train_rows": note,
                         **m, "seconds": round(time.time() - t0, 1)})
            print(f"{p:6s} {name:9s} R2={m['R2']:.3f} RMSE={m['RMSE']:.2f} "
                  f"MAE={m['MAE']:.2f} ({rows[-1]['seconds']}s, {note})")

    outdir = os.path.join(ROOT, cfg["paths"]["out_models"])
    os.makedirs(outdir, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(outdir, "model_comparison.csv"), index=False)
    # winner per pollutant by holdout RMSE
    win = df.loc[df.groupby("pollutant")["RMSE"].idxmin(), ["pollutant", "model"]]
    print("\nbest model per pollutant (holdout RMSE):")
    print(win.to_string(index=False))
    print(f"wrote {outdir}/model_comparison.csv")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--pollutants", nargs="*", default=cfg["pollutants"])
    a = ap.parse_args()
    run(cfg, a.pollutants, a.quick)


if __name__ == "__main__":
    main()
