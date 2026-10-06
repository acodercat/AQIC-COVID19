"""Event-study (leads/lags) test of the parallel-trends assumption (Reviewer 2 #2, Reviewer 5 2.5).

Regression form, same panel and inference as the revised main DiD:

    C_it = alpha_cell + gamma_eventday + sum_{k != REF} beta_k * 1[2020] * 1[eventday = k] + [meteo] + e

  - cell fixed effects (within transformation) and event-day fixed effects common to both years;
  - beta_k = 2020-minus-2019 difference on event day k (reported relative to the pre-period mean).

Inference treats DAYS as the unit of independent information. Each beta_k is identified from a
single calendar date, so the common daily (weather) shock is exactly its noise: date clustering is
undefined per coefficient (one cluster each) and cell clustering would ignore the shared daily
shock. We therefore summarise the 19 pre-period betas (event days -21..-3) as a time series:
  - pre_sd_beta: day-to-day SD of the pre-period year difference (Reviewer 5 2.5);
  - pre_slope (+ Newey-West HAC p-value, 3 lags): linear drift of the year difference before the
    LNY -- i.e. whether the difference was stable pre-lockdown (Reviewer 2 #2);
  - shift_in_pre_sd: (peri mean - pre mean) / pre_sd_beta.
Event days -2 and -1 (23-24 Jan 2020, first two days of the Wuhan lockdown) are estimated and
plotted but excluded from the pre-period statistics.

Out: outputs/analysis/event_study_coefs.csv, outputs/analysis/event_study_pretrend.csv,
     outputs/figures/event_study_<meteo|nometeo>.png  (national, 6 pollutants)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import did as DID
import significance as S

ROOT = S.ROOT
PRETTY = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$", "no2": "NO$_2$",
          "o3": "O$_3$ (MDA8)", "co": "CO"}
REF = -3                       # omitted dummy (normalisation is to the pre-period mean)
EXCLUDED = (-2, -1)            # lockdown-contaminated pre days (2020)
LO, HI = -21, 20


def estimate(sub, pol, meteo_cols=None):
    d = sub[sub.year.isin([2019, 2020]) & sub.event_day.between(LO, HI)].dropna(subset=[pol]).copy()
    d["Treat"] = (d.year == 2020).astype(float)
    days = [k for k in range(LO, HI + 1) if k != REF]
    terms = []
    for k in days:                                   # event-day FE (both years) + 2020 x day
        d[f"ed{k}"] = (d.event_day == k).astype(float)
        d[f"tx{k}"] = d[f"ed{k}"] * d["Treat"]
        terms += [f"ed{k}", f"tx{k}"]
    terms.append("Treat")
    if meteo_cols:
        terms += [c for c in meteo_cols if c in d.columns and d[c].notna().any()]
    res, ix = DID._within_ols(d, pol, terms, cluster="cell")      # point estimates only
    coefs = pd.DataFrame({"event_day": [REF] + days,
                          "beta": [0.0] + [res.params[ix[f"tx{k}"]] for k in days]}
                         ).sort_values("event_day", ignore_index=True)
    pre = coefs[(coefs.event_day <= REF) & ~coefs.event_day.isin(EXCLUDED)]
    coefs["beta"] -= pre["beta"].mean()                      # relative to the pre-period mean
    pre = coefs.loc[pre.index]
    peri = coefs[coefs.event_day >= 0]
    tr = sm.OLS(pre["beta"].to_numpy(), sm.add_constant(pre["event_day"].to_numpy(float))
                ).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    sd = float(pre["beta"].std())
    test = {"n_pre": len(pre), "pre_sd_beta": sd,
            "pre_slope_per_day": float(tr.params[1]), "pre_slope_p": float(tr.pvalues[1]),
            "peri_minus_pre": float(peri["beta"].mean()),
            "shift_in_pre_sd": float(peri["beta"].mean() / sd) if sd > 0 else np.nan}
    return coefs, test


def run(cfg):
    panel = S.build_panel(cfg)                  # unrestricted windows: event days -21..+20
    meteo = [m for m in S.METEO if m in panel.columns]
    all_coefs, tests = [], []
    for region, idx in S.regions_of(panel).items():
        sub = panel.loc[idx]
        for pol in S.POLLUTANTS:
            for adj, cols in (("nometeo", None), ("meteo", meteo)):
                try:
                    c, t = estimate(sub, pol, cols)
                except Exception as e:
                    print(f"  [skip] {region}/{pol}/{adj}: {e}"); continue
                all_coefs.append(c.assign(region=region, pollutant=pol, adj=adj))
                tests.append({"region": region, "pollutant": pol, "adj": adj, **t})
    coefs = pd.concat(all_coefs, ignore_index=True)
    tests = pd.DataFrame(tests)
    adir = os.path.join(ROOT, "outputs", "analysis")
    coefs.to_csv(os.path.join(adir, "event_study_coefs.csv"), index=False)
    tests.to_csv(os.path.join(adir, "event_study_pretrend.csv"), index=False)
    print(tests[tests.region == "National"].round(3).to_string(index=False))

    fdir = os.path.join(ROOT, "outputs", "figures")
    for adj in ("nometeo", "meteo"):
        fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2), sharex=True)
        for ax, pol in zip(axes.ravel(), S.POLLUTANTS):
            g = coefs[(coefs.region == "National") & (coefs.pollutant == pol) & (coefs.adj == adj)]
            t = tests[(tests.region == "National") & (tests.pollutant == pol) & (tests.adj == adj)].iloc[0]
            ax.axhspan(-2 * t.pre_sd_beta, 2 * t.pre_sd_beta, color="#4393c3", alpha=0.15, lw=0)
            ax.axhline(0, color="k", lw=0.6)
            ax.axvspan(-0.5, HI + 0.5, color="grey", alpha=0.12)
            ax.axvspan(EXCLUDED[0] - 0.5, EXCLUDED[1] + 0.5, color="#fdae61", alpha=0.35)
            ax.plot(g.event_day, g.beta, "o-", ms=2.5, lw=0.9, color="#d7191c")
            ax.hlines(t.peri_minus_pre, 0, HI, color="#d7191c", lw=1.2, ls="--")
            ax.set_title(f"{PRETTY[pol]}  (pre-slope p={t.pre_slope_p:.2f})", fontsize=8)
            ax.tick_params(labelsize=7); ax.grid(alpha=0.3)
        fig.supxlabel("Days from Lunar New Year", fontsize=8)
        fig.supylabel("2020 $-$ 2019 difference (rel. to pre-period mean)", fontsize=8)
        fig.tight_layout()
        out = os.path.join(fdir, f"event_study_{adj}.png")
        fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)
        print(f"wrote {out}")
    return coefs, tests


if __name__ == "__main__":
    run(S._load_cfg())
