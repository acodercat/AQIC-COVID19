"""Write the revised Supplementary Tables (LaTeX fragments) from the analysis outputs, so that every
number in the Supplementary Information is generated, not transcribed.

Out: SciRep_Submission/revision/tables/S{2,4,5,9,10,11,12}.tex  (\\input by supplementary.tex)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import significance as S
from windows import lny_windows

ROOT = S.ROOT
A = os.path.join(ROOT, "outputs", "analysis")
OUT = os.path.join(ROOT, "SciRep_Submission", "revision", "tables")
TEX = {"pm2_5": "PM$_{2.5}$", "pm10": "PM$_{10}$", "so2": "SO$_2$", "no2": "NO$_2$",
       "o3": "O$_3$", "co": "CO"}
ORDER = ["National", "BTH", "Jilin", "Jiangsu", "Guangdong", "Xinjiang",
         "Beijing", "Shanghai", "Nanjing", "Wuhan", "Guangzhou"]
POLS = ["no2", "pm2_5", "pm10", "so2", "o3", "co"]


def num(x, p):
    """Format with p decimals (CO gets one extra), keeping the sign."""
    return "--" if pd.isna(x) else f"{x:+.{p}f}".replace("-", "$-$")


def pv(x):
    """p/q-value with 3 decimals; values below 0.001 shown as <0.001."""
    return "$<$0.001" if x < 0.001 else f"{x:.3f}"


def dp(pol, base=1):
    return base + 2 if pol == "co" else base


def sortkey(df):
    return df.assign(_r=df.region.map(ORDER.index), _p=df.pollutant.map(POLS.index)) \
             .sort_values(["_r", "_p"]).drop(columns=["_r", "_p"])


def longtable(caption, header, rows, colspec):
    h = " & ".join(f"\\textbf{{{c}}}" for c in header) + "\\\\"
    body = "\n".join(" & ".join(r) + " \\\\" for r in rows)
    return (f"\\begin{{longtable}}{{{colspec}}}\n\\caption{{{caption}}}\\\\\n\\toprule\n{h}\n\\midrule\n"
            f"\\endfirsthead\n\\toprule\n{h}\n\\midrule\n\\endhead\n\\bottomrule\n\\endfoot\n{body}\n"
            f"\\end{{longtable}}\n")


def s2():
    cfg = S._load_cfg()
    w = lny_windows({int(k): v for k, v in cfg["lny_anchors"].items()}, pre_gap=2)
    g = w.groupby(["year", "period"])["date"].agg(["min", "max", "count"]).reset_index()
    g = g[g.period.isin(["placebo", "pre", "peri", "post"])]
    g["o"] = g.period.map({"placebo": 0, "pre": 1, "peri": 2, "post": 3})
    rows = [[str(r.year), r.period, r["min"], r["max"], str(r["count"])]
            for _, r in g.sort_values(["year", "o"]).iterrows()]
    t = ("\\begin{table}[H]\\centering\n\\caption{Event-time windows aligned to each year's "
         "Lunar-New-Year (LNY) anchor (5 Feb 2019, 25 Jan 2020, 12 Feb 2021). The pre window ends at "
         "LNY$-$3 in every year, so that 23--24 January 2020 (the first days of the Wuhan lockdown) are "
         "excluded. The placebo window (LNY$-$42 to LNY$-$22) is used for the pre-trend test; CHAP coverage "
         "starts on 1 January 2019, so the 2019 placebo window contains 14 of its 21 days. The year-pair "
         "placebo test uses the same definitions with anchors 10 Feb 2013, 31 Jan 2014, 19 Feb 2015, "
         "8 Feb 2016, 28 Jan 2017 and 16 Feb 2018.}\n"
         "\\begin{tabular}{llllr}\n\\toprule\n\\textbf{Year} & \\textbf{Window} & \\textbf{Start} & "
         "\\textbf{End} & \\textbf{Days}\\\\\n\\midrule\n"
         + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\\end{table}\n")
    return t


def s4():
    t = pd.read_csv(os.path.join(A, "event_study_pretrend.csv"))
    t = t[t.region == "National"]
    rows = []
    for pol in POLS:
        r = {a: t[(t.pollutant == pol) & (t.adj == a)].iloc[0] for a in ("nometeo", "meteo")}
        rows.append([TEX[pol]] + [x for a in ("nometeo", "meteo") for x in (
            f"{r[a].pre_sd_beta:.{dp(pol, 2)}f}", pv(r[a].pre_slope_p),
            num(r[a].peri_minus_pre, dp(pol, 2)))])
    return ("\\begin{table}[H]\\centering\n\\caption{Event study (national, 747 grid cells): summary of "
            "the 2020$-$2019 daily coefficients. Pre SD: standard deviation of the 19 pre-period coefficients "
            "(event days $-21$ to $-3$); slope $p$: test of a linear pre-period trend (Newey--West, 3 lags); "
            "peri$-$pre: peri-period mean relative to the pre-period mean (equal to the DiD estimate). "
            "Units: $\\upmu$g/m$^3$ (CO in mg/m$^3$). Replaces the ratio-based statistic of the original "
            "submission.}\n\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{3}{c}{\\textbf{No meteorology}}"
            " & \\multicolumn{3}{c}{\\textbf{ERA5-adjusted}}\\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
            "\\textbf{Pollutant} & \\textbf{Pre SD} & \\textbf{Slope $p$} & \\textbf{Peri$-$pre} & "
            "\\textbf{Pre SD} & \\textbf{Slope $p$} & \\textbf{Peri$-$pre}\\\\\n\\midrule\n"
            + "\n".join(" & ".join(r) + " \\\\" for r in rows) + "\n\\bottomrule\n\\end{tabular}\\end{table}\n")


def s5():
    rb = pd.read_csv(os.path.join(A, "robustness_table_rev.csv"))
    sig = {a: pd.read_csv(os.path.join(A, f"significance_table_rev_gap2_twoway_{a}.csv"))
           for a in ("nometeo", "meteo")}
    m = rb.merge(sig["nometeo"][["region", "pollutant", "placebo_p"]], on=["region", "pollutant"]) \
          .merge(sig["meteo"][["region", "pollutant", "placebo_meteo_p"]], on=["region", "pollutant"])
    rows = []
    for _, r in sortkey(m).iterrows():
        d = dp(r.pollutant)
        cells = [num(r[f"H3_{L}d{s}"], d) + ("$^*$" if r[f"p_{L}d{s}"] < 0.05 else "")
                 for L in (14, 21, 28) for s in ("", "_meteo")]
        rows.append([r.region, TEX[r.pollutant]] + cells +
                    [pv(r.placebo_p), pv(r.placebo_meteo_p)])
    cap = ("Window-length sensitivity and placebo-window pre-trend test, revised specification (pre window "
           "ending LNY$-$3; standard errors clustered by grid cell and date). Columns give the DiD estimate "
           "for 14-, 21- and 28-day windows without (nm) and with (met) ERA5 covariates; $^*$ $p<0.05$. "
           "The last two columns are $p$-values of the placebo-window DiD (LNY$-$42..$-$22 vs pre window); "
           "$p<0.05$ indicates differential pre-trends. Units: $\\upmu$g/m$^3$ (CO in mg/m$^3$).")
    return "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n" + longtable(cap, ["Region", "Pollutant", "14d nm", "14d met", "21d nm", "21d met",
                                         "28d nm", "28d met", "Plac. $p$ nm", "Plac. $p$ met"],
                                   rows, "llrrrrrrrr") + "}\n"


def s9():
    t = {a: pd.read_csv(os.path.join(A, f"significance_table_rev_gap2_twoway_{a}.csv"))
         for a in ("nometeo", "meteo")}
    m = t["nometeo"].merge(t["meteo"], on=["region", "pollutant"], suffixes=("_n", "_m"))
    rows = []
    for _, r in sortkey(m).iterrows():
        d = dp(r.pollutant)
        row = [r.region, TEX[r.pollutant]]
        for s in ("_n", "_m"):
            est = num(r["H3_lockdown_DiD" + s], d)
            if r["H3_qval" + s] < 0.05:
                est = f"\\textbf{{{est}}}"
            row += [est, f"({num(r['H3_ci_lo' + s], d)}, {num(r['H3_ci_hi' + s], d)})",
                    pv(r['H3_qval' + s])]
        rows.append(row)
    cap = ("Difference-in-differences lockdown estimate net of the holiday (2020 vs 2019; data: CHAP) for "
           "every region $\\times$ pollutant, without and with ERA5 meteorological covariates (2-m "
           "temperature, relative humidity, wind speed and direction, precipitable water, surface pressure). "
           "95\\% confidence intervals from standard errors clustered by grid cell and calendar date; $q$: "
           "Benjamini--Hochberg FDR within pollutant; bold: $q<0.05$. Units: $\\upmu$g/m$^3$ (CO in mg/m$^3$).")
    return "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n" + longtable(cap, ["Region", "Pollutant", "No met.", "95\\% CI", "$q$",
                                         "ERA5-adj.", "95\\% CI", "$q$"], rows, "llrcrrcr") + "}\n"


def s10():
    s = pd.read_csv(os.path.join(A, "placebo_years_summary.csv"))
    rows = []
    for _, r in sortkey(s).iterrows():
        d = dp(r.pollutant)
        rows.append([r.region, TEX[r.pollutant], "nm" if r.adj == "nometeo" else "met", str(r.n_pairs),
                     num(r.placebo_min, d), num(r.placebo_max, d),
                     "--" if pd.isna(r.placebo_sd) else f"{r.placebo_sd:.{d}f}",
                     num(r.lockdown_2020_DiD, d),
                     {True: "yes", False: "no"}.get(r.lockdown_beyond_all_placebos, "--"),
                     "--" if pd.isna(r.z_vs_placebo) else num(r.z_vs_placebo, 1)])
    cap = ("Year-pair placebo test. The holiday-netted DiD is estimated between consecutive years without a "
           "lockdown, using the same specification as the main analysis: five pairs (2014v2013 to 2018v2017) "
           "for NO$_2$, SO$_2$ and CO (CHAP 10-km V1 product) and one pair (2019v2018) for PM$_{2.5}$, "
           "PM$_{10}$ and O$_3$ (CHAP 1-km product). Min/max/SD summarise the placebo estimates; ``beyond'' "
           "indicates whether the 2020 estimate is more extreme than every placebo estimate in the direction of "
           "the 2020 effect; $z$ = (2020 estimate $-$ placebo mean)/placebo SD. Beyond and $z$ are reported "
           "only with at least three pairs. nm/met: without/with ERA5 covariates. Units: $\\upmu$g/m$^3$ (CO "
           "in mg/m$^3$).")
    return "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n" + longtable(cap, ["Region", "Pollutant", "Spec.", "Pairs", "Min", "Max", "SD",
                                         "2020", "Beyond", "$z$"], rows, "lllrrrrrcr") + "}\n"


def s11():
    v = pd.read_csv(os.path.join(ROOT, "outputs", "verify", "chap_lockdown_validation.csv"))
    rows = []
    v = v.assign(_r=v.region.map(ORDER.index), _p=v.pollutant.map(POLS.index)).sort_values(["_r", "_p"])
    for _, r in v.iterrows():
        d = dp(r.pollutant)
        rows.append([r.region, TEX[r.pollutant], str(r.n_cells), f"{r.A_r_daily_2020:.2f}",
                     num(r.B_change2020_cnemc, d), num(r.B_change2020_chap, d),
                     f"{r.B_attenuation_chap_over_cnemc:.2f}", f"{r.B_r_change_across_cells:.2f}",
                     num(r.C_DiD_2020v2021_cnemc, d) + ("$^*$" if r.C_p_cnemc < 0.05 else ""),
                     num(r.C_DiD_2020v2021_chap, d) + ("$^*$" if r.C_p_chap < 0.05 else "")])
    cap = ("Evaluation of CHAP against CNEMC station observations at identical grid cell-days during the "
           "lockdown period. Over December 2019 the daily correlations are 0.94 (PM$_{2.5}$), 0.93 (PM$_{10}$), "
           "0.79 (SO$_2$), 0.86 (NO$_2$), 0.80 (O$_3$) and 0.86 (CO). $r$: daily correlation in the 2020 "
           "pre+peri window; $\\Delta$: 2020 peri$-$pre change from each source; ratio: CHAP/CNEMC change; "
           "$r_\\Delta$: cross-cell correlation of the change; DiD: 2020 vs 2021 holiday-netted DiD on each "
           "source (two-way clustered, $^*$ $p<0.05$). CNEMC data for 2019 are unavailable, hence 2021 as "
           "control. CHAP O$_3$ is MDA8 whereas the CNEMC series is a daily mean. CHAP assimilates CNEMC "
           "observations, so agreement is an upper bound on accuracy.")
    return "{\\footnotesize\\setlength{\\tabcolsep}{3pt}\n" + longtable(cap, ["Region", "Pollutant", "Cells", "$r$", "$\\Delta$ CNEMC",
                                         "$\\Delta$ CHAP", "Ratio", "$r_\\Delta$", "DiD CNEMC",
                                         "DiD CHAP"], rows, "llrrrrrrrr") + "}\n"


def s12():
    t = pd.read_csv(os.path.join(A, "significance_table_rev_gap2_twoway_nometeo.csv"))
    n = (t.groupby("region").n_obs.first() // 80).reindex(ORDER)
    rows = [[r, "city" if i >= 6 else ("nation" if r == "National" else "province/region"), str(int(c))]
            for i, (r, c) in enumerate(n.items())]
    return ("\\begin{table}[H]\\centering\n\\caption{Number of $0.25^{\\circ}$ monitoring grid cells per "
            "region. Each DiD regression uses cells $\\times$ 40 days $\\times$ 2 years observations "
            "(national: 59,760). City-level estimates rest on few cells, so cluster-robust inference for "
            "cities is unreliable and city results are treated as descriptive.}\n"
            "\\begin{tabular}{llr}\n\\toprule\n\\textbf{Region} & \\textbf{Level} & \\textbf{Grid cells}\\\\\n"
            "\\midrule\n" + "\n".join(" & ".join(r) + " \\\\" for r in rows) +
            "\n\\bottomrule\n\\end{tabular}\\end{table}\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for name, fn in [("S2", s2), ("S4", s4), ("S5", s5), ("S9", s9), ("S10", s10), ("S11", s11),
                     ("S12", s12)]:
        with open(os.path.join(OUT, f"{name}.tex"), "w") as fh:
            fh.write(fn())
        print("wrote", name)
