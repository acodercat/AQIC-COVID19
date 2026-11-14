"""Compare current results with the revision-1 snapshot (outputs/_r1_snapshot/).

Lists every DiD estimate whose value, significance (q < 0.05) or pre-trend verdict (p < 0.05)
changed, plus the event-study pre-trend tests, window robustness and year-pair placebo summary.
Out: outputs/analysis/compare_r1.md
"""
from __future__ import annotations
import os
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEW = os.path.join(ROOT, "outputs", "analysis")
OLD = os.path.join(ROOT, "outputs", "_r1_snapshot", "analysis")
KEY = ["region", "pollutant"]


def load(name, key=KEY):
    o = pd.read_csv(os.path.join(OLD, name)); n = pd.read_csv(os.path.join(NEW, name))
    return o.merge(n, on=key, suffixes=("_old", "_new"))


def md(df):
    """Markdown table without the optional `tabulate` dependency."""
    if not len(df):
        return "(none)"
    df = df.round(3).astype(str)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(r) + " |" for r in df.to_numpy()]
    return "\n".join(lines)


def main():
    out = ["# Revision-1 vs current results\n"]
    for spec in ("gap2_twoway_meteo", "gap2_twoway_nometeo", "gap2_cell_meteo", "gap0_twoway_meteo"):
        m = load(f"significance_table_rev_{spec}.csv")
        m["d_DiD"] = m.H3_lockdown_DiD_new - m.H3_lockdown_DiD_old
        so, sn = m.H3_qval_old < .05, m.H3_qval_new < .05
        out.append(f"\n## {spec}\nsignificant (q<0.05): {so.sum()} -> {sn.sum()}; "
                   f"max |change in DiD| = {m.d_DiD.abs().max():.3f}\n")
        cols = KEY + ["H3_lockdown_DiD_old", "H3_lockdown_DiD_new", "H3_qval_old", "H3_qval_new"]
        out.append("\nSignificance changed:\n\n" + md(m.loc[so != sn, cols]))
        pcol = "placebo_meteo_p" if "placebo_meteo_p_old" in m else "placebo_p"
        po, pn = m[f"{pcol}_old"] < .05, m[f"{pcol}_new"] < .05
        out.append(f"\n\nPre-trend verdict ({pcol}) changed:\n\n" +
                   md(m.loc[po != pn, KEY + [f"{pcol}_old", f"{pcol}_new"]]))
        out.append("\n\nNO2 rows:\n\n" + md(m.loc[m.pollutant == "no2", cols + [f"{pcol}_old", f"{pcol}_new"]]))
    e = load("event_study_pretrend.csv", KEY + ["adj"])
    e = e[(e.adj == "meteo") & e.region.isin(["National", "BTH", "Jilin", "Xinjiang"])]
    out.append("\n\n## Event-study pre-trend (meteo)\n\n" +
               md(e[KEY + ["pre_slope_p_old", "pre_slope_p_new", "peri_minus_pre_old", "peri_minus_pre_new"]]))
    r = load("robustness_table_rev.csv")
    c = [x for x in r.columns if x.startswith("H3_") and x.endswith("meteo_new") or
         (x.startswith("H3_") and x.endswith("meteo_old"))]
    out.append("\n\n## Window robustness (meteo), NO2\n\n" + md(r.loc[r.pollutant == "no2", KEY + sorted(c)]))
    p = load("placebo_years_summary.csv", KEY + ["adj"])
    p = p[(p.pollutant == "no2") & (p.adj == "meteo")]
    out.append("\n\n## Year-pair placebo, NO2 (meteo)\n\n" + md(p[KEY + [
        "lockdown_2020_DiD_old", "lockdown_2020_DiD_new", "placebo_min_new", "placebo_max_new",
        "lockdown_beyond_all_placebos_old", "lockdown_beyond_all_placebos_new", "z_vs_placebo_new"]]))
    f = os.path.join(NEW, "compare_r1.md")
    open(f, "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
