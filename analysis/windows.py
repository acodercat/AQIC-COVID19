"""Stage 4 — Lunar-New-Year event-time windows for the 2019/2020/2021 comparison.

To compare the holiday effect across years, windows are defined RELATIVE to each
year's LNY anchor (not fixed calendar dates), so the holiday signal aligns:

    pre  = LNY-21d .. LNY-1d      (21 days before)
    peri = LNY     .. LNY+20d     (21-day window incl. the holiday; matches 2020-CLD)
    post = LNY+21d .. LNY+41d     (21 days after)

Anchors (from config): 2019-02-05, 2020-01-25, 2021-02-12.
A pre-trend placebo window (LNY-42d..LNY-22d) is provided for the DiD parallel-trends test.

`pre_gap` drops the last N days of the pre window in every year (event-time symmetric).
pre_gap=2 removes 23-24 Jan 2020 -- the first two days of the Wuhan lockdown, which otherwise
fall inside the 2020 pre-period -- giving pre = LNY-21d .. LNY-3d (19 days).
"""
from __future__ import annotations
import pandas as pd


def lny_windows(anchors: dict, pre=21, peri=21, post=21, pre_gap=0):
    """Return a tidy frame: year, period, date (one row per calendar date in-window)."""
    rows = []
    for year, anchor in anchors.items():
        a = pd.Timestamp(anchor)
        spans = {
            "placebo": (a - pd.Timedelta(days=pre + 21), a - pd.Timedelta(days=pre + 1)),
            "pre":     (a - pd.Timedelta(days=pre),       a - pd.Timedelta(days=1 + pre_gap)),
            "peri":    (a,                                 a + pd.Timedelta(days=peri - 1)),
            "post":    (a + pd.Timedelta(days=peri),       a + pd.Timedelta(days=peri + post - 1)),
        }
        for period, (s, e) in spans.items():
            for d in pd.date_range(s, e):
                rows.append({"year": int(year), "period": period,
                             "date": d.strftime("%Y-%m-%d"),
                             "event_day": (d - a).days})
    return pd.DataFrame(rows)


def attach_windows(df: pd.DataFrame, anchors: dict, **kw) -> pd.DataFrame:
    """Inner-join a (date, ...) table to the LNY windows -> adds year/period/event_day."""
    win = lny_windows(anchors, **kw)
    out = df.merge(win, on="date", how="inner")
    return out


if __name__ == "__main__":
    import yaml, os
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "pipeline", "config.yaml")))
    w = lny_windows({int(k): v for k, v in cfg["lny_anchors"].items()})
    print(w.groupby(["year", "period"])["date"].agg(["min", "max", "count"]).to_string())
