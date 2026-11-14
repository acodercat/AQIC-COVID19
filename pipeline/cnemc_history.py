"""CNEMC station history (2015-2021) over the Lunar-New-Year windows, for the multi-year control
analysis (analysis/multiyear.py).

Source: the national monitoring-station archive mirrored from the CNEMC real-time platform at
https://quotsoft.net/air/ (one CSV per day; rows = hour x type, columns = station codes).
Only the LNY windows are downloaded: LNY-42 .. LNY+41 (placebo, pre, peri, post) for 2015-2021.

Processing
  - hourly SO2, NO2, CO, PM2.5, PM10, O3 per station (Beijing time, hours 0-23);
  - daily mean if >= 18 valid hours; O3 additionally as MDA8 (8-h running means within the day,
    each with >= 6 valid hours; the day is valid if >= 14 of the 17 running means are valid);
  - physical-range screening as in pipeline/build_dataset.py (VALID);
  - reference state: from 1 Sep 2018 (MEE Announcement 2018 No. 29, amendment to GB 3095-2012)
    gaseous pollutants are reported at 25 degC instead of 0 degC. Values before that date are
    multiplied by 273.15/298.15 so that every year is on the post-2018 basis. PM is unchanged;
  - station-level balance: per pollutant, a station is kept only if it has >= 80% valid days in
    BOTH the pre and peri windows of EVERY year 2015-2021, so the network composition is the
    same in every year; kept stations are averaged to the 0.25 deg grid cell.

Run:
  python pipeline/cnemc_history.py --download     # ~590 files, ~1 GB, 1 request/s
  python pipeline/cnemc_history.py --build
Out: outputs/cnemc/cnemc_station_daily.parquet  (station_code, date, <pollutants>, o3_mean)
     outputs/cnemc/cnemc_cells.parquet          (grid_id, date, <pollutants>)  balanced panel
     outputs/cnemc/cnemc_balance.csv            stations/cells kept per pollutant
"""
from __future__ import annotations
import os, sys, time, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import load_config, ROOT, latlon_to_cell
from build_dataset import VALID

URL = "https://quotsoft.net/air/data/china_sites_{}.csv"
YEARS = range(2015, 2022)
RAW = os.path.join(ROOT, "outputs", "cnemc", "_raw")
OUT = os.path.join(ROOT, "outputs", "cnemc")
TYPES = {"PM2.5": "pm2_5", "PM10": "pm10", "SO2": "so2", "NO2": "no2", "CO": "co", "O3": "o3"}
GASES = ["so2", "no2", "co", "o3", "o3_mean"]
REF_CHANGE = pd.Timestamp("2018-09-01")
REF_FACTOR = 273.15 / 298.15
MIN_HOURS, MIN_8H, MIN_WIN = 18, 6, 14
BALANCE = 0.8


def anchors(cfg) -> dict:
    a = {int(k): v for k, v in cfg["lny_anchors"].items()}
    a.update({int(k): v for k, v in cfg["placebo_lny_anchors"].items()})
    return {y: pd.Timestamp(a[y]) for y in YEARS}


def window_dates(cfg) -> list[pd.Timestamp]:
    out = []
    for y, a in anchors(cfg).items():
        out += list(pd.date_range(a - pd.Timedelta(days=42), a + pd.Timedelta(days=41)))
    return out


def download(cfg):
    import requests
    os.makedirs(RAW, exist_ok=True)
    missing = []
    for d in window_dates(cfg):
        s = d.strftime("%Y%m%d")
        dest = os.path.join(RAW, f"china_sites_{s}.csv")
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        for attempt in range(4):
            try:
                r = requests.get(URL.format(s), timeout=60, headers={"User-Agent": "research-script"})
                if r.status_code == 404:
                    missing.append(s); break
                r.raise_for_status()
                with open(dest + ".part", "wb") as fh:
                    fh.write(r.content)
                os.replace(dest + ".part", dest)
                break
            except Exception as e:
                print(f"  [retry {attempt + 1}] {s}: {type(e).__name__}")
                time.sleep(5 * (attempt + 1))
        else:
            missing.append(s)
        time.sleep(1.0)
    print(f"download done; missing days: {missing or 'none'}")


def mda8(h: np.ndarray) -> np.ndarray:
    """h: (24, n_stations) hourly O3 with NaN -> MDA8 per station (NaN if too few windows)."""
    valid = np.isfinite(h)
    hz = np.where(valid, h, 0.0)
    cs = np.vstack([np.zeros((1, h.shape[1])), np.cumsum(hz, 0)])
    cn = np.vstack([np.zeros((1, h.shape[1])), np.cumsum(valid, 0)])
    sums = cs[8:] - cs[:-8]                       # 17 windows: hours 0-7 .. 16-23
    cnts = cn[8:] - cn[:-8]
    means = np.where(cnts >= MIN_8H, sums / np.maximum(cnts, 1), np.nan)
    ok = np.isfinite(means).sum(0) >= MIN_WIN
    with np.errstate(all="ignore"):
        mx = np.nanmax(np.where(np.isfinite(means), means, -np.inf), axis=0)
    return np.where(ok, mx, np.nan)


def parse_day(path) -> pd.DataFrame | None:
    d = pd.read_csv(path)
    if d.empty or "type" not in d:
        return None
    date = pd.to_datetime(str(int(d["date"].iloc[0])), format="%Y%m%d")
    stations = list(d.columns[3:])
    rows = {}
    for t, name in TYPES.items():
        sub = d[d["type"] == t].set_index("hour").reindex(range(24))[stations]
        h = sub.to_numpy(float)
        n = np.isfinite(h).sum(0)
        with np.errstate(all="ignore"):
            mean = np.where(n >= MIN_HOURS, np.nanmean(h, 0), np.nan)
        if name == "o3":
            rows["o3"] = mda8(h)
            rows["o3_mean"] = mean
        else:
            rows[name] = mean
    out = pd.DataFrame(rows)
    out.insert(0, "station_code", stations)
    out.insert(1, "date", date)
    return out


def build(cfg):
    files = sorted(f for f in os.listdir(RAW) if f.endswith(".csv"))
    parts = [p for p in (parse_day(os.path.join(RAW, f)) for f in files) if p is not None]
    st = pd.concat(parts, ignore_index=True)
    for p, (lo, hi) in VALID.items():
        st.loc[~st[p].between(lo, hi), p] = np.nan
    st.loc[~st["o3_mean"].between(*VALID["o3"]), "o3_mean"] = np.nan
    old = st["date"] < REF_CHANGE
    st.loc[old, GASES] = st.loc[old, GASES] * REF_FACTOR
    st["date"] = st["date"].dt.strftime("%Y-%m-%d")
    os.makedirs(OUT, exist_ok=True)
    st.to_parquet(os.path.join(OUT, "cnemc_station_daily.parquet"), index=False)
    print(f"station-days: {len(st)}, stations: {st.station_code.nunique()}, files: {len(files)}")

    # windows for the balance rule
    sys.path.insert(0, os.path.join(ROOT, "analysis"))
    from windows import lny_windows
    win = lny_windows({y: a.strftime("%Y-%m-%d") for y, a in anchors(cfg).items()})
    stw = st.merge(win, on="date", how="inner")
    meta = pd.read_csv(os.path.join(ROOT, cfg["paths"]["stations"]))
    meta["grid_id"] = latlon_to_cell(meta["lat"].values, meta["lon"].values, cfg["grid"])["grid_id"]
    cells747 = set(pd.read_csv(os.path.join(ROOT, "outputs", "grids", "modeling_grid.csv"))["grid_id"])
    meta = meta[meta.grid_id.isin(cells747)][["station_code", "grid_id"]]

    pols = list(VALID) + ["o3_mean"]
    ndays = win[win.period.isin(["pre", "peri"])].groupby(["year", "period"]).size()
    cell_frames, bal = [], []
    for p in pols:
        g = (stw[stw.period.isin(["pre", "peri"])].dropna(subset=[p])
             .groupby(["station_code", "year", "period"]).size())
        share = (g / ndays.reindex(g.index.droplevel(0)).to_numpy()).unstack(["year", "period"])
        need = [(y, per) for y in YEARS for per in ("pre", "peri")]
        share = share.reindex(columns=pd.MultiIndex.from_tuples(need)).fillna(0)
        keep = share.index[(share >= BALANCE).all(axis=1)]
        keep = meta[meta.station_code.isin(keep)]
        c = (st[st.station_code.isin(keep.station_code)][["station_code", "date", p]]
             .merge(keep, on="station_code").groupby(["grid_id", "date"], as_index=False)[p].mean())
        cell_frames.append(c)
        bal.append({"pollutant": p, "stations_in_archive": stw.dropna(subset=[p]).station_code.nunique(),
                    "stations_kept": len(keep), "cells_kept": keep.grid_id.nunique()})
    cells = cell_frames[0]
    for c in cell_frames[1:]:
        cells = cells.merge(c, on=["grid_id", "date"], how="outer")
    cells.to_parquet(os.path.join(OUT, "cnemc_cells.parquet"), index=False)
    bal = pd.DataFrame(bal)
    bal.to_csv(os.path.join(OUT, "cnemc_balance.csv"), index=False)
    print(bal.to_string(index=False))


def _selftest():
    h = np.full((24, 2), np.nan)
    h[:, 0] = np.arange(24.0)                     # increasing -> MDA8 = mean(16..23) = 19.5
    h[:5, 1] = 50.0                               # too few hours -> NaN
    m = mda8(h)
    assert abs(m[0] - 19.5) < 1e-9 and np.isnan(m[1]), m
    print("cnemc_history self-test OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    if a.download:
        download(cfg)
    if a.build:
        build(cfg)
    if not (a.download or a.build):
        _selftest()
