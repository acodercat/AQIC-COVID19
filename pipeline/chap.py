"""Stage 1 (lean) — CHAP (ChinaHighAirPollutants) -> 0.25 grid, study-region scoped.

CHAP Zenodo layout (verified): per pollutant record holds
  CHAP_<POL>_D1K_<year>_V2.zip   daily 1 km   (~2.5 GB/yr, 365 daily .nc)
  CHAP_<POL>_M1K_<year>_V2.zip   monthly 1 km (~70 MB/yr, 12 .nc)
  CHAP_<POL>_Y1K_<year>_V2.nc    yearly 1 km  (~6 MB)

For the Spring-Festival window analysis we need DAILY (D1K). To keep things small we
regrid each 1 km field to the canonical 0.25 grid but RESTRICT to a cell set (the
study-region + modeling cells), so output is a few-hundred cells x days, not all China.

Public Zenodo (no credentials). O3 is MDA8; CO in mg/m^3 (others ug/m^3).

Pre-2019 NO2/SO2/CO exist only as the older 10 km V1 product (D10K, separate Zenodo
records, ZENODO_V1 below). Used only for the year-pair placebo (analysis/placebo_years.py),
never mixed with the 1 km V2 fields in the main DiD.

Run:
  python pipeline/chap.py --pollutant no2 --year 2019 --res M1K   # quick plumbing check
  python pipeline/chap.py --pollutant no2 --year 2019 --res D1K   # real daily
  python pipeline/chap.py --pollutant pm2_5 --year 2018 --months 1,2,3 --clean
  python pipeline/chap.py --pollutant no2 --year 2017 --res D10K --months 1,2,3
Out: outputs/targets/chap_<pol>_<year>_<res>.parquet
"""
from __future__ import annotations
import os, sys, argparse, glob, zipfile, time, re
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import regrid
from grid import load_config, ROOT

ZENODO = {"pm2_5": "6398971", "pm10": "6449937", "o3": "16883507",  # O3 = open V3 record
          "no2": "15218546", "so2": "15226517", "co": "15224996"}
ZENODO_V1 = {"no2": "4641542", "so2": "4641538", "co": "4641530"}  # 10 km, pre-2019
TOKEN = {"pm2_5": "PM2.5", "pm10": "PM10", "o3": "O3", "no2": "NO2", "so2": "SO2", "co": "CO"}


def _api(rec):
    import requests
    for _ in range(6):
        try:
            r = requests.get(f"https://zenodo.org/api/records/{rec}", timeout=90,
                             headers={"User-Agent": "research-script"})
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(8)
    raise SystemExit(f"Zenodo API unreachable for record {rec}")


def _download_resumable(url, dest, expected, name, max_retries=20):
    """Stream to dest with HTTP Range resume; retry on broken connections."""
    import requests
    attempt = 0
    while True:
        have = os.path.getsize(dest) if os.path.exists(dest) else 0
        if have >= expected:
            return
        headers = {"Range": f"bytes={have}-"} if have else {}
        mode = "ab" if have else "wb"
        try:
            with requests.get(url, stream=True, timeout=120, headers=headers) as r:
                if have and r.status_code == 200:   # server ignored Range -> restart clean
                    have, mode = 0, "wb"
                r.raise_for_status()
                print(f"  downloading {name}: {have/1e6:.0f}/{expected/1e6:.0f} MB"
                      f"{' (resume)' if have else ''} ...")
                with open(dest, mode) as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
        except Exception as e:
            attempt += 1
            if attempt > max_retries:
                raise SystemExit(f"download failed after {max_retries} retries: {e}")
            got = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"  [retry {attempt}] {type(e).__name__} at {got/1e6:.0f} MB; resuming")
            time.sleep(min(5 * attempt, 30))


def fetch(pollutant, year, res, months=None) -> list[str]:
    """Download (and unzip if needed) the CHAP file for pollutant/year/res. Returns .nc paths.
    months: optional set of calendar months; only those daily files are extracted/returned."""
    import requests
    rec = ZENODO_V1[pollutant] if res.endswith("10K") else ZENODO[pollutant]
    want = f"CHAP_{TOKEN[pollutant]}_{res}_{year}_V"        # version-agnostic (V2/V4/...)
    meta = _api(rec)
    cands = [x for x in meta["files"] if x["key"].startswith(want)]
    if not cands:
        raise SystemExit(f"no file {want}* in record {rec}; files: "
                         f"{[x['key'] for x in meta['files']][:6]}")
    # prefer .zip/.nc over .rar when multiple versions exist
    f = sorted(cands, key=lambda x: (x["key"].endswith(".rar"), x["key"]))[0]
    cache = os.path.join(ROOT, "outputs", "targets", "_cache")
    os.makedirs(cache, exist_ok=True)
    dest = os.path.join(cache, f["key"])
    if not os.path.exists(dest) or os.path.getsize(dest) != f["size"]:
        _download_resumable(f["links"]["self"], dest, f["size"], f["key"])
    if dest.endswith(".nc"):
        return [dest]
    exdir = dest[:-4]
    ncs = [x for x in glob.glob(os.path.join(exdir, "**", "*.nc"), recursive=True)
           if _in_months(x, months)]
    if not ncs:
        _extract(dest, exdir, months)
        ncs = [x for x in glob.glob(os.path.join(exdir, "**", "*.nc"), recursive=True)
               if _in_months(x, months)]
    return sorted(ncs)


def _in_months(name, months) -> bool:
    if not months or not name.endswith(".nc"):
        return name.endswith(".nc") or not months
    m = re.search(r"_(\d{4})(\d{2})\d{2}_", os.path.basename(name))
    return bool(m) and int(m.group(2)) in months


def _extract(archive, exdir, months=None):
    """Extract .zip (zipfile) or .rar (libarchive) into exdir (flat); optionally only `months`."""
    os.makedirs(exdir, exist_ok=True)
    print(f"  extracting {os.path.basename(archive)}"
          f"{f' (months {sorted(months)})' if months else ''} ...")
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as z:
            z.extractall(exdir, members=[n for n in z.namelist() if _in_months(n, months)])
    elif archive.endswith(".rar"):
        import libarchive
        with libarchive.file_reader(archive) as arch:
            for entry in arch:
                if not entry.isfile or not _in_months(entry.pathname, months):
                    continue
                out = os.path.join(exdir, os.path.basename(entry.pathname))
                with open(out, "wb") as fh:
                    for block in entry.get_blocks():
                        fh.write(block)
    else:
        raise SystemExit(f"unknown archive type: {archive}")


def _restrict_cells(cfg):
    """Return the cell-id set to keep: study-region cells + all modeling cells."""
    gdir = os.path.join(ROOT, cfg["paths"]["out_grids"])
    reg = pd.read_csv(os.path.join(gdir, "cell_regions.csv"))
    keep = set(reg["grid_id"])  # all 747 modeling cells (incl. the 197 study-region ones)
    return keep


def to_grid(cfg, pollutant, paths, keep_cells) -> pd.DataFrame:
    import xarray as xr
    g = cfg["grid"]
    gk = dict(lat0=g["lat0"], lon0=g["lon0"], dx=g["dx"], ncol=g["ncol"])
    out = []
    for path in paths:
        ds = xr.open_dataset(path)
        var = list(ds.data_vars)[0]
        latn = "lat" if "lat" in ds.coords else "latitude"
        lonn = "lon" if "lon" in ds.coords else "longitude"
        lat = ds[latn].values
        lon = ds[lonn].values
        # date from filename token YYYYMMDD (daily) or YYYYMM (monthly) or YYYY (yearly)
        date = _infer_date(path)
        LON, LAT = np.meshgrid(lon, lat)
        vals = np.asarray(ds[var].squeeze().values, float)
        # CHAP fill values are large negatives / NaN
        vals = np.where(vals < -900, np.nan, vals)
        cell = regrid.reduce_mean(LAT.ravel(), LON.ravel(), vals.ravel(), **gk)
        cell = cell[cell["grid_id"].isin(keep_cells)]
        cell["date"] = date
        cell = cell.rename(columns={"value": pollutant})
        out.append(cell)
    return pd.concat(out, ignore_index=True)


def _infer_date(path):
    b = os.path.basename(path)
    m = re.search(r"_(\d{8})_", b) or re.search(r"_(\d{6})_", b) or re.search(r"_(\d{4})_", b)
    if not m:
        raise SystemExit(f"cannot infer date from {b}")
    s = m.group(1)
    fmt = {8: "%Y%m%d", 6: "%Y%m", 4: "%Y"}[len(s)]
    return pd.to_datetime(s, format=fmt).strftime("%Y-%m-%d")


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--pollutant", choices=list(ZENODO), required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--res", choices=["D1K", "M1K", "Y1K", "D10K"], default="D1K")
    ap.add_argument("--months", default="", help="e.g. 1,2,3 -> extract only these months")
    ap.add_argument("--clean", action="store_true", help="delete extracted .nc after regridding")
    a = ap.parse_args()
    months = {int(x) for x in a.months.split(",")} if a.months else None
    keep = _restrict_cells(cfg)
    paths = fetch(a.pollutant, a.year, a.res, months)
    print(f"  {len(paths)} .nc files; regridding to {len(keep)} kept cells ...")
    df = to_grid(cfg, a.pollutant, paths, keep)
    outdir = os.path.join(ROOT, "outputs", "targets"); os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, f"chap_{a.pollutant}_{a.year}_{a.res}.parquet")
    df.to_parquet(out, index=False)
    print(f"wrote {out} ({len(df)} cell-dates, {df['grid_id'].nunique()} cells, "
          f"{df['date'].min()}..{df['date'].max()})")
    if a.clean and paths and not paths[0].endswith(f"_{a.year}_V1.nc"):
        import shutil
        exdir = os.path.commonpath(paths)
        while exdir and not os.path.basename(exdir).startswith("CHAP_"):
            exdir = os.path.dirname(exdir)          # archives may nest a year folder
        if os.path.basename(exdir).startswith("CHAP_"):
            shutil.rmtree(exdir); print(f"  cleaned {exdir}")


if __name__ == "__main__":
    main()
