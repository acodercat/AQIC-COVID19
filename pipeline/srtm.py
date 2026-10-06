"""Stage 1 — SRTM 30 m elevation -> mean per 0.25 cell (static feature, via GEE).

Requires Google Earth Engine auth (`earthengine authenticate`; a Cloud project).
Reduces USGS/SRTMGL1_003 to the mean elevation within each 0.25 mapping-grid cell.

Run:  python pipeline/srtm.py --project <gee-project>
Out:  outputs/features/elevation_grid.parquet
"""
from __future__ import annotations
import os, sys, argparse
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grid import load_config, ROOT


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True, help="GEE Cloud project id")
    a = ap.parse_args()
    try:
        import ee
    except ImportError:
        raise SystemExit("pip install earthengine-api; run `earthengine authenticate`")
    ee.Initialize(project=a.project)

    g = cfg["grid"]
    grids = pd.read_csv(os.path.join(ROOT, cfg["paths"]["out_grids"], "mapping_grid.csv"))
    dem = ee.Image("USGS/SRTMGL1_003")
    half = g["dx"] / 2.0
    rows = []
    # batch cells to stay within GEE limits
    for start in range(0, len(grids), 2000):
        chunk = grids.iloc[start:start + 2000]
        feats = [ee.Feature(ee.Geometry.Rectangle(
                    [r.lon - half, r.lat - half, r.lon + half, r.lat + half]),
                    {"grid_id": int(r.grid_id)}) for r in chunk.itertuples()]
        fc = ee.FeatureCollection(feats)
        red = dem.reduceRegions(fc, ee.Reducer.mean(), scale=30)
        for f in red.getInfo()["features"]:
            p = f["properties"]
            rows.append({"grid_id": p["grid_id"], "ELEVATION": p.get("mean")})
        print(f"  elevation: {min(start+2000, len(grids))}/{len(grids)} cells")
    out = os.path.join(ROOT, "outputs", "features", "elevation_grid.parquet")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
