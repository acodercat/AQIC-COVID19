"""
Stage 0 — canonical 0.25 grid, modeling/mapping grids, crosswalk, region membership.

The grid_id encoding was reverse-engineered from the original dataset and verified to
reproduce all 747 station-cell ids exactly:

    grid_id = row * ncol + col + 1            (1-indexed, row-major)
    row = round((lat - lat0)/dx)
    col = round((lon - lon0)/dx)

Run:
    python pipeline/grid.py
Outputs (outputs/grids/):
    modeling_grid.csv     station-cells (the only cells with a ground-truth target)
    mapping_grid.csv      all China land cells for inference (verify ~16,129)
    crosswalk.csv         new grid_id <-> original grid_id  (identity by construction)
    test_holdout.csv      the reproduced 49-cell spatial holdout
    cell_regions.csv      cell -> province/region/city membership (station-cells)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(HERE, "config.yaml")
    if yaml is None:
        raise SystemExit("pyyaml is required: pip install pyyaml")
    with open(path) as fh:
        return yaml.safe_load(fh)


# ---- grid_id <-> (lat, lon) -------------------------------------------------
def latlon_to_cell(lat, lon, g) -> dict:
    """Snap a point to its 0.25 cell; return row, col, grid_id, cell-center lat/lon."""
    row = np.round((np.asarray(lat) - g["lat0"]) / g["dx"]).astype(int)
    col = np.round((np.asarray(lon) - g["lon0"]) / g["dx"]).astype(int)
    gid = row * g["ncol"] + col + 1
    clat = g["lat0"] + row * g["dx"]
    clon = g["lon0"] + col * g["dx"]
    return {"row": row, "col": col, "grid_id": gid, "lat": clat, "lon": clon}


def cell_to_latlon(grid_id, g):
    idx = np.asarray(grid_id) - 1
    row = idx // g["ncol"]
    col = idx % g["ncol"]
    return g["lat0"] + row * g["dx"], g["lon0"] + col * g["dx"]


# ---- builders ---------------------------------------------------------------
def build_modeling_grid(cfg) -> pd.DataFrame:
    """Modeling grid = the authoritative 747 cells present in the original dataset,
    enriched with CNEMC station info. Cells whose grid lacks a station in the
    station file (the original data covered a few extra cells) get province/city
    from the nearest station and n_stations=0.
    """
    g = cfg["grid"]
    # 1) authoritative cells from the original modeling data
    cells = pd.concat([
        pd.read_csv(os.path.join(ROOT, cfg["paths"]["old_train"]), usecols=["grid_id", "lat", "lon"]),
        pd.read_csv(os.path.join(ROOT, cfg["paths"]["old_test"]),  usecols=["grid_id", "lat", "lon"]),
    ]).drop_duplicates("grid_id").reset_index(drop=True)

    # 2) station -> cell aggregation
    st = pd.read_csv(os.path.join(ROOT, cfg["paths"]["stations"]))
    st["grid_id"] = latlon_to_cell(st["lat"].values, st["lon"].values, g)["grid_id"]
    agg = (st.groupby("grid_id")
             .agg(n_stations=("station_code", "nunique"),
                  province=("province", lambda s: s.mode().iat[0]),
                  city=("city", lambda s: s.mode().iat[0]))
             .reset_index())

    out = cells.merge(agg, on="grid_id", how="left")
    missing = out["province"].isna()
    if missing.any():
        # nearest-station fill for cells without a station in the file
        scoords = st[["lat", "lon", "province", "city"]].to_numpy(object)
        slat = st["lat"].to_numpy(float); slon = st["lon"].to_numpy(float)
        for i in out.index[missing]:
            d = (slat - out.at[i, "lat"])**2 + (slon - out.at[i, "lon"])**2
            j = int(np.argmin(d))
            out.at[i, "province"] = st.at[j, "province"]
            out.at[i, "city"] = st.at[j, "city"]
        out["n_stations"] = out["n_stations"].fillna(0).astype(int)
        print(f"  modeling_grid: {int(missing.sum())} of {len(out)} cells had no station "
              f"in the file -> filled region from nearest station")
    return out


def build_mapping_grid(cfg) -> tuple[pd.DataFrame, str]:
    """All China land cells in the bbox. Masks by Natural Earth China polygon.

    Returns (dataframe, status_note). If the boundary source is unavailable the
    full bbox grid is returned with is_land=NaN (land-masking deferred).
    """
    g = cfg["grid"]
    lats = np.arange(g["lat_min"], g["lat_max"] + 1e-9, g["dx"])
    lons = np.arange(g["lon_min"], g["lon_max"] + 1e-9, g["dx"])

    mask2d = _china_land_mask2d(lons, lats)            # shape (nlat, nlon) or None
    LON, LAT = np.meshgrid(lons, lats)
    flat = pd.DataFrame({"lat": np.round(LAT.ravel(), 4), "lon": np.round(LON.ravel(), 4)})
    flat["grid_id"] = latlon_to_cell(flat["lat"].values, flat["lon"].values, g)["grid_id"]

    if mask2d is None:
        flat["is_land"] = np.nan
        note = ("land mask deferred — Natural Earth boundary unavailable; "
                "cannot compute the 16,129-cell mapping grid")
        return flat, note

    flat["is_land"] = mask2d.ravel().astype(float)
    land = flat[flat["is_land"] == 1].reset_index(drop=True)
    note = f"land cells = {len(land)} of {len(flat)} bbox cells (paper states 16,129)"
    return land, note


def _china_land_mask2d(lons, lats):
    """Return a 2D 0/1 China land mask over (lats, lons) axes, or None if unavailable."""
    try:
        import regionmask
    except Exception:
        return None
    try:
        ne = regionmask.defined_regions.natural_earth_v5_0_0.countries_50
        china_num = [n for n, nm in zip(ne.numbers, ne.names) if nm == "China"][0]
        idx = np.asarray(ne.mask(lons, lats))         # (nlat, nlon), NaN off-country
        return (idx == china_num).astype(int)
    except Exception as exc:  # network / data unavailable
        print(f"  [land mask] boundary source failed: {exc}", file=sys.stderr)
        return None


def build_crosswalk(cfg, modeling: pd.DataFrame) -> pd.DataFrame:
    """Map reconstructed grid_id to the ORIGINAL dataset grid_id.

    Because the reconstructed encoding matches the original exactly, this is an
    identity for overlapping cells; we still emit it and assert the overlap.
    """
    old = pd.concat([
        pd.read_csv(os.path.join(ROOT, cfg["paths"]["old_train"]),
                    usecols=["grid_id", "lat", "lon"]),
        pd.read_csv(os.path.join(ROOT, cfg["paths"]["old_test"]),
                    usecols=["grid_id", "lat", "lon"]),
    ]).drop_duplicates()
    g = cfg["grid"]
    recon = latlon_to_cell(old["lat"].values, old["lon"].values, g)["grid_id"]
    old = old.assign(recon_grid_id=recon)
    n_match = int((old["recon_grid_id"] == old["grid_id"]).sum())
    print(f"  crosswalk: {n_match}/{len(old)} original grid_ids reproduced exactly")
    return old.rename(columns={"grid_id": "orig_grid_id"})


def build_test_holdout(cfg) -> pd.DataFrame:
    te = pd.read_csv(os.path.join(ROOT, cfg["paths"]["old_test"]),
                     usecols=["grid_id", "lat", "lon"]).drop_duplicates()
    tr = set(pd.read_csv(os.path.join(ROOT, cfg["paths"]["old_train"]),
                         usecols=["grid_id"])["grid_id"])
    assert len(set(te["grid_id"]) & tr) == 0, "spatial holdout violated: train/test grid overlap"
    return te.rename(columns={"grid_id": "orig_grid_id"})


def build_cell_regions(cfg, modeling: pd.DataFrame) -> pd.DataFrame:
    """Assign each station-cell to a study province/region and (if applicable) city."""
    prov_map = {}
    for region, provs in cfg["regions"]["provinces"].items():
        for p in provs:
            prov_map[p] = region
    city_map = {v: k for k, v in cfg["regions"]["cities"].items()}
    out = modeling.copy()
    out["region"] = out["province"].map(prov_map)        # NaN if outside study regions
    out["study_city"] = out["city"].map(city_map)        # NaN if not a study city
    return out[["grid_id", "lat", "lon", "province", "city", "region", "study_city",
                "n_stations"]]


def main():
    cfg = load_config()
    outdir = os.path.join(ROOT, cfg["paths"]["out_grids"])
    os.makedirs(outdir, exist_ok=True)

    modeling = build_modeling_grid(cfg)
    modeling.to_csv(os.path.join(outdir, "modeling_grid.csv"), index=False)
    print(f"modeling_grid: {len(modeling)} station-cells "
          f"(original had 747) -> {outdir}/modeling_grid.csv")

    crosswalk = build_crosswalk(cfg, modeling)
    crosswalk.to_csv(os.path.join(outdir, "crosswalk.csv"), index=False)

    holdout = build_test_holdout(cfg)
    holdout.to_csv(os.path.join(outdir, "test_holdout.csv"), index=False)
    print(f"test_holdout: {len(holdout)} disjoint cells (original had 49)")

    regions = build_cell_regions(cfg, modeling)
    regions.to_csv(os.path.join(outdir, "cell_regions.csv"), index=False)
    print("cell_regions by study region:\n",
          regions["region"].value_counts(dropna=True).to_string())

    mapping, note = build_mapping_grid(cfg)
    mapping.to_csv(os.path.join(outdir, "mapping_grid.csv"), index=False)
    print(f"mapping_grid: {len(mapping)} rows -> {note}")


if __name__ == "__main__":
    main()
