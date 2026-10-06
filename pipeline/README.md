# Stage 1 — raw → 0.25° daily feature pipeline

Rebuilds the full feature/target table for **2019-01-01 → 2021-11-27** (filling the
missing 2019 gap) on consistent, corrected sources. All modules write to `outputs/`.

## Credentials needed (free accounts)
| Module | Source | Setup |
|---|---|---|
| `era5.py` | Copernicus CDS | account + `~/.cdsapirc` API key — https://cds.climate.copernicus.eu |
| `aod.py`, `srtm.py` | Google Earth Engine | `pip install earthengine-api`; `earthengine authenticate`; a Cloud project id |
| `chap.py` | Zenodo (CHAP) | none (public, but large) |
| `clcd.py` | Zenodo (CLCD) | none (download the GeoTIFF once) |

## Run order
```bash
# 0. grids (done, offline)
python pipeline/grid.py

# 1. sources (each writes per-year/per-pollutant parquet, then --assemble)
python pipeline/era5.py  --year 2019            # repeat 2020, 2021; then:
python pipeline/era5.py  --assemble
python pipeline/chap.py  --pollutant no2 --year 2019   # loop 6 pollutants x 3 years; then:
python pipeline/chap.py  --assemble
python pipeline/aod.py   --project <gee> --year 2019   # loop years; then --assemble
python pipeline/clcd.py  --tif CLCD_2019.tif --year 2019
python pipeline/srtm.py  --project <gee>

# 2. assemble the model table (tolerant: uses whatever parquets exist)
python pipeline/build_dataset.py
```

## Module status
| Module | Status | Notes |
|---|---|---|
| `grid.py` | ✅ done, verified | 747/747 ids reproduced; 49-cell holdout; 15,223 land cells |
| `metcalc.py` | ✅ done, self-tested | near-surface RH, wind speed/**direction**, UTC+8 daily agg — the met bug-fix |
| `regrid.py` | ✅ done, self-tested | area-weighted fine→0.25° mean and per-class areas |
| `era5.py` | ✅ coded | needs CDS key to download; processing wired to `metcalc` |
| `chap.py` | ✅ coded | needs Zenodo file-name verification per record; large download |
| `clcd.py` | ✅ coded | needs CLCD GeoTIFF; class map → m² + % |
| `aod.py` | ✅ coded | needs GEE auth; QA-mask + gap-fill |
| `srtm.py` | ✅ coded | needs GEE auth; static |
| `build_dataset.py` | ✅ coded | joins all + outlier policy with removal logging |

## Key corrections vs the original data (baked in)
- **Meteorology**: near-surface 2 m T / 2 m RH / 10 m wind (was upper-air); **wind
  direction added** (V-wind was missing). Daily means in **Beijing time (UTC+8)**.
- **Land cover**: carried as **both m² and %** (manuscript text says %; original data was m²).
- **Outliers**: explicit valid-range + IQR with **per-pollutant removal counts logged**
  (`outputs/features/clean_report.csv`).
- **2019 gap filled** → enables the pre-COVID baseline and Spring-Festival significance test.
