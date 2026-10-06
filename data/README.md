# Processed data for "Separating the Holiday from the Lockdown"

Processed data and results for Ran, Jiang, Ma & Song, *Separating the Holiday from the Lockdown: A Multiscale Difference-in-Differences Analysis of COVID-19 Effects on Air Quality in China* (*Scientific Reports*, under revision). This folder is part of the code repository (https://github.com/acodercat/AQIC-COVID19; archived on Zenodo, DOI 10.5281/zenodo.23181067). The repository README maps every table and figure to the script that produces it. The files here are copies of the corresponding files in `outputs/`, which the scripts regenerate.

All gridded data use a regular 0.25° grid. `grid_id = row * 249 + col + 1`, with `row = (lat - 18.0) / 0.25` and `col = (lon - 73.0) / 0.25` (cell centres). The analysis uses the 747 grid cells that contain at least one CNEMC monitoring station. Units: µg m⁻³, except CO in mg m⁻³. O₃ from CHAP is the maximum daily 8-hour average (MDA8). Dates are Beijing-time days (YYYY-MM-DD).

## What is not included, and why

The 1-km CHAP NO₂, SO₂ and CO products for 2019–2021 are distributed by their providers under restricted access. We therefore do **not** redistribute gridded fields derived from them (the 2019–2021 NO₂/SO₂/CO panels and maps). All results computed from them are included in `results/` and `validation/`. To reproduce those results, request the products from the providers (Zenodo records 15218546, 15226517, 15224996) and run `pipeline/chap.py`.

## Contents

### `results/` — analysis outputs (CSV)
| File | Content |
|---|---|
| `significance_table_rev_gap2_twoway_{nometeo,meteo}.csv` | **Main DiD results** (Table 1, Fig. 5, Table S9). One row per region × pollutant. |
| `significance_table_rev_gap{0,2}_{cell,twoway}_{nometeo,meteo}.csv` | The same table under the other specifications: `gap0` keeps 23–24 Jan 2020 in the pre window; `cell` uses one-way clustering by grid cell. |
| `event_study_coefs.csv` | Event-study coefficients (2020 − 2019) by event day, relative to the pre-period mean (Fig. 6, Fig. S12). |
| `event_study_pretrend.csv` | Pre-period SD, linear-trend test and peri − pre shift (Table S4). |
| `robustness_table_rev.csv` | DiD for 14/21/28-day windows, with CIs and p-values (Table S5). |
| `placebo_years.csv`, `placebo_years_summary.csv` | Year-pair placebo DiDs and their comparison with the 2020 estimate (Fig. 7, Table S10). |
| `descriptive_numbers.csv` | Raw pre/peri means and changes quoted in the text. |
| `clean_report.csv` | Outlier screening of the station data (Table S1). |

Key columns of the significance tables:

| Column | Meaning |
|---|---|
| `H1_holiday2019_abs` / `_pct` | 2019 change, peri − pre (holiday only) |
| `H2_2020_abs` / `_pct` | 2020 change, peri − pre (holiday + lockdown) |
| `H3_lockdown_DiD`, `H3_se`, `H3_ci_lo`, `H3_ci_hi`, `H3_p`, `H3_qval` | DiD estimate β₃ (2020 change beyond the 2019 holiday change), its SE, 95% CI, p-value and Benjamini–Hochberg q-value within pollutant |
| `H4_reversion_abs` / `_pct`, `H4_p` | 2021 vs 2019 peri-window difference |
| `placebo_DiD`, `placebo_p`, `placebo_ci_*` | Placebo-window pre-trend DiD without covariates |
| `placebo_meteo_*` | The same, with ERA5 covariates |
| `n_obs` | Cell-day observations in the DiD regression |

`adj` = `nometeo` / `meteo` marks estimates without / with ERA5 covariates throughout.

### `validation/`
- `chap_lockdown_validation.csv` — CHAP vs CNEMC at identical cell-days (Table S11). Columns: `A_*` daily agreement in the 2020 window; `B_*` the 2020 peri − pre change from each source; `C_*` a 2020-vs-2021 DiD on each source.
- `sanity_report.txt` — data-integrity checks, including the December 2019 CHAP vs CNEMC correlations.

### `ml_lur/`
- `metrics.csv` — ML-LUR skill: spatial CV (`cv_*`), spatial holdout (`test_*`) and train-2019/20 → test-2021 (`temporal2021_*`) (Table S13).
- `best_params.json` — tuned LightGBM hyperparameters (Table S3).
- `model_comparison.csv` — LightGBM vs XGBoost vs SVR (Table S8).

### `ml_lur_dataset/`
- `train_set.csv.gz`, `test_set.csv.gz` — gzip-compressed (GitHub's file-size limit). To use them with the scripts: `gunzip -c data/ml_lur_dataset/train_set.csv.gz > dataset/train_set.csv` (likewise for `test_set`). Daily station-grid samples (Dec 2019 – Nov 2021) used to train and evaluate the ML-LUR model. They contain CNEMC concentrations, NCEP FNL meteorology (`*_GLL0` columns), land-use areas (m²), elevation, AOD and calendar variables. The test set holds the 49 spatially held-out grid cells.
- The CNEMC station list (coordinates, province, city) is `dataset/national_AQ_stations.csv` in the repository.

### `grids/`
| File | Content |
|---|---|
| `modeling_grid.csv` | The 747 station cells (`grid_id`, centre `lat`/`lon`, `n_stations`, `province`, `city`) |
| `cell_regions.csv` | As above, plus the study `region` (BTH, Jiangsu, Jilin, Guangdong, Xinjiang) and `study_city` |
| `mapping_grid.csv` | All 0.25° cells in the China bounding box, with `is_land` (Natural Earth v5.0.0, 1:50m) |
| `crosswalk.csv`, `test_holdout.csv` | Mapping to the grid IDs of the original dataset; the 49 held-out cells |

### `chap_panels/` — daily CHAP at the 747 cells (Parquet: `grid_id`, `date`, `<pollutant>`)
- `chap_{pm2_5,pm10,o3}_{2018..2021}_D1K.parquet` — from the 1-km products (PM₂.₅ V4, PM₁₀ V4, O₃ V3). 2018 files cover January–March only.
- `chap_{no2,so2,co}_{2013..2018}_D10K.parquet` — from the 10-km V1 products, January–March only (placebo years). 14 of the 747 cells fall outside the 10-km product's coverage.

### `chap_maps/` — 21-day window means on the full China grid (Figs 3–4)
- `mapfields_{pm2_5,pm10,o3}_{2019,2020,2021}.parquet` — columns `grid_id`, `<pollutant>`, `lat`, `lon`, `period` (`pre`/`peri`/`post`).

### `meteorology/`
- `era5_windows.parquet` — ERA5 daily means at the 747 cells for January–March 2013–2021 and December 2019. Columns: `t2m` (°C), `rh` (%), `spfh` (kg kg⁻¹), `sp` (Pa), `wind_speed` (m s⁻¹), `wind_dir` (degrees, direction wind blows from), `gust` (daily max, m s⁻¹), `pwat` (kg m⁻²).

## Sources and licences

The data in this folder are released under CC BY 4.0 (the code in the rest of the repository is MIT-licensed). Derived data inherit the attribution requirements of their sources:
- CHAP: Wei et al. (2021, *Remote Sens. Environ.* 252:112136; 2021, *Environ. Int.* 146:106290; 2022, *Remote Sens. Environ.* 270:112775; 2022, *Environ. Sci. Technol.* 56:9988; 2023, *Atmos. Chem. Phys.* 23:1511).
- ERA5: Hersbach et al. (2020), Copernicus Climate Change Service. Contains modified Copernicus Climate Change Service information.
- CNEMC: China National Environmental Monitoring Center.
- NCEP FNL: NCAR Research Data Archive.
- MODIS MCD19A2: NASA LP DAAC.
- Natural Earth: public domain.
