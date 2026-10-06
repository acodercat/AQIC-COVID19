# AQIC-COVID19

Code for **"Separating the Holiday from the Lockdown: A Multiscale Difference-in-Differences Analysis of COVID-19 Effects on Air Quality in China"** (Ran, Jiang, Ma & Song; *Scientific Reports*, under revision).

The study has two components:

1. **ML-LUR** — a LightGBM land-use-regression model that reconstructs daily fields of PM₂.₅, PM₁₀, SO₂, NO₂, O₃ and CO on a 0.25° grid from meteorology, land use, AOD and CNEMC station data (Dec 2019 – Nov 2021), evaluated by spatial cross-validation.
2. **Lockdown analysis** — a difference-in-differences (DiD) design on the ChinaHighAirPollutants (CHAP) record, using 2019 as a holiday-only control to separate the 2020 lockdown from the Lunar New Year (LNY) holiday, with event-study, window-length, year-pair placebo and station-based checks.

All DiD estimates use CHAP; the ML-LUR model is not used in the lockdown estimates.

## Repository layout

| Folder | Contents |
|---|---|
| `pipeline/` | Data acquisition and gridding: 0.25° grid (`grid.py`), CHAP (`chap.py`), ERA5 (`era5.py`, `era5_windows.py`), AOD, land cover, elevation, model table (`build_dataset.py`). Settings in `pipeline/config.yaml`. |
| `model/` | ML-LUR: hyperparameter search (`hpo.py`), training and spatial-holdout evaluation (`train.py`), LightGBM vs XGBoost vs SVR (`compare.py`). |
| `analysis/` | LNY windows (`windows.py`), DiD estimator (`did.py`), significance tables (`significance.py`), event study, window-length robustness, year-pair placebo, descriptive numbers, outlier report, supplementary-table generation. |
| `verify/` | Data-integrity checks (`sanity.py`) and CHAP-vs-station evaluation during the lockdown (`chap_lockdown_validation.py`). |
| `figures/` | Figure scripts. |
| `dataset/` | Station list (`national_AQ_stations.csv`); the ML-LUR train/test tables are downloaded separately (below). |
| `lgb.py`, `air_quality_lgb_model.ipynb` | Original single-model LightGBM scripts (first version of the study). |

Generated files are written to `outputs/` (not version-controlled; ~90 GB with the raw CHAP archives).

## Installation

```bash
uv venv --python 3.13 && uv pip sync requirements.txt     # or: pip install -r requirements.txt
```

`requirements.in` lists the top-level packages; `requirements.txt` is the locked environment.

## Data

| Data | Source | Access |
|---|---|---|
| CNEMC-based model tables (`dataset/train_set.csv`, `test_set.csv`) | this study | [Google Drive](https://drive.google.com/drive/folders/12ZRGJg0IK3h3j9HbxZFV3h8cM_vTWjvU?usp=sharing) / Zenodo (DOI below) |
| CHAP 1-km daily PM₂.₅ V4, PM₁₀ V4, O₃ V3 | Wei et al. | Zenodo, open |
| CHAP 1-km daily NO₂, SO₂, CO V2 (2019–) | Wei et al. | Zenodo, **restricted** — request access from the providers |
| CHAP 10-km daily NO₂, SO₂, CO V1 (2013–2018, placebo years) | Wei et al. | Zenodo, open |
| ERA5 single levels | Copernicus CDS | free account; `~/.cdsapirc` |
| MCD19A2 AOD, SRTM | NASA / Google Earth Engine | free account |

`pipeline/chap.py` downloads and grids CHAP (record IDs in the script); `pipeline/README.md` gives the full pipeline run order.

## Reproducing the paper

Run from the repository root, after the inputs above are in place.

```bash
# 1. CHAP and ERA5 inputs for the lockdown analysis
for p in pm2_5 pm10 so2 no2 o3 co; do for y in 2019 2020 2021; do
  python pipeline/chap.py --pollutant $p --year $y; done; done
for p in pm2_5 pm10 o3; do python pipeline/chap.py --pollutant $p --year 2018 --months 1,2,3 --clean; done
for p in no2 so2 co; do for y in 2013 2014 2015 2016 2017 2018; do
  python pipeline/chap.py --pollutant $p --year $y --res D10K --months 1,2,3; done; done
python pipeline/era5_windows.py

# 2. DiD tables (revised specification: pre window ends LNY-3, two-way clustered SE)
python analysis/significance.py --pre-gap 2 --cluster twoway --tag rev_gap2_twoway_meteo
python analysis/significance.py --pre-gap 2 --cluster twoway --no-meteo --tag rev_gap2_twoway_nometeo

# 3. Robustness and validation
python analysis/event_study.py
python analysis/robustness.py --pre-gap 2 --cluster twoway --tag rev
python analysis/placebo_years.py
python verify/chap_lockdown_validation.py
python analysis/descriptive.py

# 4. Figures and supplementary tables
python figures/revision_figs.py
python figures/regen_figs.py
python figures/fig1_stations.py
python analysis/supp_tables.py
```

The ML-LUR results (Fig. 2, Supp. Tables S3, S8 and S13) are produced by `python model/train.py` and `python model/compare.py`.

### Where each result comes from

| Manuscript item | Script | Output |
|---|---|---|
| Fig. 1 (stations and study regions) | `figures/fig1_stations.py` | `outputs/figures/fig1_stations.png` |
| Fig. 2, Supp. Fig. S7 (ML-LUR spatial validation) | `figures/validation.py` (after `model/train.py`) | `outputs/figures/fig3_validation.png`, `outputs/supplementary/FigS7_CO_SO2_validation.png` |
| Supp. Table S13 (ML-LUR skill), Supp. Table S3 (hyperparameters) | `model/train.py` | `outputs/models/metrics.csv`, `best_params.json` |
| Supp. Table S8 (model comparison) | `model/compare.py` | `outputs/models/` |
| Figs 3–4 (national maps), Supp. Fig. S10 (SO₂ by province) | `figures/regen_figs.py` | `outputs/figures/fig{5,6,7}_chap.png` |
| In-text raw changes and inter-annual means | `analysis/descriptive.py` | `outputs/analysis/descriptive_numbers.csv` |
| Table 1, Fig. 5, Supp. Table S9 (DiD with and without meteorology) | `analysis/significance.py` (rev tags), `figures/revision_figs.py` | `outputs/analysis/significance_table_rev_gap2_twoway_{nometeo,meteo}.csv`, `outputs/figures/did_forest_rev.png` |
| Fig. 6, Supp. Fig. S12, Supp. Table S4 (event study) | `analysis/event_study.py` | `outputs/figures/event_study_{nometeo,meteo}.png`, `outputs/analysis/event_study_{coefs,pretrend}.csv` |
| Supp. Table S5 (window length, placebo-window pre-trend test) | `analysis/robustness.py`, `analysis/significance.py` | `outputs/analysis/robustness_table_rev.csv` |
| Fig. 7, Supp. Table S10 (year-pair placebo) | `analysis/placebo_years.py`, `figures/revision_figs.py` | `outputs/analysis/placebo_years{,_summary}.csv`, `outputs/figures/placebo_years_no2.png` |
| Supp. Table S11 (CHAP vs CNEMC during the lockdown) | `verify/chap_lockdown_validation.py` | `outputs/verify/chap_lockdown_validation.csv` |
| Supp. Table S1 (outlier screening) | `analysis/outlier_report.py` | `outputs/analysis/clean_report.csv` |
| Supp. Table S2, S12 (windows, cells per region) | `analysis/windows.py`, `analysis/supp_tables.py` | — |
| Supp. Figs S1–S6 (event-time curves) | `figures/festival.py` | `outputs/figures/festival_<region>.png` |
| Supp. Fig. S9 (city time series) | `figures/fig4_cities.py` | `outputs/figures/fig4_cities.png` |
| Supp. Fig. S11 (2021 vs 2019) | `figures/maps.py`, `figures/reversion_composite.py` | `outputs/figures/reversion_composite.png` |
| LaTeX supplementary tables | `analysis/supp_tables.py` | `SciRep_Submission/revision/tables/*.tex` |

`analysis/did.py` contains a self-test (`python analysis/did.py`) that recovers a known effect from simulated data.

## Citation and archive

Archived version: Zenodo, DOI [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX).

## License

[MIT License](LICENSE).
