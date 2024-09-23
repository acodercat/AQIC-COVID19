# Multiscale Analysis of the Impact of Reduced Socioeconomic Activities on Air Quality in China

## Overview

This repository contains the code for our study on the impact of COVID-19 lockdowns on air quality in China. Our research leverages machine learning techniques, specifically LightGBM, to analyze changes in air pollution levels and their implications for public health and air quality management strategies.

## Methodology

Our study employs the following methods:

1. Data collection and preprocessing
2. Feature engineering
3. Model development using LightGBM
4. Model evaluation and analysis


## LightGBM Model Configuration

The LightGBM model is configured with the following parameters:

```python
params = {
    'boosting_type': 'gbdt',
    # Boosting type: Gradient Boosting Decision Tree

    'objective': 'regression',
    # Objective function: Regression

    'metric': 'rmse',
    # Evaluation metric: Root Mean Square Error

    'num_leaves': 1500,
    # Maximum number of leaves in one tree

    'max_depth': 20,
    # Maximum depth of the tree

    'learning_rate': 0.05,
    # Learning rate: Smaller values may improve generalization but require more iterations

    'feature_fraction': 0.9,
    # Randomly select 90% of features in each iteration to build trees

    'bagging_fraction': 0.8,
    # Randomly select 80% of data in each iteration to build trees

    'bagging_freq': 5,
    # Perform bagging every 5 iterations
}
```

## Dataset

Due to size constraints, the dataset for this project is hosted on Google Drive. You can access it using the following link:

[Air Quality Analysis Dataset](https://drive.google.com/drive/folders/12ZRGJg0IK3h3j9HbxZFV3h8cM_vTWjvU?usp=sharing)

Please download the dataset and place it in the `dataset` folder of this project before running the analysis scripts.

## Dataset Description

The dataset used in this project consists of several CSV files containing various types of data related to air quality in China. Here's a detailed breakdown of the data files:

### Dataset Folder

1. `grid_static_features.csv`
   - Contains static features for each grid cell in the study area
   - Columns include:
     - `grid_id`: Unique identifier for each grid cell
     - `lat`, `lon`: Latitude and longitude coordinates of the grid cell
     - Land use categories (in square meters):
       - `CultivatedLand`
       - `WoodLand`
       - `GrassLand`
       - `Waters`
       - `UrbanRural`
       - `UnusedLand`
       - `Ocean`
     - `ELEVATION`: Elevation of the grid cell (presumably in meters)
     - `AOD`: Aerosol Optical Depth, a measure of air pollution
     - `province`: The province in which the grid cell is located
   - This file provides essential geographical and environmental context for each location in the study

2. `national_AQ_stations.csv`
   - List of national air quality monitoring stations
   - Likely includes station IDs, locations, and possibly other metadata

3. `test_set.csv`
   - Test dataset used for evaluating the LightGBM model
   - Contains features and target variables for model testing

4. `train_set.csv`
   - Training dataset used for training the LightGBM model
   - Contains features and target variables for model training

### Result Files

The following CSV files contain the results of our air quality predictions for different pollutants:

1. `no2_results.csv`
   - Results for Nitrogen Dioxide (NO2) predictions

2. `o3_results.csv`
   - Results for Ozone (O3) predictions

3. `pm10_results.csv`
   - Results for Particulate Matter ≤ 10µm (PM10) predictions

4. `pm2_5_results.csv`
   - Results for Particulate Matter ≤ 2.5µm (PM2.5) predictions

### Other Files

- `air_quality_lgb_model.ipynb`
  - Jupyter notebook containing the main analysis and model implementation

- `lgb.py`
  - Python script with LightGBM model implementation and related functions

- `requirements.txt`
  - List of Python dependencies required to run the project

This dataset structure allows for a comprehensive analysis of air quality in China, incorporating various spatial, environmental, and air quality features. The grid-based approach enables high-resolution mapping and analysis of air quality across different regions of China. The separation of training and test sets enables proper model evaluation, while the result files provide insights into the model's predictions for different air pollutants.

## Dependencies

- Python 3.7+
- pandas
- numpy
- scikit-learn
- lightgbm

For a complete list of dependencies, see `requirements.txt`.

## Contributing

We welcome contributions to this project. Please feel free to submit issues or pull requests.

## License

This project is licensed under the [MIT License](LICENSE).
