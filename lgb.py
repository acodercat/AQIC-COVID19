import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import lightgbm as lgb

FEATURE_COLUMNS = ['lat', 'lon',
                        'TMP_P0_L1_GLL0', 'SPFH_P0_2L108_GLL0', 'RH_P0_L4_GLL0',
                        'PWAT_P0_L200_GLL0', 'UGRD_P0_L6_GLL0', 'GUST_P0_L1_GLL0',
                        'PRES_P0_L7_GLL0', 'CultivatedLand', 'WoodLand', 'GrassLand', 'Waters',
                        'UrbanRural', 'UnusedLand', 'Ocean', 'ELEVATION', 'AOD', 'month',
                        'year', 'weekday']
target_column = 'pm2_5'

train_set = pd.read_csv('./dataset/train_set.csv') 
test_set = pd.read_csv('./dataset/test_set.csv')
X_train, y_train, X_test, y_test = train_set[FEATURE_COLUMNS], train_set[target_column], test_set[FEATURE_COLUMNS], test_set[target_column]


params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    # 'device': 'gpu',  # Enable GPU training
    # 'gpu_platform_id': 0,  # Use the first GPU
    # 'gpu_device_id': 0,
    'num_leaves': 1500,
    'max_depth': 20,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1
}


train_data = lgb.Dataset(X_train, label=y_train)

# Train the model
num_rounds = 800
lgb_model = lgb.train(params, train_data, num_rounds)


y_pred = lgb_model.predict(X_test)

results_df = pd.DataFrame({
    'y_test': y_test,
    'predictions': y_pred
})

results_df.to_csv(f'{target_column}_results.csv', index=False)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
smape = np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test))) * 100
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)



print("RMSE:", rmse)
print("SMAPE:", smape)
print("MAE:", mae)
print("R2:", r2)