import pandas as pd
import eurostat

import json

import xgboost as xgb

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, AutoCES
from mlforecast import MLForecast

from neuralforecast.core import NeuralForecast

import optuna

from neuralforecast.models import NHITS
from neuralforecast.models import TFT

from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_regression

from sklearn.metrics import mean_absolute_error, root_mean_squared_error

import torch

from extra.constants import *
from extra.functions import *

import os

# Ensure that the 'data' and 'data/results' directories exist
os.makedirs('data/results', exist_ok=True)

# Download input time series data from eurostat

code = "ei_eteu27_2020_m"
my_filter_pars = {'startPeriod': start_period_download,
                  'geo': eu_27_countries,
                  'freq': freq,
                  'indic': 'ET-T',
                  'stk_flow': ['EXP','IMP'],
                  'partner': ['EU27_2020','EXT_EU27_2020'],
                  'unit': 'MIO-EUR-NSA'}
data = eurostat.get_data_df(code, filter_pars=my_filter_pars)
# Rename bad named column
print("Eurostat series downloaded")
data = data.rename(columns={"geo\\TIME_PERIOD": "geo"})

# Apply melt_dataframe function to our 3 series to get them whith correct format and data
intra_eu_exports = melt_dataframe(data[(data.partner == "EU27_2020") & (data.indic == "ET-T") & (data.stk_flow == "EXP") & (data.unit == "MIO-EUR-NSA")])
extra_eu_exports = melt_dataframe(data[(data.partner == "EXT_EU27_2020") & (data.indic == "ET-T") & (data.stk_flow == "EXP") & (data.unit == "MIO-EUR-NSA")])
extra_eu_imports = melt_dataframe(data[(data.partner == "EXT_EU27_2020") & (data.indic == "ET-T") & (data.stk_flow == "IMP") & (data.unit == "MIO-EUR-NSA")])

# Create a dictionary with the 3 DataFrames
df_dict = {
    "intra_eu_exports": intra_eu_exports,
    "extra_eu_exports": extra_eu_exports,
    "extra_eu_imports": extra_eu_imports
}

scalers_dict = {}  # New dict to store scaled data for each of the 3 DataFrames

for key in df_dict:
    df_ini = df_dict[key].copy()

    end_period  = df_ini.time.max().date()

    # List of all uniq dates to be included in the DataFrames
    toutes_les_dates = pd.date_range(start=start_period_download_date, end=date_2_pred, freq="MS")
    # Generate all combination countries/dates
    combinaisons = pd.MultiIndex.from_product([eu_27_countries, toutes_les_dates], names=["geo", "time"])
    # Empty dataframe with index of all combinations
    df_vide = pd.DataFrame(index=combinaisons).reset_index()
    # Melt empty dataframe with data
    df_ini = df_vide.merge(df_ini, on=["geo", "time"], how="left")

    df, scalers = standardize_target_by_group(df_ini, group_col='geo', target_col='value')
    scalers_dict[key] = scalers  # Store scaled data for  this dataframe

    # Create all lags
    for i in range(nb_lags):
        create_lag_features(df, lag=i+1)

    # keep only since start_period to drop Nan created by lag
    df = df.loc[df.time >= start_period]

    df.sort_values(by=["geo", "time"], inplace=True)

    # Rename columns to match the Nixtlaverse's expectations
    # The 'geo' becomes 'unique_id' representing the unique identifier of the time series
    # The 'time' becomes 'ds' representing the time stamp of the data points
    # The 'value' becomes 'y' representing the target variable we want to forecast
    df = df.rename(columns={
        "geo": "unique_id",
        "time": "ds",
        "value": "y"
    })

    df = df.assign(month=df.ds.dt.month).assign(week_of_year=df.ds.dt.isocalendar().week).assign(quarter=df.ds.dt.quarter).assign(year=df.ds.dt.year)

    # Standardize year
    df['year'] = (df['year'] - df['year'].mean()) / df['year'].std()

    # Standardize "cyclic" other elements of date
    df = encode_cyclic_features(df, 'month', 12)  # Mois (1 à 12)
    df = encode_cyclic_features(df, 'week_of_year', 52)  # Semaines (1 à 52)
    df = encode_cyclic_features(df, 'quarter', 4)  # Trimestres (1 à 4)

    covid = get_covid_stringency_index(start_period,date_2_pred)
    df = df.merge(covid[['ds', 'unique_id', 'stringency_index']], on=['ds', 'unique_id'], how='left')
    df['stringency_index'] = df['stringency_index'] / 100

    df['stringency_index_lag3'] = df.groupby('unique_id')['stringency_index'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['stringency_index_lag6'] = df.groupby('unique_id')['stringency_index'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)

    df['mean_2'] = df.groupby('unique_id')['lag_1'].rolling(window=2, min_periods=1).mean().reset_index(0, drop=True)
    df['mean_3'] = df.groupby('unique_id')['lag_1'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
    df['mean_6'] = df.groupby('unique_id')['lag_1'].rolling(window=6, min_periods=1).mean().reset_index(0, drop=True)
    df['mean_12'] = df.groupby('unique_id')['lag_1'].rolling(window=12, min_periods=1).mean().reset_index(0, drop=True)

    df = df.drop('lag_1', axis=1)

    df_dict[key] = df

print("feature engineering ended")

for indicator in df_dict.keys():
    print("### Computing future data for " + indicator)

    file = 'data/results/'+date_2_pred+'_'+indicator+'.json'

    # load dataframe
    df = df_dict[indicator].copy()

    # Delete lines with 'y' is NaN 
    df = df.dropna(subset=['y'])

    # dict to store best features by country
    top_features_per_country = {}

    # Function to select best feature for a country
    def select_best_features_for_country(df_country, top_k=10, importance_threshold=0.05):
        # Select exogen data columns
        exog_columns = [col for col in df_country.columns if col not in ['unique_id', 'ds', 'y']]
        X = df_country[exog_columns]
        y = df_country['y']

        if len(X) < 10:  # Quit if not enough data to compute
            return None

        # 1. LassoCV 
        pipeline = Pipeline([
            ('lasso', LassoCV(cv=5, max_iter=6000, random_state=42))
        ])
        pipeline.fit(X, y)
        lasso_coef = pipeline.named_steps['lasso'].coef_
        lasso_importance = pd.Series(lasso_coef, index=exog_columns).abs()

        # 2. Random Forest 
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        rf.fit(X, y)
        rf_importance = pd.Series(rf.feature_importances_, index=exog_columns)

        # 3. Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = pd.Series(mi_scores, index=exog_columns)

        # 4. Scale all data importances dans compute mean to combined_importance
        lasso_importance_norm = normalize_importance(lasso_importance)
        rf_importance_norm = normalize_importance(rf_importance)
        mi_importance_norm = normalize_importance(mi_importance)

        combined_importance = (lasso_importance_norm + rf_importance_norm + mi_importance_norm) / 3

        # 5. Sort importance descending
        combined_importance_sorted = combined_importance.sort_values(ascending=False)

        # 6. Select best exogen data with predefined threshold importance 
        if importance_threshold:
            top_features = combined_importance_sorted[combined_importance_sorted > importance_threshold].index.tolist()
        else:
            top_features = combined_importance_sorted.head(top_k).index.tolist()

        # If we choose a max number of features, keep only this number of exog columns
        if len(top_features) > top_k:
            top_features = top_features[:top_k]

        return top_features

    # Apply function for each country
    for country_id, df_country in df.groupby('unique_id'):
        print(f"Computing best features for country : {country_id}")
        top_features = select_best_features_for_country(df_country, top_k=8, importance_threshold=0.3)
        top_features_per_country[country_id] = filter_mean_variables(top_features)

    device = "cpu"  # Compute only with CPU, no GPU needed for these models
    torch.set_float32_matmul_precision('medium')  # 'medium' or 'high' for slightly lower precision but higher performance

    def predict_xgboost(train, futur, best_params):

        best_params["objective"] = "reg:squarederror"
        best_params["n_estimators"] = 3000
        best_params["booster"] = "gblinear"
        best_params["device"] = device

        model = MLForecast(models=[xgb.XGBRegressor(**best_params)], freq='MS')
        model.fit(df=train, static_features=[])
        h = len(futur)

        return model.predict(h=h, X_df=futur.drop(columns=['y']))

    # ------------------------------------
    # Entry_1 = xgboost
    # ------------------------------------
    print("Compute entry_1")

    def objective_xgboost(trial, train):
        params = {
            "device": device,
            "objective": "reg:squarederror",
            "n_estimators": 3000,
            "verbosity": 0,
            "booster": "gblinear",
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.05, log=True),
        }

        models = [xgb.XGBRegressor(**params)]

        model = MLForecast(models=models, freq='MS')

        # Cross-validation
        crossvalidation_df = model.cross_validation(df=train, n_windows=7, static_features=[], h=2)
        rmse = root_mean_squared_error(crossvalidation_df['y'], crossvalidation_df['XGBRegressor'])

        return rmse

    def compute_entry_1(df):

        Y_df_pred = pd.DataFrame()

        for id, df_id in df.groupby('unique_id'):
            print(f"Training for {id}")
            train = df_id.loc[df_id.ds < date_train][['unique_id', 'ds', 'y'] + top_features_per_country[id]]
            study = optuna.create_study(direction='minimize')
            study.optimize(lambda trial: objective_xgboost(trial, train), n_trials=30)

            train = df_id.loc[df_id.ds <= pd.to_datetime(end_period)][['unique_id', 'ds', 'y'] + top_features_per_country[id]]
            futur = df_id.loc[df_id.ds > pd.to_datetime(end_period)][['unique_id', 'ds', 'y'] + top_features_per_country[id]]

            pred = predict_xgboost(train, futur, study.best_params).reset_index(drop=True).merge(futur[['ds', 'unique_id', 'y']], on=['ds', 'unique_id'], how='left')

            Y_df_pred = pd.concat([Y_df_pred, pred.reset_index()])

        return Y_df_pred

    df = df_dict[indicator].copy()
    scalers = scalers_dict[indicator]

    resultat_1 = compute_entry_1(df)

    result = inverse_standardize_target_by_group(resultat_1.loc[resultat_1.ds==date_2_pred], group_col='unique_id', target_col='XGBRegressor', scalers=scalers)

    result.loc[:, 'XGBRegressor'] = result['XGBRegressor'].astype(float).round(1)
    # Create a dict with "entry_1" with key-values "unique_id" and "XGBRegressor"
    result_dict = {"entry_1": dict(zip(result['unique_id'], result['XGBRegressor']))}

    # Save entry_1 in the JSON file
    write2json(file, result_dict)
    print("The entry 'entry_1' was successfully added.")

    # ------------------------------------
    # Entry_2 = best for each country among arima, ets, theta and ces
    # ------------------------------------
    print("Compute entry_2")
    # Split data into train and test sets
    df = df_dict[indicator].copy()
    scalers = scalers_dict[indicator]

    Y_df_train = df.loc[df.ds < date_train]
    Y_df_test = df.loc[(df.ds >= date_train) & (df.ds <= pd.to_datetime(end_period))]

    season_length = 12  # Monthly data

    # Define a list of models for forecasting
    models = [
        AutoARIMA(season_length=season_length), # ARIMA model with automatic order selection and seasonal component
        AutoETS(season_length=season_length), # ETS model with automatic error, trend, and seasonal component
        AutoTheta(season_length=season_length), # Theta model with automatic seasonality detection
        AutoCES(season_length=season_length), # CES model with automatic seasonality detection
    ]

    Y_df_pred = pd.DataFrame()

    for id, df_id in df.groupby('unique_id'):
        print(f"Training for {id}")
        train = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[df_id.ds <date_train]
        test = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[(df_id.ds >= date_train) & (df_id.ds <= pd.to_datetime(end_period))]

        h = len(test)

        sf = StatsForecast(models=models, freq='MS', n_jobs=-1)

        sf.fit(df=train)
        pred = sf.predict(h=h, X_df=test.drop(columns=['y']))

        model_rmse = {}
        for model in ['AutoARIMA', 'AutoETS', 'AutoTheta', 'CES']:
            model_rmse[f"{model}_RMSE"] = rmse(test['y'].values, pred[model].values)

        # Find the model with best RMSE
        best_model = min(model_rmse, key=model_rmse.get).replace('_RMSE', '')
        model_rmse['unique_id'] = id
        model_rmse['best_model'] = best_model

        train = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[(df_id.ds >= date_train) & (df_id.ds <= pd.to_datetime(end_period))]
        test = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[(df_id.ds > pd.to_datetime(end_period))]

        sf.fit(df=train)
        final_pred = sf.predict(h=2, X_df=test.drop(columns=['y']))

        Y_df_pred = pd.concat([Y_df_pred, final_pred[['unique_id','ds',best_model]].rename(columns={best_model: 'y_hat'}).reset_index()])

        result = inverse_standardize_target_by_group(Y_df_pred.loc[Y_df_pred.ds==date_2_pred], group_col='unique_id', target_col='y_hat', scalers=scalers)

    result.loc[:, 'y_hat'] = result['y_hat'].astype(float).round(1)
    # Create a dict for "entry_2" with key values  = "unique_id" and "y_hat"
    result_dict = {"entry_2": dict(zip(result['unique_id'], result['y_hat']))}

    # Save to the JSON file
    write2json(file, result_dict)
    print("The entry 'entry_2' was successfully added.")


    # -----------------------------------
    # Now  we use GPU if any  available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('medium')  # 'medium' or 'high' for slightly lower precision but higher performance
    # ------------------------------------

    # ------------------------------------
    # Entry_3 = NHITS
    # ------------------------------------
    print("Compute entry_3")

    def predict_NHITS(train, futur, best_params):

        h = len(futur)
        best_params["batch_size"] = 7
        best_params["windows_batch_size"] = 256
        best_params["activation"] = 'ReLU'
        best_params["n_blocks"] = [1, 1, 1]
        best_params["h"] = h
        best_params["max_steps"] = max_steps_default

        model = NeuralForecast(models=[NHITS(**best_params)], freq='MS')

        model.fit(df=train)

        return model.predict(futr_df=futur.drop(columns=['y']))

    def objective_NHITS(trial):

        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.001, log=True)
        max_steps = max_steps_default  #trial.suggest_int("max_steps", 1500, 3000, step=500)
        input_size = trial.suggest_int("input_size", h, 2 * h, step=h)  
        batch_size = 7
        windows_batch_size = 256
        n_pool_kernel_size = trial.suggest_categorical("n_pool_kernel_size", [[2, 2, 2], [16, 8, 1]])
        n_freq_downsample = trial.suggest_categorical("n_freq_downsample", [[168, 24, 1], [24, 12, 1], [1, 1, 1]])
        activation = 'ReLU' # trial.suggest_categorical("activation", ['ReLU', 'LeakyReLU', 'Tanh'])
        n_blocks = [1, 1, 1]
        mlp_units = trial.suggest_categorical("mlp_units", [[[128, 128], [128, 128], [128, 128]], [[256, 256], [256, 256], [256, 256]]])
        interpolation_mode = trial.suggest_categorical("interpolation_mode", ['linear', 'nearest'])
        val_check_steps = trial.suggest_int("val_check_steps", 100, 200, step=50)
        random_seed = trial.suggest_int("random_seed", 1, 10)

        # Set model with hyper parameters found
        models = [NHITS(h=h,
                  learning_rate=learning_rate,
                  max_steps=max_steps,
                  input_size=input_size,
                  batch_size=batch_size,
                  windows_batch_size=windows_batch_size,
                  n_pool_kernel_size=n_pool_kernel_size,
                  n_freq_downsample=n_freq_downsample,
                  activation=activation,
                  n_blocks=n_blocks,
                  mlp_units=mlp_units,
                  interpolation_mode=interpolation_mode,
                  val_check_steps=val_check_steps,
                  random_seed=random_seed,
                        )]

        model = NeuralForecast(models=models, freq='MS')
        model.fit(train)

        p = model.predict(futr_df=test).reset_index()
        p = p.merge(test[['ds', 'unique_id', 'y']], on=['ds', 'unique_id'], how='left')

        loss = mean_absolute_error(p['y'], p['NHITS'])

        return loss

    # compute entry_3
    df = df_dict[indicator].copy()
    scalers = scalers_dict[indicator]

    Y_df_pred = pd.DataFrame()

    for id, df_id in df.groupby('unique_id'):
        print(f"Training for {id}")
        train = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[df_id.ds <date_train]
        test = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[(df_id.ds >= date_train) & (df_id.ds <= pd.to_datetime(end_period))]

        h = len(test)

        study = optuna.create_study(direction='minimize')
        study.optimize(objective_NHITS, n_trials=1)  # 25)

        train = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[df_id.ds <= pd.to_datetime(end_period)]
        futur = df_id[['unique_id', 'ds', 'y']+top_features_per_country[id]].loc[df_id.ds > pd.to_datetime(end_period)]

        pred = predict_NHITS(train, futur, study.best_params).reset_index(drop=True).merge(futur[['ds', 'unique_id', 'y']], on=['ds', 'unique_id'], how='left')
        print(id)

        Y_df_pred = pd.concat([Y_df_pred, pred.reset_index()])

    result = inverse_standardize_target_by_group(Y_df_pred.loc[Y_df_pred.ds == date_2_pred], group_col='unique_id', target_col='NHITS', scalers=scalers)

    result.loc[:,'NHITS'] = result['NHITS'].astype(float).round(1)
    # Create a dict for "entry_3" with key values  = "unique_id" and "NHITS"
    result_dict = {"entry_3": dict(zip(result['unique_id'], result['NHITS']))}

    # Save to JSON file
    write2json(file, result_dict)
    print("The entry 'entry_3' was successfully added.")

    # ------------------------------------
    # Entry_4 = TFT
    # ------------------------------------
    print("Compute entry_4")

    def predict_TFT(train, futur, best_params):
        """
        Train TFT model with best parameters and predict results
        """
        h = len(futur)
        best_params["h"] = h
        best_params["batch_size"] = 7
        best_params["max_steps"] = max_steps_default

        # Init model
        model = NeuralForecast(models=[TFT(**best_params)], freq='MS')
        model.fit(df=train)

        # Predict futur 
        return model.predict(futr_df=futur.drop(columns=['y']))

    def objective_TFT(trial):
        """
        Objectiv function to select best parameters for our TFT.
        """
        # Hyperparameters to optimize 
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 0.001, log=True)
        max_steps = max_steps_default
        input_size = trial.suggest_int("input_size", h, 2 * h, step=h)  
        dropout = trial.suggest_float("dropout", 0.1, 0.5, step=0.1)
        n_head = trial.suggest_int("n_head", 1, 8)  # Number of attention heads
        # `hidden_size` must be multiple of `n_head`
        hidden_size = trial.suggest_int("hidden_size", n_head * 2, n_head * 16, step=n_head)
        random_seed = trial.suggest_int("random_seed", 1, 10)
        val_check_steps = trial.suggest_int("val_check_steps", 100, 200, step=50)

        # Set TFT model with hyperparameters found
        model = NeuralForecast(models=[
            TFT(
                h=h,
                input_size=input_size,
                learning_rate=learning_rate,
                max_steps=max_steps,
                hidden_size=hidden_size,
                dropout=dropout,
                batch_size=7,
                n_head=n_head,
                val_check_steps=val_check_steps,
                random_seed=random_seed
            )
        ], freq='MS')

        # train model
        model.fit(train)

        # Predictions and loss 
        p = model.predict(futr_df=test).reset_index()
        p = p.merge(test[['ds', 'unique_id', 'y']], on=['ds', 'unique_id'], how='left')

        # loss= Mean Absolute Error
        loss = mean_absolute_error(p['y'], p['TFT'])

        return loss

    # entry_4
    df = df_dict[indicator].copy()
    scalers = scalers_dict[indicator]

    Y_df_pred = pd.DataFrame()

    for id, df_id in df.groupby('unique_id'):
        print(f"Training for {id}")
        train = df_id[['unique_id', 'ds', 'y'] + top_features_per_country[id]].loc[df_id.ds < date_train]
        test = df_id[['unique_id', 'ds', 'y'] + top_features_per_country[id]].loc[(df_id.ds >= date_train) & (df_id.ds <= pd.to_datetime(end_period))]
        h = len(test)

        # Step 1 : Optimise parameters  with Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective_TFT(trial), n_trials=1)  # 15)

        # Step 2 : train with best parameters and best features 
        train = df_id[['unique_id', 'ds', 'y'] + top_features_per_country[id]].loc[df_id.ds <= pd.to_datetime(end_period)]
        futur = df_id[['unique_id', 'ds', 'y'] + top_features_per_country[id]].loc[df_id.ds > pd.to_datetime(end_period)]

        pred = predict_TFT(train, futur, study.best_params).reset_index(drop=True).merge(futur[['ds', 'unique_id', 'y']], on=['ds', 'unique_id'], how='left')

        Y_df_pred = pd.concat([Y_df_pred, pred.reset_index()])

    # Invert data normalization 
    result = inverse_standardize_target_by_group(
        Y_df_pred.loc[Y_df_pred.ds == date_2_pred],
        group_col='unique_id',
        target_col='TFT',
        scalers=scalers
    )

    result['TFT'] = result['TFT'].astype(float).round(1)

    # Create a dict for "entry_4" with key values  = "unique_id" and "TFT"
    result_dict = {"entry_4": dict(zip(result['unique_id'], result['TFT']))}

    # Save to JSON file
    write2json(file, result_dict)
    print("The entry 'entry_4' was successfully added.")

    # ------------------------------------
    # Entry_5 = Mean of entry_1 to entry_4
    # ------------------------------------
    # Load the JSON file
    with open(file, 'r') as f:
        data = json.load(f)

    # Calculate averages for each country and create "entry_5"
    entry_5 = {}
    for country in data.get("entry_1", {}):  # Check each country in "entry_1"
        # Collect values from entry_1 to entry_4 if they exist for this country
        values = []
        for i in range(1, 5):  # Loop through entries from entry_1 to entry_4
            entry_key = f"entry_{i}"
            if entry_key in data and country in data[entry_key]:
                values.append(data[entry_key][country])
        
        # If at least one value exists, compute the average and add it to "entry_5"
        if values:
            entry_5[country] = round(sum(values) / len(values), 1)

    # Add "entry_5" to the dictionary
    data["entry_5"] = entry_5

    # Save the updated JSON file
    write2json(file, data)

    print("The entry 'entry_5' was successfully added.")



print("### The End. Files are in data/results !")
