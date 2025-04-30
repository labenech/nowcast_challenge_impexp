# Functions
import pandas as pd
import pycountry
from .constants import eu_27_countries_iso3, eu_27_countries
import json
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, root_mean_squared_error

rmse = root_mean_squared_error
mae = mean_absolute_error


def mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error (MAPE)."""
    return mean_absolute_percentage_error(y_true, y_pred) * 100


def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error (SMAPE)."""
    return 2 * np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100


def melt_dataframe(data):
    """
    Transforms a DataFrame from wide to long format.

    Args:
        data (DataFrame): Original DataFrame containing date columns.

    Returns:
        DataFrame: A new DataFrame with columns 'geo', 'time', and 'value'.
    """
    non_date_columns = ["freq", "stk_flow", "unit", "partner", "indic", "geo"]
    melted_data = pd.melt(data,
                          id_vars=non_date_columns,
                          var_name="time",
                          value_name="value")
    melted_data['time'] = pd.to_datetime(melted_data['time']).dt.normalize()
    melted_data = melted_data[["geo", "time", "value"]]
    return melted_data


def create_lag_features(df_lag, lag, value="value", geo="geo"):
    """Create lagged features grouped by country."""
    df_lag[f"lag_{lag}"] = df_lag.groupby(geo)[value].shift(lag)
    return df_lag


def get_covid_stringency_index(date1, date2):
    """
    Retrieve and preprocess COVID-19 Stringency Index data.

    Source: https://ourworldindata.org/covid-deaths
    """
    csi = pd.read_csv("data/owid-covid-data.csv")
    csi = csi[csi['iso_code'].isin(eu_27_countries_iso3)]
    csi = csi[['iso_code', 'date', 'stringency_index']]
    csi['date'] = pd.to_datetime(csi['date'])
    csi.fillna(0, inplace=True)

    iso3_to_iso2 = {
        country.alpha_3: 'EL' if country.alpha_3 == 'GRC' else country.alpha_2
        for country in pycountry.countries
    }

    csi_monthly = csi.set_index('date').groupby('iso_code').resample('MS').mean().reset_index()
    csi_monthly['iso_code'] = csi_monthly['iso_code'].map(iso3_to_iso2)
    csi_monthly = csi_monthly.rename(columns={"iso_code": "unique_id", "date": "ds", "value": "y"})

    all_dates = pd.date_range(start=date1, end=date2, freq='MS')
    combine = pd.MultiIndex.from_product([eu_27_countries, all_dates], names=["unique_id", "ds"])
    all_dates_df = pd.DataFrame(index=combine).reset_index()

    csi_full = all_dates_df.merge(csi_monthly, on=["unique_id", "ds"], how='left')
    csi_full['stringency_index'] = csi_full['stringency_index'].fillna(0)
    csi_full.sort_values(by=["unique_id", "ds"], inplace=True)

    return csi_full


def evaluate_performance(y_true, model):
    """
    Evaluate model performance by calculating MAE, MAPE, RMSE, and SMAPE.

    Args:
        y_true (DataFrame): DataFrame containing true values in column 'y'.
        model (str): Name of the column with model predictions.

    Returns:
        DataFrame: Evaluation metrics.
    """
    evaluation = {model: {}}
    for metric in [mae, mape, rmse, smape]:
        evaluation[model][metric.__name__] = metric(y_true['y'].values, y_true[model].values)
    return pd.DataFrame(evaluation).T


def evaluate_rmse(y_true, model):
    """Calculate RMSE for a model."""
    return rmse(y_true['y'].values, y_true[model].values)


def filter_mean_variables(variables):
    """Keep only one of the 'mean_2' or 'mean_3' variables based on their occurrence."""
    mean_indices = [i for i, var in enumerate(variables) if var in ['mean_2', 'mean_3']]
    if len(mean_indices) > 1:
        keep_index = mean_indices[0]
        variables = [var for i, var in enumerate(variables) if not (i in mean_indices and i != keep_index)]
    return variables


def write2json(file_path, new_data):
    """
    Updates an existing JSON file or creates a new one with provided data.

    Args:
        file_path (str): Path to JSON file.
        new_data (dict): Data to be added or updated.
    """
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            json_data = json.load(file)
        json_data.update(new_data)
    else:
        json_data = new_data

    with open(file_path, 'w') as file:
        json.dump(json_data, file, indent=2, sort_keys=True)


def encode_cyclic_features(df, column, max_value):
    """Encode cyclic features using sine and cosine transformations."""
    df[f'{column}_sin'] = np.sin(2 * np.pi * df[column] / max_value)
    df[f'{column}_cos'] = np.cos(2 * np.pi * df[column] / max_value)
    return df


def normalize_importance(importance):
    """Normalize feature importance scores to [0, 1]."""
    if importance.max() == importance.min():
        return pd.Series(1, index=importance.index)
    return (importance - importance.min()) / (importance.max() - importance.min())


def standardize_selected_columns(df, columns_to_standardize):
    """Apply standardization only to specified columns."""
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    return df_scaled, scaler


def standardize_target_by_group(df, group_col, target_col):
    """
    Standardize the target variable within each group separately.

    Returns:
        DataFrame: Standardized DataFrame.
        dict: Dictionary of scalers per group.
    """
    df_standardized = pd.DataFrame()
    scalers = {}

    for group, group_df in df.groupby(group_col):
        scaler = StandardScaler()
        non_nan_mask = group_df[target_col].notna()
        group_df.loc[non_nan_mask, target_col] = scaler.fit_transform(
            group_df.loc[non_nan_mask, target_col].values.reshape(-1, 1)
        )
        scalers[group] = scaler
        df_standardized = pd.concat([df_standardized, group_df])

    return df_standardized, scalers


def inverse_standardize_target_by_group(df, group_col, target_col, scalers):
    """
    Revert standardized target variable back to original scale for each group.

    Returns:
        DataFrame: DataFrame with target variable in original scale.
    """
    df_original = pd.DataFrame()

    for group, group_df in df.groupby(group_col):
        scaler = scalers[group]
        group_df[target_col] = scaler.inverse_transform(group_df[target_col].values.reshape(-1, 1))
        df_original = pd.concat([df_original, group_df])

    return df_original
