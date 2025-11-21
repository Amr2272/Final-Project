from preprocessing import full_preprocessing_pipeline
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.stattools import adfuller
import pickle
from Monitoring import (
    integrate_monitoring_with_prophet,
    continuous_monitoring_workflow
)
import matplotlib.pyplot as plt

# ======================================================
# LOAD PROCESSED DATA
# ======================================================
def load_and_preprocess(data_path="."):
    """
    This function calls the full preprocessing pipeline
    and returns clean train and test data.
    """
    train_df, test_df = full_preprocessing_pipeline(data_path)
    return train_df, test_df


# ======================================================
# STEP 0 — Convert df_final to Daily Time Series
# ======================================================
def build_daily_series(df_final):
    """
    Converts df_final (store-level rows) into a daily aggregated
    time-series suitable for Prophet.
    """
    ts = df_final.copy()

    # Aggregate total sales per day
    daily = ts['sales'].resample('D').sum().to_frame()

    return daily



# ======================================================
# STEP 1 — RESAMPLING + CHECK STATIONARITY
# ======================================================
def prepare_time_series(train_df):
    df = train_df["sales"]
    df = df.resample("D").sum().asfreq("D")

    result = adfuller(df.sales)
    print(f"p-value: {result[1]}")

    if result[1] > 0.05:
        print("The time series is not stationary")
    else:
        print("The time series is stationary")

    return df


# ======================================================
# STEP 2 — TRAIN/TEST SPLIT FOR PROPHET
# ======================================================
def split_for_prophet(df):
    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]

    train = train.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
    test = test.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})

    train['ds'] = pd.to_datetime(train['ds'])
    test['ds'] = pd.to_datetime(test['ds'])

    return train, test


# ======================================================
# STEP 3 — HOLIDAYS
# ======================================================
def build_holidays(train):
    years = train['ds'].dt.year.unique()
    holiday_dates = [f"{y}-12-25" for y in years]

    holidays = pd.DataFrame({
        'holiday': 'christmas_25_dec',
        'ds': pd.to_datetime(holiday_dates),
        'lower_window': 0,
        'upper_window': 0
    })
    return holidays


# ======================================================
# STEP 4 — BASE PROPHET MODEL
# ======================================================
def train_prophet_model(train, holidays):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        holidays=holidays
    )
    model.fit(train)
    return model


# ======================================================
# STEP 5 — FORECAST + PLOT
# ======================================================
def forecast_and_plot(model, test):
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)

    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

    model.plot(forecast)
    plt.show()

    model.plot_components(forecast)
    plt.show()

    return forecast


# ======================================================
# STEP 6 — EVALUATION METRICS
# ======================================================
def calculate_metrics(test, forecast):
    y_pred = forecast['yhat'].iloc[-len(test):].values
    y_true = test['y'].values

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    rmse_pct = (rmse / np.mean(y_true)) * 100

    print(f"MAE (original scale): {mae:.2f}")
    print(f"RMSE (original scale): {rmse:.2f}")
    print(f"RMSE (% of mean): {rmse_pct:.2f}%")

    return mae, rmse, mape


# ======================================================
# STEP 7 — PARAMETER GRID SEARCH
# ======================================================
def tune_prophet(train, test, holidays):
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.1, 1.0, 10.0]
    }

    best_rmse = float('inf')
    best_params = None

    for params in ParameterGrid(param_grid):
        print(f"Training with parameters: {params}")

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=True,
            seasonality_mode='additive',
            holidays=holidays,
            **params
        )

        model.fit(train)

        future = model.make_future_dataframe(periods=len(test))
        forecast = model.predict(future)

        y_true = test['y'].values
        y_pred = forecast['yhat'].iloc[-len(test):].values

        rmse = np.sqrt(np.mean((y_true - y_pred)**2))

        print(f"RMSE: {rmse:.2f}")

        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params

    print(f"\nBest RMSE: {best_rmse:.2f}")
    print(f"Best parameters: {best_params}")

    return best_params


# ======================================================
# STEP 8 — TRAIN FINAL MODEL WITH BEST PARAMS
# ======================================================
def train_best_model(train, holidays, best_params):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=True,
        seasonality_mode='additive',
        holidays=holidays,
        **best_params
    )
    model.fit(train)
    return model


# ======================================================
# STEP 9 — MONITORING
# ======================================================
def run_monitoring(train, test, model):
    forecast = model.predict(test[['ds']])
    forecast = forecast[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')

    monitor = integrate_monitoring_with_prophet(train, test, model, forecast)

    health = monitor.check_model_health()
    print("Model Health:", health)

    return monitor


def monitoring_update(monitor, test, model, new_date="2025-01-01"):
    new_date = pd.to_datetime(new_date)
    new_predicted = float(model.predict(pd.DataFrame({'ds': [new_date]}))['yhat'].iloc[0])
    new_actual = float(test['y'].iloc[-1])

    status = continuous_monitoring_workflow(
        monitor=monitor,
        new_actual=new_actual,
        new_predicted=new_predicted,
        new_date=new_date
    )

    print("Monitoring Status:", status)
    return status


# ======================================================
# STEP 10 — SAVE TRAINED MODEL
# ======================================================
def save_model(model, filename="prophet_model.pkl"):
    """
    Saves the trained Prophet model using pickle.
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {filename}")
