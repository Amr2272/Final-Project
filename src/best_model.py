from preprocessing import full_preprocessing_pipeline
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.model_selection import ParameterGrid
from statsmodels.tsa.stattools import adfuller
import pickle
from Monitoring import (
    integrate_monitoring_with_prophet,
    continuous_monitoring_workflow)

import matplotlib.pyplot as plt

# ======================================================
# LOAD PROCESSED DATA
# ======================================================

def load_and_preprocess(data_path="."):
    """
    This function calls the full preprocessing pipeline
    and returns clean train and test data.
    
    Returns:
    --------
    train_cleaned : DataFrame with original structure (date, sales, features)
    train_encoded : Scaled/encoded features with date index
    test_encoded : Scaled/encoded test features with date index
    """
    train_cleaned, train_encoded, test_encoded = full_preprocessing_pipeline(data_path)
    
    return train_cleaned, train_encoded, test_encoded


# ======================================================
# STEP 0 – Convert to Daily Time Series for Prophet
# ======================================================
def build_daily_series(train_cleaned):
    """
    Converts train_cleaned to a daily aggregated time-series suitable for Prophet.
    
    Parameters:
    -----------
    train_cleaned : DataFrame with 'date' and 'sales' columns
    
    Returns:
    --------
    daily : DataFrame with aggregated daily sales
    """
    # Set date as index if it isn't already
    if 'date' in train_cleaned.columns:
        ts = train_cleaned.set_index('date').copy()
    else:
        ts = train_cleaned.copy()
    
    # Aggregate total sales per day
    daily = ts['sales'].resample('D').sum().to_frame()
    
    return daily


# ======================================================
# STEP 1 – RESAMPLING + CHECK STATIONARITY
# ======================================================
def prepare_time_series(train_cleaned):
    """
    Prepare time series data and check for stationarity.
    
    Parameters:
    -----------
    train_cleaned : DataFrame with 'date' and 'sales' columns
    
    Returns:
    --------
    df : Daily resampled sales data
    """
    # Ensure date is the index
    if 'date' in train_cleaned.columns:
        df = train_cleaned.set_index('date').copy()
    else:
        df = train_cleaned.copy()
    
    # Resample to daily frequency and sum sales
    df = df['sales'].resample('D').sum().to_frame()
    
    # Check stationarity
    result = adfuller(df['sales'].dropna())
    print(f"ADF Test p-value: {result[1]:.4f}")
    
    if result[1] > 0.05:
        print("⚠️  The time series is NOT stationary")
    else:
        print("✓ The time series is stationary")
    
    return df


# ======================================================
# STEP 2 – TRAIN/TEST SPLIT FOR PROPHET
# ======================================================
def split_for_prophet(df, train_ratio=0.8):
    """
    Split data for Prophet training and testing.
    
    Parameters:
    -----------
    df : DataFrame with date index and 'sales' column
    train_ratio : float, proportion of data for training
    
    Returns:
    --------
    train : DataFrame with 'ds' and 'y' columns (Prophet format)
    test : DataFrame with 'ds' and 'y' columns (Prophet format)
    """
    train_size = int(len(df) * train_ratio)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    
    # Convert to Prophet format
    train = train.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
    test = test.reset_index().rename(columns={'date': 'ds', 'sales': 'y'})
    
    # Ensure datetime format
    train['ds'] = pd.to_datetime(train['ds'])
    test['ds'] = pd.to_datetime(test['ds'])
    
    print(f"Training samples: {len(train)}")
    print(f"Testing samples: {len(test)}")
    print(f"Training period: {train['ds'].min()} to {train['ds'].max()}")
    print(f"Testing period: {test['ds'].min()} to {test['ds'].max()}")
    
    return train, test


# ======================================================
# STEP 3 – HOLIDAYS
# ======================================================
def build_holidays(train):
    """
    Build holidays dataframe for Prophet.
    """
    years = train['ds'].dt.year.unique()
    
    # Christmas
    christmas_dates = [f"{y}-12-25" for y in years]
    
    holidays = pd.DataFrame({
        'holiday': 'christmas',
        'ds': pd.to_datetime(christmas_dates),
        'lower_window': 0,
        'upper_window': 1
    })
    
    return holidays


# ======================================================
# STEP 4 – BASE PROPHET MODEL
# ======================================================
def train_prophet_model(train, holidays):
    """
    Train a base Prophet model.
    """
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        holidays=holidays
    )
    
    print("Training Prophet model...")
    model.fit(train)
    print("✓ Model training complete")
    
    return model


# ======================================================
# STEP 5 – FORECAST + PLOT
# ======================================================
def forecast_and_plot(model, test):
    """
    Generate forecasts and visualize results.
    """
    future = model.make_future_dataframe(periods=len(test))
    forecast = model.predict(future)
    
    print("\nForecast summary (last 5 days):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Plot forecast
    fig1 = model.plot(forecast)
    plt.title('Prophet Forecast')
    plt.tight_layout()
    plt.savefig('prophet_forecast.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot components
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('prophet_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return forecast


# ======================================================
# STEP 6 – EVALUATION METRICS
# ======================================================
def calculate_metrics(test, forecast):
    """
    Calculate evaluation metrics for the forecast.
    """
    y_pred = forecast['yhat'].iloc[-len(test):].values
    y_true = test['y'].values
    
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    rmse_pct = (rmse / np.mean(y_true)) * 100
    
    print("\n" + "="*50)
    print("MODEL EVALUATION METRICS")
    print("="*50)
    print(f"MAE (Mean Absolute Error):    {mae:.2f}")
    print(f"RMSE (Root Mean Squared):     {rmse:.2f}")
    print(f"RMSE (% of mean):             {rmse_pct:.2f}%")
    print(f"MAPE (Mean Absolute % Error): {mape:.2f}%")
    print("="*50)
    
    return mae, rmse, mape


# ======================================================
# STEP 7 – PARAMETER GRID SEARCH
# ======================================================
def tune_prophet(train, test, holidays):
    """
    Hyperparameter tuning for Prophet model.
    """
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.1, 1.0, 10.0]
    }
    
    best_rmse = float('inf')
    best_params = None
    
    print("\nStarting hyperparameter tuning...")
    print(f"Total combinations to test: {len(list(ParameterGrid(param_grid)))}")
    
    for i, params in enumerate(ParameterGrid(param_grid), 1):
        print(f"\n[{i}] Testing parameters: {params}")
        
        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
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
        
        print(f"    RMSE: {rmse:.2f}")
        
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            print("    ✓ New best parameters!")
    
    print("\n" + "="*50)
    print("HYPERPARAMETER TUNING RESULTS")
    print(f"Best RMSE: {best_rmse:.2f}")
    print(f"Best parameters: {best_params}")
    print("="*50)
    
    return best_params


# ======================================================
# STEP 8 – TRAIN FINAL MODEL WITH BEST PARAMS
# ======================================================
def train_best_model(train, holidays, best_params):
    """
    Train final model with best hyperparameters.
    """
    print("\nTraining final model with best parameters...")
    
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        holidays=holidays,
        **best_params
    )
    
    model.fit(train)
    print("✓ Final model training complete")
    
    return model


# ======================================================
# STEP 9 – MONITORING
# ======================================================
def run_monitoring(train, test, model):
    """
    Initialize and run monitoring system.
    """
    print("\nInitializing monitoring system...")
    
    forecast = model.predict(test[['ds']])
    forecast = forecast[['ds', 'yhat']].merge(test[['ds', 'y']], on='ds')
    
    monitor = integrate_monitoring_with_prophet(train, test, model, forecast)
    
    health = monitor.check_model_health()
    print("\n" + "="*60)
    print("MODEL HEALTH CHECK")
    print(f"Status: {health['status'].upper()}")
    print("="*60)
    
    return monitor


def monitoring_update(monitor, test, model, new_date="2025-01-01"):
    """
    Update monitoring with new prediction.
    """
    new_date = pd.to_datetime(new_date)
    new_predicted = float(model.predict(pd.DataFrame({'ds': [new_date]}))['yhat'].iloc[0])
    new_actual = float(test['y'].iloc[-1])
    
    status = continuous_monitoring_workflow(
        monitor=monitor,
        new_actual=new_actual,
        new_predicted=new_predicted,
        new_date=new_date
    )
    
    print(f"Monitoring Status: {status}")
    return status


# ======================================================
# STEP 10 – SAVE TRAINED MODEL
# ======================================================
def save_model(model, filename="prophet_model.pkl"):
    """
    Saves the trained Prophet model using pickle.
    """
    with open(filename, "wb") as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filename}")


# ======================================================
# MAIN EXECUTION WORKFLOW
# ======================================================
def main():
    """
    Complete workflow from data loading to model deployment.
    """
    print("PROPHET FORECASTING PIPELINE")
    print("="*60)
    
    # Step 1: Load and preprocess data
    print("\n[1/10] Loading and preprocessing data...")
    train_cleaned, train_encoded, test_encoded = load_and_preprocess()
    
    # Step 2: Prepare time series
    print("\n[2/10] Preparing time series data...")
    daily_ts = prepare_time_series(train_cleaned)
    
    # Step 3: Split data
    print("\n[3/10] Splitting data for Prophet...")
    train, test = split_for_prophet(daily_ts)
    
    # Step 4: Build holidays
    print("\n[4/10] Building holidays dataframe...")
    holidays = build_holidays(train)
    
    # Step 5: Train base model
    print("\n[5/10] Training base Prophet model...")
    base_model = train_prophet_model(train, holidays)
    
    # Step 6: Generate forecast
    print("\n[6/10] Generating forecasts...")
    forecast = forecast_and_plot(base_model, test)
    
    # Step 7: Calculate metrics
    print("\n[7/10] Calculating evaluation metrics...")
    mae, rmse, mape = calculate_metrics(test, forecast)
    
    # Step 8: Hyperparameter tuning (optional - uncomment if needed)
    print("\n[8/10] Hyperparameter tuning...")
    tune = input("Perform hyperparameter tuning? (y/n): ").lower()
    if tune == 'y':
        best_params = tune_prophet(train, test, holidays)
        final_model = train_best_model(train, holidays, best_params)
    else:
        print("Skipping tuning, using base model as final model")
        final_model = base_model
    
    # Step 9: Monitoring
    print("\n[9/10] Setting up monitoring system...")
    monitor = run_monitoring(train, test, final_model)
    
    # Step 10: Save model
    print("\n[10/10] Saving final model...")
    save_model(final_model)
    
    print("✓ PIPELINE COMPLETE!")
    print("="*60)
    
    return final_model, monitor


if __name__ == "__main__":
    model, monitor = main()
