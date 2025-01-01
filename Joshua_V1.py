import logging
import time
from datetime import datetime, timedelta
from math import sqrt

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

# Prophet import (choose the correct one)
from prophet import Prophet

import warnings
# Silencing warnings about 'force_all_finite'
warnings.filterwarnings(
    "ignore",
    message=".*force_all_finite.*",
    category=FutureWarning
)

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_bitcoin_data(retries=3, delay=5, days=90):
    """
    Fetch Bitcoin price data from Yahoo Finance over a specified number of days.
    Retries a few times in case of network issues.
    Increase 'days' to 180 or 365 if you want more historical depth for Prophet.
    """
    for attempt in range(retries):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download('BTC-USD', start=start_date, end=end_date, interval='1h')
            if df.empty:
                raise ValueError("No data fetched from yfinance.")
            # Force hourly frequency
            df = df.asfreq('h')
            return df
        except Exception as e:
            logging.error(f"Attempt {attempt + 1} failed to fetch data: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                logging.error("All retries failed. Exiting.")
                return None

def validate_data(df):
    """
    Validate data for continuity and fill missing values as needed.
    Return None if validation fails.
    """
    if df is None or df.empty:
        return None

    # Fill missing values forward, then backward if necessary
    if df.isna().any().any():
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)
        logging.warning("Data contained NaN values that have been filled.")

    # Check hourly intervals
    diffs = df.index.to_series().diff().dropna()
    if not all(diffs == timedelta(hours=1)):
        logging.warning("Data does not have consistent hourly intervals.")
        return None

    # Optional outlier handling (Volume, Close, etc.). Adjust thresholds as needed.
    for col in ['Volume', 'Close']:
        upper_q = df[col].quantile(0.999)
        lower_q = df[col].quantile(0.001)
        df = df[(df[col] >= lower_q) & (df[col] <= upper_q)]

    return df

def calculate_indicators(df):
    """
    Calculate a set of technical indicators (SMA, EMA, RSI, Bollinger Bands, etc.).
    """
    result = pd.DataFrame(index=df.index)
    result['Close'] = df['Close']
    result['High'] = df['High']
    result['Low'] = df['Low']
    result['Open'] = df['Open']
    result['Volume'] = df['Volume']

    # Moving Averages
    for period in [20, 50, 100]:
        result[f'SMA_{period}'] = result['Close'].rolling(window=period).mean()
        result[f'EMA_{period}'] = result['Close'].ewm(span=period, adjust=False).mean()

    # MACD
    ema_12 = result['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = result['Close'].ewm(span=26, adjust=False).mean()
    result['MACD'] = ema_12 - ema_26
    result['Signal_Line'] = result['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    delta = result['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14).mean()
    result['RSI'] = 100 - (100 / (1 + (avg_gain / avg_loss)))

    # Bollinger Bands
    rolling_window = 20
    rolling_std = result['Close'].rolling(window=rolling_window).std()
    result['Bollinger_Upper'] = result[f'SMA_{rolling_window}'] + 2 * rolling_std
    result['Bollinger_Lower'] = result[f'SMA_{rolling_window}'] - 2 * rolling_std

    # Time-based features
    result['Hour'] = result.index.hour
    result['DayOfWeek'] = result.index.dayofweek

    # Drop rows that don't have enough data for all calculations
    return result.dropna()

def create_feature_matrix(df, lookback=1):
    """
    Convert time series into supervised learning format for ML models (Gradient Boosting, MLP).
    lookback: how many lags to include as features.
    """
    df_features = df[['Close', 'RSI', 'MACD', 'Signal_Line']].copy()

    # Create lag features
    for lag in range(1, lookback + 1):
        for col in ['Close', 'RSI', 'MACD', 'Signal_Line']:
            df_features[f'{col}_lag_{lag}'] = df_features[col].shift(lag)

    # Drop rows with NaN after shifting
    df_features.dropna(inplace=True)

    # Our target is the future close price (the next time step)
    X = df_features.drop('Close', axis=1)
    y = df_features['Close']

    return X, y

# ------------------------- PROPHET FUNCTIONS ------------------------- #

def train_prophet(df):
    """
    Train a Prophet model on the entire time series of Close prices.
    Return the fitted model, removing any timezone info from the index.
    """
    try:
        # Remove timezone (tz_localize(None)) so Prophet doesn't complain
        df_no_tz = df.copy()
        df_no_tz.index = df_no_tz.index.tz_localize(None)  # remove timezone if present

        prophet_df = pd.DataFrame({
            'ds': df_no_tz.index,  # datetimes without tz
            'y': df_no_tz['Close']
        }).reset_index(drop=True)

        model = Prophet(
            daily_seasonality=True,
            yearly_seasonality=True,
            weekly_seasonality=True
            # Add more advanced config if needed
        )
        model.fit(prophet_df)
        return model
    except Exception as e:
        logging.error(f"Prophet training error: {e}")
        return None

def make_prophet_prediction(model, df_original, steps=1):
    """
    Generate a future forecast from a trained Prophet model, for a given number of steps.
    Returns an array of predictions for the final 'steps' periods at hourly frequency.
    """
    if model is None:
        return None

    try:
        # Remove timezone from original index as well
        df_no_tz = df_original.copy()
        df_no_tz.index = df_no_tz.index.tz_localize(None)

        last_date = df_no_tz.index[-1]
        future_df = model.make_future_dataframe(periods=steps, freq='h')

        forecast = model.predict(future_df)
        # Only take the tail for the requested steps
        forecast_tail = forecast.tail(steps)

        return forecast_tail['yhat'].values
    except Exception as e:
        logging.error(f"Prophet prediction error: {e}")
        return None

# ------------------------ ML MODELS (GBR, MLP) ----------------------- #

def train_gradient_boosting(X_train, y_train):
    gbr = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gbr.fit(X_train, y_train)
    return gbr

def train_mlp(X_train, y_train):
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 64),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42
    )
    mlp.fit(X_train, y_train)
    return mlp

# ---------------------- RECOMMENDATIONS ---------------------- #

def recommend_action(df_ind, ensemble_prediction_list):
    last_rsi = df_ind['RSI'].iloc[-1]
    last_price = df_ind['Close'].iloc[-1]

    if last_rsi < 30:
        rsi_signal = "RSI suggests OVERSOLD conditions."
    elif last_rsi > 70:
        rsi_signal = "RSI suggests OVERBOUGHT conditions."
    else:
        rsi_signal = "RSI is in a neutral range."

    if ensemble_prediction_list is not None and len(ensemble_prediction_list) > 0:
        forecast_price = ensemble_prediction_list[-1]
        if forecast_price > last_price * 1.02:
            forecast_signal = "Ensemble forecast is significantly ABOVE the current price. Potential uptrend."
        elif forecast_price < last_price * 0.98:
            forecast_signal = "Ensemble forecast is significantly BELOW the current price. Potential downtrend."
        else:
            forecast_signal = "Ensemble forecast is near the current price. No strong trend indicated."
    else:
        forecast_signal = "No ensemble forecast available."

    return rsi_signal, forecast_signal

def recommend_for_non_expert(rsi_signal, forecast_signal):
    advice = []

    if "OVERSOLD" in rsi_signal:
        advice.append("Price may be lower than normal relative to recent trends, which can sometimes indicate a good buying opportunity.")
    elif "OVERBOUGHT" in rsi_signal:
        advice.append("Price may be higher than normal relative to recent trends, which can sometimes indicate a good selling opportunity.")
    else:
        advice.append("Price doesn't appear strongly overbought or oversold right now. No strong signal from RSI alone.")

    if "UPTREND" in forecast_signal.upper():
        advice.append("The forecast suggests that prices might go up soon.")
    elif "DOWNTREND" in forecast_signal.upper():
        advice.append("The forecast suggests that prices might go down soon.")
    elif "NEAR THE CURRENT PRICE" in forecast_signal.upper():
        advice.append("The forecast suggests no significant immediate price movement.")
    else:
        advice.append("No clear forecast signal is available at this moment.")

    user_friendly_recommendation = (
        "For someone not deeply familiar with trading indicators:\n\n"
        f"1) {advice[0]}\n"
        f"2) {advice[1]}"
    )

    return user_friendly_recommendation

# ------------------------- MAIN PIPELINE ------------------------- #

def main():
    # STEP 1: Fetch and validate data
    df = fetch_bitcoin_data(days=90)
    df = validate_data(df)
    if df is None:
        logging.error("Data fetching or validation failed.")
        return

    # STEP 2: Calculate indicators
    df_ind = calculate_indicators(df)

    # STEP 3: Feature Matrix (ML)
    X, y = create_feature_matrix(df_ind, lookback=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # STEP 4: Train Prophet (removing any timezone from index)
    prophet_model = train_prophet(df_ind)
    if prophet_model:
        prophet_preds = make_prophet_prediction(prophet_model, df_ind, steps=len(X_test))
    else:
        prophet_preds = None

    if prophet_preds is None or len(prophet_preds) != len(X_test):
        prophet_preds = np.zeros(len(X_test))
    prophet_preds = np.array(prophet_preds)

    # STEP 5: Train Gradient Boosting and MLP
    gbr_model = train_gradient_boosting(X_train, y_train)
    mlp_model = train_mlp(X_train, y_train)

    # STEP 6: Predictions
    gbr_preds = np.array(gbr_model.predict(X_test))
    mlp_preds = np.array(mlp_model.predict(X_test))

    # STEP 7: Ensemble
    min_len = min(len(prophet_preds), len(gbr_preds), len(mlp_preds))
    ensemble_preds = (
        prophet_preds[:min_len] +
        gbr_preds[:min_len] +
        mlp_preds[:min_len]
    ) / 3.0
    ensemble_preds_list = ensemble_preds.tolist()
    y_test_aligned = y_test.iloc[-min_len:].values

    # STEP 8: Evaluate
    prophet_mse = mean_squared_error(y_test_aligned, prophet_preds[:min_len])
    prophet_rmse = sqrt(prophet_mse)

    gbr_mse = mean_squared_error(y_test_aligned, gbr_preds[:min_len])
    gbr_rmse = sqrt(gbr_mse)

    mlp_mse = mean_squared_error(y_test_aligned, mlp_preds[:min_len])
    mlp_rmse = sqrt(mlp_mse)

    ensemble_mse = mean_squared_error(y_test_aligned, ensemble_preds)
    ensemble_rmse = sqrt(ensemble_mse)

    logging.info(f"Prophet Test RMSE: {prophet_rmse:.4f}")
    logging.info(f"GBR Test RMSE:     {gbr_rmse:.4f}")
    logging.info(f"MLP Test RMSE:     {mlp_rmse:.4f}")
    logging.info(f"Ensemble RMSE:     {ensemble_rmse:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test_aligned, label='Actual', color='blue')
    plt.plot(prophet_preds[:min_len], label='Prophet', color='orange', linestyle='--')
    plt.plot(gbr_preds[:min_len], label='Gradient Boosting', color='green', linestyle='--')
    plt.plot(mlp_preds[:min_len], label='MLP', color='red', linestyle='--')
    plt.plot(ensemble_preds, label='Ensemble', color='purple', linestyle='--')
    plt.title('Model Predictions vs. Actual')
    plt.xlabel('Test Data Index')
    plt.ylabel('Bitcoin Price (USD)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # STEP 9: Recommendations
    rsi_signal, forecast_signal = recommend_action(df_ind, ensemble_preds_list)
    logging.info(f"\n--- Technical Recommendation ---\nRSI Info: {rsi_signal}\nForecast Info: {forecast_signal}")

    user_friendly_recommendation = recommend_for_non_expert(rsi_signal, forecast_signal)
    logging.info(f"\n--- Recommendation for Non-Expert ---\n{user_friendly_recommendation}")

    # STEP 10: 24-Hour Forecast with Prophet
    if prophet_model:
        prophet_24_preds = make_prophet_prediction(prophet_model, df_ind, steps=24)
        if prophet_24_preds is not None and len(prophet_24_preds) == 24:
            logging.info(f"Prophet 24-hr raw predictions array: {prophet_24_preds}")

            # Create a future date range for the next 24 hours
            df_no_tz = df_ind.copy()
            df_no_tz.index = df_no_tz.index.tz_localize(None)
            last_date = df_no_tz.index[-1]

            future_24_dates = pd.date_range(
                start=last_date + pd.Timedelta(hours=1),
                periods=24,
                freq='h'
            )
            forecast_24_df = pd.DataFrame({'Predicted_Close': prophet_24_preds}, index=future_24_dates)
            logging.info("\n--- 24-Hour Prophet Forecast ---")
            logging.info(f"\n{forecast_24_df}")

            # **Additional Lines** for the 24-hour forecast display
            logging.info("Here is the extended 24-hour forecast in plain text:")
            for ts, val in forecast_24_df['Predicted_Close'].items():
                logging.info(f"Date/Time: {ts}, Forecasted Price: {val:.2f}")

            plt.figure(figsize=(12, 5))
            plt.plot(df_no_tz.index, df_no_tz['Close'], label='Historical Price', color='blue')
            plt.plot(forecast_24_df.index, forecast_24_df['Predicted_Close'],
                     label='24h Prophet Forecast', color='orange', linestyle='--')
            plt.title('24-Hour Prophet Price Forecast')
            plt.xlabel('Date')
            plt.ylabel('Bitcoin Price (USD)')
            plt.legend()
            plt.tight_layout()
            plt.show()
        else:
            logging.info("Could not generate valid 24-hour Prophet forecast.")
    else:
        logging.info("Prophet model is unavailable for 24-hour forecasting.")

if __name__ == "__main__":
    main()
