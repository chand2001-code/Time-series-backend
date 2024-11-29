from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBRegressor
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

# Function to calculate metrics
def calculate_metrics(y_test, y_pred):
    return {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'SMAPE': np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))) * 100
    }

# Function to prepare time series data
def prepare_time_series(df: pd.DataFrame, target_col: str, datetime_col: str, past_steps: int = 30, future_steps: int = 30, test_size: float = 0.2):
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")
    
    # Check and process datetime column
    if df.index.name == datetime_col:
        df.reset_index(inplace=True)
    elif datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")
    
    # Convert datetime column to datetime, handle mixed formats
    df[datetime_col] = df[datetime_col].astype(str)
    
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce', dayfirst=True)  # Automatically infer formats, with dayfirst=True
    except Exception as e:
        raise ValueError(f"Error in converting datetime: {e}")
    
    # Drop rows where datetime conversion failed (if any)
    df.dropna(subset=[datetime_col], inplace=True)
    
    df.drop_duplicates(subset=[datetime_col], inplace=True)
    df.sort_values(by=datetime_col, inplace=True)
    df.set_index(datetime_col, inplace=True)
    
    # Forward fill missing values
    df = df.ffill()

    # Generate past steps lag features
    for i in range(1, past_steps + 1):
        df[f'lag_{i}'] = df[target_col].shift(i)
    
    # Generate the target feature (shifted by future_steps)
    df['target'] = df[target_col].shift(-future_steps)
    df.dropna(inplace=True)

    # Split data into features (X) and target (y)
    X = df[[f'lag_{i}' for i in range(1, past_steps + 1)]]
    y = df['target']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    # Dictionary to store results
    results = {}

    # ARIMA
    try:
        model_arima = ARIMA(y_train, order=(5, 1, 0)).fit()
        y_pred_arima = model_arima.forecast(steps=len(y_test))
        results['ARIMA'] = calculate_metrics(y_test, y_pred_arima)
    except Exception as e:
        results['ARIMA'] = f"Error: {e}"

    # SARIMA
    try:
        model_sarima = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()
        y_pred_sarima = model_sarima.forecast(steps=len(y_test))
        results['SARIMA'] = calculate_metrics(y_test, y_pred_sarima)
    except Exception as e:
        results['SARIMA'] = f"Error: {e}"

    # Holt-Winters
    try:
        model_hw = ExponentialSmoothing(y_train, seasonal='add', seasonal_periods=12).fit()
        y_pred_hw = model_hw.forecast(steps=len(y_test))
        results['Holt-Winters'] = calculate_metrics(y_test, y_pred_hw)
    except Exception as e:
        results['Holt-Winters'] = f"Error: {e}"

    # Linear Regression with GridSearchCV
    try:
        param_grid_lr = {'fit_intercept': [True, False]}
        model_lr = GridSearchCV(LinearRegression(), param_grid_lr, cv=3)
        model_lr.fit(X_train, y_train)
        y_pred_lr = model_lr.predict(X_test)
        results['Linear Regression'] = calculate_metrics(y_test, y_pred_lr)
    except Exception as e:
        results['Linear Regression'] = f"Error: {e}"

    # Random Forest with GridSearchCV
    try:
        param_grid_rf = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
        model_rf = GridSearchCV(RandomForestRegressor(), param_grid_rf, cv=3)
        model_rf.fit(X_train, y_train)
        y_pred_rf = model_rf.predict(X_test)
        results['Random Forest'] = calculate_metrics(y_test, y_pred_rf)
    except Exception as e:
        results['Random Forest'] = f"Error: {e}"

    # SVM with GridSearchCV
    try:
        param_grid_svm = {'C': [0.1, 1], 'kernel': ['linear', 'rbf']}
        model_svm = GridSearchCV(SVR(), param_grid_svm, cv=3)
        model_svm.fit(X_train, y_train)
        y_pred_svm = model_svm.predict(X_test)
        results['SVM'] = calculate_metrics(y_test, y_pred_svm)
    except Exception as e:
        results['SVM'] = f"Error: {e}"

    # XGBoost with GridSearchCV
    try:
        param_grid_xgb = {'n_estimators': [50, 100], 'learning_rate': [0.1, 0.01]}
        model_xgb = GridSearchCV(XGBRegressor(), param_grid_xgb, cv=3)
        model_xgb.fit(X_train, y_train)
        y_pred_xgb = model_xgb.predict(X_test)
        results['XGBoost'] = calculate_metrics(y_test, y_pred_xgb)
    except Exception as e:
        results['XGBoost'] = f"Error: {e}"

    # LSTM
    try:
        # Reshape input data for LSTM (samples, time steps, features)
        X_train_lstm = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test_lstm = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Define model
        model_lstm = Sequential([
            LSTM(50, activation='relu', input_shape=(1, past_steps)),
            Dense(1)
        ])
        model_lstm.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        # Early stopping
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        # Fit model
        history = model_lstm.fit(
            X_train_lstm, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )

        # Predict
        y_pred_lstm = model_lstm.predict(X_test_lstm).flatten()
        results['LSTM'] = calculate_metrics(y_test, y_pred_lstm)
    except Exception as e:
        results['LSTM'] = f"Error: {e}"


    return results



