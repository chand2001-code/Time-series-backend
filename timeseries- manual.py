import numpy as np
import pandas as pd
import itertools
import xgboost as xgb
from pmdarima import auto_arima
from sklearn.model_selection import train_test_split
from tbats import TBATS
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score, RandomizedSearchCV, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,  precision_score, recall_score
from sklearn.svm import SVR
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
import time

import pandas as pd
from sklearn.model_selection import train_test_split
from dateutil import parser

 
import pandas as pd
from sklearn.model_selection import train_test_split
from dateutil import parser














def prepare_time_series_data(df: pd.DataFrame, target_col: str, datetime_col: str, past_steps: int, future_steps: int, test_size: float):
    # Check if the datetime column exists
    if datetime_col not in df.columns:
        raise ValueError(f"Column '{datetime_col}' not found in DataFrame.")

    # Check if the datetime column is specifically of ns64 datetime type
    if not pd.api.types.is_datetime64_ns_dtype(df[datetime_col]):
        # Convert to datetime type
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')  # {{ edit_1 }}

    # Check if the DataFrame is empty after cleaning
    if df.empty:
        raise ValueError("DataFrame is empty after cleaning. Please check the input data.")

    # Check if the target column exists
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in DataFrame.")

    # Ensure there are enough rows for lag features
    if len(df) < past_steps + future_steps:
        raise ValueError("Not enough data to create lag features and target variable.")

    # Generate past steps lag features
    for i in range(1, past_steps + 1):
        df[f'lag_{i}'] = df[target_col].shift(i)

    # Generate the target feature
    df['target'] = df[target_col].shift(-future_steps)

    # Drop rows with NaN values created by shifting
    df.dropna(inplace=True)  # {{ edit_2 }}

    # Split data into features and target
    X = df[[f'lag_{i}' for i in range(1, past_steps + 1)]]
    y = df['target']

    # Perform train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test



def compute_metrics(y_pred, y_test, execution_time):
    def safe_compute(func, *args):
        try:
            result = func(*args)
            if np.isnan(result) or np.isinf(result):
                return None
            return float(result)  # Ensure the result is a Python float
        except Exception:
            return None

    metrics = {
        'RMSE': safe_compute(lambda: np.sqrt(mean_squared_error(y_test, y_pred))),
        'R2': safe_compute(r2_score, y_test, y_pred),
        'MAE': safe_compute(mean_absolute_error, y_test, y_pred),
        'MSE': safe_compute(mean_squared_error, y_test, y_pred),
        'SMAPE': safe_compute(lambda: np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_test) + np.abs(y_pred))) * 100),
        'Duration': float(execution_time)  # Ensure execution_time is a Python float
    }

    # Remove any metrics that couldn't be computed
    metrics = {k: v for k, v in metrics.items() if v is not None}

    return metrics

def arima_time_series(X_train, X_test, y_train, y_test, perform_cv=False):
    # Hyperparameter tuning for ARIMA
    p = d = q = range(0, 2)
    pdq = [(x[0], x[1], x[2]) for x in itertools.product(p, d, q)]
    best_aic = np.inf
    best_order = None
    model = None

    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    start_time = time.time()

    for order in pdq:
        try:
            model2 = ARIMA(y_train, order=order)
            model_fit = model2.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = order
                model = model_fit
        except Exception as e:
            print(f"Error fitting ARIMA model with order {order}: {e}")
            continue
            
    if perform_cv:
        # Perform cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_index, test_index in tscv.split(y_train):
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

            if len(y_train_cv) == 0 or len(y_test_cv) == 0:
                continue  # Skip if any split is empty

            cv_model = ARIMA(y_train_cv, order=best_order)
            cv_model_fit = cv_model.fit(disp=False)
            y_pred_cv = cv_model_fit.forecast(steps=len(y_test_cv))
            mse = mean_squared_error(y_test_cv, y_pred_cv)
            cv_scores.append(mse)

        cv_scores = np.array(cv_scores)
    else:
        cv_scores = None

    # Forecasting with the best model
    if model is None:
        raise ValueError("No suitable ARIMA model found. Please check your data and parameters.")

    y_pred = model.forecast(steps=len(y_test))
    execution_time = time.time() - start_time
    
    # Return predictions, actual values, and execution time (and CV scores if applicable)
    return y_pred, y_test, execution_time, cv_scores


def sarimax_time_series(X_train, X_test, y_train, y_test, perform_cv=False):
    # Hyperparameter tuning for SARIMAX
    p = d = q = range(0, 2)  # Reduced range to decrease model complexity
    pdq = [(x[0], x[1], x[2]) for x in itertools.product(p, d, q)]
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in itertools.product(p, d, q)]
    
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    model = None

    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    start_time = time.time() 

    for order in pdq:
        for seasonal_order in seasonal_pdq:
            try:
                model2 = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
                model_fit = model2.fit(disp=False)
                if model_fit.aic < best_aic:
                    best_aic = model_fit.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
                    model = model_fit
            except Exception as e:
                print(f"Error fitting SARIMAX model with order {order} and seasonal order {seasonal_order}: {e}")
                continue
                
    if perform_cv:
        # Perform cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_index, test_index in tscv.split(y_train):
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

            if len(y_train_cv) == 0 or len(y_test_cv) == 0:
                continue  # Skip if any split is empty

            cv_model = SARIMAX(y_train_cv, order=best_order, seasonal_order=best_seasonal_order)
            cv_model_fit = cv_model.fit(disp=False)
            y_pred_cv = cv_model_fit.get_forecast(steps=len(y_test_cv)).predicted_mean
            mse = mean_squared_error(y_test_cv, y_pred_cv)
            cv_scores.append(mse)

        cv_scores = np.array(cv_scores)
    else:
        cv_scores = None

    # Check if a suitable model was found
    if model is None:
        raise ValueError("No suitable SARIMAX model found. Please check your data and parameters.")

    # Forecasting with the best model
    y_pred = model.get_forecast(steps=len(y_test)).predicted_mean
    execution_time = time.time() - start_time
    
    # Return predictions, actual values, execution time, and CV scores if applicable
    return y_pred, y_test, execution_time, cv_scores

def holt_winters_time_series(X_train, X_test, y_train, y_test, perform_cv=False):
    def evaluate_model(seasonal, seasonal_periods):
        if seasonal == 'mul' and np.any(y_train <= 0):
            raise ValueError("Data contains non-positive values; multiplicative seasonal component requires strictly positive data.")
        
        model_hw = ExponentialSmoothing(y_train, seasonal=seasonal, seasonal_periods=seasonal_periods)
        hw_model = model_hw.fit()
        y_pred = hw_model.forecast(len(y_test))
        mse = mean_squared_error(y_test, y_pred)
        return mse, hw_model

    # Define the parameter grid for hyperparameter tuning
    seasonal_options = ['add', 'mul']
    seasonal_periods_options = [12, 24, 36]
    
    best_mse = float('inf')
    best_params = None
    best_model = None

    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    start_time = time.time()

    for seasonal in seasonal_options:
        for seasonal_periods in seasonal_periods_options:
            try:
                mse, model = evaluate_model(seasonal, seasonal_periods)
                if mse < best_mse:
                    best_mse = mse
                    best_params = {'seasonal': seasonal, 'seasonal_periods': seasonal_periods}
                    best_model = model
            except Exception as e:
                print(f"Error evaluating model with seasonal={seasonal} and seasonal_periods={seasonal_periods}: {e}")
                continue

    if perform_cv:
        # Perform cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        cv_scores = []

        for train_index, test_index in tscv.split(y_train):
            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

            if len(y_train_cv) == 0 or len(y_test_cv) == 0:
                continue  # Skip if any split is empty

            cv_model = ExponentialSmoothing(y_train_cv, seasonal=best_params['seasonal'], seasonal_periods=best_params['seasonal_periods'])
            cv_model_fit = cv_model.fit()
            y_pred_cv = cv_model_fit.forecast(len(y_test_cv))
            mse = mean_squared_error(y_test_cv, y_pred_cv)
            cv_scores.append(mse)

        cv_scores = np.array(cv_scores)
    else:
        cv_scores = None

    # Check if a suitable model was found
    if best_model is None:
        raise ValueError("No suitable Holt-Winters model found. Please check your data and parameters.")

    # Predict on the test data using the best model
    y_pred = best_model.forecast(len(y_test))
    execution_time = time.time() - start_time

    return y_pred, y_test, execution_time, cv_scores



def tbats_time_series(X_train, X_test, y_train, y_test, perform_cv=False, seasonal_periods=[365.25], **kwargs):
    # Check if training data is empty
    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    start_time = time.time()
    
    # Hyperparameter tuning
    tbats_params = {
        'seasonal_periods': kwargs.get('seasonal_periods', seasonal_periods),
        'use_box_cox': kwargs.get('use_box_cox', [False, True]),
        'use_trend': kwargs.get('use_trend', [False, True]),
        'use_damped_trend': kwargs.get('use_damped_trend', [False, True]),
        'spike_and_slab': kwargs.get('spike_and_slab', [False, True])
    }

    cv_scores = None  # Initialize cv_scores

    if perform_cv:
        tscv = TimeSeriesSplit(n_splits=3)  # Reduced splits for faster execution
        metrics = {'mse': []}

        # Use a simpler model for faster execution during CV
        tbats_model = TBATS(**tbats_params)

        grid_search = GridSearchCV(tbats_model, param_grid={'seasonal_periods': seasonal_periods}, cv=tscv, n_jobs=-1)  # Parallel processing
        try:
            grid_search.fit(X_train)
            best_model = grid_search.best_estimator_
            y_pred = best_model.forecast(steps=len(y_test))
            # Calculate metrics here if needed
            # metrics['mse'].append(mean_squared_error(test, y_pred))
            cv_scores = grid_search.best_score_  # Store best score
        except Exception as e:
            print(f"Error during cross-validation: {e}")

    else:
        tbats_model = TBATS(seasonal_periods=seasonal_periods, **kwargs)
        model = tbats_model.fit(y_train)
        y_pred = model.forecast(steps=len(y_test))

    execution_time = time.time() - start_time
        
    return y_pred, y_test, execution_time, cv_scores

def ensemble_time_series(X_train, X_test, y_train, y_test, perform_cv=False):
    # Define parameter grids for SARIMAX tuning
    seasonal_order_list = [(1, 1, 1, 12), (0, 1, 1, 12), (1, 0, 1, 12)]
    order_list = [(2, 1, 2), (1, 1, 1), (2, 0, 2)]
    
    best_sarimax_rmse = float('inf')
    best_sarimax_model = None
    best_sarimax_params = None

    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    start_time = time.time()

    for seasonal_order in seasonal_order_list:
        for order in order_list:
            try:
                model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order)
                model_fit = model.fit(disp=False)
                y_pred = model_fit.get_forecast(steps=len(y_test)).predicted_mean
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))

                if rmse < best_sarimax_rmse:
                    best_sarimax_rmse = rmse
                    best_sarimax_model = model_fit
                    best_sarimax_params = (order, seasonal_order)
            except Exception as e:
                print(f"Error fitting SARIMAX model with order {order} and seasonal order {seasonal_order}: {e}")
                continue

    if best_sarimax_model is None:
        raise ValueError("No suitable SARIMAX model found. Please check your data and parameters.")

    # Define parameter grid for RandomForestRegressor
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30]
    }

    cv_scores = None  # Initialize cv_scores

    if perform_cv:
        # Use TimeSeriesSplit for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        rf = RandomForestRegressor()
        grid_search = GridSearchCV(estimator=rf, param_grid=rf_param_grid, scoring='neg_mean_squared_error', cv=tscv)
        grid_search.fit(X_train, y_train)

        # Best RandomForest model
        model = grid_search.best_estimator_
        best_rf_params = grid_search.best_params_

        # Calculate cross-validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
    else:
        # Train RandomForest without cross-validation
        model = RandomForestRegressor(n_estimators=100, max_depth=None)
        model.fit(X_train, y_train)
        best_rf_params = {'n_estimators': 100, 'max_depth': None}

    rf_y_pred = model.predict(X_test)
    execution_time = time.time() - start_time

    # Forecast with the best SARIMAX model
    sarimax_y_pred = best_sarimax_model.get_forecast(steps=len(y_test)).predicted_mean

    # Combine predictions
    y_pred = (sarimax_y_pred + rf_y_pred) / 2

    # Return y_pred, y_test, execution_time, and cv_scores
    return y_pred, y_test, execution_time, cv_scores

def random_forest_timeseries(X_train, X_test, y_train, y_test, perform_cv=False):
    # Check if training data is empty
    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    start_time = time.time()
    
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    
    # Initialize GridSearchCV with RandomForestRegressor
    if perform_cv:
        # Initialize GridSearchCV with cross-validation
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    else:
        # Initialize GridSearchCV without cross-validation (use default split)
        grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=[(slice(None), slice(None))], n_jobs=-1, scoring='neg_mean_squared_error')

    # Fit GridSearchCV to the training data
    grid_search.fit(X_train_scaled, y_train)
    
    # Get the best model from GridSearchCV
    model = grid_search.best_estimator_

    if perform_cv:
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        cv_scores = None

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)
    execution_time = time.time() - start_time
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    return y_pred, y_test, execution_time, cv_scores

def lstm_timeseries(X_train, X_test, y_train, y_test, perform_cv=False):
    def create_lstm_model(units=50, dropout_rate=0.2):
        model = Sequential()
        model.add(LSTM(units=units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units=units))
        model.add(Dropout(dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')
        return model

    # Check if training data is empty
    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    # Convert DataFrames to NumPy arrays
    X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    y_test = y_test.values if isinstance(y_test, pd.Series) else y_test
    start_time = time.time()

    # Reshape the data for LSTM [samples, time steps, features]
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

    param_grid = {
        'model__units': [50, 100],
        'model__dropout_rate': [0.2, 0.3],
        'batch_size': [32, 64],
        'epochs': [50, 100]
    }

    model = KerasRegressor(build_fn=create_lstm_model, verbose=0)
    
    if perform_cv:
        # Initialize GridSearchCV with cross-validation
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train_scaled, y_train)

        # Perform cross-validation
        cv_scores = cross_val_score(grid_search.best_estimator_, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
    else:
        # Initialize GridSearchCV without cross-validation (use default split)
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=[(slice(None), slice(None))], scoring='neg_mean_squared_error', verbose=1)
        grid_search.fit(X_train_scaled, y_train)
        cv_scores = None

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_scaled)

    best_params = grid_search.best_params_
    execution_time = time.time() - start_time
    best_score = grid_search.best_score_

    return y_pred, y_test, execution_time, cv_scores


def xgboost_timeseries(X_train, X_test, y_train, y_test, perform_cv=False):
    # Check if training data is empty
    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    start_time = time.time()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'booster': ['gbtree', 'gblinear', 'dart'],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }

    # Initialize GridSearchCV
    if perform_cv:
        # Initialize GridSearchCV with cross-validation
        grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    else:
        # Initialize GridSearchCV without cross-validation (use default split)
        grid_search = GridSearchCV(xgb.XGBRegressor(), param_grid, cv=[(slice(None), slice(None))], n_jobs=-1, scoring='neg_mean_squared_error')

    grid_search.fit(X_train_scaled, y_train)

    # Get the best estimator
    model = grid_search.best_estimator_

    if perform_cv:
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        cv_scores = None

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test_scaled)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    execution_time = time.time() - start_time

    return y_pred, y_test, execution_time, cv_scores


def linear_regression_timeseries(X_train, X_test, y_train, y_test, perform_cv=False):
    # Check if training data is empty
    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    start_time = time.time()

    # Define the parameter grid for hyperparameter tuning (although LinearRegression has no hyperparameters in GridSearchCV)
    param_grid = {
        'fit_intercept': [True, False]
    }

    # Initialize GridSearchCV
    if perform_cv:
        # Initialize GridSearchCV with cross-validation
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    else:
        # Initialize GridSearchCV without cross-validation (use default split)
        grid_search = GridSearchCV(LinearRegression(), param_grid, cv=[(slice(None), slice(None))], n_jobs=-1, scoring='neg_mean_squared_error')

    grid_search.fit(X_train_scaled, y_train)

    # Get the best estimator
    model = grid_search.best_estimator_

    if perform_cv:
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        cv_scores = None

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test_scaled)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    execution_time = time.time() - start_time

    return y_pred, y_test, execution_time, cv_scores


def svm_timeseries(X_train, X_test, y_train, y_test, perform_cv=False):
    # Check if training data is empty
    if len(y_train) == 0:
        raise ValueError("Training data is empty. Please provide valid data.")

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    start_time = time.time()

    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'kernel': ['rbf', 'linear', 'poly'],
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 1]
    }

    # Initialize GridSearchCV
    if perform_cv:
        # Initialize GridSearchCV with cross-validation
        grid_search = GridSearchCV(SVR(), param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')
    else:
        # Initialize GridSearchCV without cross-validation (use default split)
        grid_search = GridSearchCV(SVR(), param_grid, cv=[(slice(None), slice(None))], n_jobs=-1, scoring='neg_mean_squared_error')

    grid_search.fit(X_train_scaled, y_train)

    # Get the best estimator
    model = grid_search.best_estimator_

    if perform_cv:
        # Perform cross-validation
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error')
    else:
        cv_scores = None

    # Fit the model to the training data
    model.fit(X_train_scaled, y_train)

    # Predict on the test data
    y_pred = model.predict(X_test_scaled)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    execution_time = time.time() - start_time

    return y_pred, y_test, execution_time, cv_scores