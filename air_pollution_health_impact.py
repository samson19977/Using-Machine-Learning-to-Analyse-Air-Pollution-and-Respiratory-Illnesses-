#!/usr/bin/env python3
"""
Air Pollution and Respiratory Health Impact Prediction

This script uses synthetic air quality data to predict respiratory illness cases.
It trains and compares Linear Regression, Random Forest, and XGBoost models.
The project can be extended to use real-world data.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# -------------------- Configuration --------------------
CONFIG = {
    # Data generation
    'synthetic_n': 500,
    'random_seed': 42,
    'date_start': '2022-01-01',

    # Model parameters
    'test_size': 0.2,
    'models': {
        'LinearRegression': {
            'class': LinearRegression,
            'params': {}
        },
        'RandomForest': {
            'class': RandomForestRegressor,
            'params': {'n_estimators': [50, 100], 'max_depth': [3, 5]}
        },
        'XGBoost': {
            'class': XGBRegressor,
            'params': {'n_estimators': [50, 100], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}
        }
    },

    # Paths
    'data_dir': Path('./data'),
    'models_dir': Path('./models'),
    'plots_dir': Path('./plots'),
    'data_file': 'synthetic_air_quality.csv',
    'best_model_file': 'best_model.pkl',
}

# Create directories
for d in [CONFIG['data_dir'], CONFIG['models_dir'], CONFIG['plots_dir']]:
    d.mkdir(parents=True, exist_ok=True)

# -------------------- Logging --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# -------------------- Data Generation --------------------
def generate_data(n, seed, date_start):
    """Generate synthetic air quality and health data."""
    np.random.seed(seed)
    date_range = pd.date_range(start=date_start, periods=n)

    # Simulate pollutants and weather
    pm25 = np.random.normal(loc=50, scale=10, size=n)   # PM2.5
    no2 = np.random.normal(loc=20, scale=5, size=n)     # NO2
    ozone = np.random.normal(loc=30, scale=7, size=n)   # Ozone
    temperature = np.random.normal(loc=25, scale=5, size=n)  # Celsius
    humidity = np.random.uniform(40, 80, size=n)        # %

    # Respiratory cases: linear combination + noise
    respiratory_cases = (10 + 0.5 * pm25 + 0.3 * no2 + 0.2 * ozone
                         - 0.1 * temperature + 0.05 * humidity
                         + np.random.normal(scale=5, size=n))

    data = pd.DataFrame({
        'Date': date_range,
        'PM2.5': pm25,
        'NO2': no2,
        'Ozone': ozone,
        'Temperature': temperature,
        'Humidity': humidity,
        'Respiratory_Cases': respiratory_cases
    })
    return data

# -------------------- Data Exploration --------------------
def explore_data(data):
    """Print basic info and create scatter plot."""
    logger.info("Data sample:")
    print(data.head())
    logger.info("Data statistics:")
    print(data.describe())

    # Scatter plot: PM2.5 vs Respiratory Cases
    plt.figure(figsize=(8, 5))
    plt.scatter(data['PM2.5'], data['Respiratory_Cases'], alpha=0.7, color='red')
    plt.title('Respiratory Cases vs PM2.5 Concentration')
    plt.xlabel('PM2.5 (µg/m³)')
    plt.ylabel('Respiratory Cases')
    plt.tight_layout()
    plt.savefig(CONFIG['plots_dir'] / 'scatter_pm25_cases.png', dpi=300)
    plt.show()

# -------------------- Model Training --------------------
def train_model(model_name, model_class, X_train, y_train, param_grid=None):
    """Train a model with optional grid search."""
    if param_grid:
        logger.info(f"Running GridSearchCV for {model_name}...")
        gs = GridSearchCV(model_class(random_state=CONFIG['random_seed']),
                          param_grid, cv=3, scoring='r2', n_jobs=-1)
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        logger.info(f"Best params for {model_name}: {gs.best_params_}")
    else:
        best_model = model_class(random_state=CONFIG['random_seed'])
        best_model.fit(X_train, y_train)
    return best_model

def evaluate_model(y_test, y_pred, model_name):
    """Compute and print evaluation metrics."""
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    logger.info(f"{model_name} - MSE: {mse:.2f}, R²: {r2:.2f}")
    return mse, r2

def plot_actual_vs_predicted(y_test, y_pred, model_name, save=False):
    """Plot actual vs predicted values."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Actual', marker='o', alpha=0.7)
    plt.plot(y_pred, label='Predicted', marker='x', alpha=0.7)
    plt.title(f'Actual vs Predicted Respiratory Cases ({model_name})')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Respiratory Cases')
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(CONFIG['plots_dir'] / f'actual_vs_predicted_{model_name}.png', dpi=300)
    plt.show()

def plot_feature_importance(model, feature_names, model_name, save=False):
    """Plot feature importance (if model supports it)."""
    if hasattr(model, 'feature_importances_'):
        importances = pd.Series(model.feature_importances_, index=feature_names)
        plt.figure(figsize=(8, 5))
        importances.nlargest(5).plot(kind='barh', color='teal')
        plt.title(f'Feature Importance ({model_name})')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        if save:
            plt.savefig(CONFIG['plots_dir'] / f'feature_importance_{model_name}.png', dpi=300)
        plt.show()
    else:
        logger.warning(f"{model_name} does not provide feature importance.")

# -------------------- Main --------------------
def main():
    parser = argparse.ArgumentParser(description='Air Pollution Health Impact Prediction')
    parser.add_argument('--mode', type=str, required=True,
                        choices=['generate', 'train', 'full'],
                        help='Mode: generate synthetic data, train models, or full pipeline')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to existing CSV data (if not generating)')
    args = parser.parse_args()

    if args.mode == 'generate':
        data = generate_data(CONFIG['synthetic_n'], CONFIG['random_seed'], CONFIG['date_start'])
        data.to_csv(CONFIG['data_dir'] / CONFIG['data_file'], index=False)
        logger.info(f"Synthetic data saved to {CONFIG['data_dir'] / CONFIG['data_file']}")

    elif args.mode == 'train':
        if args.data_path:
            data_path = Path(args.data_path)
        else:
            data_path = CONFIG['data_dir'] / CONFIG['data_file']
        if not data_path.exists():
            logger.error(f"Data file not found: {data_path}. Please generate data first.")
            sys.exit(1)
        data = pd.read_csv(data_path, parse_dates=['Date'])
        logger.info(f"Loaded data from {data_path}, shape {data.shape}")

        # Prepare features and target
        X = data[['PM2.5', 'NO2', 'Ozone', 'Temperature', 'Humidity']]
        y = data['Respiratory_Cases']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'],
                                                            random_state=CONFIG['random_seed'])

        # Train and evaluate each model
        results = {}
        best_model = None
        best_r2 = -np.inf
        for name, cfg in CONFIG['models'].items():
            logger.info(f"Training {name}...")
            model = train_model(name, cfg['class'], X_train, y_train,
                                param_grid=cfg.get('params') if cfg['params'] else None)
            y_pred = model.predict(X_test)
            mse, r2 = evaluate_model(y_test, y_pred, name)
            results[name] = {'model': model, 'mse': mse, 'r2': r2}
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
            # Plot actual vs predicted for this model
            plot_actual_vs_predicted(y_test, y_pred, name, save=True)
            # Feature importance (if applicable)
            plot_feature_importance(model, X.columns, name, save=True)

        # Save the best model
        import joblib
        joblib.dump(best_model, CONFIG['models_dir'] / CONFIG['best_model_file'])
        logger.info(f"Best model ({best_name}) saved to {CONFIG['models_dir'] / CONFIG['best_model_file']}")

        # Summary table
        print("\n=== Model Comparison ===")
        for name, res in results.items():
            print(f"{name:20} MSE: {res['mse']:6.2f}  R²: {res['r2']:.3f}")
        print(f"\nBest model: {best_name} with R² = {best_r2:.3f}")

    elif args.mode == 'full':
        # Generate data and then train
        logger.info("Running full pipeline: generate data and train models")
        data = generate_data(CONFIG['synthetic_n'], CONFIG['random_seed'], CONFIG['date_start'])
        data.to_csv(CONFIG['data_dir'] / CONFIG['data_file'], index=False)
        logger.info(f"Synthetic data saved to {CONFIG['data_dir'] / CONFIG['data_file']}")

        # Explore data (optional)
        explore_data(data)

        # Train models (reuse train mode logic)
        X = data[['PM2.5', 'NO2', 'Ozone', 'Temperature', 'Humidity']]
        y = data['Respiratory_Cases']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=CONFIG['test_size'],
                                                            random_state=CONFIG['random_seed'])

        results = {}
        best_model = None
        best_r2 = -np.inf
        for name, cfg in CONFIG['models'].items():
            logger.info(f"Training {name}...")
            model = train_model(name, cfg['class'], X_train, y_train,
                                param_grid=cfg.get('params') if cfg['params'] else None)
            y_pred = model.predict(X_test)
            mse, r2 = evaluate_model(y_test, y_pred, name)
            results[name] = {'model': model, 'mse': mse, 'r2': r2}
            if r2 > best_r2:
                best_r2 = r2
                best_model = model
                best_name = name
            plot_actual_vs_predicted(y_test, y_pred, name, save=True)
            plot_feature_importance(model, X.columns, name, save=True)

        import joblib
        joblib.dump(best_model, CONFIG['models_dir'] / CONFIG['best_model_file'])
        logger.info(f"Best model ({best_name}) saved to {CONFIG['models_dir'] / CONFIG['best_model_file']}")

        print("\n=== Model Comparison ===")
        for name, res in results.items():
            print(f"{name:20} MSE: {res['mse']:6.2f}  R²: {res['r2']:.3f}")
        print(f"\nBest model: {best_name} with R² = {best_r2:.3f}")

if __name__ == '__main__':
    main()
