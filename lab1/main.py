"""
MDI3003 - Lab 01: House Price Prediction & Multiple Regression Analysis
========================================================================
This script performs an end-to-end machine learning pipeline for house price prediction
on two datasets:
1. California Housing Dataset (Regression models & tuning)
2. Ames Housing Dataset (High-dimensional regression & tree ensembles/XGBoost)

Run directly via terminal:
    python main.py
"""

import os
import sys
import warnings

# Ensure UTF-8 output encoding for Windows command line compatibility
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing, fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Suppress warnings for clean terminal output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

# Ensure images directory exists for output plots
OS_IMAGES_DIR = "images"
os.makedirs(OS_IMAGES_DIR, exist_ok=True)

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_section(title):
    print("\n" + "-" * 50)
    print(f"[*] {title}")
    print("-" * 50)

# ==============================================================================
# PART 1: CALIFORNIA HOUSING DATASET ANALYSIS
# ==============================================================================

def run_california_housing_analysis():
    print_header("PART 1: CALIFORNIA HOUSING DATASET ANALYSIS")

    # 1. Load Data
    print_section("1. Loading California Housing Dataset")
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame.rename(columns={'MedHouseVal': 'Price'})

    print(f"Dataset Loaded Successfully! Shape: {df.shape} (rows, cols)")
    print(f"Features (8): {list(df.columns[:-1])}")
    print(f"Target: Price ($100,000s)")

    # Data Quality Audit
    missing_vals = df.isna().sum().sum()
    duplicate_rows = df.duplicated().sum()
    print(f"Missing Values: {missing_vals} | Duplicate Rows: {duplicate_rows}")

    # 2. Train-Test Split (80/20)
    X = df.drop(columns=['Price'])
    y = df['Price'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED
    )
    print(f"Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")

    # Train DF for correlation analysis
    train_df = X_train.copy()
    train_df['Price'] = y_train

    # Correlation Analysis
    corr_matrix = train_df.corr()
    price_corr = corr_matrix['Price'].sort_values(ascending=False)
    print("\nTop Feature Correlations with Price:")
    print(price_corr.round(4).to_string())

    # 3. Model 1: Naive Baseline
    print_section("2. Model Building & Evaluation")

    naive = DummyRegressor(strategy='mean')
    naive.fit(X_train, y_train)
    naive_pred = naive.predict(X_test)
    naive_mae = mean_absolute_error(y_test, naive_pred)
    naive_rmse = np.sqrt(mean_squared_error(y_test, naive_pred))
    naive_r2 = r2_score(y_test, naive_pred)

    # Model 2: Simple Linear Regression (MedInc)
    simple_feature = 'MedInc'
    simple_model = LinearRegression()
    simple_model.fit(X_train[[simple_feature]], y_train)
    simple_pred = simple_model.predict(X_test[[simple_feature]])
    simple_mae = mean_absolute_error(y_test, simple_pred)
    simple_rmse = np.sqrt(mean_squared_error(y_test, simple_pred))
    simple_r2 = r2_score(y_test, simple_pred)

    # Model 3: Multiple Linear Regression (with StandardScaler)
    numeric_features = X_train.columns.tolist()
    preprocess = ColumnTransformer([
        ('num', Pipeline([('scaler', StandardScaler())]), numeric_features)
    ])

    linear_pipeline = Pipeline([
        ('preprocess', preprocess),
        ('model', LinearRegression())
    ])
    linear_pipeline.fit(X_train, y_train)
    linear_pred = linear_pipeline.predict(X_test)
    linear_mae = mean_absolute_error(y_test, linear_pred)
    linear_rmse = np.sqrt(mean_squared_error(y_test, linear_pred))
    linear_r2 = r2_score(y_test, linear_pred)

    # Model 4: Ridge Regression (Tuned)
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    ridge_pipe = Pipeline([('preprocess', preprocess), ('model', Ridge())])
    ridge_grid = {'model__alpha': np.logspace(-3, 3, 10)}
    ridge_search = GridSearchCV(ridge_pipe, ridge_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    ridge_search.fit(X_train, y_train)

    best_ridge = ridge_search.best_estimator_
    ridge_pred = best_ridge.predict(X_test)
    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_pred))
    ridge_r2 = r2_score(y_test, ridge_pred)

    # Model 5: Lasso Regression (Tuned)
    lasso_pipe = Pipeline([('preprocess', preprocess), ('model', Lasso(max_iter=50000, random_state=SEED))])
    lasso_grid = {'model__alpha': np.logspace(-4, 0, 10)}
    lasso_search = GridSearchCV(lasso_pipe, lasso_grid, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    lasso_search.fit(X_train, y_train)

    best_lasso = lasso_search.best_estimator_
    lasso_pred = best_lasso.predict(X_test)
    lasso_mae = mean_absolute_error(y_test, lasso_pred)
    lasso_rmse = np.sqrt(mean_squared_error(y_test, lasso_pred))
    lasso_r2 = r2_score(y_test, lasso_pred)

    # Model 6: Random Forest Regressor (Tuned)
    rf_pipe = Pipeline([('preprocess', preprocess), ('model', RandomForestRegressor(random_state=SEED, n_jobs=-1))])
    rf_grid = {
        'model__n_estimators': [100, 200],
        'model__max_depth': [10, 15],
        'model__min_samples_leaf': [2, 4]
    }
    rf_search = GridSearchCV(rf_pipe, rf_grid, scoring='neg_root_mean_squared_error', cv=3, n_jobs=-1)
    rf_search.fit(X_train, y_train)

    best_rf = rf_search.best_estimator_
    rf_pred = best_rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    rf_r2 = r2_score(y_test, rf_pred)

    # Model Comparison Table
    comparison_df = pd.DataFrame({
        'Model': ['Naive Baseline', 'Simple Linear', 'Multiple Linear', 'Ridge (Tuned)', 'Lasso (Tuned)', 'Random Forest'],
        'MAE': [naive_mae, simple_mae, linear_mae, ridge_mae, lasso_mae, rf_mae],
        'RMSE': [naive_rmse, simple_rmse, linear_rmse, ridge_rmse, lasso_rmse, rf_rmse],
        'R²': [naive_r2, simple_r2, linear_r2, ridge_r2, lasso_r2, rf_r2]
    })
    baseline_rmse = comparison_df.loc[0, 'RMSE']
    comparison_df['RMSE_Improvement_%'] = ((baseline_rmse - comparison_df['RMSE']) / baseline_rmse) * 100

    print("\nCalifornia Housing Model Performance Summary:")
    print(comparison_df.round(4).to_string(index=False))

    # 4. Cross-Validation Results & Overfitting Check
    print_section("3. 5-Fold Cross-Validation Results")
    rf_best_params = {k.replace('model__', ''): v for k, v in rf_search.best_params_.items()}
    models_cv = {
        'Linear Regression': LinearRegression(),
        'Ridge': Ridge(alpha=ridge_search.best_params_['model__alpha']),
        'Lasso': Lasso(alpha=lasso_search.best_params_['model__alpha'], max_iter=50000, random_state=SEED),
        'Random Forest': RandomForestRegressor(**rf_best_params, random_state=SEED, n_jobs=-1)
    }

    cv_results = {}
    for name, model in models_cv.items():
        pipe = Pipeline([('preprocess', preprocess), ('model', model)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
        cv_results[name] = {
            'CV_RMSE_Mean': -scores.mean(),
            'CV_RMSE_Std': scores.std(),
            'CV_RMSE_Min': -scores.max(),
            'CV_RMSE_Max': -scores.min()
        }
    cv_df = pd.DataFrame(cv_results).T
    print(cv_df.round(4).to_string())

    # Generate and Save Plots for California Housing
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, rf_pred, alpha=0.4, s=20, color='steelblue')
    min_val, max_val = min(y_test.min(), rf_pred.min()), max(y_test.max(), rf_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price ($100,000s)')
    ax.set_ylabel('Predicted Price ($100,000s)')
    ax.set_title('California Housing: Actual vs Predicted (Random Forest)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OS_IMAGES_DIR, 'california_actual_vs_predicted.png'))
    plt.close()

    print("\n[+] Saved diagnostic plot to 'images/california_actual_vs_predicted.png'")


# ==============================================================================
# PART 2: AMES HOUSING DATASET ANALYSIS
# ==============================================================================

def run_ames_housing_analysis():
    print_header("PART 2: AMES HOUSING DATASET ANALYSIS")

    print_section("1. Loading Ames Housing Dataset (OpenML ID: 42165)")
    try:
        ames = fetch_openml(data_id=42165, as_frame=True, parser='auto')
    except Exception as e:
        print(f"Error loading OpenML dataset: {e}")
        return

    df = ames.frame
    target_col = 'SalePrice'
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float)

    print(f"Dataset Loaded Successfully! Shape: {df.shape} (rows, cols)")
    print(f"Samples: {len(df)} | Features: {len(X.columns)}")
    print(f"Target Range: ${y.min():,.0f} - ${y.max():,.0f} | Mean: ${y.mean():,.0f}")

    # Preprocessing
    print_section("2. Data Preprocessing & Categorical Encoding")
    for col in X.select_dtypes(include=['category', 'object']).columns:
        X[col] = X[col].astype('category').cat.codes

    X = X.fillna(X.median())
    y_log = np.log1p(y)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_log, test_size=0.2, random_state=SEED)
    y_test_orig = np.expm1(y_test)

    # Model 1: Random Forest
    print_section("3. Model Building & Comparison")
    rf = RandomForestRegressor(n_estimators=200, max_depth=15, min_samples_leaf=4, random_state=SEED, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred_log = rf.predict(X_test)
    rf_pred = np.expm1(rf_pred_log)

    rf_mae = mean_absolute_error(y_test_orig, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test_orig, rf_pred))
    rf_r2 = r2_score(y_test_orig, rf_pred)

    # Model 2: XGBoost Regressor
    xgb_installed = False
    try:
        from xgboost import XGBRegressor
        xgb_installed = True
    except ImportError:
        print("XGBoost library not found. Installing or falling back...")

    if xgb_installed:
        xgb = XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=SEED,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        xgb_pred_log = xgb.predict(X_test)
        xgb_pred = np.expm1(xgb_pred_log)

        xgb_mae = mean_absolute_error(y_test_orig, xgb_pred)
        xgb_rmse = np.sqrt(mean_squared_error(y_test_orig, xgb_pred))
        xgb_r2 = r2_score(y_test_orig, xgb_pred)

        comparison_df = pd.DataFrame({
            'Model': ['Random Forest', 'XGBoost'],
            'MAE ($)': [rf_mae, xgb_mae],
            'RMSE ($)': [rf_rmse, xgb_rmse],
            'R²': [rf_r2, xgb_r2]
        })
    else:
        comparison_df = pd.DataFrame({
            'Model': ['Random Forest'],
            'MAE ($)': [rf_mae],
            'RMSE ($)': [rf_rmse],
            'R²': [rf_r2]
        })

    print("\nAmes Housing Model Performance Comparison:")
    print(comparison_df.round(2).to_string(index=False))

    # Feature Importance (XGBoost or Random Forest)
    print_section("4. Top Feature Importances")
    if xgb_installed:
        imp_values = xgb.feature_importances_
        model_name = "XGBoost"
    else:
        imp_values = rf.feature_importances_
        model_name = "Random Forest"

    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': imp_values
    }).sort_values('Importance', ascending=False)

    print(f"Top 10 Features by Importance ({model_name}):")
    print(importance_df.head(10).to_string(index=False))

    # Save Ames plot
    fig, ax = plt.subplots(figsize=(10, 6))
    pred_to_plot = xgb_pred if xgb_installed else rf_pred
    ax.scatter(y_test_orig, pred_to_plot, alpha=0.4, s=20, color='coral')
    min_v, max_v = min(y_test_orig.min(), pred_to_plot.min()), max(y_test_orig.max(), pred_to_plot.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Price ($)')
    ax.set_ylabel('Predicted Price ($)')
    ax.set_title(f'Ames Housing: Actual vs Predicted ({model_name})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OS_IMAGES_DIR, 'ames_actual_vs_predicted.png'))
    plt.close()

    print("\n[+] Saved diagnostic plot to 'images/ames_actual_vs_predicted.png'")


# ==============================================================================
# MAIN EXECUTION ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    print_header("STARTING HOUSE PRICE PREDICTION LAB PIPELINE")
    run_california_housing_analysis()
    run_ames_housing_analysis()
    print_header("LAB PIPELINE EXECUTION COMPLETE! ALL RESULTS & PLOTS GENERATED SUCCESSFULLY.")
