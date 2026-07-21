# House Price Prediction & Regression Modeling

An end-to-end predictive machine learning lab project that implements, tunes, and evaluates multiple regression models on two distinct datasets: **California Housing** and **Ames Housing**.

---

##  Quick Execution Guide (Faculty & Evaluator Instructions)

To execute the entire machine learning pipeline, train/tune all regression models, print full performance evaluation tables, and generate diagnostic plot artifacts for **both datasets**, simply run the standalone Python script from your terminal:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the complete pipeline script
python main.py
```

### What `main.py` executes:
- **Part 1 (California Housing)**: Data audit, correlation analysis, 80/20 split, training/tuning 6 regression models (Baseline, Simple Linear, Multiple Linear, Ridge, Lasso, Random Forest), 5-Fold Cross-Validation, and saves `images/california_actual_vs_predicted.png`.
- **Part 2 (Ames Housing)**: OpenML data load, categorical encoding, median imputation, log transformation $\log(1+y)$, training Random Forest & **XGBoost** models, feature importance calculation, and saves `images/ames_actual_vs_predicted.png`.

Alternatively, you can open and run the interactive Jupyter Notebook:
```bash
jupyter notebook Lab__1.ipynb
```

---

##  Project Overview

This repository contains comprehensive exploratory data analysis (EDA), data cleaning, feature engineering, model selection, hyperparameter tuning, cross-validation, and diagnostic analysis across two real-world housing datasets:

1. **California Housing Analysis**: Focuses on baseline regressor, simple/multiple linear regressions, L1/L2 regularized regressions (Lasso & Ridge), and Random Forest.
2. **Ames Housing Analysis**: Advanced analysis handling high-dimensional data, missing value imputation, log-target transformation, Random Forest, and Gradient Boosting (**XGBoost**).

### Machine Learning Pipeline Workflow
- **Data Audit & Quality Check**: Inspection of data types, missing values, duplicates, and statistical distribution.
- **Exploratory Data Analysis (EDA)**: Target distribution analysis, correlation matrix heatmaps, and feature-target relationship scatter plots with trend lines.
- **Data Preprocessing & Feature Engineering**: Standard scaling via Scikit-Learn Pipelines, categorical encoding, median imputation, and log transformation ($\log(1+y)$).
- **Model Training & Hyperparameter Tuning**: 5-Fold Cross-Validation (`GridSearchCV`) for regularized models and tree ensembles.
- **Diagnostic & Residual Analysis**: Residuals vs. Fitted plots, Q-Q plots, price-segment error distribution, and train-test gap analysis (overfitting/underfitting).

---

##  Repository Structure

```
.
├── main.py              # Standalone Python script to run complete pipeline & output all results/plots
├── Lab__1.ipynb         # Main Jupyter Notebook containing complete interactive analysis
├── requirements.txt     # Python environment requirements file
├── images/              # Generated diagnostic plot images (created upon running main.py)
└── README.md            # Comprehensive project documentation
```

---

##  Dataset 1: California Housing Dataset

### Summary Statistics
- **Samples**: 20,640 (16,512 Train / 4,128 Test split — 80/20 ratio)
- **Features**: 8 numerical features (`MedInc`, `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`)
- **Target Variable**: `Price` (Median House Value in units of $100,000s)
  - **Mean**: $2.07 ($206,856)
  - **Median**: $1.80 ($179,700)
  - **Range**: $0.15 to $5.00 ($14,999 to $500,001)
  - **Distribution**: Right-skewed with a ceiling cap at $5.00.

### Top Feature Correlations with Price
| Feature | Description | Correlation with Price |
| :--- | :--- | :---: |
| **MedInc** | Median Income in block group | **+0.6906** (Strongest predictor) |
| **AveRooms** | Average number of rooms | +0.1585 |
| **HouseAge** | Median house age | +0.1037 |
| **AveOccup** | Average household members | -0.0220 |
| **Population** | Block group population | -0.0260 |
| **Longitude** | Block group longitude | -0.0463 |
| **AveBedrms** | Average number of bedrooms | -0.0514 |
| **Latitude** | Block group latitude | -0.1430 |

### California Housing Model Performance Comparison

| Model | Hyperparameters / Details | MAE ($100k) | RMSE ($100k) | $R^2$ Score | RMSE Improvement % |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Naive Baseline** | Strategy: `mean` | 0.9061 | 1.1449 | -0.0002 | Baseline (0%) |
| **Simple Linear Regression** | Single Feature (`MedInc`) | 0.6299 | 0.8421 | 0.4589 | +26.4% |
| **Multiple Linear Regression** | Standard Scaled Features | 0.5332 | 0.7456 | 0.5758 | +34.9% |
| **Ridge Regression** | L2 Regularized ($\alpha = 2.1544$) | 0.5332 | 0.7455 | 0.5758 | +34.9% |
| **Lasso Regression** | L1 Regularized ($\alpha = 0.0008$) | 0.5332 | 0.7448 | 0.5766 | +34.9% |
| **Random Forest Regressor** | `max_depth=15`, `min_samples_leaf=2`, `n_estimators=200` | **0.3304** | **0.5077** | **0.8033** | **+55.7%** |

### 5-Fold Cross-Validation & Generalization Analysis
- **Linear Regression**: CV RMSE Mean = `0.7205` ($\pm 0.0103$) \| Train RMSE: `0.7197` vs Test RMSE: `0.7456` (Reasonable generalization)
- **Ridge Regression**: CV RMSE Mean = `0.7205` ($\pm 0.0103$) \| Train RMSE: `0.7197` vs Test RMSE: `0.7455` (Reasonable generalization)
- **Lasso Regression**: CV RMSE Mean = `0.7205` ($\pm 0.0102$) \| Train RMSE: `0.7197` vs Test RMSE: `0.7448` (Reasonable generalization)
- **Random Forest**: CV RMSE Mean = `0.5123` ($\pm 0.0051$) \| Train RMSE: `0.2672` vs Test RMSE: `0.5077` (Moderate overfitting, but best test accuracy)

### Segment Error Analysis (Random Forest)
| Price Segment | MAE ($100,000s) | Std Error | Max Absolute Error |
| :--- | :---: | :---: | :---: |
| **Low** | 0.2262 | 0.2723 | 2.9602 |
| **Medium-Low** | 0.2533 | 0.2832 | 2.7748 |
| **Medium-High** | 0.2930 | 0.2976 | 2.3920 |
| **High** | 0.5491 | 0.5347 | 3.1973 |

---

##  Dataset 2: Ames Housing Dataset

### Summary Statistics
- **Data Source**: OpenML (Dataset ID: `42165`)
- **Samples**: 1,460 (1,168 Train / 292 Test split — 80/20 ratio)
- **Features**: 80 features (categorical and numerical predictors)
- **Data Processing**: Missing values (6,965 values across 19 columns) handled via median imputation and categorical code encoding. Target variable transformed using $\log(1 + \text{SalePrice})$.
- **Target Variable**: `SalePrice` (in USD)
  - **Mean**: $180,921
  - **Median**: $163,000
  - **Range**: $34,900 to $755,000

### Top Features Correlated with SalePrice
1. **OverallQual** (Overall Material & Finish Quality): `+0.791`
2. **GrLivArea** (Above grade ground living area): `+0.709`
3. **GarageCars** (Garage size in car capacity): `+0.640`
4. **GarageArea** (Garage size in sq ft): `+0.623`
5. **TotalBsmtSF** (Total basement area): `+0.614`
6. **1stFlrSF** (First floor area): `+0.606`

### Ames Housing Model Performance Comparison

| Model | Hyperparameters / Details | MAE ($) | RMSE ($) | $R^2$ Score |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest Regressor** | `n_estimators=200`, `max_depth=15`, `min_samples_leaf=4` | $17,444 | $30,019 | 0.8825 |
| **XGBoost Regressor** | `n_estimators=500`, `learning_rate=0.05`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8` | **$15,761** | **$25,975** | **0.9120** |

### Top Feature Importances (XGBoost Model)
1. **OverallQual**: `0.1998` (19.98%)
2. **GarageQual**: `0.1372` (13.72%)
3. **GarageCars**: `0.0997` (9.97%)
4. **FullBath**: `0.0573` (5.73%)
5. **CentralAir**: `0.0416` (4.16%)
6. **GrLivArea**: `0.0366` (3.66%)

---

##  Summary of Key Findings

1. **Primary Price Drivers**:
   - **California Dataset**: Median Income (`MedInc`) is overwhelmingly the single strongest linear predictor ($r = 0.691$).
   - **Ames Dataset**: Overall Quality (`OverallQual`, importance: $19.98\%$) and Garage Quality/Capacity dominate model decisions.
2. **Linear vs. Non-linear Models**:
   - Linear models (Linear, Ridge, Lasso) achieve $R^2 \approx 0.576$ on California Housing.
   - Non-linear ensemble models significantly outperform linear counterparts:
     - **Random Forest** on California Housing achieves $R^2 = 0.8033$ (a 55.7% reduction in RMSE over baseline).
     - **XGBoost** on Ames Housing achieves $R^2 = 0.9120$ with an RMSE of **$25,975**.
3. **Target Log Transformation**: Log-transforming price targets ($\log(1+y)$) mitigates right-skewness and reduces high-value prediction error leverage.

---

##  Environment & Prerequisites

To run the project (`main.py` or `Lab__1.ipynb`), ensure Python 3.8+ is installed:

```bash
pip install -r requirements.txt
```
