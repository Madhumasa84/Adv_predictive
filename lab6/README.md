# MDI3003 Lab 06: Time-Series Analysis and Forecasting of Reported Crime Incidents by Time and Location using AR and ARIMA Models

A comprehensive, reproducible predictive analytics system developed in accordance with the **MDI3003 Lab 06 Manual (Fall 2026)**:
1. **Official Open Data Portal Dataset**: City of Chicago Public Safety Portal (Chicago Police Department, 2018–2023, 76,175 records for District 1, 90,297 records for District 8) ingested via Socrata SODA API URL (`https://data.cityofchicago.org/resource/ijzp-q8t2.csv`).
2. **Regular Time-Series Construction**: Weekly incident count aggregation (`W-MON`, 314 continuous weeks) with zero-fill handling for non-event periods, verified for monotonicity and uniqueness.
3. **Leakage-Safe Chronological Partitioning**: 294-week Training History → 8-week Validation Window → 12-week Locked Test Horizon (Zero lookahead / No shuffling).
4. **Stationarity & Autocorrelation Diagnostics**: Augmented Dickey-Fuller (ADF) unit-root test (Original $p=0.2520 \rightarrow$ Differenced $p=0.0000$) and ACF/PACF analysis conducted strictly on training evidence.
5. **Autoregressive Baseline**: `AutoReg` AR(4) ($p=4$ motivated by training PACF lag decay).
6. **Information-Theoretic Model Selection**: 6-candidate ARIMA grid ranked by training Akaike Information Criterion (AIC). Selected optimal specification: `ARIMA(0, 1, 1)` ($\text{AIC}=3281.49$).
7. **Locked Holdout Horizon Evaluation (12 Weeks)**:
   - **Naive Persistence Baseline**: $\text{MAE} = 21.33$, $\text{RMSE} = 26.29$
   - **AutoReg AR(4)**: $\text{MAE} = 24.22$, $\text{RMSE} = 27.84$, $\text{AIC} = 3252.64$
   - **ARIMA(0, 1, 1)**: $\text{MAE} = 28.01$, $\text{RMSE} = 35.66$, $\text{AIC} = 3281.49$, 95% Confidence Intervals calculated
8. **Diagnostic Residual Audit**: Ljung-Box test at lag 5 ($p=0.7741$), lag 10 ($p=0.9775$), and lag 20 ($p=0.9988$) confirming white-noise residuals with zero unmodeled serial correlation.
9. **Walk-Forward Rolling-Origin Backtesting**: 22 expanding-window evaluation folds yielding mean $\text{MAE} = 29.84 \pm 25.74$.
10. **Cross-Spatial Generalization Replication**: Replicated identical pipeline on Chicago Police District 8 (West Side / High Volume, $\bar{y}=287.57$ incidents/week), achieving $\text{Test MAE} = 38.54$ and $\text{Test RMSE} = 44.11$.
11. **Seasonal Modeling Extension**: $\text{SARIMA}(1,1,1)(1,0,1,52)$ capturing annual summer crime peaks ($\text{MAE} = 25.04$, $\text{RMSE} = 28.95$, $\text{AIC} = 2653.81$).

---

## Quick Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Machine Learning & Time-Series Pipeline
```bash
python main.py
```

### 3. Launch Interactive Jupyter Notebooks
```bash
jupyter notebook 23MID0444_Lab06_Crime_AR_ARIMA.ipynb
# or
jupyter notebook lab6da.ipynb
```

---

## Repository Structure

```
lab6/
├── 23MID0444_Lab06_Report.pdf                   # Master PDF Laboratory Technical Report
├── 23MID0444_Lab06_Crime_AR_ARIMA.ipynb         # Official assessed student notebook (100% executed, 1.04 MB)
├── lab6da.ipynb                                 # Standardized interactive Jupyter notebook
├── 23MID0444_Lab06_README.md                    # Standardized submission documentation
├── README.md                                    # Project documentation
├── 23MID0444_Lab06_Model_Comparison.csv         # Locked test evaluation metrics
├── 23MID0444_Lab06_Test_Predictions.csv         # 12-week test actuals and all model forecasts + 95% CI
├── 23MID0444_Lab06_Location_Comparison.csv      # District 1 vs District 8 spatial comparison
├── 23MID0444_Lab06_Rolling_Origin.csv           # 22-fold walk-forward validation results
├── 23MID0444_Lab06_Manifest.json                # Complete experiment configuration & reproducibility manifest
├── main.py                                      # End-to-end reproducible time-series ML pipeline
├── requirements.txt                             # Python package dependencies
├── asi_6qn.pdf                                  # Faculty Laboratory Manual (Dr. Durgesh Kumar)
└── lab06_outputs/                               # Structured output directory
    ├── manifest.json
    ├── artifacts/
    │   ├── versions.json
    │   ├── dataset_card.json
    │   ├── weekly_crime_series.csv
    │   ├── chicago_crimes_district001_extract.csv
    │   ├── chicago_crimes_district008_extract.csv
    │   └── responsible_analytics_statement.txt
    ├── figures/
    │   ├── eda_series_overview.png
    │   ├── chronological_split.png
    │   ├── acf_pacf_diagnostics.png
    │   ├── forecast_comparison.png
    │   ├── residual_diagnostics.png
    │   ├── rolling_origin_errors.png
    │   ├── two_location_comparison.png
    │   ├── sarima_advanced.png
    │   └── model_comparison_bars.png
    ├── models/
    │   ├── ar_model.joblib
    │   └── arima_model.joblib
    └── results/
        ├── adf_stationarity_results.csv
        ├── arima_candidate_comparison.csv
        ├── model_comparison.csv
        ├── test_predictions.csv
        ├── test_results.csv
        ├── rolling_origin_backtesting.csv
        ├── location_comparison.csv
        ├── ljung_box_results.csv
        └── sarima_advanced_comparison.csv
```

---

## Dataset Profile & Governance

| Metadata Dimension | Specification |
| :--- | :--- |
| **Dataset Name** | Chicago Crimes — 2001 to Present (D1 Extract) |
| **Source Agency** | Chicago Police Department / City of Chicago Data Portal |
| **API Endpoint** | `https://data.cityofchicago.org/resource/ijzp-q8t2.csv` |
| **Time Horizon** | January 1, 2018 to December 31, 2023 (6 Complete Calendar Years) |
| **SHA-256 Digest** | `16c3c052a42087a165d4be858a79e5ad25907df80aedfa0e9fc4aa7b1ae101f9` |
| **Total Ingested Records** | 76,175 incident rows (District 1), 90,297 rows (District 8) |
| **Temporal Frequency** | Weekly Monday-anchored (`W-MON`, 314 periods) |
| **Zero-Count Intervals** | Filled with 0 (true absence of recorded events, not missing data) |
| **Spatial Scope** | Police District 001 (Central) core; District 008 (West Side) replication |
| **Privacy Safeguards** | Zero individual PII; aggregated geographic district level only |

---

## Model Benchmark Results

### 1. Training-Period Model Selection (AIC Comparison)

| Candidate Order | Training AIC | Training BIC | Validation MAE | Validation RMSE | Selection Status |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **ARIMA(0, 1, 1)** | **3281.49** | **3288.91** | 38.86 | 43.25 | **Selected (Minimum AIC)** |
| **ARIMA(1, 1, 1)** | 3283.30 | 3294.42 | 42.60 | 46.63 | Candidate |
| **ARIMA(2, 1, 1)** | 3284.23 | 3299.06 | 44.31 | 48.65 | Candidate |
| **ARIMA(2, 0, 0)** | 3337.11 | 3351.95 | 19.53 | 31.82 | Candidate |
| **ARIMA(1, 1, 0)** | 3348.17 | 3355.59 | 91.96 | 94.36 | Candidate |
| **ARIMA(1, 0, 0)** | 3373.83 | 3384.97 | 40.91 | 45.54 | Candidate |

### 2. Locked-Test Horizon Performance (12 Weeks)

| Model Architecture | Order / Lags | Train AIC | Validation MAE | Test MAE | Test RMSE | Ljung-Box p (Lag 5) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Naive (Persistence)** | Last Observed | — | 16.75 | **21.33** | **26.29** | — |
| **AutoReg AR(4)** | $p=4$ | 3252.64 | 20.22 | **24.22** | **27.84** | — |
| **ARIMA(0, 1, 1)** | $(0, 1, 1)$ | 3281.49 | 38.86 | **28.01** | **35.66** | **0.7741** |
| **SARIMA(1,1,1)(1,0,1,52)** | Seasonal $s=52$ | 2653.81 | — | **25.04** | **28.95** | 0.8120 |

---

## Responsible Analytics & Ethical Interpretation

1. **Administrative Measurement**: Forecasts capture counts of *reported and recorded incidents*, which reflect reporting propensity, enforcement allocation, and recording protocols rather than actual unobserved crime prevalence.
2. **Spatial Modeling Integrity**: Geographic districts define distinct time series; location is never injected as an arbitrary numeric feature into a univariate ARIMA model.
3. **Temporal Drift**: Macro disruptions (e.g., pandemic stay-at-home orders) induce structural breaks, requiring walk-forward rolling-origin recalibration rather than static long-term deployment.
4. **Count-Data Boundaries**: Continuous Gaussian assumptions in ARIMA can theoretically generate negative point forecasts in sparse regimes, which must be documented transparently as modeling limitations.
5. **Deployment Boundary**: These models are strictly intended for macro-level academic analysis and resource planning; they must **never** be used for individual risk scoring, predictive profiling, or autonomous punitive dispatch.

---

## Author & Academic Information

- **Student:** Madhusudhanan G (Registration No: `23MID0444`)
- **Course:** MDI3003 - Advanced Predictive Analytics
- **Faculty Instructor:** Dr. Durgesh Kumar, Assistant Professor (Senior), SCOPE
- **School:** School of Computer Science and Engineering (SCOPE), VIT Vellore
- **Laboratory Manual:** Experiment 06 — Fall Semester 2026-2027
- **Repository:** [Madhumasa84/Adv_predictive](https://github.com/Madhumasa84/Adv_predictive/tree/main/lab6)
