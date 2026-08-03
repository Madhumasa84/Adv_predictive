# MDI3003 Lab 03: Email Classification & Housing Price Prediction

A comprehensive machine learning project combining multi-dataset email classification with housing price prediction using regression, classification, and deep learning models.

## Quick Execution Guide

### 1. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline
```bash
python main.py
```

This will execute the entire pipeline including:
- **Part 1**: Email Classification on 3 datasets with 5 benchmark classifiers
- **Part 2**: Housing Price Prediction with regression, classification, and deep learning models

### Alternative: Run Jupyter Notebook
```bash
jupyter notebook lab3da.ipynb
```

---

## Project Overview

### Part 1: Email Classification Pipeline

A multi-dataset text classification system implementing:
- **3 Email Datasets**: Business Intent, Enron Spam, SpamAssassin
- **5 Classifiers**: Dummy Baseline, Multinomial NB, Complement NB, Logistic Regression, Linear SVC
- **Evaluation**: 5-Fold Stratified Cross-Validation with Accuracy, F1, Precision, Recall metrics
- **Transfer Learning**: Cross-dataset spam detection evaluation

**Architecture Flow**:
```
Data Loading → Audit & Split → TF-IDF Vectorization → 
5-Fold Cross-Validation → Model Selection → 
Test Evaluation → Transfer Analysis → Visualization & Results
```

### Part 2: Housing Price Prediction

A comprehensive housing price prediction system with:
- **Dataset**: California Housing (20,640 samples, 8 features)
- **Regression Models** (7 models):
  1. Naive Baseline
  2. Simple Linear Regression
  3. Multiple Linear Regression
  4. Ridge Regression
  5. Lasso Regression
  6. Random Forest
  7. **BiLSTM (Deep Learning)** - NEW

- **Classification Models** (4 models):
  1. K-Nearest Neighbors (KNN)
  2. Gaussian Naive Bayes
  3. Bernoulli Naive Bayes
  4. Multinomial Naive Bayes

- **Features**: 
  - TF-IDF feature extraction with unigrams and bigrams (email classification)
  - Standard scaling preprocessing
  - Cross-validation for model selection
  - Hyperparameter tuning (k-value optimization for KNN)

---

## Repository Structure

```
.
├── main.py                      # Complete standalone pipeline
├── lab3da.ipynb                 # Interactive Jupyter notebook
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── Lab03_report.pdf            # Lab report
├── outputs/                     # Generated outputs
│   ├── models/                 # Saved model binaries (.joblib)
│   ├── figures/                # Visualizations
│   ├── cv_results_all.csv
│   ├── test_results_all.csv
│   ├── regression_results.csv
│   ├── classification_results.csv
│   ├── feature_importance.csv
│   └── dataset_summary.csv
└── images/                      # Diagnostic plots

```

---

## Dataset Summary

### Email Classification Datasets

| Dataset | Task | Samples | Train/Test | Classes | 
|---------|------|---------|-----------|---------|
| **Business Intent** | Multi-class Intent | 800 | 640/160 | 6 (request, meeting, complaint, information, urgent_action, spam) |
| **Enron Spam** | Binary Classification | 500 | 400/100 | 2 (legitimate, spam) |
| **SpamAssassin** | Binary Classification | 400 | 320/80 | 2 (legitimate, spam) |

### Housing Price Prediction Dataset

| Metric | Value |
|--------|-------|
| **Dataset** | California Housing |
| **Samples** | 20,640 |
| **Features** | 8 (MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude) |
| **Target** | MedHouseVal (Median House Value) |
| **Train/Test** | 80/20 split |

---

## Model Performance Results

### Part 1: Email Classification - Cross-Validation Results

Mean Macro F1 Scores (5-Fold Stratified CV):

| Dataset | Dummy | Multinomial NB | Complement NB | Logistic Reg | Linear SVC |
|---------|-------|----------------|--------------|--------------|-----------|
| **Business Intent** | 0.065 | **1.000** | **1.000** | **1.000** | **1.000** |
| **Enron Spam** | 0.358 | **1.000** | **1.000** | **1.000** | **1.000** |
| **SpamAssassin** | 0.336 | **1.000** | **1.000** | **1.000** | **1.000** |

### Test Set Results

| Dataset | Best Model | Accuracy | Macro F1 | Weighted F1 |
|---------|-----------|----------|----------|-------------|
| **Business Intent** | MultinomialNB | **1.0000** | **1.0000** | **1.0000** |
| **Enron Spam** | MultinomialNB | **1.0000** | **1.0000** | **1.0000** |
| **SpamAssassin** | MultinomialNB | **1.0000** | **1.0000** | **1.0000** |

### Cross-Dataset Transfer

| Train Dataset | Test Dataset | Model | Accuracy | Macro F1 |
|---------------|--------------|-------|----------|----------|
| **Enron Spam** | **SpamAssassin** | LinearSVC | **1.0000** | **1.0000** |
| **SpamAssassin** | **Enron Spam** | LinearSVC | **1.0000** | **1.0000** |

### Part 2: Housing Price Prediction - Regression Models

| Model | MAE | RMSE | R² |
|-------|-----|------|-----|
| Naive Baseline | 0.9061 | 1.1449 | -0.0002 |
| Simple Linear | 0.6299 | 0.8421 | 0.4589 |
| Multiple Linear | 0.5332 | 0.7456 | 0.5758 |
| Ridge | 0.5332 | 0.7456 | 0.5758 |
| Lasso | 0.5331 | 0.7446 | 0.5769 |
| **Random Forest** | **0.3268** | **0.5038** | **0.8063** |
| BiLSTM | 0.5606 | 0.7257 | 0.5981 |

**Best Regression Model**: Random Forest (RMSE: 0.5038, R²: 0.8063)
**Second-Best Regression Model (Deep Learning)**: BiLSTM (RMSE: 0.7257, R²: 0.5981)

### Classification Models (Price Tiers)

| Model | Accuracy |
|-------|----------|
| **KNN (k=16)** | **0.7900** |
| Bernoulli Naive Bayes | 0.6909 |
| Gaussian Naive Bayes | 0.6269 |
| Multinomial Naive Bayes | 0.6221 |

**Best Classification Model**: K-Nearest Neighbors (k=16, Accuracy: 0.7900)

---

## Key Features

### Part 1: Email Classification
- ✅ Multi-dataset evaluation (3 different email datasets)
- ✅ 5-Fold Stratified Cross-Validation
- ✅ TF-IDF n-gram feature extraction (unigrams + bigrams)
- ✅ 5 diverse classifiers (baseline + NB variants + linear models)
- ✅ Cross-dataset transfer learning evaluation
- ✅ Comprehensive performance metrics
- ✅ Confusion matrices & visualization

### Part 2: Housing Price Prediction
- ✅ 7 Regression models (including BiLSTM)
- ✅ 4 Classification models (multi-class price tier prediction)
- ✅ KNN hyperparameter optimization
- ✅ Naive Bayes variants (Gaussian, Bernoulli, Multinomial)
- ✅ Deep learning with BiLSTM neural networks
- ✅ Feature importance analysis
- ✅ Correlation analysis with target variable
- ✅ Model comparison and visualization

---

## Dependencies

- **Core**: pandas, numpy, scipy
- **ML**: scikit-learn
- **Visualization**: matplotlib, seaborn
- **Deep Learning**: tensorflow
- **Serialization**: joblib
- **NLP**: sentence-transformers, openai (optional)

See `requirements.txt` for detailed version specifications.

---

## Outputs Generated

### CSV Files
- `cv_results_all.csv` - Cross-validation results for all models/datasets
- `test_results_all.csv` - Locked test set evaluation results
- `regression_results.csv` - Regression model comparison
- `classification_results.csv` - Classification model comparison
- `feature_importance.csv` - Random Forest feature importance
- `dataset_summary.csv` - Email dataset summaries

### Model Files
- `business_intent_best_model.joblib`
- `enron_spam_best_model.joblib`
- `spamassassin_best_model.joblib`
- `housing_pipeline.joblib`

### Visualizations
- Class distribution plots
- Cross-validation performance curves
- Model performance heatmaps
- Confusion matrices
- Training history (BiLSTM)
- Feature importance bar charts

---

## Usage Examples

### Running the Pipeline
```bash
python main.py
```

### Loading Trained Models
```python
import joblib
from pathlib import Path

# Load email classifier
model = joblib.load('outputs/models/business_intent_best_model.joblib')
predictions = model.predict(['This is a test email'])

# Load housing pipeline
pipeline = joblib.load('outputs/models/housing_pipeline.joblib')
house_price_prediction = pipeline.predict(X_test)
```

### Working with Results
```python
import pandas as pd

# Load cross-validation results
cv_results = pd.read_csv('outputs/cv_results_all.csv')
print(cv_results.pivot_table(index='dataset', columns='model', values='macro_f1_mean'))

# Load regression results
reg_results = pd.read_csv('outputs/regression_results.csv')
print(reg_results.sort_values('RMSE'))
```

---

## Notes

- All models are trained with consistent random seeds (RANDOM_STATE=42) for reproducibility
- Email datasets may use mock data if OpenML datasets are unavailable
- BiLSTM training uses early stopping (patience=5) for efficiency
- Feature scaling is applied consistently across all models
- Class weights are balanced in classifier training to handle imbalanced datasets

---

## Requirements & Environment

- **Python**: 3.8+
- **scikit-learn**: >= 1.0.0
- **pandas**: >= 1.3.0
- **numpy**: >= 1.21.0
- **tensorflow**: >= 2.10.0
- **matplotlib**: >= 3.4.0
- **seaborn**: >= 0.11.0

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Lab Information

- **Course**: MDI3003 - Advanced Predictive Analytics
- **Lab**: Lab 03
- **Focus**: Multi-Dataset Email Classification & Housing Price Prediction
- **Models**: 5 Email Classifiers + 7 Regression + 4 Classification Models for Housing
- **Evaluation**: Cross-Validation, Locked Test Sets, Transfer Learning

---

## Contact & Support

For questions or issues, please refer to the lab documentation or contact the course instructor.
