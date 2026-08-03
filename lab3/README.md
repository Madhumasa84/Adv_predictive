# MDI3003 Lab 03: Benchmark-Aligned Multi-Dataset Email Classification & LLM Automatic Response Drafting

A comprehensive, reproducible predictive analytics system implementing:
1. **Multi-Dataset Email Classification** on 3 email datasets (Business Intent D1, Enron Spam D2, SpamAssassin D3).
2. **Classifier Benchmark**: Dummy Baseline, Multinomial Naive Bayes, Complement Naive Bayes, Logistic Regression, Linear SVC, K-Nearest Neighbors (KNN), and a **Trainable Word-Embedding BiLSTM** text classifier (Manual Rev 3.1).
3. **Leakage-Safe Evaluation**: 5-Fold Stratified Cross-Validation & Corrected Model Selection on training data only.
4. **Selective Prediction & Review Routing**: Decision score / margin calculations ($p_1 - p_2$), low-confidence flags ($<0.15$), mandatory review escalation for `urgent_action`, and automatic spam reply suppression.
5. **LLM API Automatic Draft Generation**: PII redaction (`[EMAIL_REDACTED]`, `[PHONE_REDACTED]`), prompt injection defense (`<email_data>` tags), and JSON audit log storage.

---

## Repository Structure

```
lab3/
├── main.py                      # Executable standalone pipeline
├── lab3da.ipynb                 # Interactive Jupyter notebook
├── requirements.txt             # Project dependencies
├── README.md                    # Project documentation
├── Lab03_report.pdf             # Official Lab 03 System Report
├── MDI3003_Lab03...Manual.pdf  # Instructor Laboratory Manual (Rev 3.1)
└── outputs/                     # Generated artifacts & outputs
    ├── cv_results_all.csv       # 5-Fold CV metrics for all models
    ├── test_results_all.csv     # Locked holdout test metrics
    ├── cross_dataset_transfer.csv# Enron <-> SpamAssassin transfer test
    ├── dataset_summary.csv      # Data audit summary
    ├── draft_quality_ratings.csv# Human draft evaluation worksheet
    ├── figures/                 # Plot figures
    │   ├── email_class_distributions.png
    │   ├── email_cv_performance.png
    │   ├── email_model_heatmap.png
    │   └── email_confusion_matrices.png
    ├── models/                  # Saved fitted model binaries (.joblib)
    └── drafts/                  # Local JSON audit logs for generated drafts
```

---

## Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Complete Pipeline
```bash
python main.py
```

### 3. Open Interactive Notebook
```bash
jupyter notebook lab3da.ipynb
```

---

## Dataset Summary

| Dataset ID | Domain / Task | Samples | Classes | Prevalence | Median Text Length |
|------------|---------------|---------|---------|------------|-------------------|
| **business_intent (D1)** | Multiclass Intent | 800 | 6 (`request`, `meeting`, `complaint`, `information`, `urgent_action`, `spam`) | 24.4% (`request`) | 115 chars |
| **enron_spam (D2)** | Binary Spam | 500 | 2 (`legitimate`, `spam`) | 50.0% (`spam`) | 151 chars |
| **spamassassin (D3)** | Binary Spam | 400 | 2 (`legitimate`, `spam`) | 50.0% (`spam`) | 161 chars |

---

## Model Performance Results

### 1. 5-Fold Stratified Cross-Validation (Macro F1)

| Model | Business Intent (D1) | Enron Spam (D2) | SpamAssassin (D3) |
|-------|----------------------|-----------------|-------------------|
| **Multinomial Naive Bayes** | **1.0000** | **1.0000** | **1.0000** |
| **Complement Naive Bayes** | **1.0000** | **1.0000** | **1.0000** |
| **Logistic Regression** | **1.0000** | **1.0000** | **1.0000** |
| **Linear SVC** | **1.0000** | **1.0000** | **1.0000** |
| **K-Nearest Neighbors (k=15)** | **1.0000** | **1.0000** | **1.0000** |
| **Word-Embedding BiLSTM** | **0.9862** | N/A | N/A |
| Dummy Majority Baseline | 0.0699 | 0.3333 | 0.3333 |

### 2. Locked Holdout Test Evaluation

| Dataset ID | Selected Best Model | Test Accuracy | Test Macro F1 | Test Weighted F1 |
|------------|--------------------|---------------|---------------|------------------|
| **business_intent** | MultinomialNB | **1.0000** | **1.0000** | **1.0000** |
| **enron_spam** | MultinomialNB | **1.0000** | **1.0000** | **1.0000** |
| **spamassassin** | MultinomialNB | **1.0000** | **1.0000** | **1.0000** |

### 3. Cross-Dataset Spam Transfer Test

| Train Dataset | Test Dataset | Model | Accuracy | Macro F1 |
|---------------|--------------|-------|----------|----------|
| **Enron Spam (D2)** | **SpamAssassin (D3)** | LinearSVC | **1.0000** | **1.0000** |
| **SpamAssassin (D3)** | **Enron Spam (D2)** | LinearSVC | **1.0000** | **1.0000** |

---

## Selective Routing & Draft Generation Policies

1. **Spam Suppression**: Spam messages receive **zero** reply draft (`status: suppressed`).
2. **Mandatory Human Review**: Triggered if margin ($p_1 - p_2$) is $< 0.15$ or predicted class is `urgent_action`.
3. **PII Redaction**: Email addresses and phone numbers masked before external API call (`[EMAIL_REDACTED]`, `[PHONE_REDACTED]`).
4. **Prompt Injection Isolation**: Email text wrapped inside `<email_data>` tags with system instructions forbidding invented commitments or dates.
5. **No Auto-Send Guarantee**: System generates reviewable local drafts stored in `outputs/drafts/*.json`.

---

## Author & Course Info

- **Student**: Madhusudhanan G (23MID0444)
- **Course**: MDI3003 - Advanced Predictive Analytics
- **Manual Revision**: 3.1 (28 July 2026)
- **Repository**: [Madhumasa84/Adv_predictive](https://github.com/Madhumasa84/Adv_predictive/tree/main/lab3)
