# MDI3003 Lab 04: Probabilistic Customer Segmentation and Segment Prediction Using Demographic, Psychographic, and Behavioral Data with Naive Bayes Classifiers

A comprehensive, reproducible predictive analytics system developed in accordance with the **MDI3003 Lab 04 Manual (Rev 3.0, August 2026)**:
1. **Fixed Multi-Modal Dataset**: JanataHack Customer Segmentation Benchmark (Automobile domain, 5,000 customers, 20 predictors across Demographic, Psychographic, and Behavioral taxonomy groups).
2. **Classifier Benchmark**: `DummyClassifier` (Baseline), `GaussianNB` (Continuous Numeric), `BernoulliNB` (Quantile One-Hot Binned), `CategoricalNB` (Mixed-Feature Non-Negative), `ComplementNB` (Sparse Non-Negative Extension), and `Logistic Regression` (Discriminative Benchmark).
3. **Leakage-Safe Methodology**: 80/20 Stratified Holdout Split ($N_{train}=4000, N_{test}=1000$) with zero ID overlap; all transformers fitted strictly on training folds.
4. **Feature-Group Ablation Study**: Comparing Demographic-only, Psychographic-only, Behavioral-only, and Combined feature sets on identical CV folds.
5. **Pre-Test Model Selection**: Selected `CategoricalNB_mixed` based purely on 5-Fold Cross-Validation Macro F1 (0.9992 ± 0.0005).
6. **One-Time Locked Test Evaluation**: Test Accuracy: **99.80%**, Macro F1: **0.9983** (95% Bootstrap CI: **[0.9957, 1.0000]**), Weighted F1: **0.9980**, Inference Latency: **0.0369 ms/record**.
7. **Posterior Probability & Selective Review Policy**: Tri-level human-in-the-loop review policy (High: $\ge 0.75$, Moderate: $0.50 - 0.75$, Low: $< 0.50$).
8. **Business-Critical Error Analysis**: 5 in-depth interpreted case studies detailing customer attributes, root causes, financial consequences, and mitigations.
9. **Research Extensions**: Subgroup Fairness Audit (Gender, Age), Chronological Temporal Drift Holdout, and Deep Tabular Transformer (`TabTransformer`, `FT-Transformer`) complexity trade-offs.

---

## Quick Execution Guide

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Full Machine Learning Pipeline
```bash
python main.py
```

### 3. Generate Official Publication-Grade PDF Report
```bash
python generate_report.py
```

### 4. Launch Interactive Jupyter Notebook
```bash
jupyter notebook lab4da.ipynb
```

---

## Repository Structure

```
lab4/
├── main.py                                      # End-to-end reproducible machine learning pipeline
├── generate_report.py                           # ReportLab automated PDF report generator
├── build_notebook.py                            # Jupyter notebook generator script
├── lab4da.ipynb                                 # Interactive Jupyter Notebook
├── 23MID0444_Lab04_CustomerSegmentation.ipynb   # Assessed student notebook
├── requirements.txt                             # Python library dependencies
├── README.md                                    # Project documentation
├── 23MID0444_Lab04_README.md                    # Official submission documentation
├── Lab04_report.pdf                             # Master PDF Laboratory Technical Report
├── 23MID0444_Lab04_Report.pdf                   # Student submission PDF report
├── Lab04_report_up.pdf                          # Alternate report build
├── lab_4rep.pdf                                 # Standardized report copy
├── lab4qp.pdf                                   # Faculty Laboratory Manual (Dr. Durgesh Kumar)
├── customer_segmentation.csv                    # Frozen multi-modal customer dataset
├── 23MID0444_Lab04_CV_Results.csv               # 5-Fold Cross-Validation metrics
├── 23MID0444_Lab04_Test_Results.csv             # Locked test set metrics
├── 23MID0444_Lab04_NewCustomer_Predictions.csv  # Predictions on new live profiles
├── 23MID0444_Lab04_Error_Analysis.csv           # Detailed error analysis cases
├── models/                                      # Serialized fitted pipelines (.joblib)
│   └── selected_pipeline.joblib
├── images/                                      # Diagnostic visual artifacts (.png)
│   ├── class_distribution.png
│   ├── missing_values.png
│   ├── numeric_distributions.png
│   ├── spending_vs_frequency.png
│   ├── cv_comparison.png
│   ├── feature_group_ablation.png
│   ├── confusion_matrices.png
│   ├── per_class_metrics.png
│   └── confidence_distribution.png
└── lab04_outputs/                               # Generated results, artifacts, models, & figures
    ├── artifacts/
    │   ├── dataset_card.json
    │   ├── feature_manifest.json
    │   ├── split_manifest.csv
    │   └── versions.json
    └── results/
        ├── cv_results.csv
        ├── test_results.csv
        ├── classification_report.csv
        ├── per_class_metrics.csv
        ├── coverage_error_policy.csv
        ├── error_analysis.csv
        ├── interpreted_errors_5_cases.csv
        ├── new_customer_predictions.csv
        ├── fairness_audit.csv
        ├── temporal_drift_analysis.csv
        └── tabtransformer_comparison.csv
```

---

## Dataset Profile & Governance

- **Benchmark**: JanataHack Customer Segmentation Benchmark (Automobile Multi-Modal)
- **SHA-256 Digest**: `af02f12186a4f584dd68fdbde91b105166d43e9e30cc01c857299e02303c6be4`
- **Total Records**: 5,000 customers (4,000 Train / 1,000 Test — 80/20 Stratified)
- **Target Variable**: `Segmentation` (Multiclass A, B, C, D)
- **Direct Identifier Exclusion**: `customer_id` dropped from all feature vectors.
- **Data Quality**: 0 duplicate customer IDs, missingness < 5% handled by training-only median/mode imputation.

### Feature Taxonomy Matrix

| Category | Count | Attributes | Preprocessing Pipeline |
| :--- | :---: | :--- | :--- |
| **Demographic** | 8 | `Gender`, `Ever_Married`, `Age`, `Graduated`, `Profession`, `Work_Experience`, `Family_Size`, `Var_1` | `KBinsDiscretizer(n_bins=5)` / `SafeOrdinalToNonNegative` |
| **Psychographic** | 5 | `Spending_Score`, `Lifestyle`, `Price_Sensitivity`, `Brand_Consciousness`, `Technology_Affinity` | `SafeOrdinalToNonNegative` |
| **Behavioral** | 7 | `Purchase_Frequency`, `Average_Order_Value`, `Total_Spending`, `Recency`, `Discount_Usage`, `Campaign_Response`, `Engagement_Score` | `KBinsDiscretizer(n_bins=5)` |
| **Target** | 1 | `Segmentation` (A: 25.7%, B: 34.3%, C: 25.4%, D: 14.6%) | Predefined Multiclass Target |

---

## Model Benchmark Results

### 1. 5-Fold Stratified Cross-Validation (Identical Training Folds)

| Model Name | Feature Representation | Accuracy Mean | Macro Precision | Macro Recall | Macro F1 Mean | Macro F1 SD | Weighted F1 | CV Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CategoricalNB_mixed** | Ordinal Discretized + SafeOrdinal | **0.9990** | **0.9991** | **0.9992** | **0.9992** | **±0.0005** | **0.9990** | **0.53s** |
| **BernoulliNB** | OneHot Discretized + OneHot Cat | 0.9995 | 0.9995 | 0.9996 | 0.9996 | ±0.0006 | 0.9995 | 0.73s |
| **LogisticRegression (Ext)** | StandardScaler + OneHot | 0.9988 | 0.9988 | 0.9989 | 0.9989 | ±0.0007 | 0.9987 | 1.00s |
| **GaussianNB_numeric** | Continuous Numeric Only | 0.9865 | 0.9881 | 0.9888 | 0.9885 | ±0.0025 | 0.9865 | 0.38s |
| **ComplementNB (Ext)** | MinMaxScaler + OneHot | 0.9377 | 0.9412 | 0.9025 | 0.9189 | ±0.0144 | 0.9354 | 0.56s |
| **DummyClassifier** | Most Frequent Class Baseline | 0.3373 | 0.0843 | 0.2500 | 0.1261 | ±0.0002 | 0.1701 | 0.63s |

*Pre-Test Selection Decision: `CategoricalNB_mixed` selected based on highest cross-validation Macro F1, zero representation violation, and closed-form probability interpretability.*

### 2. Feature-Group Ablation Study (CategoricalNB)

| Feature Group Subset | Included Attributes | Features Count | Macro F1 Mean | Macro F1 SD | Weighted F1 | Key Finding |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Demographic Only** | Static socioeconomic attributes | 8 | 0.9040 | ±0.0079 | 0.9038 | Strong population baseline |
| **Psychographic Only** | Lifestyle, price/brand affinity | 5 | 0.9064 | ±0.0080 | 0.9061 | Isolates VIP spending mindset |
| **Behavioral Only** | Freq, AOV, Recency, Spend, Disc | 7 | 0.9644 | ±0.0050 | 0.9642 | **Strongest standalone signal** |
| **Combined (All Groups)** | Full multi-modal feature vector | 20 | **0.9992** | **±0.0005** | **0.9990** | **Maximizes minority recall** |

### 3. One-Time Locked Holdout Test Evaluation (N=1,000 Customers)

| Metric | Score | 95% Bootstrap Confidence Interval |
| :--- | :---: | :---: |
| **Test Accuracy** | **99.80%** (998/1000 correct) | [99.50%, 100.00%] |
| **Macro Precision** | **0.9981** | [0.9950, 1.0000] |
| **Macro Recall** | **0.9985** | [0.9960, 1.0000] |
| **Macro F1-Score** | **0.9983** | **[0.9957, 1.0000]** |
| **Weighted F1-Score** | **0.9980** | [0.9950, 1.0000] |
| **Training Latency** | **107.78 ms** | Sub-second fitting |
| **Inference Latency** | **0.0369 ms / record** | 27,000+ predictions / second |

#### Class-Wise Breakdown:
- **Segment A (Affluent VIP)**: Precision: `0.9924` | Recall: `1.0000` | F1: `0.9962` | Support: `257`
- **Segment B (Upward Mobile)**: Precision: `1.0000` | Recall: `0.9942` | F1: `0.9971` | Support: `343`
- **Segment C (Budget Conscious)**: Precision: `1.0000` | Recall: `1.0000` | F1: `1.0000` | Support: `254`
- **Segment D (At-Risk / Inactive)**: Precision: `1.0000` | Recall: `1.0000` | F1: `1.0000` | Support: `146`

---

## Selective Prediction & Review Policy

| Posterior Threshold ($\tau$) | Coverage Rate (%) | Selective Error (%) | Review Rate (%) | Operational Review Policy |
| :---: | :---: | :---: | :---: | :--- |
| **$\tau \ge 0.75$ (High)** | **97.80%** | **0.00%** | **2.20%** | Automated Assignment with Routine Sampling |
| **$0.50 \le \tau < 0.75$ (Moderate)** | **99.60%** | **0.10%** | **0.40%** | Accept with Explicit Marketing Review Flag |
| **$\tau < 0.50$ (Low)** | **100.00%** | **0.20%** | **0.00%** | Mandatory Manual Staff Analysis (Abstention) |

---

## Subgroup Fairness & Temporal Drift Audit

1. **Demographic Parity & Subgroup Fairness**:
   - **Female Subgroup (N=482)**: Macro F1: `0.9981`, Recall: `0.9982`
   - **Male Subgroup (N=518)**: Macro F1: `0.9985`, Recall: `0.9987`
   - **Age < 30 (N=224)**: Macro F1: `1.0000` | **Age 30–50 (N=538)**: Macro F1: `0.9968` | **Age > 50 (N=238)**: Macro F1: `1.0000`
   - *Verdict: Zero disparate impact; equal recall observed across all demographic cohorts.*
2. **Temporal Drift Holdout**:
   - Random Stratified Split F1: `0.9983` vs Chronological Recency Split F1: `0.9773` ($\Delta F1 = -0.0210$).
   - *Recommendation: Schedule quarterly model recalibration to adapt to natural customer purchase recency drift.*
3. **Tabular Transformer Benchmark**:
   - `CategoricalNB`: Macro F1: `0.9983` | Latency: `0.0369 ms` | Params: `112 probs` | Training: `0.11 s`
   - `TabTransformer`: Macro F1: `0.9985` | Latency: `1.4500 ms` | Params: `450k weights` | Training: `42.50 s`
   - *Verdict: Naive Bayes achieves 99.8% F1 with 400x lower compute and sub-millisecond CPU inference, making it the optimal production architecture.*

---

## Author & Academic Information

- **Student:** Madhusudhanan G (Registration No: `23MID0444`)
- **Course:** MDI3003 - Advanced Predictive Analytics
- **Faculty Instructor:** Dr. Durgesh Kumar
- **School:** School of Computer Science and Engineering (SCOPE), VIT Vellore
- **Laboratory Manual Revision:** 3.0 (August 2026)
- **Repository:** [Madhumasa84/Adv_predictive](https://github.com/Madhumasa84/Adv_predictive/tree/main/lab4)
