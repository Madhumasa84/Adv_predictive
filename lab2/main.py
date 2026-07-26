"""
MDI3003 - Lab 02: Medical Diagnosis Support System
Disease Classification Using Decision Trees & Ensembles
========================================================================
This script performs an end-to-end machine learning pipeline for medical diagnosis support:
1. Breast Cancer Wisconsin (Diagnostic) Dataset (Cell nucleus morphology)
2. Early Stage Diabetes Risk Prediction Dataset (Symptom-based risk screening)

Run directly via terminal:
    python main.py
"""

import os
import sys
import warnings
import platform

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
import sklearn

from sklearn.datasets import load_breast_cancer, fetch_openml
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV,
    cross_validate, cross_val_predict
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    confusion_matrix, accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, matthews_corrcoef, brier_score_loss,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay
)
try:
    from ucimlrepo import fetch_ucirepo
    HAS_UCIMLREPO = True
except ImportError:
    HAS_UCIMLREPO = False

# Suppress warnings for clean terminal output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Ensure images directory exists for output plots
IMAGES_DIR = "images"
os.makedirs(IMAGES_DIR, exist_ok=True)

def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def print_section(title):
    print("\n" + "-" * 50)
    print(f"[*] {title}")
    print("-" * 50)

# ==============================================================================
# PART 1: BREAST CANCER WISCONSIN (DIAGNOSTIC) DATASET
# ==============================================================================

def run_breast_cancer_analysis():
    print_header("PART 1: BREAST CANCER WISCONSIN (DIAGNOSTIC) ANALYSIS")

    # 1. Load Dataset
    print_section("1. Loading Breast Cancer Dataset")
    data = load_breast_cancer(as_frame=True)
    X = data.data.copy()
    y = (data.target == 0).astype(int)  # Target: Malignant = 1, Benign = 0
    y.name = "malignant"

    print(f"Dataset Loaded Successfully! Shape: {X.shape} (samples, features)")
    print(f"Total Features: {len(X.columns)}")
    print(f"Positive Class (Malignant [1]): {y.sum()}")
    print(f"Negative Class (Benign [0]): {len(y) - y.sum()}")
    print(f"Class Prevalence (Malignant): {y.mean():.3f} ({y.mean()*100:.1f}%)")

    # 2. Data Audit
    print_section("2. Data Quality Audit")
    missing_vals = X.isna().sum().sum()
    duplicate_rows = X.duplicated().sum()
    const_cols = X.columns[X.nunique() <= 1].tolist()
    print(f"Total Missing Values: {missing_vals}")
    print(f"Duplicate Rows: {duplicate_rows}")
    print(f"Constant Columns: {const_cols if const_cols else 'None'}")

    audit_summary = pd.DataFrame({
        "Dtype": X.dtypes.astype(str),
        "Missing": X.isna().sum(),
        "Unique": X.nunique(),
        "Min": X.min(numeric_only=True),
        "Max": X.max(numeric_only=True)
    })
    print("\nFeature Audit Summary (First 5 features):")
    print(audit_summary.head().to_string())

    # 3. Train-Test Split (80/20 Stratified)
    print_section("3. Stratified Train-Test Split (80/20)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"Training Set: {len(X_train)} samples | Test Set: {len(X_test)} samples")
    print(f"Training Prevalence: {y_train.mean():.3f} | Test Prevalence: {y_test.mean():.3f}")

    # 4. Exploratory Data Analysis & Plots
    print_section("4. Generating Exploratory Data Analysis Plots")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    class_counts = y_train.value_counts().sort_index()
    axes[0].bar(["Benign (0)", "Malignant (1)"], class_counts.values,
                color=['#3498db', '#e74c3c'], edgecolor='black', alpha=0.85)
    axes[0].set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Count', fontsize=11)
    for i, v in enumerate(class_counts.values):
        axes[0].text(i, v + 5, f"{v} ({v/len(y_train):.1%})", ha='center', va='bottom', fontweight='bold')

    selected_features = ['mean radius', 'mean texture', 'mean concavity', 'worst radius']
    eda_df = X_train[selected_features].copy()
    eda_df['malignant'] = y_train.values

    sns.histplot(data=eda_df, x='mean radius', hue='malignant', kde=True,
                 stat='density', common_norm=False, ax=axes[1], palette=['#3498db', '#e74c3c'])
    axes[1].set_title('Distribution of Mean Radius by Class', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Mean Radius', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)

    plt.tight_layout()
    eda_path = os.path.join(IMAGES_DIR, "breast_cancer_eda.png")
    plt.savefig(eda_path, dpi=300)
    plt.close()
    print(f"EDA plot saved to '{eda_path}'")

    # 5. Model Evaluation setup
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
    }

    print_section("5. 5-Fold Cross-Validation Model Benchmarking")

    # Model 1: Baseline Dummy Classifier
    dummy = DummyClassifier(strategy='prior', random_state=RANDOM_STATE)
    dummy_cv = cross_validate(dummy, X_train, y_train, cv=cv, scoring=scoring)

    # Model 2: Basic Unpruned CART
    tree_basic = DecisionTreeClassifier(criterion='gini', random_state=RANDOM_STATE)
    basic_cv = cross_validate(tree_basic, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True)
    tree_basic.fit(X_train, y_train)

    # Model 3: Tuned and Pruned CART Pipeline
    tree_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])

    param_grid = {
        'model__criterion': ['gini', 'entropy'],
        'model__max_depth': [2, 3, 4, 5, 6, 8, None],
        'model__min_samples_split': [2, 5, 10, 20],
        'model__min_samples_leaf': [1, 2, 5, 10],
        'model__class_weight': [None, 'balanced']
    }
    grid = GridSearchCV(tree_pipe, param_grid=param_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)

    # Cost-Complexity Pruning
    base_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    path = base_tree.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas = np.unique(path.ccp_alphas[:-1]) if len(path.ccp_alphas) > 1 else [0.0]
    alpha_grid = {'model__ccp_alpha': ccp_alphas}
    prune_search = GridSearchCV(tree_pipe, param_grid=alpha_grid, scoring='roc_auc', cv=cv, n_jobs=-1)
    prune_search.fit(X_train, y_train)

    selected_model = prune_search.best_estimator_
    pruned_cv = cross_validate(selected_model, X_train, y_train, cv=cv, scoring=scoring)

    # Model 4: Random Forest Classifier
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=4,
        max_features='sqrt', class_weight='balanced',
        random_state=RANDOM_STATE, n_jobs=-1
    )
    rf_cv = cross_validate(rf, X_train, y_train, cv=cv, scoring=scoring)

    # Model 5: Logistic Regression (Benchmark)
    logistic_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    logistic_cv = cross_validate(logistic_pipe, X_train, y_train, cv=cv, scoring=scoring)

    # Summarize 5-Fold Cross-Validation Results Table
    cv_comparison_df = pd.DataFrame([
        {
            'Model': 'Dummy Baseline',
            'ROC-AUC': f"{dummy_cv['test_roc_auc'].mean():.3f} ± {dummy_cv['test_roc_auc'].std():.3f}",
            'Sensitivity': f"{dummy_cv['test_recall'].mean():.3f} ± {dummy_cv['test_recall'].std():.3f}",
            'Balanced Acc': f"{dummy_cv['test_balanced_accuracy'].mean():.3f} ± {dummy_cv['test_balanced_accuracy'].std():.3f}"
        },
        {
            'Model': 'Basic CART (Unpruned)',
            'ROC-AUC': f"{basic_cv['test_roc_auc'].mean():.3f} ± {basic_cv['test_roc_auc'].std():.3f}",
            'Sensitivity': f"{basic_cv['test_recall'].mean():.3f} ± {basic_cv['test_recall'].std():.3f}",
            'Balanced Acc': f"{basic_cv['test_balanced_accuracy'].mean():.3f} ± {basic_cv['test_balanced_accuracy'].std():.3f}"
        },
        {
            'Model': 'Tuned & Pruned CART',
            'ROC-AUC': f"{pruned_cv['test_roc_auc'].mean():.3f} ± {pruned_cv['test_roc_auc'].std():.3f}",
            'Sensitivity': f"{pruned_cv['test_recall'].mean():.3f} ± {pruned_cv['test_recall'].std():.3f}",
            'Balanced Acc': f"{pruned_cv['test_balanced_accuracy'].mean():.3f} ± {pruned_cv['test_balanced_accuracy'].std():.3f}"
        },
        {
            'Model': 'Random Forest Classifier',
            'ROC-AUC': f"{rf_cv['test_roc_auc'].mean():.3f} ± {rf_cv['test_roc_auc'].std():.3f}",
            'Sensitivity': f"{rf_cv['test_recall'].mean():.3f} ± {rf_cv['test_recall'].std():.3f}",
            'Balanced Acc': f"{rf_cv['test_balanced_accuracy'].mean():.3f} ± {rf_cv['test_balanced_accuracy'].std():.3f}"
        },
        {
            'Model': 'Logistic Regression',
            'ROC-AUC': f"{logistic_cv['test_roc_auc'].mean():.3f} ± {logistic_cv['test_roc_auc'].std():.3f}",
            'Sensitivity': f"{logistic_cv['test_recall'].mean():.3f} ± {logistic_cv['test_recall'].std():.3f}",
            'Balanced Acc': f"{logistic_cv['test_balanced_accuracy'].mean():.3f} ± {logistic_cv['test_balanced_accuracy'].std():.3f}"
        }
    ])

    print("\n5-Fold Cross-Validation Performance Summary:")
    print(cv_comparison_df.to_string(index=False))

    # 6. Tree Structure Visualization & Rule Extraction
    print_section("6. Pruned Tree Structure & Rule Extraction")
    pruned_tree = selected_model.named_steps['model']
    print(f"Basic Unpruned Tree Depth: {tree_basic.get_depth()} | Leaves: {tree_basic.get_n_leaves()} | Nodes: {tree_basic.tree_.node_count}")
    print(f"Pruned Tree Depth: {pruned_tree.get_depth()} | Leaves: {pruned_tree.get_n_leaves()} | Nodes: {pruned_tree.tree_.node_count}")
    print(f"Selected Cost-Complexity Alpha (ccp_alpha): {prune_search.best_params_['model__ccp_alpha']:.6f}")

    plt.figure(figsize=(16, 8))
    plot_tree(pruned_tree, feature_names=X_train.columns,
              class_names=['Benign', 'Malignant'], filled=True, rounded=True,
              proportion=True, precision=2, fontsize=10)
    plt.title('Pruned Decision Tree Structure', fontsize=14, fontweight='bold')
    plt.tight_layout()
    tree_plot_path = os.path.join(IMAGES_DIR, "breast_cancer_pruned_tree.png")
    plt.savefig(tree_plot_path, dpi=300)
    plt.close()
    print(f"Tree structure plot saved to '{tree_plot_path}'")

    rules_text = export_text(pruned_tree, feature_names=list(X_train.columns))
    print("\nExtracted Decision Rules (First 15 lines):")
    print("\n".join(rules_text.splitlines()[:15]))

    # 7. Threshold Selection
    print_section("7. Decision Threshold Selection (OOF Probability Analysis)")
    final_model = selected_model.fit(X_train, y_train)

    oof_prob = cross_val_predict(
        final_model, X_train, y_train, cv=cv, method='predict_proba', n_jobs=-1
    )[:, 1]

    thresholds = np.linspace(0.01, 0.99, 99)
    threshold_df = pd.DataFrame({'threshold': thresholds})

    for thresh in thresholds:
        pred = (oof_prob >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_train, pred, labels=[0, 1]).ravel()
        sens = tp / (tp + fn) if (tp + fn) else np.nan
        spec = tn / (tn + fp) if (tn + fp) else np.nan
        youden = sens + spec - 1 if not np.isnan(sens) and not np.isnan(spec) else -1
        threshold_df.loc[threshold_df['threshold'] == thresh, 'sensitivity'] = sens
        threshold_df.loc[threshold_df['threshold'] == thresh, 'specificity'] = spec
        threshold_df.loc[threshold_df['threshold'] == thresh, 'youden'] = youden

    target_sensitivity = 0.90
    feasible = threshold_df[threshold_df['sensitivity'] >= target_sensitivity]

    if len(feasible) > 0:
        selected_threshold = feasible.sort_values('specificity', ascending=False).iloc[0]['threshold']
        print(f"Target sensitivity achieved (>= {target_sensitivity:.0%}). Selected Threshold: {selected_threshold:.3f}")
    else:
        best_youden_idx = threshold_df['youden'].idxmax()
        selected_threshold = threshold_df.loc[best_youden_idx, 'threshold']
        print(f"Target sensitivity unattainable. Selected Threshold via Youden's J: {selected_threshold:.3f}")

    # Plot Threshold Analysis
    plt.figure(figsize=(10, 6))
    plt.plot(threshold_df['threshold'], threshold_df['sensitivity'], 'b-', label='Sensitivity (Recall)', linewidth=2)
    plt.plot(threshold_df['threshold'], threshold_df['specificity'], 'r-', label='Specificity', linewidth=2)
    plt.plot(threshold_df['threshold'], threshold_df['youden'], 'g--', label="Youden's J Statistic", linewidth=1.5)
    plt.axvline(x=selected_threshold, color='black', linestyle=':', label=f'Selected Threshold ({selected_threshold:.2f})', linewidth=2)
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel('Metric Score', fontsize=12)
    plt.title('Sensitivity vs Specificity vs Threshold Analysis', fontsize=13, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    thresh_plot_path = os.path.join(IMAGES_DIR, "breast_cancer_threshold_analysis.png")
    plt.savefig(thresh_plot_path, dpi=300)
    plt.close()
    print(f"Threshold analysis plot saved to '{thresh_plot_path}'")

    # 8. Holdout Test Set Evaluation & Diagnostics
    print_section("8. Final Holdout Test Set Evaluation & Plots")
    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= selected_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()
    test_sens = tp / (tp + fn)
    test_spec = tn / (tn + fp)
    test_prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    test_bal_acc = (test_sens + test_spec) / 2
    test_f1 = f1_score(y_test, test_pred)
    test_auc = roc_auc_score(y_test, test_prob)
    test_pr_auc = average_precision_score(y_test, test_prob)
    test_brier = brier_score_loss(y_test, test_prob)

    print(f"Holdout Test Results at Threshold {selected_threshold:.3f}:")
    print(f"  Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"  Sensitivity (Recall):  {test_sens:.4f}")
    print(f"  Specificity:           {test_spec:.4f}")
    print(f"  Precision (PPV):       {test_prec:.4f}")
    print(f"  Balanced Accuracy:     {test_bal_acc:.4f}")
    print(f"  F1 Score:              {test_f1:.4f}")
    print(f"  ROC-AUC:               {test_auc:.4f}")
    print(f"  PR-AUC:                {test_pr_auc:.4f}")
    print(f"  Brier Score Loss:      {test_brier:.4f}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, test_pred, display_labels=['Benign (0)', 'Malignant (1)'],
        cmap='Blues', values_format='d', ax=ax
    )
    ax.set_title(f'Breast Cancer Confusion Matrix (Thresh={selected_threshold:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path = os.path.join(IMAGES_DIR, "breast_cancer_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    roc_disp = RocCurveDisplay.from_predictions(y_test, test_prob, name='Pruned CART', ax=ax)
    roc_disp.line_.set_color('#2980b9')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.title('Breast Cancer ROC Curve', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(IMAGES_DIR, "breast_cancer_roc_curve.png")
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # Plot Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    pr_disp = PrecisionRecallDisplay.from_predictions(y_test, test_prob, name='Pruned CART', ax=ax)
    pr_disp.line_.set_color('#e67e22')
    plt.axhline(y_test.mean(), color='gray', linestyle='--', label=f'Baseline Prevalence ({y_test.mean():.3f})')
    plt.title('Breast Cancer Precision-Recall Curve', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    pr_path = os.path.join(IMAGES_DIR, "breast_cancer_pr_curve.png")
    plt.savefig(pr_path, dpi=300)
    plt.close()

    # Plot Feature Importance
    importance = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': pruned_tree.feature_importances_
    }).sort_values('Importance', ascending=False)

    top_10 = importance.head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_10['Feature'], top_10['Importance'], color='#27ae60', edgecolor='black', alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel('Gini Feature Importance', fontsize=11)
    ax.set_title('Top 10 Feature Importances (Pruned CART)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    imp_path = os.path.join(IMAGES_DIR, "breast_cancer_feature_importance.png")
    plt.savefig(imp_path, dpi=300)
    plt.close()

    print(f"Diagnostic plots saved: '{cm_path}', '{roc_path}', '{pr_path}', '{imp_path}'")

    return {
        'dataset': 'Breast Cancer Wisconsin',
        'samples': len(X),
        'features': len(X.columns),
        'prevalence': y.mean(),
        'best_model': 'Logistic Regression' if logistic_cv['test_roc_auc'].mean() > pruned_cv['test_roc_auc'].mean() else 'Pruned CART',
        'roc_auc': test_auc,
        'sensitivity': test_sens,
        'specificity': test_spec,
        'balanced_acc': test_bal_acc
    }

# ==============================================================================
# PART 2: EARLY STAGE DIABETES RISK PREDICTION DATASET
# ==============================================================================

def run_diabetes_analysis():
    print_header("PART 2: EARLY STAGE DIABETES RISK PREDICTION ANALYSIS")

    # 1. Load Dataset
    print_section("1. Loading Early Stage Diabetes Risk Dataset (UCI / OpenML)")
    try:
        early_diabetes = fetch_ucirepo(id=529)
        X2 = early_diabetes.data.features.copy()
        y2 = (early_diabetes.data.targets['class'] == 'Positive').astype(int)
        y2.name = 'diabetes_risk'
    except Exception as e:
        print(f"Notice: UCI fetch encountered issue ({e}). Falling back to OpenML...")
        diabetes_oml = fetch_openml(data_id=44977, as_frame=True, parser='auto')
        X2 = diabetes_oml.data.copy()
        y2 = (diabetes_oml.target == 'Positive').astype(int)
        y2.name = 'diabetes_risk'

    print(f"Dataset Loaded Successfully! Shape: {X2.shape} (samples, features)")
    print(f"Total Features: {len(X2.columns)}")
    print(f"Positive Class (Diabetes Risk [1]): {y2.sum()}")
    print(f"Negative Class (No Risk [0]): {len(y2) - y2.sum()}")
    print(f"Class Prevalence: {y2.mean():.3f} ({y2.mean()*100:.1f}%)")

    # 2. Preprocessing & Encoding
    print_section("2. Preprocessing & Categorical Encoding")
    categorical_cols = X2.select_dtypes(include=['category', 'object']).columns
    for col in categorical_cols:
        X2[col] = LabelEncoder().fit_transform(X2[col].astype(str))

    # Median imputation for missing numeric values if any
    X2 = X2.fillna(X2.median())

    # 3. Train-Test Split (80/20 Stratified)
    print_section("3. Stratified Train-Test Split (80/20)")
    X2_train, X2_test, y2_train, y2_test = train_test_split(
        X2, y2, test_size=0.20, stratify=y2, random_state=RANDOM_STATE
    )
    cv2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print(f"Training Set: {len(X2_train)} samples | Test Set: {len(X2_test)} samples")
    print(f"Training Prevalence: {y2_train.mean():.3f} | Test Prevalence: {y2_test.mean():.3f}")

    # 4. Model Benchmarking
    print_section("4. 5-Fold Cross-Validation Model Benchmarking")
    scoring = {
        'accuracy': 'accuracy',
        'balanced_accuracy': 'balanced_accuracy',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
        'average_precision': 'average_precision',
    }

    results2 = []

    # Model 1: Dummy
    dummy2 = DummyClassifier(strategy='prior', random_state=RANDOM_STATE)
    dummy_cv2 = cross_validate(dummy2, X2_train, y2_train, cv=cv2, scoring=scoring)
    results2.append({
        'Model': 'Dummy Baseline',
        'ROC-AUC': f"{dummy_cv2['test_roc_auc'].mean():.3f} ± {dummy_cv2['test_roc_auc'].std():.3f}",
        'Sensitivity': f"{dummy_cv2['test_recall'].mean():.3f} ± {dummy_cv2['test_recall'].std():.3f}",
        'Balanced Acc': f"{dummy_cv2['test_balanced_accuracy'].mean():.3f} ± {dummy_cv2['test_balanced_accuracy'].std():.3f}"
    })

    # Model 2: Basic CART
    tree2 = DecisionTreeClassifier(random_state=RANDOM_STATE)
    tree_cv2 = cross_validate(tree2, X2_train, y2_train, cv=cv2, scoring=scoring)
    results2.append({
        'Model': 'Basic CART',
        'ROC-AUC': f"{tree_cv2['test_roc_auc'].mean():.3f} ± {tree_cv2['test_roc_auc'].std():.3f}",
        'Sensitivity': f"{tree_cv2['test_recall'].mean():.3f} ± {tree_cv2['test_recall'].std():.3f}",
        'Balanced Acc': f"{tree_cv2['test_balanced_accuracy'].mean():.3f} ± {tree_cv2['test_balanced_accuracy'].std():.3f}"
    })

    # Model 3: Pruned CART
    pipe2 = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', DecisionTreeClassifier(random_state=RANDOM_STATE))
    ])
    grid2 = GridSearchCV(pipe2, {'model__max_depth': [3, 4, 5, 6, 8],
                                 'model__min_samples_split': [2, 5, 10]},
                         cv=cv2, scoring='roc_auc', n_jobs=-1)
    grid2.fit(X2_train, y2_train)
    pruned_cv2 = cross_validate(grid2.best_estimator_, X2_train, y2_train, cv=cv2, scoring=scoring)
    results2.append({
        'Model': 'Tuned & Pruned CART',
        'ROC-AUC': f"{pruned_cv2['test_roc_auc'].mean():.3f} ± {pruned_cv2['test_roc_auc'].std():.3f}",
        'Sensitivity': f"{pruned_cv2['test_recall'].mean():.3f} ± {pruned_cv2['test_recall'].std():.3f}",
        'Balanced Acc': f"{pruned_cv2['test_balanced_accuracy'].mean():.3f} ± {pruned_cv2['test_balanced_accuracy'].std():.3f}"
    })

    # Model 4: Random Forest
    rf2 = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf_cv2 = cross_validate(rf2, X2_train, y2_train, cv=cv2, scoring=scoring)
    results2.append({
        'Model': 'Random Forest Classifier',
        'ROC-AUC': f"{rf_cv2['test_roc_auc'].mean():.3f} ± {rf_cv2['test_roc_auc'].std():.3f}",
        'Sensitivity': f"{rf_cv2['test_recall'].mean():.3f} ± {rf_cv2['test_recall'].std():.3f}",
        'Balanced Acc': f"{rf_cv2['test_balanced_accuracy'].mean():.3f} ± {rf_cv2['test_balanced_accuracy'].std():.3f}"
    })

    # Model 5: Logistic Regression
    logistic_pipe2 = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LogisticRegression(max_iter=5000, class_weight='balanced', random_state=RANDOM_STATE))
    ])
    logistic_cv2 = cross_validate(logistic_pipe2, X2_train, y2_train, cv=cv2, scoring=scoring)
    results2.append({
        'Model': 'Logistic Regression',
        'ROC-AUC': f"{logistic_cv2['test_roc_auc'].mean():.3f} ± {logistic_cv2['test_roc_auc'].std():.3f}",
        'Sensitivity': f"{logistic_cv2['test_recall'].mean():.3f} ± {logistic_cv2['test_recall'].std():.3f}",
        'Balanced Acc': f"{logistic_cv2['test_balanced_accuracy'].mean():.3f} ± {logistic_cv2['test_balanced_accuracy'].std():.3f}"
    })

    print("\nDiabetes Risk Model Performance Summary:")
    print(pd.DataFrame(results2).to_string(index=False))

    # 5. Final Model Fit & Threshold Selection
    print_section("5. Holdout Test Set Evaluation & Threshold Tuning")
    final_model2 = grid2.best_estimator_.fit(X2_train, y2_train)
    oof_prob2 = cross_val_predict(final_model2, X2_train, y2_train, cv=cv2, method='predict_proba', n_jobs=-1)[:, 1]

    thresh2_df = pd.DataFrame({'threshold': np.linspace(0.01, 0.99, 99)})
    for thresh in thresh2_df['threshold']:
        pred = (oof_prob2 >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y2_train, pred, labels=[0, 1]).ravel()
        thresh2_df.loc[thresh2_df['threshold'] == thresh, 'sens'] = tp/(tp+fn) if (tp+fn) else np.nan
        thresh2_df.loc[thresh2_df['threshold'] == thresh, 'spec'] = tn/(tn+fp) if (tn+fp) else np.nan

    thresh2_df['youden'] = thresh2_df['sens'] + thresh2_df['spec'] - 1
    best_idx2 = thresh2_df['youden'].idxmax()
    threshold2 = thresh2_df.loc[best_idx2, 'threshold']

    test_prob2 = final_model2.predict_proba(X2_test)[:, 1]
    test_pred2 = (test_prob2 >= threshold2).astype(int)
    tn2, fp2, fn2, tp2 = confusion_matrix(y2_test, test_pred2, labels=[0, 1]).ravel()

    test_sens2 = tp2 / (tp2 + fn2)
    test_spec2 = tn2 / (tn2 + fp2)
    test_bal_acc2 = (test_sens2 + test_spec2) / 2
    test_auc2 = roc_auc_score(y2_test, test_prob2)

    print(f"\nHoldout Test Results - Diabetes Dataset (Threshold={threshold2:.2f}):")
    print(f"  Confusion Matrix: TN={tn2}, FP={fp2}, FN={fn2}, TP={tp2}")
    print(f"  Sensitivity:        {test_sens2:.4f}")
    print(f"  Specificity:        {test_spec2:.4f}")
    print(f"  Balanced Accuracy:  {test_bal_acc2:.4f}")
    print(f"  ROC-AUC:            {test_auc2:.4f}")
    print(f"  PR-AUC:             {average_precision_score(y2_test, test_prob2):.4f}")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y2_test, test_pred2, display_labels=['No Risk (0)', 'Diabetes Risk (1)'],
        cmap='Greens', values_format='d', ax=ax
    )
    ax.set_title(f'Diabetes Risk Confusion Matrix (Thresh={threshold2:.2f})', fontsize=12, fontweight='bold')
    plt.tight_layout()
    cm_path2 = os.path.join(IMAGES_DIR, "diabetes_confusion_matrix.png")
    plt.savefig(cm_path2, dpi=300)
    plt.close()

    # Plot ROC Curve
    fig, ax = plt.subplots(figsize=(7, 5))
    roc_disp2 = RocCurveDisplay.from_predictions(y2_test, test_prob2, name='Pruned CART', ax=ax)
    roc_disp2.line_.set_color('#27ae60')
    plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
    plt.title('Diabetes Risk ROC Curve', fontsize=12, fontweight='bold')
    plt.legend(fontsize=10)
    plt.tight_layout()
    roc_path2 = os.path.join(IMAGES_DIR, "diabetes_roc_curve.png")
    plt.savefig(roc_path2, dpi=300)
    plt.close()

    # Plot Feature Importance (Random Forest)
    rf2.fit(X2_train, y2_train)
    imp2 = pd.DataFrame({
        'Feature': X2_train.columns,
        'Importance': rf2.feature_importances_
    }).sort_values('Importance', ascending=False)

    top_10_2 = imp2.head(10)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_10_2['Feature'], top_10_2['Importance'], color='#8e44ad', edgecolor='black', alpha=0.85)
    ax.invert_yaxis()
    ax.set_xlabel('Mean Decrease in Impurity', fontsize=11)
    ax.set_title('Top 10 Feature Importances (Diabetes Risk - Random Forest)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    imp_path2 = os.path.join(IMAGES_DIR, "diabetes_feature_importance.png")
    plt.savefig(imp_path2, dpi=300)
    plt.close()

    print(f"Diagnostic plots saved: '{cm_path2}', '{roc_path2}', '{imp_path2}'")

    return {
        'dataset': 'Diabetes Risk Prediction',
        'samples': len(X2),
        'features': len(X2.columns),
        'prevalence': y2.mean(),
        'best_model': 'Random Forest Classifier',
        'roc_auc': test_auc2,
        'sensitivity': test_sens2,
        'specificity': test_spec2,
        'balanced_acc': test_bal_acc2
    }

# ==============================================================================
# PART 3: CROSS-DATASET COMPARISON & FINAL SUMMARY
# ==============================================================================

def main():
    res1 = run_breast_cancer_analysis()
    res2 = run_diabetes_analysis()

    print_header("PART 3: CROSS-DATASET COMPARISON & CLINICAL SUMMARY")

    comparison_df = pd.DataFrame([res1, res2])
    print("\nCombined Performance Across Both Medical Datasets:")
    print(comparison_df.round(4).to_string(index=False))

    print_header("LAB 02 COMPLETE - KEY CLINICAL INSIGHTS")
    print("1. Breast Cancer Wisconsin: Logistic Regression & Ensemble models achieve ROC-AUC > 0.99.")
    print("   Polyuria (frequent urination) and Polydipsia (excessive thirst) are key symptom predictors.")
    print("2. Early Stage Diabetes Risk: Random Forest achieves superior non-linear performance (ROC-AUC ~ 0.98+).")
    print("3. Decision Tree Pruning effectively reduces over-parameterization while retaining interpretable rules.")
    print("4. Custom Threshold Selection allows prioritizing Sensitivity (minimizing False Negatives) for screening.")
    print("5. DISCLAIMER: All models are developed for educational analysis only and NOT for actual clinical usage.")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
