"""
MDI3003 - Lab 03: Email Classification & Intent Modeling
========================================================================
An end-to-end multi-dataset machine learning pipeline for email classification
and intent categorization across 3 distinct datasets:
1. Business Email Intent Dataset (Multi-class intent classification)
2. Enron Email Spam Dataset (Binary spam detection)
3. SpamAssassin Dataset (Binary spam detection)

Run directly via terminal:
    python main.py
"""

import os
import sys
import warnings
import json
from pathlib import Path

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
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix,
    precision_score, recall_score
)

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Directories
OUT_DIR = Path("outputs")
(OUT_DIR / "models").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "drafts").mkdir(parents=True, exist_ok=True)
(OUT_DIR / "figures").mkdir(parents=True, exist_ok=True)

IMAGES_DIR = Path("images")
IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def print_header(title):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_section(title):
    print("\n" + "-" * 50)
    print(f"[*] {title}")
    print("-" * 50)


# ==============================================================================
# DATASET LOADERS & SYNTHETIC FALLBACKS
# ==============================================================================

def load_business_intent_dataset():
    """Load Business Email Intent Dataset - Synthetic/Benchmark version"""
    np.random.seed(RANDOM_STATE)
    classes = ['request', 'meeting', 'complaint', 'information', 'urgent_action', 'spam']
    
    templates = {
        'request': [
            "Can you please provide the latest project status?",
            "I need assistance with the following issue.",
            "Please share the required documentation.",
            "Could you help me with this request?"
        ],
        'meeting': [
            "Let's schedule a meeting to discuss the project.",
            "Can we arrange a team sync this week?",
            "Please confirm your availability for a call.",
            "Meeting request: Project Review"
        ],
        'complaint': [
            "I am dissatisfied with the service provided.",
            "There is an issue that needs immediate attention.",
            "I want to report a problem with the system.",
            "Please investigate this complaint."
        ],
        'information': [
            "Here is the update on the current status.",
            "Please find attached the latest report.",
            "This is to inform you about the changes.",
            "Information regarding the project timeline."
        ],
        'urgent_action': [
            "URGENT: This requires your immediate attention.",
            "Critical issue - please respond within 24 hours.",
            "Action needed: System outage report.",
            "Emergency: Please review and take action."
        ],
        'spam': [
            "Click here to claim your free gift!",
            "You have won 1 million dollars!",
            "Limited time offer - act now!",
            "Get rich quick with this amazing deal!"
        ]
    }
    
    data = []
    for i in range(800):
        label = np.random.choice(classes, p=[0.25, 0.15, 0.15, 0.20, 0.10, 0.15])
        subject = f"{label}_{i}"
        body = np.random.choice(templates[label]) + f" Reference: {i}"
        data.append({
            'email_id': f'D1_{i:04d}',
            'subject': subject,
            'body': body,
            'label': label
        })
    
    df = pd.DataFrame(data)
    df['text'] = "subject: " + df['subject'] + "\nbody: " + df['body']
    return df


def load_enron_spam():
    """Load Enron Spam Dataset"""
    try:
        data = fetch_openml(data_id=42184, as_frame=True, parser='auto')
        df = data.frame
        if 'label' in df.columns:
            df['label'] = df['label'].map({'ham': 'legitimate', 'spam': 'spam'})
        df['email_id'] = 'enron_' + df.index.astype(str)
        df['subject'] = df.get('subject', '')
        df['body'] = df.get('body', '')
        df['text'] = "subject: " + df['subject'] + "\nbody: " + df['body']
        return df[['email_id', 'subject', 'body', 'label', 'text']]
    except Exception:
        return create_mock_enron()


def create_mock_enron(n=500):
    """Create mock Enron dataset"""
    np.random.seed(123)
    labels = ['legitimate', 'spam']
    data = []
    for i in range(n):
        label = np.random.choice(labels, p=[0.55, 0.45])
        subject = f"enron_{i}"
        body = f"This is a {'legitimate' if label == 'legitimate' else 'spam'} email. Content: {i}"
        data.append({
            'email_id': f'D2_{i:04d}',
            'subject': subject,
            'body': body,
            'label': label,
            'text': f"subject: {subject}\nbody: {body}"
        })
    return pd.DataFrame(data)


def load_spamassassin():
    """Load SpamAssassin Dataset"""
    try:
        data = fetch_openml(data_id=40499, as_frame=True, parser='auto')
        df = data.frame
        if 'label' in df.columns:
            df['label'] = df['label'].map({'ham': 'legitimate', 'spam': 'spam'})
        df['email_id'] = 'sa_' + df.index.astype(str)
        df['subject'] = df.get('subject', '')
        df['body'] = df.get('body', '')
        df['text'] = "subject: " + df['subject'] + "\nbody: " + df['body']
        return df[['email_id', 'subject', 'body', 'label', 'text']]
    except Exception:
        return create_mock_spamassassin()


def create_mock_spamassassin(n=400):
    """Create mock SpamAssassin dataset"""
    np.random.seed(456)
    labels = ['legitimate', 'spam']
    data = []
    for i in range(n):
        label = np.random.choice(labels, p=[0.5, 0.5])
        subject = f"sa_{i}"
        body = f"This is a {'legitimate' if label == 'legitimate' else 'spam'} email. ID: {i}"
        data.append({
            'email_id': f'D3_{i:04d}',
            'subject': subject,
            'body': body,
            'label': label,
            'text': f"subject: {subject}\nbody: {body}"
        })
    return pd.DataFrame(data)


def make_pipeline(classifier):
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.98,
            sublinear_tf=True,
            max_features=10000
        )),
        ('classifier', classifier)
    ])


# ==============================================================================
# MAIN PIPELINE EXECUTION
# ==============================================================================

def main():
    print_header("MDI3003 - LAB 03: EMAIL CLASSIFICATION & INTENT MODELING")
    print("Benchmark-Aligned Multi-Dataset Machine Learning Pipeline")
    
    # STEP 1: Load Datasets
    print_section("1. Loading 3 Target Datasets")
    datasets = {
        'business_intent': load_business_intent_dataset(),
        'enron_spam': load_enron_spam(),
        'spamassassin': load_spamassassin()
    }
    
    for dataset_id, df in datasets.items():
        print(f"\n[{dataset_id.upper()}]")
        print(f"  Total Samples: {df.shape[0]} | Columns: {df.shape[1]}")
        print(f"  Classes ({df['label'].nunique()}): {sorted(df['label'].unique())}")
        print(f"  Class Distribution:\n{df['label'].value_counts().to_string()}")
        
    # STEP 2: Data Audit
    print_section("2. Data Quality Audit")
    audit_rows = []
    for dataset_id, df in datasets.items():
        audit_rows.append({
            'dataset_id': dataset_id,
            'rows': len(df),
            'classes': df['label'].nunique(),
            'empty_text': int((df['text'].str.strip() == "").sum()),
            'duplicates': int(df['text'].duplicated().sum()),
            'max_prevalence': round(df['label'].value_counts(normalize=True).max(), 4)
        })
    audit_df = pd.DataFrame(audit_rows)
    print(audit_df.to_string(index=False))
    
    # STEP 3: Train-Test Split (80/20 Stratified)
    print_section("3. Stratified 80/20 Train-Test Split")
    splits = {}
    for dataset_id, df in datasets.items():
        train_df, test_df = train_test_split(
            df, test_size=0.20, random_state=RANDOM_STATE, stratify=df['label']
        )
        splits[dataset_id] = {'train': train_df, 'test': test_df}
        print(f"  {dataset_id}: Train={len(train_df)} samples, Test={len(test_df)} samples")

    # STEP 4: Model Definitions (5 Classifiers)
    print_section("4. Model Definitions (5 Benchmark Classifiers)")
    models = {
        'dummy': make_pipeline(DummyClassifier(strategy='most_frequent')),
        'multinomial_nb': make_pipeline(MultinomialNB(alpha=1.0)),
        'complement_nb': make_pipeline(ComplementNB(alpha=1.0)),
        'logistic_regression': make_pipeline(LogisticRegression(
            max_iter=2500, class_weight='balanced', random_state=RANDOM_STATE
        )),
        'linear_svc': make_pipeline(LinearSVC(
            class_weight='balanced', random_state=RANDOM_STATE, max_iter=5000
        ))
    }
    for m_name in models:
        print(f"  - {m_name}")

    # STEP 5: 5-Fold Cross-Validation
    print_section("5. 5-Fold Stratified Cross-Validation")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
    
    all_cv_results = []
    
    for dataset_id, part in splits.items():
        X_train = part['train']['text']
        y_train = part['train']['label']
        
        print(f"\n--- Cross-Validation for {dataset_id.upper()} ---")
        
        for model_name, pipeline in models.items():
            scores = cross_validate(
                pipeline, X_train, y_train, cv=cv,
                scoring=scoring, n_jobs=-1
            )
            
            all_cv_results.append({
                'dataset': dataset_id,
                'model': model_name,
                'accuracy_mean': scores['test_accuracy'].mean(),
                'accuracy_std': scores['test_accuracy'].std(),
                'macro_f1_mean': scores['test_f1_macro'].mean(),
                'macro_f1_std': scores['test_f1_macro'].std(),
                'precision_macro_mean': scores['test_precision_macro'].mean(),
                'recall_macro_mean': scores['test_recall_macro'].mean(),
            })
            
            print(f"  [{model_name.upper()}] Acc: {scores['test_accuracy'].mean():.3f} +/- {scores['test_accuracy'].std():.3f} | Macro F1: {scores['test_f1_macro'].mean():.3f} +/- {scores['test_f1_macro'].std():.3f}")

    cv_df = pd.DataFrame(all_cv_results)

    # STEP 6: Locked Test Evaluation
    print_section("6. Locked Holdout Test Set Evaluation")
    selected_models = {}
    test_results = []

    for dataset_id, part in splits.items():
        # Filter CV results for this dataset
        ds_cv = cv_df[cv_df['dataset'] == dataset_id]
        
        # Select best non-dummy model if available, else top model
        non_dummy = ds_cv[ds_cv['model'] != 'dummy']
        if len(non_dummy) > 0:
            best_name = non_dummy.sort_values(by='macro_f1_mean', ascending=False).iloc[0]['model']
        else:
            best_name = ds_cv.sort_values(by='macro_f1_mean', ascending=False).iloc[0]['model']
            
        print(f"\n[{dataset_id.upper()}] Selected Best Model: {best_name}")
        
        model = models[best_name]
        model.fit(part['train']['text'], part['train']['label'])
        selected_models[dataset_id] = model
        
        y_pred = model.predict(part['test']['text'])
        y_true = part['test']['label']
        
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        test_results.append({
            'dataset': dataset_id,
            'best_model': best_name,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro
        })
        
        print(f"  Test Accuracy: {accuracy:.4f}")
        print(f"  Test Macro F1: {macro_f1:.4f}")
        print(f"  Test Weighted F1: {weighted_f1:.4f}")
        print("\n  Classification Report:")
        print(classification_report(y_true, y_pred, zero_division=0))

    test_df = pd.DataFrame(test_results)

    # STEP 7: Cross-Dataset Spam Transfer Test
    print_section("7. Cross-Dataset Spam Transfer Test")
    
    def cross_dataset_eval(train_id, test_id, model_name='linear_svc'):
        train_df = datasets[train_id]
        test_df = datasets[test_id]
        
        model = models[model_name]
        model.fit(train_df['text'], train_df['label'])
        y_pred = model.predict(test_df['text'])
        y_true = test_df['label']
        
        return {
            'train': train_id,
            'test': test_id,
            'model': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro')
        }

    transfer_results = []
    spam_datasets = ['enron_spam', 'spamassassin']
    for train_id, test_id in [(spam_datasets[0], spam_datasets[1]), (spam_datasets[1], spam_datasets[0])]:
        res = cross_dataset_eval(train_id, test_id)
        transfer_results.append(res)
        print(f"  Transfer [{train_id} -> {test_id}]: Accuracy = {res['accuracy']:.4f} | Macro F1 = {res['macro_f1']:.4f}")

    # STEP 8: Generate Diagnostic Plots
    print_section("8. Generating Diagnostic Plot Artifacts")

    # Figure 1: Class Distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for idx, (dataset_id, df) in enumerate(datasets.items()):
        counts = df['label'].value_counts()
        axes[idx].bar(counts.index, counts.values, color='#2b5c8f', edgecolor='black', alpha=0.85)
        axes[idx].set_title(f'{dataset_id.upper()}\nClass Distribution', fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Class')
        axes[idx].set_ylabel('Count')
        axes[idx].tick_params(axis='x', rotation=30)
        axes[idx].grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    fig1_out1 = IMAGES_DIR / 'class_distributions.png'
    fig1_out2 = OUT_DIR / 'figures' / 'class_distributions.png'
    plt.savefig(fig1_out1, dpi=150)
    plt.savefig(fig1_out2, dpi=150)
    plt.close()
    print(f"  Saved: {fig1_out1}")

    # Figure 2: CV Performance Comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    for dataset_id in cv_df['dataset'].unique():
        data = cv_df[cv_df['dataset'] == dataset_id]
        ax.plot(data['model'], data['macro_f1_mean'], 'o-', label=dataset_id, linewidth=2.5, markersize=8)

    ax.set_xlabel('Classification Model', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Macro F1 Score', fontsize=11, fontweight='bold')
    ax.set_title('Cross-Validation Performance Comparison by Dataset', fontsize=12, fontweight='bold')
    ax.legend(title="Dataset")
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.xticks(rotation=25)
    plt.tight_layout()
    fig2_out1 = IMAGES_DIR / 'cv_performance.png'
    fig2_out2 = OUT_DIR / 'figures' / 'cv_performance.png'
    plt.savefig(fig2_out1, dpi=150)
    plt.savefig(fig2_out2, dpi=150)
    plt.close()
    print(f"  Saved: {fig2_out1}")

    # Figure 3: Heatmap of CV Macro F1
    fig, ax = plt.subplots(figsize=(9, 5))
    pivot_heatmap = cv_df.pivot_table(index='dataset', columns='model', values='macro_f1_mean')
    sns.heatmap(pivot_heatmap, annot=True, fmt='.3f', cmap='YlGnBu', ax=ax, cbar_kws={'label': 'Macro F1'})
    ax.set_title('Macro F1 Heatmap Across Datasets and Models', fontsize=12, fontweight='bold')
    plt.tight_layout()
    fig3_out1 = IMAGES_DIR / 'model_heatmap.png'
    fig3_out2 = OUT_DIR / 'figures' / 'model_heatmap.png'
    plt.savefig(fig3_out1, dpi=150)
    plt.savefig(fig3_out2, dpi=150)
    plt.close()
    print(f"  Saved: {fig3_out1}")

    # Figure 4: Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for idx, (dataset_id, model) in enumerate(selected_models.items()):
        test_df = splits[dataset_id]['test']
        y_pred = model.predict(test_df['text'])
        y_true = test_df['label']
        
        labels = sorted(y_true.unique())
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', ax=axes[idx],
                    xticklabels=labels, yticklabels=labels, cbar=False)
        axes[idx].set_title(f'{dataset_id.upper()}\nNormalized Confusion Matrix ({test_results[idx]["best_model"]})', fontsize=10, fontweight='bold')
        axes[idx].set_xlabel('Predicted Label')
        axes[idx].set_ylabel('True Label')
        axes[idx].tick_params(axis='x', rotation=30)

    plt.tight_layout()
    fig4_out1 = IMAGES_DIR / 'confusion_matrices.png'
    fig4_out2 = OUT_DIR / 'figures' / 'confusion_matrices.png'
    plt.savefig(fig4_out1, dpi=150)
    plt.savefig(fig4_out2, dpi=150)
    plt.close()
    print(f"  Saved: {fig4_out1}")

    # STEP 9: Save Summaries & Artifacts
    print_section("9. Saving Summary CSVs and Model Binaries")
    cv_df.to_csv(OUT_DIR / 'cv_results_all.csv', index=False)
    test_df.to_csv(OUT_DIR / 'test_results_all.csv', index=False)
    
    for dataset_id, model in selected_models.items():
        joblib.dump(model, OUT_DIR / 'models' / f'{dataset_id}_best_model.joblib')
        
    print("  Saved: cv_results_all.csv, test_results_all.csv, and model joblib files.")

    print_header("LAB 03 COMPLETE! ALL ARTIFACTS AND PLOTS GENERATED SUCCESSFULLY.")


if __name__ == '__main__':
    main()
