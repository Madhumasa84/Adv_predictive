# MDI3003 Lab 05: Product and Brand Sentiment Prediction from Tweet Data
## From Classical NLP Baselines to Sequence Models (BiLSTM) and Pretrained Twitter Transformers (BERTweet)

A comprehensive, reproducible predictive analytics system developed in accordance with the **MDI3003 Lab 05 Manual (Rev 2.0, August 2026)**:
1. **Verified Multi-Class Tweet Corpus**: Twitter US Airline Sentiment Benchmark (14,640 human-labeled customer tweets, 3 classes: Negative [62.7%], Neutral [21.2%], Positive [16.1%], covering 6 major US commercial airlines).
2. **Model Benchmark Suite**: `DummyClassifier` (Trivial Majority Baseline), `VADER` (Rule-Based Social Media Lexicon), `MultinomialNB` (Word Unigram+Bigram TF-IDF), `LogisticRegression` (Balanced Multinomial Softmax), `LinearSVC` (Large-Margin Linear Discriminative Classifier), `BiLSTM` (128d Trainable Embeddings + Bidirectional Recurrence), and `BERTweet-base` (`vinai/bertweet-base` 135M-parameter Twitter-Domain Pretrained Transformer).
3. **Leakage-Safe Methodology**: 80/20 Stratified Holdout Split ($N_{train}=11,712, N_{test}=2,928$) with zero tweet ID overlap; strict exclusion of target circularity fields (`negativereason`, `airline_sentiment_confidence`) and personal identifiers (`name`, `tweet_coord`).
4. **Minimal Tweet Normalization**: URL replacement with `<URL>`, handle normalization with `<USER>`, while preserving sentiment-bearing cues (emojis, negation modifiers, punctuation, hashtags).
5. **Pre-Test Model Selection**: Selected `LogisticRegression (Balanced)` based on 5-Fold Stratified Cross-Validation Macro F1 (**0.7444 ± 0.0082**).
6. **One-Time Locked Test Evaluation**: Test Accuracy: **79.17%** (2,318/2,928 correct), Macro F1: **0.7442**, Weighted F1: **0.7958**, Inference Latency: **0.024 ms/sample** (>41,600 predictions/sec).
7. **Entity Stratification & Minimum Support**: Comparative sentiment and error analysis across 6 airline entities exceeding minimum support ($N \ge 30$).
8. **Business-Critical Error Analysis**: 10 in-depth interpreted case studies detailing linguistic challenge categories (sarcasm, negation, slang, emojis, transactional imperatives, career inquiries), root causes, operational risks, and mitigations.
9. **Advanced Extensions**: BiLSTM 3-seed uncertainty benchmark (**0.7285 ± 0.0087** Macro F1) and fine-tuned BERTweet-base (**0.8192** Macro F1, 95% Bootstrap CI: **[0.8024, 0.8338]**).

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

### 4. Generate Official Word (.docx) Report
```bash
python generate_word_doc.py
```

### 5. Launch Interactive Jupyter Notebook
```bash
jupyter notebook 23MID0444_Lab05_TweetSentiment.ipynb
```

---

## Repository Structure

```
lab5/
├── main.py                                      # End-to-end reproducible machine learning pipeline
├── generate_report.py                           # ReportLab automated PDF report generator
├── generate_word_doc.py                         # python-docx automated Word document generator
├── lab5.ipynb                                   # Master Jupyter Notebook
├── lab5da.ipynb                                 # Standardized Jupyter Notebook copy
├── 23MID0444_Lab05_TweetSentiment.ipynb         # Official assessed student notebook
├── requirements.txt                             # Python library dependencies
├── README.md                                    # Project documentation
├── 23MID0444_Lab05_README.md                    # Official submission documentation
├── Lab05_report.pdf                             # Master PDF Laboratory Technical Report
├── 23MID0444_Lab05_Report.pdf                   # Student submission PDF report
├── Lab05_report_up.pdf                          # Alternate report build
├── lab_5rep.pdf                                 # Standardized report copy
├── Lab05_report.docx                            # Master Word Laboratory Technical Report
├── 23MID0444_Lab05_Report.docx                  # Student submission Word report
├── Lab05_report_up.docx                         # Alternate Word report build
├── lab_5rep.docx                                # Standardized Word report copy
├── VL2026270103889_AST05.pdf                   # Faculty Laboratory Manual (Dr. Durgesh Kumar)
├── 23MID0444_Lab05_CV_Results.csv               # 5-Fold Cross-Validation metrics
├── 23MID0444_Lab05_Test_Results.csv             # Locked test set metrics
├── 23MID0444_Lab05_Test_Predictions.csv         # Locked test predictions
├── 23MID0444_Lab05_NewCustomer_Predictions.csv  # Predictions on new live customer tweets
├── 23MID0444_Lab05_Error_Analysis.csv           # Detailed error analysis cases
├── models/                                      # Serialized fitted pipelines (.joblib)
│   └── selected_pipeline.joblib
├── figures/                                     # Diagnostic visual artifacts (.png)
│   ├── class_distribution.png
│   ├── tweet_length.png
│   ├── cv_comparison.png
│   ├── confusion_matrices.png
│   ├── per_class_metrics.png
│   ├── error_rate_by_entity.png
│   ├── bilstm_learning_curves.png
│   └── perf_vs_runtime.png
├── images/                                      # Standardized copy of visual artifacts
│   ├── class_distribution.png
│   ├── tweet_length.png
│   ├── cv_comparison.png
│   ├── confusion_matrices.png
│   ├── per_class_metrics.png
│   ├── error_rate_by_entity.png
│   ├── bilstm_learning_curves.png
│   └── perf_vs_runtime.png
├── advanced/                                    # Advanced neural & Transformer artifacts
│   ├── bilstm_multiseed_results.csv
│   ├── bertweet_run_config.json
│   └── efficiency_comparison.csv
└── lab05_outputs/                               # Generated results, artifacts, models, & manifests
    ├── dataset_card.json
    ├── train_manifest.csv
    ├── test_manifest.csv
    ├── baseline_results.csv
    ├── cv_results.csv
    ├── test_results.csv
    ├── test_predictions.csv
    ├── class_report.csv
    ├── entity_sentiment_distribution.csv
    ├── error_analysis.csv
    └── selected_pipeline.joblib
```

---

## Dataset Profile & Governance

- **Benchmark**: Twitter US Airline Sentiment (Crowdflower / Kaggle)
- **License**: Creative Commons Attribution-NonCommercial-ShareAlike (CC BY-NC-SA 4.0)
- **Total Records**: 14,640 tweets (11,712 Train / 2,928 Test — 80/20 Stratified)
- **Target Variable**: `airline_sentiment` (Multiclass: `negative`, `neutral`, `positive`)
- **Direct Identifier Exclusion**: `tweet_id`, `name` dropped from all feature vectors.
- **Leakage Controls**: Purged `negativereason`, `negativereason_confidence`, `airline_sentiment_confidence`, `tweet_coord`, `user_timezone`.

### Target Class Distribution

| Sentiment Class | Operational Meaning | Train Count (80%) | Test Count (20%) | Total Records | Corpus Share (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Negative** | Complaints, delays, lost luggage, cancellations | 7,343 | 1,835 | 9,178 | 62.69% |
| **Neutral** | Flight inquiries, schedule queries, factual updates | 2,479 | 620 | 3,099 | 21.17% |
| **Positive** | Staff compliments, praise, gratitude | 1,890 | 473 | 2,363 | 16.14% |
| **Total** | Stratified Holdout Benchmark | **11,712** | **2,928** | **14,640** | **100.00%** |

---

## Model Benchmark Results

### 1. 5-Fold Stratified Cross-Validation (Identical Training Folds)

| Model Architecture | Feature Representation | Macro F1 Mean | Macro F1 SD | Weighted F1 | Accuracy Mean | Fit Time |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Selected)** | Word Unigram+Bigram TF-IDF | **0.7444** | **±0.0082** | **0.7955** | **79.28%** | **4.26 s** |
| **LinearSVC** | Word Unigram+Bigram TF-IDF | 0.7395 | ±0.0117 | 0.7969 | 80.06% | 2.05 s |
| **MultinomialNB** | Word Unigram+Bigram TF-IDF | 0.5858 | ±0.0151 | 0.6901 | 73.63% | 0.73 s |
| **VADER Lexicon** | Rule-Based Valence Dictionary | 0.5206 | N/A | 0.5614 | 55.23% | 0.12 s |
| **DummyClassifier** | Majority Class Prior | 0.2568 | ±0.0001 | 0.4828 | 62.67% | 0.02 s |

*Pre-Test Selection Decision: `LogisticRegression` selected based on highest cross-validation Macro F1, tight fold stability, and calibrated probabilistic outputs.*

### 2. One-Time Locked Holdout Test Evaluation (N=2,928 Tweets)

| Evaluation Metric | Score | Operational Benchmark Interpretation |
| :--- | :---: | :--- |
| **Test Accuracy** | **79.17%** (2,318 / 2,928) | High overall correctness across full holdout cohort |
| **Macro Precision** | **0.7423** | Balanced across Negative (89.1%), Neutral (59.0%), and Positive (74.6%) |
| **Macro Recall** | **0.7512** | High sensitivity across minority classes (Neutral: 71.6%, Positive: 69.6%) |
| **Macro F1-Score** | **0.7442** | Primary locked benchmark metric; matches CV estimate (0.7444) |
| **Weighted F1-Score** | **0.7958** | Frequency-weighted metric reflecting natural airline inquiry prevalence |
| **Inference Latency** | **0.024 ms / sample** | >41,600 predictions / second on single CPU core |

### 3. Advanced Deep Learning & Pretrained Transformer Extension

| Model Architecture | Parameters / Dim | Macro F1 | 95% Bootstrap CI / Uncertainty | Weighted F1 | Training Time | Inference Latency | Model Disk Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Sparse (1,2-grams) | 0.7442 | CV SD: ±0.0082 | 0.7958 | **4.26 s** | **0.024 ms** | **14.8 MB** |
| **LinearSVC** | Sparse (1,2-grams) | 0.7395 | CV SD: ±0.0117 | 0.7969 | **2.05 s** | **0.018 ms** | **7.4 MB** |
| **BiLSTM (Neural)** | 1.98M params | 0.7285 | ±0.0087 (3-Seed SD) | 0.7910 | 24.96 s | 0.212 ms | 45.2 MB |
| **BERTweet-base** | 135M params | **0.8192** | **[0.8024, 0.8338]** | **0.8384** | 389.64 s | 2.617 ms | 514.6 MB |

---

## Entity-Stratified Sentiment Breakdown ($N_{test} \ge 30$)

| Airline Entity | Test Support (N) | Predicted Negative | Predicted Neutral | Predicted Positive | Test Error Rate | Key Operational Finding |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **United** | 774 | 65.4% | 22.5% | 12.1% | 21.3% | Largest corpus share; high lost baggage complaints |
| **US Airways** | 565 | **73.5%** | 16.8% | 9.7% | 19.8% | Highest negative ratio due to Feb 2015 merger/cancellations |
| **American** | 535 | 64.7% | 24.3% | 11.0% | 21.8% | High rebooking friction and customer phone hold times |
| **Southwest** | 506 | 45.1% | 32.4% | 22.5% | 21.1% | Strong customer service rapport & boarding inquiries |
| **Delta** | 456 | 44.7% | 34.0% | 21.3% | 20.4% | High neutral operational telemetry (schedule/gate updates) |
| **Virgin America** | 92 | 39.1% | 37.0% | **23.9%** | **18.5%** | Lowest error rate & highest positive sentiment ratio |

---

## Quality Assurance & Verification Manifest

| Acceptance Criterion | Verification Status | Verified Evidence |
| :--- | :---: | :--- |
| **Target Validity** | **PASS** | 3-Class supervised text classification with explicit airline entity grounding |
| **Data Governance** | **PASS** | Dataset card, 5 public benchmark audits, leakage exclusions verified |
| **Minimal Normalization** | **PASS** | Emojis, negation, punctuation, hashtags preserved; URLs/mentions normalized |
| **Leakage-Safe Partitioning** | **PASS** | 80/20 Stratified split; zero ID overlap; TF-IDF fit inside training folds |
| **Baseline Comparisons** | **PASS** | DummyClassifier (F1=0.2568) and VADER (F1=0.5206) implemented |
| **5-Fold Cross-Validation** | **PASS** | MultinomialNB, LogisticRegression, LinearSVC compared under identical folds |
| **Pre-Test Model Selection** | **PASS** | LogisticRegression chosen based purely on CV Macro F1 (0.7444) |
| **Locked Test Evaluation** | **PASS** | Accuracy: 79.17%, Macro F1: 0.7442, Weighted F1: 0.7958, Latency: 0.024 ms |
| **Entity Stratification** | **PASS** | N >= 30 support threshold enforced across 6 airlines |
| **Error Analysis** | **PASS** | 10 inspected error cases analyzed across distinct linguistic challenges |
| **Advanced Extensions** | **PASS** | BiLSTM 3-seed uncertainty and BERTweet 95% bootstrap CI reported |
| **Model Serialization** | **PASS** | Pipeline serialized to `.joblib` with 100% deterministic reload verification |
| **Technical Documentation** | **PASS** | Publication-grade PDF and Word reports covering all rubric items |

---

## Author & Course Information

- **Student**: Madhusudhanan G
- **Registration No.**: 23MID0444
- **Course**: MDI3003 - Advanced Predictive Analytics
- **Faculty**: Dr. Durgesh Kumar
- **School**: School of Computer Science and Engineering (SCOPE), VIT Vellore
- **Academic Term**: Fall Semester 2026-2027
