#  House Price Prediction using Multiple Regression Models

An end-to-end machine learning project that predicts house prices using multiple regression algorithms on the California Housing Dataset.

---

##  Project Overview

This project implements and compares multiple regression algorithms for predicting house prices using socioeconomic and geographical features from the California Housing dataset.

The complete machine learning workflow includes:

- Business understanding
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Development
- Hyperparameter Tuning
- Model Evaluation
- Residual Analysis
- Model Comparison
- Report Documentation

The objective is to evaluate the performance of different regression models and identify the most suitable algorithm for house price prediction.

---

##  Repository Structure

```
House-Price-Prediction/
│
├── README.md
├── LICENSE
├── requirements.txt
│
├── notebook/
│   └── lab1.ipynb
│
├── report/
│   └── MDI3003_Lab01_HousePrice_Report.pdf
│
├── images/
│   ├── target_distribution.png
│   ├── correlation_heatmap.png
│   ├── scatterplots.png
│   ├── actual_vs_predicted.png
│   ├── residual_plot.png
│   └── feature_importance.png
│
└── outputs/
    └── model_results.csv
```

---

##  Dataset

**Dataset:** California Housing Dataset

Source:

https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html

### Dataset Statistics

- Samples: **20,640**
- Features: **8**
- Target: Median House Value
- Missing Values: **0**
- Duplicate Rows: **0**

Features:

- Median Income
- House Age
- Average Rooms
- Average Bedrooms
- Population
- Average Occupancy
- Latitude
- Longitude

---

##  Regression Models Implemented

The following regression models were implemented and evaluated:

- Dummy Regressor (Baseline)
- Simple Linear Regression
- Multiple Linear Regression
- Ridge Regression
- Lasso Regression
- Random Forest Regressor

---

##  Machine Learning Workflow

1. Load Dataset
2. Data Inspection
3. Exploratory Data Analysis
4. Feature Scaling
5. Train/Test Split
6. Model Training
7. Hyperparameter Tuning
8. Cross Validation
9. Model Evaluation
10. Residual Analysis
11. Model Comparison

---

##  Evaluation Metrics

Model performance was evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R² Score)

---

## Final Results

| Model | MAE | RMSE | R² |
|-------|------:|------:|------:|
| Dummy Regressor | 0.9061 | 1.1449 | -0.0002 |
| Simple Linear Regression | 0.6299 | 0.8421 | 0.4589 |
| Multiple Linear Regression | 0.5332 | 0.7456 | 0.5758 |
| Ridge Regression | 0.5332 | 0.7455 | 0.5758 |
| Lasso Regression | 0.5332 | 0.7448 | 0.5766 |
| Random Forest Regressor | **0.3304** | **0.5077** | **0.8033** |

---

##  Best Model

**Random Forest Regressor**

Performance:

- RMSE: **0.5077**
- MAE: **0.3304**
- R² Score: **0.8033**

Random Forest achieved the highest predictive performance among all implemented models due to its ability to capture nonlinear relationships and complex feature interactions.

---

##  Exploratory Data Analysis

The project includes:

- Distribution of Target Variable
- Correlation Heatmap
- Scatter Plots
- Feature Importance
- Actual vs Predicted Plot
- Residual Analysis

---

##  Technologies Used

- Python 3
- Google Colab
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

---

##  Running the Project

Clone the repository

```bash
git clone https://github.com/<your-username>/House-Price-Prediction.git
```

Move into the project

```bash
cd House-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the notebook

```
Open notebook/lab1.ipynb
```

---

##  Learning Outcomes

This project demonstrates:

- End-to-end Machine Learning workflow
- Regression Analysis
- Feature Engineering
- Hyperparameter Tuning
- Cross Validation
- Model Evaluation
- Data Visualization
- Reproducible ML Pipelines

---

##  References

- Scikit-learn Documentation
- California Housing Dataset
- An Introduction to Statistical Learning
- The Elements of Statistical Learning

---
