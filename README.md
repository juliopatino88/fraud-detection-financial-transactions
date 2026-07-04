# Fraud Detection in Financial Transactions: Logistic Regression vs. Naive Bayes

## Overview
This project analyzes **6.3 million simulated financial transactions** to detect fraudulent activity using two supervised learning algorithms — Logistic Regression and Naive Bayes. The goal is to compare model performance on a highly imbalanced dataset where fraud accounts for only **0.13%** of all transactions.

This is a two-part analysis completed as coursework for **DATA 430 (Machine Learning)** at the University of Maryland Global Campus.

## Business Problem
Financial institutions lose billions annually to fraudulent transactions. Detecting fraud is challenging because:
- Fraudulent transactions are extremely rare (class imbalance)
- False negatives (missed fraud) are costly
- Models must balance precision and recall — flagging every transaction as fraud isn't useful

## Dataset
- **Source:** [Kaggle — Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)
- **Size:** 6,362,620 transactions
- **Features:** Transaction type, amount, origin/destination balances, and fraud label
- **Class distribution:** 99.87% legitimate, 0.13% fraudulent

## Methods

### Part 1: Logistic Regression
- Exploratory data analysis and feature engineering
- Train/test split (80/20)
- Logistic regression with default and tuned hyperparameters
- Evaluation using accuracy, precision, recall, F1-score, and confusion matrix

### Part 2: Naive Bayes
- Gaussian Naive Bayes classifier on the same dataset
- Direct comparison with logistic regression results
- Analysis of how each algorithm handles class imbalance

## Key Findings
- Both models achieved high overall accuracy (>99%), but accuracy is misleading with imbalanced classes
- **Logistic Regression** performed better at identifying actual fraud cases (higher recall)
- The **class imbalance** (0.13% fraud) is the primary challenge — techniques like SMOTE or undersampling could improve recall
- Precision-recall tradeoff analysis showed that optimizing for recall (catching more fraud) comes at the cost of more false positives

## Tools & Libraries
- Python (pandas, NumPy, scikit-learn, matplotlib, seaborn)
- Jupyter Notebook

## Files
- `Assignment_1_Logistic_Regression.ipynb` — Logistic regression analysis
- `Assignment_2_Naive_Bayes.ipynb` — Naive Bayes analysis and model comparison

## Author
**Julio Patiño**
- [LinkedIn](https://www.linkedin.com/in/juliopatino88/)
- [GitHub](https://github.com/juliopatino88)
