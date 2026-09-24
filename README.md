# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using transaction data and a Random Forest classification model.

## Project Overview

Credit card fraud detection is a challenging classification problem because fraudulent transactions are very small compared to normal transactions.

In this project, the transaction dataset is cleaned and analyzed before training a machine learning model. The trained model is then used in a Streamlit web application to predict whether a transaction is normal or potentially fraudulent.

## Features

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Fraud and normal transaction analysis
- Random Forest classification
- Model evaluation using ROC-AUC and PR-AUC
- Fraud probability prediction
- Risk level display
- Interactive Streamlit web application

## Dataset

The project uses the Credit Card Fraud Detection dataset from Kaggle.

The dataset contains:

- 284,807 transactions originally
- 30 input features
- `Time` and `Amount` features
- `V1` to `V28` anonymized features
- `Class` as the target variable

Where:

- `0` = Normal transaction
- `1` = Fraudulent transaction

After removing duplicate records, 283,726 transactions were used for analysis and model training.

## Data Analysis

The following analysis was performed:

- Checked missing values
- Checked duplicate records
- Removed duplicate transactions
- Analyzed class distribution
- Studied transaction amounts
- Compared average transaction amounts
- Analyzed fraud transactions over time
- Checked feature correlation with the target variable

After cleaning:

- Normal transactions: 283,253
- Fraud transactions: 473

## Machine Learning

A Random Forest Classifier was used for fraud detection.

The data was divided into training and testing sets using an 80:20 split. StandardScaler was also applied before training the model.

The model was configured with:

- 100 decision trees
- Balanced class weights
- Random state: 42

## Model Performance

The model achieved the following results on the test data:

| Metric | Score |
|--------|------:|
| Fraud Precision | 0.97 |
| Fraud Recall | 0.71 |
| Fraud F1-Score | 0.82 |
| ROC-AUC | 0.9246 |
| PR-AUC | 0.7958 |

The confusion matrix was:

```text
[[56649     2]
 [   28    67]]