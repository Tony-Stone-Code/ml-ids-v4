# Defense One-Pager (Plain English)

## Project Title
AI-Based Intrusion Detection System (UNSW-NB15)

## Problem
Networks are attacked frequently. Manual analysis is too slow, so we need automated detection and classification.

## Solution (In One Sentence)
We trained machine learning models to detect and classify attacks from network traffic data and built a dashboard to display results.

## Dataset
UNSW-NB15 benchmark dataset with 10 classes (Normal + 9 attack types). Official train/test split used for fair evaluation.

## Method (Simple Steps)
1. Clean data (fix missing or inconsistent values).
2. Convert text fields to numbers and scale features.
3. Add behavior-based ratio features.
4. Train models and compare performance.
5. Deploy a Streamlit app for inference and visualization.

## Models Compared
- Logistic Regression
- Decision Tree
- XGBoost (best)

## Results (Key Numbers)
- Best accuracy: 76.60% (XGBoost)
- Macro precision/recall/F1 reported to reflect rare classes

## Demo Steps (What to Do)
1. Run: streamlit run app.py
2. Upload: data/UNSW_NB15_testing-set.csv
3. Explain: accuracy, precision/recall/F1, confusion matrix

## Why Macro and Weighted Metrics
- Macro treats all classes equally (good for rare attacks).
- Weighted reflects real class proportions.

## Limitations
- Research dataset, not live traffic.
- Performance may drop on very different data.

## Future Work
- Real-time ingestion pipeline
- Retraining on fresh data
- Integration into monitoring tools
