# Demo Script (Step-by-Step)

## Goal
Show the system running end-to-end and explain results in simple terms.

## Before you start
- Ensure Python is installed.
- The dataset is in data/ (use the official test set for metrics).

## 1) Launch the app
```bash
streamlit run app.py
```
Open the URL shown in the terminal.

## 2) Upload a dataset
Recommended for metrics:
- data/UNSW_NB15_testing-set.csv

Optional for class-imbalance demo:
- data/demo_samples/UNSW_NB15_testing-set_balanced_2000_seed42.csv

## 3) Explain what appears
Say:
- "The model predicts the attack type for each record."
- "Because this file has labels, we can compute accuracy, precision, recall, and F1."
- "The confusion matrix shows where the model confuses classes."

## 4) If asked about fairness
Say:
- "Macro scores treat each class equally, so rare attacks are not ignored."
- "Weighted scores show overall performance with real class proportions."

## 5) Wrap-up
Say:
- "This is a proof-of-concept IDS. With live data and tuning, it can be extended into a real system."

## Common issues
- If metrics do not show, the file likely does not have attack_cat labels.
- If results look odd, use the official test set.
