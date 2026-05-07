# AI-IDS (UNSW-NB15)

AI-based Intrusion Detection System that classifies network traffic into Normal + attack categories using the UNSW-NB15 benchmark dataset.

## Problem Statement (2026 Context)

By 2026, enterprise and campus networks face faster, stealthier attacks and higher traffic volume (cloud services, IoT, and remote access). Manual inspection of logs and alerts cannot scale, and most student IDS projects stop at offline modeling without a usable pipeline. The goal here is to deliver an end-to-end, deployable IDS that detects and classifies attack types so responders can prioritize actions quickly and reliably.

## Why This Project Matters

- Faster detection reduces the time attackers stay inside a network.
- Classification helps prioritize response actions by attack type.
- The solution is reproducible and can be extended to real deployments.

## Approach Summary

1. **Use a trusted benchmark** (UNSW-NB15) with an official train/test split to avoid evaluation bias.
2. **Clean and normalize raw data** to handle missing values, mixed types, and inconsistent labels.
3. **Encode categorical features** and scale numeric features for stable model behavior.
4. **Engineer traffic behavior ratios** to capture asymmetric attack patterns.
5. **Train and compare multiple models** (Logistic Regression, Decision Tree, XGBoost).
6. **Evaluate with fairness-aware metrics** (macro/weighted precision, recall, F1) and confusion matrices.
7. **Deploy a Streamlit app** so non-technical users can run and interpret results.

## Improvements Over Typical Student IDS Projects

- End-to-end pipeline from raw data through preprocessing, training, and deployment.
- Multiple models compared with consistent evaluation metrics.
- Clear UI for non-technical users with built-in charts and reports.
- Balanced demo subsets for class-imbalance walkthroughs.

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Models and Data

- The app expects trained model artifacts in models/.
- If models are missing, train them first:

```bash
python src/convert_to_parquet.py
python src/train.py
```

- Recommended demo input (with labels): data/UNSW_NB15_testing-set.csv

## Demo Subsets (Balanced)

Generate balanced demo subsets for class-imbalance walkthroughs:

```bash
python tools/make_balanced_demo.py --input data/UNSW_NB15_testing-set.csv --seed 42 --allow-upsample --out-dir data/demo_samples
```

## Documentation

- Full technical write-up: documentation.md
- Defense materials: documents/

## Streamlit Cloud Notes

- Ensure models/ is included in the repo, or train on first run.
- The app will stop with a clear error if model artifacts are missing.

## Dataset Citation

- N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems", MILCOM 2015.
- Dataset page: https://research.unsw.edu.au/projects/unsw-nb15-dataset
