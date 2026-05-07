# AI-IDS (UNSW-NB15)

AI-based Intrusion Detection System that classifies network traffic into Normal + attack categories using the UNSW-NB15 benchmark dataset.

## Problem Statement

Modern networks face frequent cyber attacks, and manual inspection of network logs is too slow to keep up. The goal of this project is to automate detection and classify attack types so a defender can respond faster and more accurately.

## Why This Project Matters

- Faster detection reduces the time attackers stay inside a network.
- Classification helps prioritize response actions by attack type.
- The solution is reproducible and can be extended to real deployments.

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
