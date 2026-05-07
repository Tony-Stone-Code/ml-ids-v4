# AI-IDS (UNSW-NB15)

AI-based Intrusion Detection System that classifies network traffic into Normal + attack categories using the UNSW-NB15 benchmark dataset.

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
