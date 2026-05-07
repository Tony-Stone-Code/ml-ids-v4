# UNSW-NB15 Network Intrusion Detection System

A Machine Learning-based **Network Intrusion Detection System (NIDS)** that classifies network traffic into 10 categories (Normal + 9 attack types) using the UNSW-NB15 benchmark dataset.

## Project Overview

| Component          | Details                                                                               |
| ------------------ | ------------------------------------------------------------------------------------- |
| **Dataset**        | UNSW-NB15 (82,332 training / 175,341 testing samples)                                 |
| **Models**         | Logistic Regression, Decision Tree, XGBoost                                           |
| **Best Accuracy**  | 76.60% (XGBoost) on official test split                                               |
| **Attack Classes** | Analysis, Backdoor, DoS, Exploits, Fuzzers, Generic, Reconnaissance, Shellcode, Worms |
| **Web App**        | Streamlit-based inference dashboard                                                   |

## Project Structure

```
ai_ids/
├── app.py                      # Streamlit web application
├── data/                       # Dataset files (CSV + Parquet)
├── documents/                  # Defense materials (overview, Q&A, slides)
├── models/                     # Trained model artifacts (.joblib)
├── plots/                      # Generated visualizations
├── notebooks/
│   └── EDA.ipynb               # Exploratory Data Analysis
├── src/
│   ├── preprocessing.py        # Data cleaning, feature engineering, encoding
│   ├── train.py                # Model training, evaluation, visualization
│   └── convert_to_parquet.py   # CSV -> Parquet converter
├── tools/
│   └── make_balanced_demo.py   # Balanced demo subset generator
├── requirements.txt
├── README.md
└── documentation.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv` in `data/`.

```bash
python src/convert_to_parquet.py
```

### 3. Train Models

```bash
python src/train.py
```

This generates trained models in `models/` and evaluation plots in `plots/`.

### 4. Launch Web App

```bash
streamlit run app.py
```

Open `http://localhost:8501` and upload any UNSW-NB15 CSV file for classification.

## Balanced Demo Subsets (Class-Imbalance Demos)

Generate a balanced multiclass demo subset from the official test set. This is useful for UI demos and class-imbalance walkthroughs, but not for reporting final model accuracy.

```bash
python tools/make_balanced_demo.py --input data/UNSW_NB15_testing-set.csv --total-size 2000 --seed 42 --allow-upsample --out-dir data/demo_samples
```

The script balances on `attack_cat` by default and writes a randomized subset to `data/demo_samples/`.

Fully automated (generates multiple sizes by default: 200, 500, 1000, 2000, 5000):

```bash
python tools/make_balanced_demo.py --input data/UNSW_NB15_testing-set.csv --seed 42 --allow-upsample --out-dir data/demo_samples
```

Generate multiple sizes in one run:

```bash
python tools/make_balanced_demo.py --input data/UNSW_NB15_testing-set.csv --total-sizes 200,1000,5000 --seed 42 --allow-upsample --out-dir data/demo_samples
```

Command template:

```bash
python tools/make_balanced_demo.py --input <CSV_PATH> [--total-size N | --total-sizes N1,N2,...] --seed SEED --allow-upsample --out-dir <OUTPUT_DIR>
```

Arguments:

- `--input` (required): Path to the source CSV file.
- `--total-size`: Create a single balanced subset with this total row count.
- `--total-sizes`: Create multiple balanced subsets at once (comma-separated list).
- `--seed`: Random seed for reproducible sampling.
- `--class-col`: Column to balance on (default: `attack_cat`).
- `--out-dir`: Output directory for generated subsets (default: `data/demo_samples`).
- `--allow-upsample`: Allow sampling with replacement when a class has too few rows.

## Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology:

```
1. Business Understanding    ->  Define IDS requirements & attack taxonomy
2. Data Understanding        ->  EDA on UNSW-NB15, class distribution analysis
3. Data Preparation          ->  Cleaning, encoding, scaling, feature engineering
4. Modeling                  ->  Train & compare 3 ML classifiers
5. Evaluation                ->  Accuracy, precision, recall, F1, confusion matrices
6. Deployment                ->  Streamlit web app for real-time inference
```

---

## End-to-End Pipeline

### High-Level Workflow

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Raw CSV     │---->│  Data        │---->│  Feature        │---->│  Model       │
│  Upload      │     │  Cleaning    │     │  Engineering    │     │  Inference   │
└─────────────┘     └──────────────┘     └─────────────────┘     └──────┬───────┘
                                                                        │
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐           │
│  Results     │<----│  Label       │<----│  Attack          │<----------┘
│  Dashboard   │     │  Decoding    │     │  Classification  │
└─────────────┘     └──────────────┘     └─────────────────┘
```

### Training Pipeline (Offline)

```
CSV/Parquet Data
       │
       ▼
┌─────────────────────────────────────────┐
│  1. DATA CLEANING                       │
│  • Strip whitespace from categoricals   │
│  • Coerce all numerics to float64       │
│  • Replace inf/NaN with 0               │
│  • Normalize attack_cat labels          │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  2. FEATURE ENGINEERING                 │
│  • 5 traffic asymmetry ratio features:  │
│    - byte_ratio  = sbytes/(sbytes+dby)  │
│    - pkt_ratio   = spkts/(spkts+dpkts)  │
│    - load_ratio  = sload/(sload+dload)  │
│    - mean_ratio  = smean/(smean+dmean)  │
│    - srv_density = ct_srv_src/(ct+1)    │
│  Total: 42 original + 5 engineered     │
│       = 47 features                     │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  3. ENCODING                            │
│  • LabelEncoder for categoricals:       │
│    - proto   (131 unique -> integers)   │
│    - service (13 unique -> integers)    │
│    - state   (7 unique -> integers)     │
│  • LabelEncoder for target (attack_cat) │
│    - 10 classes -> integers 0-9         │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  4. NORMALIZATION                       │
│  • MinMaxScaler fitted on training data │
│  • All 47 features scaled to [0, 1]    │
│  • Same scaler applied to test data     │
│    (no data leakage)                    │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  5. CLASS IMBALANCE HANDLING            │
│  • Compute sample weights per class:    │
│    weight = n_total / (n_classes x n_i) │
│  • Rare classes get higher weight:      │
│    Worms: 187x | Shellcode: 21.8x      │
│    Backdoor: 14.1x | Normal: 0.22x     │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  6. MODEL TRAINING                      │
│  • Logistic Regression (class_weight)   │
│  • Decision Tree (no constraints)       │
│  • XGBoost (1000 trees, sample_weight)  │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  7. EVALUATION & VISUALIZATION          │
│  • Accuracy, Classification Report      │
│  • Confusion Matrix Heatmaps            │
│  • Feature Importance Chart             │
│  • Model Comparison Bar Charts          │
│  • Per-Class Recall Comparison          │
└──────────────────┬──────────────────────┘
                   ▼
              Save to models/
         (14 .joblib artifacts)
```

### Inference Pipeline (Real-time in Streamlit)

```
User Uploads CSV
       │
       ▼
┌─────────────────────────────────────────┐
│  AUTO-DETECT FORMAT                     │
│  • Headered CSV -> read normally         │
│  • Headerless raw UNSW-NB15 CSV ->       │
│    assign 49 official column names      │
│  • Normalize variant names:             │
│    Backdoors->Backdoor, " Fuzzers"->Fuzz  │
└──────────────────┬──────────────────────┘
                   ▼
┌─────────────────────────────────────────┐
│  LOAD SAVED ARTIFACTS                   │
│  • le_proto.joblib, le_service.joblib   │
│  • le_state.joblib, scaler.joblib       │
│  • feature_columns.joblib               │
│  • selected model .joblib               │
└──────────────────┬──────────────────────┘
                   ▼
  transform_new() -> model.predict() -> Decode labels
       │
       ▼
  Dashboard: Accuracy, Confusion Matrix,
  Classification Report, Download CSV
```

---

## Techniques Used

### Data Preprocessing

| Technique               | Implementation                       | Rationale                                                                                                                                                 |
| ----------------------- | ------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LabelEncoder**        | `sklearn.preprocessing.LabelEncoder` | Converts categorical text to ordinal integers; tree-based models handle this natively via splits without needing one-hot encoding                         |
| **MinMaxScaler**        | `sklearn.preprocessing.MinMaxScaler` | Scales features to [0,1]; prevents large-range features (e.g., `sbytes` in millions) from dominating small-range features (e.g., binary flags)            |
| **Feature Engineering** | Custom ratio features                | Traffic asymmetry ratios are domain-specific indicators — attacks typically create one-directional traffic patterns unlike symmetric normal communication |
| **Type Coercion**       | `pd.to_numeric(errors='coerce')`     | Handles mixed-type columns from raw CSV files; replaces unparseable values with NaN -> 0                                                                   |

### Class Imbalance Strategies Evaluated

| Technique                                      | Tried    | Result                                                                                      | Kept       |
| ---------------------------------------------- | -------- | ------------------------------------------------------------------------------------------- | ---------- |
| **SMOTE** (Synthetic Minority Oversampling)    | Yes v1-v2 | Generated synthetic samples that didn't generalize to the differently-distributed test set | No |
| **SMOTE + class_weight** (Double Compensation) | Yes v1    | Over-compensated for rare classes, suppressing majority class accuracy                      | No |
| **class_weight='balanced'** (Native)           | Yes v3    | Sklearn-native weighting in LR and DT; improved rare class recall but hurt overall accuracy | Yes (LR only) |
| **sample_weight** (XGBoost)                    | Yes v4    | Manually computed balanced weights passed to XGBoost; best accuracy + rare class balance    | Yes (XGBoost) |

### Feature Encoding Approaches Evaluated

| Approach                       | Tried    | Features                       | Result                                                              |
| ------------------------------ | -------- | ------------------------------ | ------------------------------------------------------------------- |
| **One-Hot Encoding**           | Yes v1-v2 | 42 numeric + 151 one-hot = 193 | Created sparse, noisy features; 75.85% accuracy                     |
| **One-Hot + Variance Pruning** | Yes v2-v3 | 45 numeric + 40 one-hot = 85   | Pruned rare categories; 75.83% accuracy                             |
| **LabelEncoder** (final)       | Yes v4    | 42 + 5 engineered = 47         | Dense features, XGBoost handles ordinal splits; **76.60% accuracy** |

### Model Configurations

**Logistic Regression** — `max_iter=3000, solver='saga', penalty='l2', class_weight='balanced'`

**Decision Tree** — `max_depth=None, min_samples_split=2, min_samples_leaf=1`

**XGBoost (Best)** — `n_estimators=1000, max_depth=12, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, gamma=0.5, min_child_weight=3, reg_alpha=0.1, reg_lambda=1.0`

---

## Training Evolution History

The model pipeline went through **4 major iterations**, each addressing a specific bottleneck:

### v1 — Baseline

```
Pipeline:  One-Hot Encoding (193 features) + MinMaxScaler + SMOTE + class_weight
XGBoost:   200 trees, max_depth=6
Result:    ~65% accuracy
Problem:   Double compensation (SMOTE + class_weight) suppressed majority class accuracy
```

### v2 — Remove Double Compensation

```
Pipeline:  One-Hot + Variance Pruning (85 features) + MinMaxScaler + SMOTE only
XGBoost:   500 trees, max_depth=8, lr=0.05, regularization added
Result:    75.83% accuracy (+11%)
Problem:   SMOTE synthetic samples didn't generalize to differently-distributed test set
```

### v3 — Remove SMOTE

```
Pipeline:  One-Hot + Variance Pruning (85 features) + MinMaxScaler + native class_weight
XGBoost:   500 trees, max_depth=10, sample_weight
Result:    76.58% accuracy (+0.75%)
Insight:   Training on real data improved generalization
```

### v4 — LabelEncoder + Aggressive Tuning (Final)

```
Pipeline:  LabelEncoder (47 features) + MinMaxScaler + 5 ratio features + sample_weight
XGBoost:   1000 trees, max_depth=12, gamma=0.5
Result:    76.60% accuracy (best)
Key:       Replaced One-Hot with LabelEncoder, reducing noise from 85->47 features
```

### Accuracy Progression

```
v1 ████████████████████████████████░░░░░░░░░░░░░░░░░░  65.0%
v2 ████████████████████████████████████████░░░░░░░░░░  75.8%
v3 █████████████████████████████████████████░░░░░░░░░  76.6%
v4 █████████████████████████████████████████░░░░░░░░░  76.6%  <- Best
```

---

## Final Results

### Model Comparison (Official Test Set — 175,341 samples)

| Model               | Accuracy   | Macro Precision | Macro Recall | Macro F1 | Training Time |
| ------------------- | ---------- | --------------- | ------------ | -------- | ------------- |
| Logistic Regression | 63.47%     | 0.35            | 0.45         | 0.31     | ~936s         |
| Decision Tree       | 74.22%     | 0.64            | 0.50         | 0.50     | ~2s           |
| **XGBoost**         | **76.60%** | **0.61**        | **0.58**     | **0.56** | ~118s         |

### Per-Class Performance (XGBoost)

| Attack Category | Train Samples | Precision | Recall   | F1-Score |
| --------------- | ------------- | --------- | -------- | -------- |
| Analysis        | 677           | 0.05      | 0.04     | 0.05     |
| Backdoor        | 583           | 0.36      | 0.08     | 0.13     |
| DoS             | 4,089         | 0.34      | 0.73     | 0.47     |
| Exploits        | 11,132        | 0.81      | 0.56     | 0.66     |
| Fuzzers         | 6,062         | 0.71      | 0.20     | 0.32     |
| **Generic**     | 18,871        | **0.99**  | **0.98** | **0.99** |
| **Normal**      | 37,000        | 0.79      | **0.97** | 0.87     |
| Reconnaissance  | 3,496         | 0.88      | 0.77     | 0.82     |
| Shellcode       | 378           | 0.55      | 0.73     | 0.63     |
| **Worms**       | **44**        | 0.64      | **0.75** | **0.69** |

### Generated Visualizations (plots/)

| Plot                              | Description                                     |
| --------------------------------- | ----------------------------------------------- |
| `confusion_matrix_*.png`          | Normalized confusion matrix per model (3 plots) |
| `feature_importance_xgboost.png`  | Top 20 most important features                  |
| `model_comparison.png`            | Accuracy & training time bar charts             |
| `per_class_recall_comparison.png` | Per-class recall across all 3 models            |
| `eda_class_distribution.png`      | Train vs test attack distribution               |
| `eda_correlation_matrix.png`      | Full feature correlation heatmap                |
| `eda_top_correlations.png`        | Top 15 features correlated with attack label    |
| `eda_attack_patterns.png`         | Feature distributions by attack type            |
| `eda_engineered_features.png`     | Ratio feature boxplots by attack type           |

---

## Streamlit Web Application

The deployment interface provides:

- **Model Selection**: Choose between Logistic Regression, Decision Tree, or XGBoost
- **Auto-Format Detection**: Handles both headered and raw headerless UNSW-NB15 CSVs
- **Real-time Inference**: Upload CSV -> preprocess -> predict -> display results
- **Evaluation Dashboard**: Accuracy metrics, confusion matrix heatmap, classification report
- **Prediction Export**: Download predictions as CSV for further analysis

---

## Technologies

| Category            | Tools                 |
| ------------------- | --------------------- |
| **Language**        | Python 3.13           |
| **ML Framework**    | scikit-learn, XGBoost |
| **Data Processing** | Pandas, NumPy         |
| **Visualization**   | Matplotlib, Seaborn   |
| **Web App**         | Streamlit             |
| **Serialization**   | Joblib                |
| **Data Format**     | Parquet (via PyArrow) |

## Dataset Citation

- N. Moustafa and J. Slay, "UNSW-NB15: a comprehensive data set for network intrusion detection systems", MILCOM 2015.
- Dataset page: https://research.unsw.edu.au/projects/unsw-nb15-dataset

## License

Academic project — UNSW-NB15 dataset by the Australian Centre for Cyber Security (ACCS), UNSW Canberra.
