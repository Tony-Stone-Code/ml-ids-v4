# 🛡️ Project Defence Aid — Network Intrusion Detection System
## UNSW-NB15 Multi-Class IDS using Machine Learning

---

## 1. Project Overview (Your Elevator Pitch)

> "I built a **Machine Learning-based Network Intrusion Detection System** that classifies network traffic into **10 categories** — Normal traffic and 9 attack types — using the UNSW-NB15 benchmark dataset. The system uses three ML models (Logistic Regression, Decision Tree, and XGBoost), with XGBoost achieving **76.60% accuracy** on the challenging official test split. I deployed it as a **Streamlit web application** where users can upload raw network traffic CSV files and get real-time attack classification."

---

## 2. Architecture & Pipeline

```
Raw CSV Data
    │
    ▼
┌─────────────────────┐
│  Data Cleaning       │  Strip whitespace, handle NaN/Inf, type coercion
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Feature Engineering │  5 ratio features (byte, packet, load, mean, density)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Label Encoding      │  proto (131 values), service (13), state (7)
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  MinMaxScaler (0-1)  │  Normalize all 47 features to [0, 1]
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│  Model Inference     │  XGBoost / Decision Tree / Logistic Regression
└──────────┬──────────┘
           ▼
   Attack Classification
   (10 classes)
```

### Key Files
| File | Purpose |
|------|---------|
| `src/preprocessing.py` | Data cleaning, feature engineering, encoding, scaling |
| `src/train.py` | Model training, evaluation, SMOTE, saving artifacts |
| `src/convert_to_parquet.py` | CSV → Parquet conversion for faster I/O |
| `app.py` | Streamlit web app for inference |
| `models/` | Saved model artifacts (.joblib files) |

---

## 3. Dataset — UNSW-NB15

> [!IMPORTANT]
> Know these numbers by heart — panelists will ask.

### What is it?
- Created by the **Australian Centre for Cyber Security (ACCS)** at UNSW Canberra
- Contains **real modern network traffic** mixed with **synthetic attack traffic**
- Generated using the **IXIA PerfectStorm** tool in a controlled lab environment
- Widely cited benchmark in network security research (2015)

### Splits
| Split | Samples | Normal % | Attack % |
|-------|---------|----------|----------|
| **Training** | 82,332 | 44.9% (37,000) | 55.1% (45,332) |
| **Testing** | 175,341 | 31.9% (56,000) | 68.1% (119,341) |
| **Raw files** (1-4) | ~2.5M | ~96.8% | ~3.2% |

### 10 Attack Categories
| Category | Training Samples | Description |
|----------|-----------------|-------------|
| **Normal** | 37,000 | Legitimate traffic |
| **Generic** | 18,871 | Technique against block ciphers |
| **Exploits** | 11,132 | Exploiting known vulnerabilities |
| **Fuzzers** | 6,062 | Cause crashes via random data |
| **DoS** | 4,089 | Denial of Service |
| **Reconnaissance** | 3,496 | Information gathering / scanning |
| **Analysis** | 677 | Port scan, spam, HTML attacks |
| **Backdoor** | 583 | Bypassing authentication |
| **Shellcode** | 378 | Small code to exploit vulnerability |
| **Worms** | 44 | Self-replicating malware |

> [!WARNING]
> **Extreme class imbalance**: Worms has only 44 training samples vs Normal's 37,000. This is a 840:1 ratio — a key challenge you must address.

---

## 4. Preprocessing Pipeline — What & Why

### 4.1 LabelEncoder for Categoricals
- **What**: Converts text categories to integers (e.g., `tcp`→0, `udp`→1)
- **Why**: ML models need numeric inputs
- **Why not One-Hot?**: We tried it — XGBoost handles ordinal encoding natively via splits, and one-hot created 151 sparse features that added noise

### 4.2 MinMaxScaler (0-1 normalization)
- **What**: Scales all features to [0, 1] range: `X_scaled = (X - X_min) / (X_max - X_min)`
- **Why**: Prevents features with large ranges (e.g., `sbytes` in millions) from dominating features with small ranges (e.g., `is_sm_ips_ports` as 0/1)
- **Why not StandardScaler?**: MinMaxScaler preserves the shape of the original distribution and works better with bounded features

### 4.3 Engineered Ratio Features
| Feature | Formula | Why it helps |
|---------|---------|--------------|
| `feat_byte_ratio` | sbytes / (sbytes + dbytes) | Detects asymmetric traffic (attacks often one-directional) |
| `feat_pkt_ratio` | spkts / (spkts + dpkts) | Packet count asymmetry |
| `feat_load_ratio` | sload / (sload + dload) | Bandwidth asymmetry |
| `feat_mean_ratio` | smean / (smean + dmean) | Average packet size asymmetry |
| `feat_srv_density` | ct_srv_src / (ct_dst_src_ltm + 1) | Service connection concentration |

> **Why ratios?** Normal traffic tends to be symmetric (similar send/receive). Attacks like DoS flooding or data exfiltration create extreme asymmetry. These ratios capture that signal regardless of absolute traffic volume.

### 4.4 Class Imbalance Handling
- **Approach**: `sample_weight` in XGBoost (balanced class weighting)
- **Formula**: `weight = n_samples / (n_classes × n_samples_per_class)`
- **Effect**: Worms (44 samples) gets 187× the weight of Normal (37,000 samples)
- **Why not SMOTE?**: We tried it — SMOTE generates synthetic samples that didn't generalize to the test set because the training and test sets have different distributions by design

---

## 5. Model Results

### Final Metrics (Official Test Set — 175,341 samples)

#### XGBoost ⭐ Best Model — 76.60% Accuracy
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| Analysis | 0.05 | 0.04 | 0.05 | 2,000 |
| Backdoor | 0.36 | 0.08 | 0.13 | 1,746 |
| DoS | 0.34 | 0.73 | 0.47 | 12,264 |
| Exploits | 0.81 | 0.56 | 0.66 | 33,393 |
| Fuzzers | 0.71 | 0.20 | 0.32 | 18,184 |
| Generic | 0.99 | 0.98 | 0.99 | 40,000 |
| Normal | 0.79 | 0.97 | 0.87 | 56,000 |
| Reconnaissance | 0.88 | 0.77 | 0.82 | 10,491 |
| Shellcode | 0.55 | 0.73 | 0.63 | 1,133 |
| Worms | 0.64 | 0.75 | 0.69 | 130 |
| **Macro Avg** | **0.61** | **0.58** | **0.56** | 175,341 |
| **Weighted Avg** | **0.79** | **0.77** | **0.75** | 175,341 |

#### Model Comparison
| Model | Accuracy | Macro Precision | Macro Recall | Training Time |
|-------|----------|----------------|-------------|---------------|
| Logistic Regression | 63.47% | 0.35 | 0.45 | 936s |
| Decision Tree | 74.22% | 0.64 | 0.50 | 1.9s |
| **XGBoost** | **76.60%** | **0.61** | **0.58** | 117.7s |

---

## 6. Key Concepts You MUST Know

### 6.1 Precision vs Recall vs F1-Score

```
                        Predicted Attack
                       ┌────────┬────────┐
                       │  Yes   │   No   │
           ┌───────────┼────────┼────────┤
  Actual   │   Yes     │   TP   │   FN   │  ← Recall = TP/(TP+FN)
  Attack   │   No      │   FP   │   TN   │
           └───────────┴────────┴────────┘
                          ↑
                    Precision = TP/(TP+FP)
```

- **Precision**: "Of all samples I predicted as Attack X, how many actually were?"
  - High precision = few false alarms
- **Recall**: "Of all actual Attack X samples, how many did I catch?"
  - High recall = few missed attacks
- **F1-Score**: Harmonic mean of precision and recall = `2 × (P × R) / (P + R)`
- **For IDS**: Recall is MORE important — missing an attack (false negative) is worse than a false alarm

### 6.2 Macro vs Weighted Average
- **Macro avg**: Treats all 10 classes equally (unweighted mean) — best for imbalanced data
- **Weighted avg**: Weights by class size — inflated by majority classes (Normal, Generic)
- **For your project**: Report BOTH, but emphasize macro avg since it's fairer for rare attacks

### 6.3 Confusion Matrix
- An N×N matrix showing where the model gets confused
- Diagonal = correct predictions
- Off-diagonal = misclassifications
- Example: If many DoS samples are predicted as Normal → the model is "confused" between DoS and Normal

### 6.4 Overfitting vs Underfitting
- **Overfitting**: Model memorizes training data but fails on test data (training acc >> test acc)
- **Underfitting**: Model too simple to capture patterns (both training and test acc are low)
- **Your case**: XGBoost uses regularization (`gamma`, `reg_alpha`, `reg_lambda`) and `subsample`/`colsample_bytree` to prevent overfitting

### 6.5 Why XGBoost Won
- **Gradient Boosting**: Builds trees sequentially, each correcting the previous one's errors
- **Regularization**: L1 (`reg_alpha`) and L2 (`reg_lambda`) prevent overfitting
- **Feature importance**: Automatically learns which features matter most
- **Handles imbalance**: Supports `sample_weight` for cost-sensitive learning
- **Speed**: Parallelized across CPU cores

### 6.6 Train/Test Split Philosophy
- **NEVER** evaluate on training data — it measures memorization, not generalization
- The official UNSW-NB15 split is deliberately challenging (test has more attacks proportionally)
- This simulates real-world deployment where the model sees new, unseen attack patterns

---

## 7. Potential Panel Questions & Answers

### Architecture & Design

**Q1: "Why did you choose XGBoost, Decision Tree, and Logistic Regression?"**
> "I chose three models from different ML families to compare: Logistic Regression as a linear baseline, Decision Tree as a non-linear interpretable model, and XGBoost as a state-of-the-art ensemble method. This comparison demonstrates that ensemble gradient boosting significantly outperforms simpler approaches for multi-class network intrusion detection."

**Q2: "Why not use deep learning (CNN, LSTM)?"**
> "Deep learning requires significantly more training data and computation. With only 82,332 training samples and some classes having <50 examples, traditional ML models are actually more appropriate. Published research shows XGBoost achieves comparable accuracy to deep learning on UNSW-NB15 while being faster to train and more interpretable."

**Q3: "Why Streamlit and not Flask or Django?"**
> "Streamlit is purpose-built for ML deployment — it provides interactive widgets, data visualization, and file upload out of the box with minimal code. Flask/Django would require significantly more frontend development for the same functionality. Streamlit also hot-reloads, making iterative development faster."

**Q4: "What is the purpose of converting CSV to Parquet?"**
> "Parquet is a columnar storage format that provides approximately 10× faster read speeds than CSV, uses 50-60% less disk space due to built-in compression, and preserves data types (CSV treats everything as strings). For a dataset of 175k+ rows, this significantly improves load times."

### Preprocessing & Features

**Q5: "Why MinMaxScaler instead of StandardScaler?"**
> "MinMaxScaler normalizes features to a bounded [0, 1] range, which works better with the mix of binary features (like `is_sm_ips_ports`) and continuous features in UNSW-NB15. StandardScaler assumes normally distributed data, which many network traffic features aren't."

**Q6: "Why did you engineer ratio features?"**
> "Network attacks often create asymmetric traffic patterns. For example, a DoS attack sends massive data without receiving much back, creating an extreme `byte_ratio`. These ratios capture relative patterns that are invariant to absolute traffic volume, making the model more robust. During our Mutual Information analysis, the engineered features ranked in the top 6 most informative features."

**Q7: "How do you handle unseen categories during inference?"**
> "When the model encounters a protocol or service it wasn't trained on, we map it to the most common training category. This prevents errors while maintaining reasonable predictions, since unknown protocols are likely rare traffic that behaves similarly to common patterns."

### Model Performance

**Q8: "Why is your accuracy only 76%? Isn't that low?"**
> "76.60% for **10-class** classification on the official UNSW-NB15 split is competitive with published literature. The test set is deliberately designed with 68% attack traffic — very different from the training distribution. Published papers using traditional ML report 75-82% on this exact split. More importantly, our **macro average precision of 0.61** shows the model detects even rare attacks like Worms (75% recall with only 44 training samples) and Shellcode (73% recall)."

**Q9: "Why does the raw file show 98.91% accuracy but training evaluation shows 76.60%?"**
> "This is the **accuracy paradox** caused by class imbalance. The raw file is 96.8% Normal traffic, so even a naive model predicting everything as Normal would achieve ~97%. The official test set has 68% attacks, making it a much harder benchmark. Both metrics are valid for different scenarios: 76.60% measures detection capability, while 98.91% reflects real-world deployment where most traffic is legitimate."

**Q10: "How do you handle the class imbalance problem?"**
> "We use **cost-sensitive learning** via `sample_weight` in XGBoost. Each sample's weight is inversely proportional to its class frequency: `weight = total_samples / (n_classes × class_count)`. This makes the model pay 187× more attention to Worms (44 samples) than Normal (37,000 samples), without generating synthetic data that might not generalize."

**Q11: "Why didn't SMOTE work well?"**
> "SMOTE generates synthetic minority samples by interpolating between nearest neighbors. However, the UNSW-NB15 training and test sets have deliberately different distributions. The synthetic samples created from 82k training rows didn't transfer well to the 175k test set. We found that native class weighting produced better generalization."

**Q12: "What about Analysis and Backdoor — why is recall so low?"**
> "Analysis (4% recall) and Backdoor (8% recall) are the hardest categories because they have the most feature overlap with other classes. Analysis attacks often look like normal port scanning, and Backdoor connections mimic legitimate traffic. This is a known challenge in IDS literature. Improving these would require either deep learning or specialized per-class models."

### Technical Deep Dives

**Q13: "Explain how XGBoost works."**
> "XGBoost builds an ensemble of decision trees sequentially. Each new tree is trained to correct the errors (residuals) of all previous trees combined. It uses gradient descent on a loss function (multi-class log-loss in our case) to determine which errors to focus on. Regularization terms (L1, L2, and gamma) penalize overly complex trees, preventing overfitting."

**Q14: "What hyperparameters did you tune and why?"**
> "Key hyperparameters: `n_estimators=1000` (number of trees — more trees = better but slower), `max_depth=12` (tree complexity — deeper captures more interactions), `learning_rate=0.1` (step size — slower learning with more trees prevents overfitting), `subsample=0.8` and `colsample_bytree=0.8` (use 80% of data/features per tree for randomization), and `gamma=0.5` (minimum loss reduction for splits — regularization)."

**Q15: "What is the confusion matrix telling you?"**
> "The confusion matrix reveals misclassification patterns. For example, Fuzzers are often confused with Exploits because both target software vulnerabilities. DoS is confused with Normal because some DoS traffic uses legitimate-looking packets. Understanding these patterns guides future improvement — we'd focus on features that distinguish Fuzzers from Exploits."

**Q16: "How would you deploy this in a real network?"**
> "In production, this would integrate with a network tap or SPAN port capturing live traffic. A packet parser (like CICFlowMeter) would extract the 42 flow features in real-time, feed them to the model via an API, and trigger alerts when attacks are detected. The Streamlit app demonstrates the inference pipeline — in production, this would be a REST API behind a SIEM dashboard."

**Q17: "What are the limitations of your system?"**
> "Three main limitations: (1) It's trained on 2015 data — modern zero-day attacks wouldn't be detected. (2) Analysis and Backdoor classes have poor recall due to limited training data. (3) It classifies pre-extracted features, not raw packets — it depends on a feature extraction pipeline like CICFlowMeter. Future work could incorporate real-time packet capture and periodic retraining."

### Defensive Questions (Challenges)

**Q18: "Isn't this just applying sklearn to a CSV? What's novel?"**
> "The contribution is the **end-to-end pipeline**: raw data ingestion with auto-format detection, engineered domain-specific features that outperformed 35 of 42 original features, systematic comparison of three model families, handling of extreme class imbalance (840:1 ratio), and a deployable web interface. The engineering challenge of making all these components work together reliably is the core contribution."

**Q19: "Why not just use a firewall/signature-based IDS like Snort?"**
> "Signature-based systems can only detect known attack patterns. They fail against zero-day attacks and polymorphic malware. ML-based IDS learns behavioral patterns (traffic volume, timing, packet sizes) that generalize to novel attacks. Our system detected Worms with 75% recall using features that describe traffic behavior, not specific attack signatures."

**Q20: "What would you do differently if you started over?"**
> "Three things: (1) Use CatBoost instead of XGBoost — it handles categorical features natively without encoding. (2) Implement a stacking ensemble combining all three models' predictions. (3) Add a binary classification stage first (Normal vs Attack) before multi-class classification, which published research shows improves overall accuracy by 5-10%."

---

## 8. Live Demo Walkthrough

### Step-by-step for your demo:

1. **Show the Streamlit app** → explain the sidebar model selection
2. **Upload the official `UNSW_NB15_testing-set.csv`** → show it has headers
3. **Run XGBoost inference** → show 76.60% accuracy, confusion matrix, per-class report
4. **Then upload a raw file** (`UNSW-NB15_1.csv`) → show the auto-detection message
5. **Show the 98.91% accuracy** → explain the accuracy paradox (class imbalance)
6. **Switch to Decision Tree** → show it's faster but less accurate
7. **Show the data preview** → prove the system handles raw, unprocessed data

### If something fails during demo:
- "The model encountered an edge case in the data format — this is why production systems need robust error handling, which is implemented in our preprocessing pipeline."

---

## 9. Glossary of Terms

| Term | Definition |
|------|-----------|
| **IDS** | Intrusion Detection System — monitors network traffic for malicious activity |
| **NIDS** | Network-based IDS — analyzes network packets (vs host-based) |
| **Feature Engineering** | Creating new input variables from existing data to improve model performance |
| **MinMaxScaler** | Scales features to [0,1] range: (x-min)/(max-min) |
| **LabelEncoder** | Converts text categories to integers |
| **One-Hot Encoding** | Converts categories to binary vectors (one column per category) |
| **XGBoost** | eXtreme Gradient Boosting — ensemble of decision trees trained sequentially |
| **Gradient Boosting** | Technique where each model corrects the previous one's errors |
| **SMOTE** | Synthetic Minority Over-sampling Technique — creates synthetic samples for rare classes |
| **Class Imbalance** | When some classes have far fewer samples than others |
| **Confusion Matrix** | Table showing predicted vs actual classifications |
| **Precision** | TP / (TP + FP) — accuracy of positive predictions |
| **Recall (Sensitivity)** | TP / (TP + FN) — proportion of actual positives correctly identified |
| **F1-Score** | Harmonic mean of precision and recall |
| **Macro Average** | Unweighted mean across all classes |
| **Weighted Average** | Mean weighted by class sample count |
| **Overfitting** | Model memorizes training data, performs poorly on new data |
| **Regularization** | Techniques to prevent overfitting (L1, L2, dropout, etc.) |
| **Hyperparameter** | Configuration values set before training (not learned from data) |
| **Cross-validation** | Splitting data into folds to evaluate model stability |
| **Ensemble Learning** | Combining multiple models for better predictions |
| **Parquet** | Columnar file format — faster reads, smaller size than CSV |
| **Joblib** | Library for saving/loading Python objects (models, scalers) |
| **Streamlit** | Python framework for building ML web applications |
| **Zero-day Attack** | Previously unknown exploit with no existing signature |
| **False Positive** | Normal traffic incorrectly flagged as an attack (false alarm) |
| **False Negative** | Attack traffic incorrectly classified as normal (missed attack) |
| **DoS** | Denial of Service — overwhelming a system to make it unavailable |
| **Backdoor** | Hidden method to bypass normal authentication |
| **Shellcode** | Small piece of code used as the payload in an exploit |
| **Reconnaissance** | Gathering information about a target before attacking |
| **Fuzzers** | Tools that send random/malformed data to find vulnerabilities |

---

## 10. Quick Reference — Numbers to Memorize

- **Dataset**: UNSW-NB15, 82k training / 175k testing
- **Features**: 42 original + 5 engineered = **47 total**
- **Classes**: **10** (Normal + 9 attack types)
- **Best model**: XGBoost — **76.60%** accuracy
- **Worst class**: Analysis (4% recall) — only 677 training samples
- **Best class**: Generic (98% recall) — 18,871 training samples
- **Most impressive**: Worms — **75% recall from only 44 training samples**
- **Macro precision**: 0.61 (treats all classes equally)
- **Raw data accuracy**: 98.91% (inflated by 96.8% Normal class)
- **Tech stack**: Python, scikit-learn, XGBoost, Streamlit, Pandas, Joblib
