"""
Preprocessing module for the UNSW-NB15 IDS project.

Enhanced pipeline (v3 — max accuracy):
  - LabelEncoder for categorical features (XGBoost handles ordinal splits natively)
  - MinMaxScaler normalization
  - Engineered ratio features (traffic asymmetry indicators)
  - All features retained (no pruning — tree models handle dimensionality)

Target: attack_cat (multi-class: Normal + 9 attack categories)
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Paths -----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Columns to drop (not used as features) --------------------------------------
DROP_COLS = ["id", "attack_cat", "label"]

# Categorical columns ----------------------------------------------------------
CATEGORICAL_COLS = ["proto", "service", "state"]

# Target column ----------------------------------------------------------------
TARGET_COL = "attack_cat"

# Raw UNSW-NB15 column names (49 columns, in order) ----------------------------
# The raw CSVs (UNSW-NB15_1.csv to _4.csv) have NO headers.
RAW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "sload", "dload", "spkts", "dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smean", "dmean", "trans_depth", "response_body_len", "sjit", "djit",
    "stime", "ltime", "sinpkt", "dinpkt", "tcprtt", "synack", "ackdat",
    "is_sm_ips_ports", "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login",
    "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "label",
]

# Mapping from raw/variant column names to the training set column names -------
COLUMN_RENAME_MAP = {
    "Sload": "sload", "Dload": "dload",
    "Spkts": "spkts", "Dpkts": "dpkts",
    "Sjit": "sjit", "Djit": "djit",
    "Sintpkt": "sinpkt", "Dintpkt": "dinpkt",
    "Stime": "stime", "Ltime": "ltime",
    "Label": "label",
    "smeansz": "smean", "dmeansz": "dmean",
    "res_bdy_len": "response_body_len",
    "ct_src_ ltm": "ct_src_ltm",
}

# Extra columns in raw data that aren't in the training set --------------------
RAW_EXTRA_COLS = ["srcip", "sport", "dstip", "dsport", "stime", "ltime"]


def _ensure_models_dir():
    """Create models/ directory if it doesn't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names to match the training set format.
    Handles:
      - Headerless raw CSVs (49 columns → assign RAW_COLUMNS)
      - Variant column names (Sload → sload, smeansz → smean, etc.)
      - Extra columns not in training set (srcip, sport, dstip, dsport)
    """
    df = df.copy()

    # Detect headerless CSV: columns are integers (0, 1, 2, ...) or 'Unnamed'
    if all(isinstance(c, int) for c in df.columns) or str(df.columns[0]).startswith("Unnamed"):
        if len(df.columns) == 49:
            df.columns = RAW_COLUMNS
            print(f"[AUTO] Assigned 49 raw UNSW-NB15 column names (headerless CSV detected)")
        elif len(df.columns) == 48:
            # Some raw files omit the label column
            df.columns = RAW_COLUMNS[:48]
            print(f"[AUTO] Assigned 48 raw UNSW-NB15 column names (no label column)")
        else:
            print(f"[WARN] Headerless CSV with {len(df.columns)} columns (expected 48-49)")

    # Rename variant column names to match training set
    rename_needed = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    if rename_needed:
        df = df.rename(columns=rename_needed)
        print(f"[AUTO] Renamed {len(rename_needed)} columns: {rename_needed}")

    # Lowercase all column names for consistency
    df.columns = [c.strip().lower() if isinstance(c, str) else c for c in df.columns]

    # Drop extra columns not used in training
    extra_cols = [c for c in RAW_EXTRA_COLS if c in df.columns]
    if extra_cols:
        df = df.drop(columns=extra_cols)
        print(f"[AUTO] Dropped {len(extra_cols)} extra columns: {extra_cols}")

    return df


def _clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the raw dataframe."""
    df = _normalize_columns(df)

    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("", "-").fillna("-")

    if TARGET_COL in df.columns:
        df[TARGET_COL] = df[TARGET_COL].astype(str).str.strip()
        df[TARGET_COL] = df[TARGET_COL].replace({"": "Normal", "nan": "Normal"})

    # Force numeric columns to float64 and handle inf/nan
    for col in df.columns:
        if col not in CATEGORICAL_COLS and col != TARGET_COL and col != "id":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)

    return df


def _engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """
    Create domain-specific engineered features.
    Traffic asymmetry ratios are strong indicators of attack patterns.
    """
    X = X.copy()

    # Byte ratio: sbytes / (sbytes + dbytes)
    if "sbytes" in X.columns and "dbytes" in X.columns:
        total = X["sbytes"] + X["dbytes"]
        X["feat_byte_ratio"] = np.where(total != 0, X["sbytes"] / total, 0.5)

    # Packet ratio: spkts / (spkts + dpkts)
    if "spkts" in X.columns and "dpkts" in X.columns:
        total = X["spkts"] + X["dpkts"]
        X["feat_pkt_ratio"] = np.where(total != 0, X["spkts"] / total, 0.5)

    # Load ratio: sload / (sload + dload)
    if "sload" in X.columns and "dload" in X.columns:
        total = X["sload"] + X["dload"]
        X["feat_load_ratio"] = np.where(total != 0, X["sload"] / total, 0.5)

    # Mean packet size ratio: smean / (smean + dmean)
    if "smean" in X.columns and "dmean" in X.columns:
        total = X["smean"] + X["dmean"]
        X["feat_mean_ratio"] = np.where(total != 0, X["smean"] / total, 0.5)

    # Connection density: ct_srv_src / (ct_dst_src_ltm + 1)
    if "ct_srv_src" in X.columns and "ct_dst_src_ltm" in X.columns:
        denom = X["ct_dst_src_ltm"] + 1
        X["feat_srv_density"] = np.where(denom != 0, X["ct_srv_src"] / denom, 0.0)
        X["feat_srv_density"] = X["feat_srv_density"].replace([np.inf, -np.inf], 0).fillna(0)

    n_feats = len([c for c in X.columns if c.startswith("feat_")])
    print(f"[FEAT] Engineered {n_feats} ratio features")
    return X


def fit_and_save(df_train: pd.DataFrame):
    """
    Fit encoders and scaler on the training data and save all artifacts.
    Returns X_final, y, le_target.
    """
    _ensure_models_dir()
    df = _clean_dataframe(df_train)

    # --- Encode target ---
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[TARGET_COL])
    joblib.dump(le_target, os.path.join(MODELS_DIR, "le_attack_cat.joblib"))
    print(f"[SAVE] Target classes: {list(le_target.classes_)}")

    # --- Prepare features ---
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # --- Encode categoricals with LabelEncoder ---
    # XGBoost and Decision Trees handle ordinal encoding natively via splits
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        joblib.dump(le, os.path.join(MODELS_DIR, f"le_{col}.joblib"))
        print(f"[SAVE] LabelEncoder for '{col}' - {len(le.classes_)} unique values")

    # --- Engineer ratio features ---
    X = _engineer_features(X)

    # --- Save feature column order ---
    feature_columns = list(X.columns)
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "feature_columns.joblib"))
    joblib.dump(feature_columns, os.path.join(MODELS_DIR, "selected_features.joblib"))
    print(f"[SAVE] Feature columns: {len(feature_columns)} total")

    # --- MinMaxScaler ---
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print(f"[SAVE] MinMaxScaler fitted and saved")

    return X_scaled, y, le_target


def transform_new(df: pd.DataFrame):
    """
    Transform new (unseen) data using saved artifacts.
    IMPORTANT: This NEVER re-fits.
    """
    df = _clean_dataframe(df)

    # --- Extract true labels if available ---
    y_true = None
    if TARGET_COL in df.columns:
        le_target = joblib.load(os.path.join(MODELS_DIR, "le_attack_cat.joblib"))
        known_classes = set(le_target.classes_)
        df[TARGET_COL] = df[TARGET_COL].apply(
            lambda x: x if x in known_classes else "Normal"
        )
        y_true = le_target.transform(df[TARGET_COL])

    # --- Prepare features ---
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)

    # --- Encode categoricals using saved encoders ---
    for col in CATEGORICAL_COLS:
        le = joblib.load(os.path.join(MODELS_DIR, f"le_{col}.joblib"))
        known = set(le.classes_)
        if col not in X.columns:
            # Column missing from uploaded data — fill with most common class
            X[col] = le.classes_[0]
            print(f"[WARN] Column '{col}' not found in data, filled with '{le.classes_[0]}'")
        else:
            X[col] = X[col].astype(str).str.strip().replace("", "-").fillna("-")
            X[col] = X[col].apply(lambda x: x if x in known else le.classes_[0])
        X[col] = le.transform(X[col])

    # --- Engineer ratio features ---
    X = _engineer_features(X)

    # --- Align feature columns ---
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    for col in feature_columns:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_columns]

    # --- Scale ---
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
    X_scaled = scaler.transform(X)

    return X_scaled, y_true
