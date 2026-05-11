"""
Train 3 ML models for multi-class intrusion detection on UNSW-NB15.

Enhanced pipeline:
  - One-Hot Encoding for categorical features
  - MinMaxScaler normalization
  - Pearson correlation-based feature selection
  - SMOTE oversampling for class imbalance
  - Class weighting for all models

Models:
  1. Logistic Regression (Traditional)
  2. Decision Tree Classifier (Traditional)
  3. XGBoost Classifier (Ensemble)

Usage:
    python src/train.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Add project root to path so we can import preprocessing
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)
from src.preprocessing import fit_and_save, transform_new

# Paths
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def load_data(name: str) -> pd.DataFrame:
    """Load dataset, preferring Parquet over CSV for speed."""
    parquet_path = os.path.join(DATA_DIR, f"{name}.parquet")
    csv_path = os.path.join(DATA_DIR, f"{name}.csv")

    if os.path.exists(parquet_path):
        print(f"[LOAD] {name}.parquet")
        return pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        print(f"[LOAD] {name}.csv (parquet not found)")
        return pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No data file found for {name} in {DATA_DIR}")


def print_class_distribution(y, le_target, title="Class distribution"):
    """Print class counts and percentages."""
    print(f"   {title}:")
    unique, counts = np.unique(y, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        cls_name = le_target.classes_[cls_idx]
        pct = count / len(y) * 100
        print(f"     {cls_name:20s}: {count:6,} ({pct:5.1f}%)")
    print(f"     {'TOTAL':20s}: {len(y):6,}")


def apply_smote(X_train, y_train, le_target):
    """
    Apply SMOTE oversampling to balance the training set.

    SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic
    samples for minority classes by interpolating between existing samples
    and their k-nearest neighbors. This helps models learn rare attack
    patterns without simply duplicating rows.

    Uses k_neighbors=5 (default) for richer synthetic sample diversity
    in the expanded one-hot encoded feature space.
    """
    print("\n-- Applying SMOTE Oversampling --")
    print(f"   Before SMOTE: {X_train.shape[0]:,} samples")

    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    print(f"   After SMOTE:  {X_resampled.shape[0]:,} samples")
    print_class_distribution(y_resampled, le_target, "Resampled distribution")

    return X_resampled, y_resampled


def train_and_evaluate():
    print("=" * 70)
    print("  UNSW-NB15 Intrusion Detection System - Model Training")
    print("  (One-Hot + MinMax + Correlation Selection + SMOTE + Weights)")
    print("=" * 70)

    # 1. Load Data
    print("\n-- Loading Data --")
    df_train = load_data("UNSW_NB15_training-set")
    df_test = load_data("UNSW_NB15_testing-set")
    print(f"   Training samples: {len(df_train):,}")
    print(f"   Testing samples:  {len(df_test):,}")

    # 2. Preprocess Training Data (One-Hot + MinMax + Feature Selection)
    print("\n-- Preprocessing Training Data --")
    X_train, y_train, le_target = fit_and_save(df_train)
    print(f"   X_train shape: {X_train.shape}")
    print(f"   Classes: {list(le_target.classes_)}")
    print_class_distribution(y_train, le_target, "Original distribution")

    # 3. Preprocess Test Data
    print("\n-- Preprocessing Test Data --")
    X_test, y_test = transform_new(df_test)
    print(f"   X_test shape: {X_test.shape}")

    # 4. Compute sample weights for XGBoost (sklearn-style balanced)
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    n_samples = len(y_train)
    n_classes = len(unique_classes)
    sample_weight_map = {}
    for cls, count in zip(unique_classes, class_counts):
        sample_weight_map[cls] = n_samples / (n_classes * count)
    xgb_sample_weights = np.array([sample_weight_map[y] for y in y_train])

    # 5. Define Models
    models = {
        "logistic_regression": LogisticRegression(
            max_iter=3000,
            solver="saga",
            penalty="l2",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        ),
        "xgboost": XGBClassifier(
            n_estimators=1000,
            max_depth=12,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.5,
            min_child_weight=3,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1,
        ),
    }

    # 6. Train & Evaluate
    results = {}
    os.makedirs(MODELS_DIR, exist_ok=True)

    for name, model in models.items():
        print(f"\n{'-' * 70}")
        print(f"  Training: {name.replace('_', ' ').title()}")
        print(f"{'-' * 70}")

        start = time.time()
        if name == "xgboost":
            model.fit(X_train, y_train, sample_weight=xgb_sample_weights)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start
        print(f"  Training time: {train_time:.1f}s")

        # Predict
        y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(
            y_test, y_pred,
            target_names=le_target.classes_,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "time": train_time,
            "report": report,
            "confusion_matrix": cm,
        }

        print(f"  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
        print(f"\n  Classification Report:\n{report}")

        # Save model
        model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"  [SAVE] {model_path}")



    # 8. Summary
    print(f"\n{'=' * 70}")
    print("  MODEL COMPARISON SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Model':<25s} {'Accuracy':>10s} {'Train Time':>12s}")
    print(f"  {'-' * 25} {'-' * 10} {'-' * 12}")
    for name, res in results.items():
        display_name = name.replace("_", " ").title()
        print(
            f"  {display_name:<25s} {res['accuracy']:>9.4f} {res['time']:>10.1f}s"
        )
    print(f"{'=' * 70}")

    # ── Generate Visualizations ──────────────────────────────────────
    PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    print("\n-- Generating Visualizations --")

    class_names = list(le_target.classes_)

    # 8a. Confusion Matrix Heatmaps
    for name, res in results.items():
        fig, ax = plt.subplots(figsize=(10, 8))
        cm = res["confusion_matrix"]
        # Normalize by row (true labels) for better visualization
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(
            cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5,
        )
        display_name = name.replace("_", " ").title()
        ax.set_title(f"Confusion Matrix — {display_name}\n(Normalized by True Label)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Predicted Label", fontsize=12)
        ax.set_ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, f"confusion_matrix_{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PLOT] {path}")

    # 8b. XGBoost Feature Importance (Top 20)
    xgb_model = models.get("xgboost")
    if xgb_model is not None:
        feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
        importances = xgb_model.feature_importances_
        feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
        top_n = 20

        fig, ax = plt.subplots(figsize=(10, 8))
        feat_imp.head(top_n).sort_values().plot(kind="barh", ax=ax, color="#2196F3", edgecolor="white")
        ax.set_title(f"XGBoost — Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
        ax.set_xlabel("Importance Score", fontsize=12)
        ax.set_ylabel("")
        for i, (val, name_) in enumerate(zip(feat_imp.head(top_n).sort_values(), feat_imp.head(top_n).sort_values().index)):
            ax.text(val + 0.001, i, f"{val:.3f}", va="center", fontsize=9)
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "feature_importance_xgboost.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [PLOT] {path}")

    # 8c. Model Comparison Bar Charts
    model_names = [n.replace("_", " ").title() for n in results.keys()]
    accuracies = [res["accuracy"] for res in results.values()]
    train_times = [res["time"] for res in results.values()]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy comparison
    bars = axes[0].bar(model_names, accuracies, color=colors, edgecolor="white", linewidth=1.5)
    axes[0].set_title("Model Accuracy Comparison", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=max(accuracies), color="gray", linestyle="--", alpha=0.5)
    for bar, acc in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f"{acc:.2%}", ha="center", fontsize=11, fontweight="bold")

    # Training time comparison
    bars = axes[1].bar(model_names, train_times, color=colors, edgecolor="white", linewidth=1.5)
    axes[1].set_title("Training Time Comparison", fontsize=14, fontweight="bold")
    axes[1].set_ylabel("Time (seconds)", fontsize=12)
    for bar, t in zip(bars, train_times):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{t:.1f}s", ha="center", fontsize=11, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {path}")

    # 8d. Per-Class Recall Comparison (all models)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(class_names))
    width = 0.25

    for i, (name, res) in enumerate(results.items()):
        cm = res["confusion_matrix"]
        recall_per_class = np.diag(cm) / cm.sum(axis=1)
        display_name = name.replace("_", " ").title()
        ax.bar(x + i * width, recall_per_class, width, label=display_name, color=colors[i], edgecolor="white")

    ax.set_title("Per-Class Recall Comparison", fontsize=14, fontweight="bold")
    ax.set_ylabel("Recall", fontsize=12)
    ax.set_xticks(x + width)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "per_class_recall_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] {path}")

    print(f"\n[DONE] All models trained and saved to models/")
    print(f"[DONE] All plots saved to plots/")


if __name__ == "__main__":
    train_and_evaluate()
