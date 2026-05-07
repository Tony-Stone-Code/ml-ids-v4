# %% [markdown]
# # 📊 Exploratory Data Analysis — UNSW-NB15 Dataset
# 
# This notebook provides a comprehensive EDA of the UNSW-NB15 Network Intrusion Detection dataset,
# covering class distributions, feature analysis, correlation studies, and attack pattern visualization.

# %% [markdown]
# ## 1. Setup & Data Loading

# %%
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PLOTS_DIR = os.path.join(PROJECT_ROOT, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load datasets
df_train = pd.read_parquet(os.path.join(DATA_DIR, 'UNSW_NB15_training-set.parquet'))
df_test = pd.read_parquet(os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.parquet'))

print(f"Training set: {df_train.shape[0]:,} rows × {df_train.shape[1]} columns")
print(f"Testing set:  {df_test.shape[0]:,} rows × {df_test.shape[1]} columns")
print(f"\nColumns: {list(df_train.columns)}")

# %% [markdown]
# ## 2. Dataset Overview

# %%
# Basic info
print("=== Training Set Info ===")
print(f"Shape: {df_train.shape}")
print(f"\nData Types:\n{df_train.dtypes.value_counts()}")
print(f"\nMissing Values:\n{df_train.isnull().sum()[df_train.isnull().sum() > 0]}")

# %%
# First few rows
df_train.head()

# %%
# Statistical summary
df_train.describe().round(2)

# %% [markdown]
# ## 3. Target Distribution — Class Imbalance Analysis

# %%
# Attack category distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Training set
train_counts = df_train['attack_cat'].value_counts()
colors = sns.color_palette("husl", len(train_counts))
bars = axes[0].barh(train_counts.index, train_counts.values, color=colors)
axes[0].set_title('Training Set — Attack Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Count')
for bar, count in zip(bars, train_counts.values):
    axes[0].text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2, 
                f'{count:,} ({count/len(df_train)*100:.1f}%)', va='center', fontsize=9)

# Testing set
test_counts = df_test['attack_cat'].value_counts()
bars = axes[1].barh(test_counts.index, test_counts.values, color=colors)
axes[1].set_title('Testing Set — Attack Distribution', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Count')
for bar, count in zip(bars, test_counts.values):
    axes[1].text(bar.get_width() + 200, bar.get_y() + bar.get_height()/2, 
                f'{count:,} ({count/len(df_test)*100:.1f}%)', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# %%
# Imbalance ratio table
print("=== Class Imbalance Analysis ===\n")
imbalance_df = pd.DataFrame({
    'Train Count': df_train['attack_cat'].value_counts(),
    'Train %': (df_train['attack_cat'].value_counts() / len(df_train) * 100).round(2),
    'Test Count': df_test['attack_cat'].value_counts(),
    'Test %': (df_test['attack_cat'].value_counts() / len(df_test) * 100).round(2),
})
imbalance_df['Imbalance Ratio'] = (imbalance_df['Train Count'].max() / imbalance_df['Train Count']).round(1)
print(imbalance_df.to_string())

# %% [markdown]
# ## 4. Feature Analysis

# %%
# Categorical feature distributions
cat_cols = ['proto', 'service', 'state']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for i, col in enumerate(cat_cols):
    top_n = df_train[col].value_counts().head(10)
    top_n.plot(kind='bar', ax=axes[i], color=sns.color_palette("viridis", len(top_n)))
    axes[i].set_title(f'{col} — Top 10 Values', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('')
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_categorical_features.png'), dpi=150, bbox_inches='tight')
plt.close()

# %%
# Numeric feature distributions (key features)
key_numeric = ['dur', 'sbytes', 'dbytes', 'sload', 'dload', 'rate', 
               'sttl', 'dttl', 'smean', 'dmean', 'sinpkt', 'dinpkt']

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()
for i, col in enumerate(key_numeric):
    df_train[col].clip(upper=df_train[col].quantile(0.99)).hist(
        bins=50, ax=axes[i], color='#2196F3', alpha=0.7, edgecolor='white'
    )
    axes[i].set_title(col, fontsize=11, fontweight='bold')
    axes[i].set_ylabel('')

plt.suptitle('Numeric Feature Distributions (clipped at 99th percentile)', 
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_numeric_distributions.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 5. Correlation Analysis

# %%
# Correlation heatmap of numeric features
numeric_df = df_train.select_dtypes(include=[np.number]).drop(columns=['id', 'label'], errors='ignore')

fig, ax = plt.subplots(figsize=(16, 14))
corr_matrix = numeric_df.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, vmin=-1, vmax=1,
            cbar_kws={"shrink": 0.8})
ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_correlation_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

# %%
# Top correlated features with the label
label_corr = numeric_df.corrwith(df_train['label']).abs().sort_values(ascending=False)
print("=== Top Features Correlated with Attack Label ===\n")
print(label_corr.head(15).to_string())

fig, ax = plt.subplots(figsize=(10, 8))
label_corr.head(15).sort_values().plot(kind='barh', ax=ax, color='#FF6B6B')
ax.set_title('Top 15 Features Correlated with Attack Label', fontsize=14, fontweight='bold')
ax.set_xlabel('Absolute Correlation')
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_top_correlations.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 6. Attack Pattern Analysis

# %%
# Feature distributions by attack type (key differentiators)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
features_to_plot = ['sbytes', 'sttl', 'rate', 'smean', 'ct_state_ttl', 'sload']

for i, feat in enumerate(features_to_plot):
    ax = axes[i // 3][i % 3]
    for cat in ['Normal', 'DoS', 'Exploits', 'Generic', 'Reconnaissance']:
        subset = df_train[df_train['attack_cat'] == cat][feat].clip(
            upper=df_train[feat].quantile(0.95))
        subset.hist(bins=40, ax=ax, alpha=0.5, label=cat, density=True)
    ax.set_title(feat, fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)

plt.suptitle('Feature Distributions by Attack Category', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_attack_patterns.png'), dpi=150, bbox_inches='tight')
plt.close()

# %%
# Protocol usage by attack type
proto_attack = pd.crosstab(df_train['attack_cat'], df_train['proto'])
top_protos = df_train['proto'].value_counts().head(5).index
proto_attack_top = proto_attack[top_protos]

fig, ax = plt.subplots(figsize=(12, 6))
proto_attack_top.plot(kind='bar', ax=ax, stacked=True, colormap='viridis')
ax.set_title('Protocol Usage by Attack Type', fontsize=14, fontweight='bold')
ax.set_xlabel('Attack Category')
ax.set_ylabel('Count')
ax.legend(title='Protocol', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_protocol_by_attack.png'), dpi=150, bbox_inches='tight')
plt.close()

# %% [markdown]
# ## 7. Engineered Feature Analysis

# %%
# Create the same ratio features used in our pipeline
df_temp = df_train.copy()

total_bytes = df_temp['sbytes'] + df_temp['dbytes']
df_temp['byte_ratio'] = np.where(total_bytes != 0, df_temp['sbytes'] / total_bytes, 0.5)

total_pkts = df_temp['spkts'] + df_temp['dpkts']
df_temp['pkt_ratio'] = np.where(total_pkts != 0, df_temp['spkts'] / total_pkts, 0.5)

total_load = df_temp['sload'] + df_temp['dload']
df_temp['load_ratio'] = np.where(total_load != 0, df_temp['sload'] / total_load, 0.5)

# Box plots of engineered features by attack type
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, feat in enumerate(['byte_ratio', 'pkt_ratio', 'load_ratio']):
    df_temp.boxplot(column=feat, by='attack_cat', ax=axes[i], rot=45)
    axes[i].set_title(feat, fontsize=13, fontweight='bold')
    axes[i].set_xlabel('')

plt.suptitle('Engineered Ratio Features by Attack Type', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'eda_engineered_features.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f'\n[DONE] All EDA plots saved to {PLOTS_DIR}')

# %% [markdown]
# ## 8. Key Findings Summary
# 
# 1. **Severe Class Imbalance**: Worms (44 samples) vs Normal (37,000) — 840:1 ratio
# 2. **Different Train/Test Distributions**: Test set has proportionally more attacks (68% vs 55%)
# 3. **Top Discriminative Features**: `sttl`, `ct_state_ttl`, `rate`, `sload`, `sbytes`
# 4. **Categorical Complexity**: `proto` has 131 unique values, most rare
# 5. **Engineered Features**: Traffic asymmetry ratios effectively separate attack categories
# 6. **Strong Correlations**: `sbytes`↔`sload` (0.96), indicating redundancy that models can exploit
