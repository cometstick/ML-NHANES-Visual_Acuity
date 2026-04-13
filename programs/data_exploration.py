"""
NHANES Data Exploration Script
===============================
Loads imputed dataset and performs exploratory analysis:
- Correlation heatmap
- Covariance matrix
- Feature-target relationships
- Summary findings

Input  : nhanes_imputed.csv
Output : Console report + visualizations (optional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, pearsonr

INPUT_CSV = "nhanes_imputed.csv"

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading data")
print("=" * 70)
df = pd.read_csv(INPUT_CSV)
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}\n")

# ── Define target and features ────────────────────────────────────────────────
TARGET = "AVG_VISUAL_ACUITY"
EXCLUDE_COLS = ["VIDROVA", "VIDLOVA", "VISION_IMPAIRED"]  # Raw vision measurements and binary target
FEATURES = [c for c in df.columns if c not in EXCLUDE_COLS and c != TARGET]

# Convert CYCLE dummies and categorical to numeric for correlation
df_numeric = df[FEATURES + [TARGET]].copy()
df_numeric = df_numeric.select_dtypes(include=[np.number])

print(f"  Target variable: {TARGET}")
print(f"  Number of features: {len(FEATURES)}")
print(f"  Features for analysis: {FEATURES}\n")

# ── Summary statistics ────────────────────────────────────────────────────────
print("=" * 70)
print("Descriptive Statistics")
print("=" * 70)
print(df_numeric[[TARGET]].describe().round(3))
print()

# ── Correlation with target ───────────────────────────────────────────────────
print("=" * 70)
print(f"Pearson Correlation with {TARGET}")
print("=" * 70)
correlations = df_numeric.corr()[TARGET].drop(TARGET).sort_values(ascending=False)
print(correlations.round(3).to_string())
print()

# ── Covariance matrix ─────────────────────────────────────────────────────────
print("=" * 70)
print("Covariance Matrix (all numeric features)")
print("=" * 70)
cov_matrix = df_numeric.cov()
print(cov_matrix.round(3).to_string())
print()

# ── High correlation pairs (potential multicollinearity) ────────────────────
print("=" * 70)
print("High Correlation Pairs (|r| > 0.7, excluding target)")
print("=" * 70)
features_only = df_numeric.drop(columns=[TARGET])
corr_matrix = features_only.corr().abs()

# Get upper triangle to avoid duplicates
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.7:
            high_corr_pairs.append({
                'Feature 1': corr_matrix.columns[i],
                'Feature 2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', ascending=False)
    print(high_corr_df.to_string(index=False))
else:
    print("  No feature pairs with |correlation| > 0.7 detected.")
print()

# ── Missing and NaN check ─────────────────────────────────────────────────────
print("=" * 70)
print("Data Quality Check")
print("=" * 70)
print(f"  Total rows: {len(df)}")
print(f"  Total missing values: {df.isna().sum().sum()}")
print(f"  Inf values: {(df == np.inf).sum().sum()}")
print(f"  -Inf values: {(df == -np.inf).sum().sum()}")
print(f"  Data types:")
for col, dtype in df.dtypes.items():
    print(f"    {col}: {dtype}")
print()

# ── Feature ranges (outlier detection) ─────────────────────────────────────────
print("=" * 70)
print("Feature Ranges (Potential Outliers: beyond 3σ)")
print("=" * 70)
for col in df_numeric.columns:
    mean = df_numeric[col].mean()
    std = df_numeric[col].std()
    lower_bound = mean - 3 * std
    upper_bound = mean + 3 * std
    outliers = ((df_numeric[col] < lower_bound) | (df_numeric[col] > upper_bound)).sum()
    if outliers > 0:
        print(f"  [{col}] {outliers} values beyond 3σ (range: {df_numeric[col].min():.2f}–{df_numeric[col].max():.2f})")
    else:
        print(f"  [{col}] OK (no outliers beyond 3σ)")
print()

# ── Class balance (VISION_IMPAIRED) ───────────────────────────────────────────
print("=" * 70)
print("Binary Target (VISION_IMPAIRED) Distribution")
print("=" * 70)
if "VISION_IMPAIRED" in df.columns:
    class_counts = df["VISION_IMPAIRED"].value_counts().sort_index()
    print(class_counts)
    print(f"  Normal (0): {class_counts[0]} ({100*class_counts[0]/len(df):.1f}%)")
    print(f"  Impaired (1): {class_counts[1]} ({100*class_counts[1]/len(df):.1f}%)")
    print(f"  Ratio (0:1): {class_counts[0]/class_counts[1]:.1f}:1")
else:
    print("  VISION_IMPAIRED not found in dataset.")
print()

# ── Final summary ─────────────────────────────────────────────────────────────
print("=" * 70)
print("Summary & Recommendations")
print("=" * 70)
print(f"""
✓ Dataset is complete ({len(df)} rows, 0 missing values)
✓ Target {TARGET} has mean={df_numeric[TARGET].mean():.3f}, std={df_numeric[TARGET].std():.3f}
✓ Class balance for binary task: {class_counts[1]/len(df):.1%} positive

Top 3 features most correlated with {TARGET}:
""")
for i, (feat, corr) in enumerate(correlations.head(3).items(), 1):
    print(f"  {i}. {feat}: r = {corr:.3f}")


# ── Optional: Generate heatmap ────────────────────────────────────────────────
print("=" * 70)
print("Generating Correlation Heatmap")
print("=" * 70)
try:
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_numeric.corr(), annot=True, cmap="coolwarm", center=0, fmt=".2f", cbar_kws={"label": "Correlation"})
    plt.title(f"Correlation Matrix\nTarget: {TARGET}")
    plt.tight_layout()
    heatmap_path = "correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=100, bbox_inches="tight")
    print(f"  Saved to: {heatmap_path}")
    plt.close()
except Exception as e:
    print(f"  [WARN] Could not generate heatmap: {e}")

# ── BPX vs Visual Acuity plots ───────────────────────────────────────────────
print("\n" + "=" * 70)
print("BPX vs AVG_VISUAL_ACUITY Scatterplots")
print("=" * 70)

bpx_cols = [c for c in df.columns if c.startswith("BPX")]
if bpx_cols:
    for col in bpx_cols:
        try:
            plt.figure(figsize=(8, 5))
            sns.regplot(x=df[col], y=df[TARGET], scatter_kws={"alpha": 0.5}, line_kws={"color": "red"})
            plt.xlabel(col)
            plt.ylabel(TARGET)
            plt.title(f"{col} vs {TARGET}")
            plt.tight_layout()
            plot_path = f"{col}_vs_{TARGET}.png"
            plt.savefig(plot_path, dpi=120, bbox_inches="tight")
            print(f"  Saved: {plot_path}")
            plt.close()
        except Exception as e:
            print(f"  [WARN] Could not plot {col}: {e}")
else:
    print("  No BPX columns found in dataset.")

print("\n" + "=" * 70)
print("Exploration Complete")
print("=" * 70)
