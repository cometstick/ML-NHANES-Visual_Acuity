"""
NHANES Imputation + Pre-modelling Prep Script
==============================================
Inputs : nhanes_merged.csv  (output of nhanes_pipeline.py)
Outputs: nhanes_imputed.csv

Steps
-----
0. Pre-imputation fix  → BPXDI1 zeros recoded to NaN (NHANES coding artefact)
1. Mode imputation     → categorical/ordinal cols per cycle (DIQ010, DIQ050, INDHHINR)
2. KNN imputation      → continuous cols (k=5, distance-weighted)
3. Plausibility checks → flag any physiologically implausible imputed values
4. CYCLE encoding      → one-hot encode the survey cycle string column
5. Target engineering  → continuous target (AVG_VISUAL_ACUITY kept as-is)
                         binary target (VISION_IMPAIRED: LogMAR > 0.3, ~20/40 Snellen)
6. Summary + save
"""

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

INPUT_CSV  = "nhanes_merged.csv"
OUTPUT_CSV = "nhanes_imputed.csv"
KNN_K      = 5          # neighbours — 5 is the standard default for this sample size

# ── Load ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Loading data")
print("=" * 60)
nhanes = pd.read_csv(INPUT_CSV)
print(f"  Shape: {nhanes.shape}")
print(f"  Columns: {nhanes.columns.tolist()}")

# ── Classify columns by imputation strategy ───────────────────────────────────
# Categorical: ordinal or nominal codes — impute by mode
# Continuous : numeric measurements — impute by KNN
# Passthrough : no missingness or non-numeric identifiers — leave alone

CATEGORICAL = ["DIQ010", "DIQ050", "INDHHINR"]

CONTINUOUS  = [
    "LBXTR", "LBDHDD",
    "BPXSY1", "BPXDI1",
    "BMXWAIST", "BMXBMI",
    "VIDRVA", "VIDLVA",
    "AVG_VISUAL_ACUITY",
    "DIABETES_DURATION_YRS",
    "RIDAGEYR",            # 0% missing but include so KNN can use it as a distance feature
]

PASSTHROUGH = ["RIAGENDR", "RIDRETH1", "CYCLE"]   # 0% missing, kept as-is

# Only keep columns that actually exist in this run
CATEGORICAL = [c for c in CATEGORICAL if c in nhanes.columns]
CONTINUOUS  = [c for c in CONTINUOUS  if c in nhanes.columns]

# Separate KNN features from targets to prevent data leakage
# AVG_VISUAL_ACUITY is the target
FEATURES_FOR_KNN = [c for c in CONTINUOUS if c not in ["AVG_VISUAL_ACUITY"]]

print(f"\n  Categorical cols ({len(CATEGORICAL)}): {CATEGORICAL}")
print(f"  Continuous cols  ({len(CONTINUOUS)}):  {CONTINUOUS}")
print(f"  Features for KNN ({len(FEATURES_FOR_KNN)}): {FEATURES_FOR_KNN}")

# ── Step 0: Pre-imputation data fixes ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 0 — Pre-imputation fixes")
print("=" * 60)

# Since Snellen values are non-linear,
# convert the valid readings to LogMAR: log10(denominator / 20).
# This makes the 0.3 threshold correct: log10(40/20) = 0.301 ≈ 20/40.
for col in ["VIDRVA", "VIDLVA"]:
    if col not in nhanes.columns:
        continue
    
    # Guard: detect and replace invalid values (zero or negative) with NaN
    invalid_mask = nhanes[col] <= 0
    n_invalid = invalid_mask.sum()
    if n_invalid > 0:
        nhanes.loc[invalid_mask, col] = float("nan")
        print(f"  {col}: {n_invalid} zero/negative values replaced with NaN")
    
    nhanes[col] = np.log10(nhanes[col] / 20)
    print(f"  {col}: All valid values converted to LogMAR")

# Recompute AVG_VISUAL_ACUITY now that both eyes are in LogMAR
if "VIDRVA" in nhanes.columns and "VIDLVA" in nhanes.columns:
    nhanes["AVG_VISUAL_ACUITY"] = nhanes[["VIDRVA", "VIDLVA"]].mean(axis=1, skipna=True)
    print(f"  AVG_VISUAL_ACUITY recomputed in LogMAR — "
          f"mean: {nhanes['AVG_VISUAL_ACUITY'].mean():.3f}, "
          f"range: {nhanes['AVG_VISUAL_ACUITY'].min():.3f} to "
          f"{nhanes['AVG_VISUAL_ACUITY'].max():.3f}")

print("\n" + "=" * 60)
print("Step 1 — Mode imputation for categorical columns")
print("=" * 60)

for col in CATEGORICAL:
    missing_before = nhanes[col].isna().sum()
    if missing_before == 0:
        print(f"  {col}: no missing values, skipped")
        continue

    # Impute per cycle so cohort-level response distributions are respected
    for cycle in nhanes["CYCLE"].unique():
        mask_cycle = nhanes["CYCLE"] == cycle
        mode_val   = nhanes.loc[mask_cycle, col].mode()
        if mode_val.empty:
            continue
        missing_cycle = mask_cycle & nhanes[col].isna()
        nhanes.loc[missing_cycle, col] = mode_val.iloc[0]

    missing_after = nhanes[col].isna().sum()
    print(f"  {col}: {missing_before} → {missing_after} missing "
          f"(filled {missing_before - missing_after})")

# ── Step 2: KNN-impute continuous columns ─────────────────────────────────────
print("\n" + "=" * 60)
print(f"Step 2 — KNN imputation for predictor columns (k={KNN_K})")
print("=" * 60)

print("\n  Missing before KNN imputation:")
for col in FEATURES_FOR_KNN:
    n = nhanes[col].isna().sum()
    pct = 100 * n / len(nhanes)
    print(f"    {col}: {n} ({pct:.1f}%)")

# KNNImputer operates on a numeric matrix — extract, impute, put back
# Target and label-engineered features (AVG_VISUAL_ACUITY, DIABETES_DURATION_YRS) are excluded to prevent data leakage
knn_matrix = nhanes[FEATURES_FOR_KNN].copy()

imputer = KNNImputer(n_neighbors=KNN_K, weights="distance")
imputed_matrix = imputer.fit_transform(knn_matrix)

nhanes[FEATURES_FOR_KNN] = imputed_matrix

print("\n  Missing after KNN imputation:")
for col in FEATURES_FOR_KNN:
    n = nhanes[col].isna().sum()
    print(f"    {col}: {n}")

# ── Step 3: Sanity checks ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3 — Sanity checks")
print("=" * 60)

total_missing = nhanes.isna().sum().sum()
print(f"\n  Total remaining missing values across all columns: {total_missing}")
if total_missing > 0:
    print("  Columns still missing:")
    print(nhanes.isna().sum()[nhanes.isna().sum() > 0].to_string())

# Corrected plausibility ranges — all units match NHANES raw coding
checks = {
    "BMXBMI":                (10,   80,  "BMI"),
    "BMXWAIST":              (40,  200,  "Waist circumference (cm)"),
    "BPXSY1":                (60,  250,  "Systolic BP (mmHg)"),
    "BPXDI1":                (20,  150,  "Diastolic BP (mmHg)"),
    "LBDHDD":                (10,  100,  "HDL cholesterol (mg/dL)"),   # NHANES uses mg/dL
    "LBXTR":                 (10, 2000,  "Triglycerides (mg/dL)"),
    "DIABETES_DURATION_YRS": ( 0,  100,  "Diabetes duration (yrs)"),   # 100 allows elderly early-onset
}
print("\n  Plausibility checks on imputed continuous values:")
for col, (lo, hi, label) in checks.items():
    if col not in nhanes.columns:
        continue
    n_low  = (nhanes[col] < lo).sum()
    n_high = (nhanes[col] > hi).sum()
    status = "OK" if (n_low + n_high) == 0 else "WARN"
    print(f"  [{status}] {label}: {n_low} below {lo}, {n_high} above {hi}")

# ── Step 4: One-hot encoding CYCLE, RIAGENDR, RIDRETH1 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 4 — One-hot encoding CYCLE, RIAGENDR, RIDRETH1")
print("=" * 60)

cycle_dummies = pd.get_dummies(nhanes["CYCLE"], prefix="CYCLE", drop_first=False)
# drop_first=False keeps all cycles explicit — easier to interpret in SHAP plots.
# Multicollinearity is not a concern for tree-based models.
nhanes = pd.concat([nhanes.drop(columns=["CYCLE"]), cycle_dummies], axis=1)
print(f"  CYCLE expanded into columns: {cycle_dummies.columns.tolist()}")
print(f"  Shape after encoding: {nhanes.shape}")

# --- Gender ---
gender_map = {
    1: "MALE",
    2: "FEMALE"
}

gender_dummies = pd.get_dummies(
    nhanes["RIAGENDR"].map(gender_map),
    prefix="IS",
    prefix_sep="_"
)

nhanes = pd.concat(
    [nhanes.drop(columns=["RIAGENDR"]), gender_dummies],
    axis=1
)

print(f"  Gender columns: {gender_dummies.columns.tolist()}")
print(f"  Shape after gender encoding: {nhanes.shape}")


# --- Ethnicity ---
ethnicity_map = {
    1: "MEXICAN_AMERICAN",
    2: "OTHER_HISPANIC",
    3: "NON_HISPANIC_WHITE",
    4: "NON_HISPANIC_BLACK",
    5: "OTHER_RACE"
}

ethnicity_dummies = pd.get_dummies(
    nhanes["RIDRETH1"].map(ethnicity_map),
    prefix="IS",
    prefix_sep="_"
)

nhanes = pd.concat(
    [nhanes.drop(columns=["RIDRETH1"]), ethnicity_dummies],
    axis=1
)

print(f"  Ethnicity columns: {ethnicity_dummies.columns.tolist()}")
print(f"  Shape after ethnicity encoding: {nhanes.shape}")


# ── Step 5: Target engineering ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5 — Target engineering")
print("=" * 60)

# Continuous target: AVG_VISUAL_ACUITY already exists (LogMAR scale).
# Lower = better vision; 0.0 = 20/20 equivalent.
print(f"  Continuous target AVG_VISUAL_ACUITY — "
      f"mean: {nhanes['AVG_VISUAL_ACUITY'].mean():.3f}, "
      f"std: {nhanes['AVG_VISUAL_ACUITY'].std():.3f}")

# Binary target: LogMAR > 0.3 corresponds approximately to Snellen 20/40,
# the standard clinical threshold for "visual impairment" used in NHANES research.
LOGMAR_THRESHOLD = 0.3
nhanes["VISION_IMPAIRED"] = (nhanes["AVG_VISUAL_ACUITY"] > LOGMAR_THRESHOLD).astype(int)
n_impaired = nhanes["VISION_IMPAIRED"].sum()
n_total    = len(nhanes)
pct        = 100 * n_impaired / n_total
print(f"  Binary target VISION_IMPAIRED (LogMAR > {LOGMAR_THRESHOLD}):")
print(f"    Impaired (1): {n_impaired:,}  ({pct:.1f}%)")
print(f"    Normal   (0): {n_total - n_impaired:,}  ({100-pct:.1f}%)")
print(f"    Class ratio 1:0 = 1:{(n_total-n_impaired)/max(n_impaired,1):.1f}")
if pct < 10:
    print("  [NOTE] Prevalence below 10% — class imbalance correction will be "
          "needed inside CV folds at training time (SMOTE or class_weight).")


# ── Step 6: Final summary and output ─────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 6 — Final summary")
print("=" * 60)

print(f"\n  Final shape: {nhanes.shape}")
print(f"\n  Descriptive statistics (continuous):")
pd.set_option("display.float_format", "{:.3f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(nhanes[CONTINUOUS].describe().to_string())

print("\n" + "=" * 60)
print("HEAD (first 10 rows)")
print("=" * 60)
print(nhanes.head(10).to_string(index=False))

nhanes.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Saved to: {OUTPUT_CSV}")