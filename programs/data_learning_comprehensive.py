"""
NHANES Ophthalmology-Metabolic ML Training
==========================================
Comprehensive training script combining baseline and advanced models.

Analytical strategy
-------------------
Three parallel model runs answer three distinct research questions:

  RUN A — Full feature set
    All metabolic + demographic + cycle features.
    Establishes overall predictive ceiling and exposes confounder dominance.

  RUN B — Metabolic-only feature set
    Drops age, sex, race/ethnicity, income, and cycle dummies entirely.
    Isolates the pure metabolic contribution — the XAI story of the study.

  RUN C — Age-stratified (40-59, 60-75+)
    Re-runs the full model within each age band.
    Tests whether metabolic features rise in importance once age variance
    is removed by design rather than by exclusion.

Imbalance handling
------------------
SMOTE is applied INSIDE each CV fold via an imblearn Pipeline so synthetic
samples never leak into validation folds. The held-out test set is never
oversampled — it stays as the real-world class distribution.

Input  : nhanes_imputed.csv
Outputs: plots saved to working directory, console metrics
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, precision_score, recall_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# ── Configuration ──────────────────────────────────────────────────────────────
INPUT_CSV    = "nhanes_imputed.csv"
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_FOLDS     = 5
SHAP_SAMPLE  = 1000   # rows to sample for SHAP (full set is slow)

# SMOTE sampling_strategy controls the target minority:majority ratio after
# oversampling. 1.0 = full 50:50 balance. In a low-signal feature space
# (Run B — metabolic-only) full balance causes the model to collapse toward
# predicting everything positive (recall→1, precision→base rate). A more
# conservative ratio keeps the minority class meaningfully underrepresented
# so the model must actually discriminate rather than just predict positive.
# Run A uses a moderate ratio; Run B uses a conservative one.
SMOTE_RATIO_FULL = 0.5   # minority brought to 50% of majority count (Run A / C)
SMOTE_RATIO_META = 0.3   # minority brought to 30% of majority count (Run B)

# Features to exclude from all runs — they are either raw inputs to the target
# or the continuous version of the target itself
TARGET          = "VISION_IMPAIRED"

# original:
#ALWAYS_EXCLUDE  = ["VIDRVA", "VIDLVA", "AVG_VISUAL_ACUITY"]
# For ablation experiment (removing bp and lipids) use this instead:
#ALWAYS_EXCLUDE = ["VIDRVA", "VIDLVA", "AVG_VISUAL_ACUITY", "BPXSY1", "BPXDI1", "LBXTR", "LBDHDD"]
# For year confounding experiment use this instead:
ALWAYS_EXCLUDE  = ["VIDRVA", "VIDLVA", "AVG_VISUAL_ACUITY","CYCLE_1999-2000", "CYCLE_2001-2002", "CYCLE_2003-2004", "CYCLE_2005-2006", "CYCLE_2007-2008" ]

# Features to exclude from the metabolic-only run (Run B)
# CONFOUNDER_COLS = [
#     "RIDAGEYR", "RIAGENDR", "RIDRETH1", "INDHHINR",
#     "CYCLE_1999-2000", "CYCLE_2001-2002", "CYCLE_2003-2004",
#     "CYCLE_2005-2006", "CYCLE_2007-2008",
# ]
CONFOUNDER_COLS = [
    "RIDAGEYR", "INDHHINR",
    "IS_FEMALE", "IS_MALE",
    "IS_MEXICAN_AMERICAN", "IS_NON_HISPANIC_BLACK",
    "IS_NON_HISPANIC_WHITE", "IS_OTHER_HISPANIC", "IS_OTHER_RACE",
    "CYCLE_1999-2000", "CYCLE_2001-2002", "CYCLE_2003-2004",
    "CYCLE_2005-2006", "CYCLE_2007-2008",
]

AGE_BANDS = [(40, 59, "40-59"), (60, 120, "60+")]

# ── Load ───────────────────────────────────────────────────────────────────────
print("=" * 70)
print("Loading data")
print("=" * 70)
df = pd.read_csv(INPUT_CSV)
print(f"  Shape: {df.shape}")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Target distribution:")
print(df[TARGET].value_counts().to_string())
print(f"  Positive class rate: {df[TARGET].mean():.1%}")

# ── Feature sets ───────────────────────────────────────────────────────────────
FEATURES_FULL  = [c for c in df.columns
                  if c not in ALWAYS_EXCLUDE + [TARGET]]
FEATURES_META  = [c for c in FEATURES_FULL
                  if c not in CONFOUNDER_COLS]

print(f"\n  Run A — full feature set ({len(FEATURES_FULL)}): {FEATURES_FULL}")
print(f"  Run B — metabolic-only  ({len(FEATURES_META)}): {FEATURES_META}")

# ── Train / test split ─────────────────────────────────────────────────────────
X_full  = df[FEATURES_FULL].copy()
X_meta  = df[FEATURES_META].copy()
y       = df[TARGET].copy()

(X_train_full, X_test_full,
 X_train_meta, X_test_meta,
 y_train,      y_test) = (
    lambda idx_tr, idx_te: (
        X_full.iloc[idx_tr],  X_full.iloc[idx_te],
        X_meta.iloc[idx_tr],  X_meta.iloc[idx_te],
        y.iloc[idx_tr],       y.iloc[idx_te],
    )
)(
    *[list(x) for x in train_test_split(
        range(len(df)), test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y
    )]
)

print(f"\n  Train: {len(y_train)} rows  |  Test: {len(y_test)} rows")
print(f"  Train positive rate: {y_train.mean():.1%}")
print(f"  Test  positive rate: {y_test.mean():.1%}")

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

# ── Model factory ──────────────────────────────────────────────────────────────
def make_models():
    return {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_split=20, min_samples_leaf=10,
            random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=20,
            min_samples_leaf=10, random_state=RANDOM_STATE,
            class_weight="balanced", n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE, eval_metric="logloss",
            verbosity=0
        ),
    }

# ── Helper: evaluate one model with SMOTE-in-CV ────────────────────────────────
def evaluate(name, model, X_tr, X_te, y_tr, y_te, cv,
             smote_ratio=SMOTE_RATIO_FULL):
    """
    Wraps model in an imblearn Pipeline so SMOTE only sees training folds.
    smote_ratio: target minority/majority ratio after oversampling.
      Use SMOTE_RATIO_FULL for rich feature sets, SMOTE_RATIO_META for
      metabolic-only runs where full balance causes recall collapse.
    Returns a results dict with metrics and predictions.
    """
    pipe = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=smote_ratio,
                        random_state=RANDOM_STATE)),
        ("clf",   model)
    ])

    cv_res = cross_validate(
        pipe, X_tr, y_tr, cv=cv,
        scoring=["f1", "roc_auc"],
        return_train_score=False, n_jobs=-1
    )

    # Final fit on full training set with SMOTE, evaluate on raw test set
    pipe.fit(X_tr, y_tr)
    y_pred  = pipe.predict(X_te)
    y_proba = pipe.predict_proba(X_te)[:, 1]

    # Extract the inner classifier for SHAP / feature importance
    clf = pipe.named_steps["clf"]

    return {
        "name":        name,
        "pipe":        pipe,
        "clf":         clf,
        "cv_f1_mean":  cv_res["test_f1"].mean(),
        "cv_f1_std":   cv_res["test_f1"].std(),
        "cv_auc_mean": cv_res["test_roc_auc"].mean(),
        "test_f1":     f1_score(y_te, y_pred),
        "test_prec":   precision_score(y_te, y_pred),
        "test_rec":    recall_score(y_te, y_pred),
        "test_auc":    roc_auc_score(y_te, y_proba),
        "y_true":      y_te,
        "y_pred":      y_pred,
        "y_proba":     y_proba,
        "cm":          confusion_matrix(y_te, y_pred),
        "features":    list(X_tr.columns),
    }

# ── Helper: print run summary table ────────────────────────────────────────────
def print_summary(run_results):
    rows = []
    for r in run_results.values():
        rows.append({
            "Model":       r["name"],
            "CV F1":       f"{r['cv_f1_mean']:.4f} ±{r['cv_f1_std']:.4f}",
            "CV AUC":      f"{r['cv_auc_mean']:.4f}",
            "Test F1":     f"{r['test_f1']:.4f}",
            "Test Prec":   f"{r['test_prec']:.4f}",
            "Test Rec":    f"{r['test_rec']:.4f}",
            "Test AUC":    f"{r['test_auc']:.4f}",
        })
    print(pd.DataFrame(rows).to_string(index=False))

# ══════════════════════════════════════════════════════════════════════════════
# RUN A — Full feature set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RUN A — Full feature set (all metabolic + demographic + cycle)")
print("=" * 70)

cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results_A = {}
for name, model in make_models().items():
    print(f"  Training {name}...")
    results_A[name] = evaluate(name, model, X_train_full, X_test_full,
                               y_train, y_test, cv)
    r = results_A[name]
    print(f"    CV F1: {r['cv_f1_mean']:.4f} ±{r['cv_f1_std']:.4f}  "
          f"| Test F1: {r['test_f1']:.4f}  AUC: {r['test_auc']:.4f}")

print("\n  Run A summary:")
print_summary(results_A)

# ══════════════════════════════════════════════════════════════════════════════
# RUN B — Metabolic-only feature set
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RUN B — Metabolic-only (confounders removed)")
print("=" * 70)
print(f"  Dropped: {[c for c in FEATURES_FULL if c not in FEATURES_META]}")

results_B = {}
for name, model in make_models().items():
    print(f"  Training {name}...")
    results_B[name] = evaluate(name, model, X_train_meta, X_test_meta,
                               y_train, y_test, cv,
                               smote_ratio=SMOTE_RATIO_META)
    r = results_B[name]
    print(f"    CV F1: {r['cv_f1_mean']:.4f} ±{r['cv_f1_std']:.4f}  "
          f"| Test F1: {r['test_f1']:.4f}  AUC: {r['test_auc']:.4f}")

print("\n  Run B summary:")
print_summary(results_B)

# AUC delta — how much do confounders add?
best_A_auc = max(r["test_auc"] for r in results_A.values())
best_B_auc = max(r["test_auc"] for r in results_B.values())
print(f"\n  AUC delta (full vs metabolic-only): "
      f"{best_A_auc:.4f} - {best_B_auc:.4f} = {best_A_auc - best_B_auc:.4f}")
print("  → This is the incremental predictive value of demographics over metabolism.")

# ══════════════════════════════════════════════════════════════════════════════
# RUN C — Age-stratified analysis
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("RUN C — Age-stratified models (full feature set within each band)")
print("=" * 70)

results_C = {}
for lo, hi, label in AGE_BANDS:
    age_mask_all   = (df["RIDAGEYR"] >= lo) & (df["RIDAGEYR"] < hi)
    stratum        = df[age_mask_all].copy()
    X_s = stratum[FEATURES_FULL]
    y_s = stratum[TARGET]

    if y_s.sum() < 20:
        print(f"  [{label}] Too few positives ({y_s.sum()}) — skipping.")
        continue

    X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
        X_s, y_s, test_size=TEST_SIZE,
        random_state=RANDOM_STATE, stratify=y_s
    )
    print(f"\n  [{label}]  n={len(stratum)}  positives={y_s.sum()} "
          f"({100*y_s.mean():.1f}%)")

    band_res = {}
    for name, model in make_models().items():
        print(f"    Training {name}...")
        r = evaluate(name, model, X_tr_s, X_te_s, y_tr_s, y_te_s, cv)
        r["X_te"] = X_te_s
        band_res[name] = r
        print(f"      CV F1: {r['cv_f1_mean']:.4f}  "
              f"Test F1: {r['test_f1']:.4f}  AUC: {r['test_auc']:.4f}")

    results_C[label] = band_res
    print(f"  [{label}] summary:")
    print_summary(band_res)

# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter tuning — best model from Run A
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Hyperparameter tuning — XGBoost (Run A)")
print("=" * 70)

param_dist = {
    "clf__n_estimators":     [100, 200, 300],
    "clf__max_depth":        [4, 6, 8],
    "clf__learning_rate":    [0.01, 0.05, 0.1],
    "clf__subsample":        [0.7, 0.8, 1.0],
    "clf__colsample_bytree": [0.7, 0.8, 1.0],
    "clf__min_child_weight": [1, 3, 5],
}

xgb_pipe = ImbPipeline([
    ("smote", SMOTE(sampling_strategy=SMOTE_RATIO_FULL,
                    random_state=RANDOM_STATE)),
    ("clf", XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss",
        verbosity=0
    ))
])

search = RandomizedSearchCV(
    xgb_pipe, param_distributions=param_dist,
    n_iter=20, scoring="f1", cv=cv,
    random_state=RANDOM_STATE, n_jobs=-1, verbose=0
)
search.fit(X_train_full, y_train)

print(f"  Best CV F1: {search.best_score_:.4f}")
print(f"  Best params: {search.best_params_}")

best_pipe = search.best_estimator_
y_pred_tuned  = best_pipe.predict(X_test_full)
y_proba_tuned = best_pipe.predict_proba(X_test_full)[:, 1]

tuned_f1  = f1_score(y_test, y_pred_tuned)
tuned_auc = roc_auc_score(y_test, y_proba_tuned)
print(f"  Tuned XGBoost — Test F1: {tuned_f1:.4f}  AUC: {tuned_auc:.4f}")
print(f"  vs baseline XGBoost  — Test F1: {results_A['XGBoost']['test_f1']:.4f}"
      f"  AUC: {results_A['XGBoost']['test_auc']:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Class-weighted XGBoost comparison (no SMOTE)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Class-weighted XGBoost comparison (no SMOTE)")
print("=" * 70)

neg = (y_train == 0).sum()
pos = (y_train == 1).sum()
cw_scale_pos_weight = neg / pos

cw_xgb = XGBClassifier(
    **{k.replace("clf__", ""): v for k, v in search.best_params_.items()},
    scale_pos_weight=cw_scale_pos_weight,
    random_state=RANDOM_STATE,
    eval_metric="logloss", verbosity=0
)
cw_xgb.fit(X_train_full, y_train)


xgb_y_pred = cw_xgb.predict(X_test_full)
cv_auc = roc_auc_score(y_test, cw_xgb.predict_proba(X_test_full)[:, 1])
cw_f1 = f1_score(y_test, xgb_y_pred)
cw_prec = precision_score(y_test, xgb_y_pred)
cw_rec = recall_score(y_test, xgb_y_pred)

print(f"  Class-weighted XGBoost — Test F1: {cw_f1:.4f}  AUC: {cv_auc:.4f}")
print(f"    Precision: {cw_prec:.4f}  Recall: {cw_rec:.4f}  "
      f"Scale pos weight: {cw_scale_pos_weight:.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# Threshold tuning on tuned XGBoost
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Threshold tuning — tuned XGBoost")
print("=" * 70)

precision, recall, _ = precision_recall_curve(y_test, y_proba_tuned)
thresholds = np.linspace(0.1, 0.9, 50)
best_f1 = 0.0
best_thresh = thresholds[0]
for t in thresholds:
    preds = (y_proba_tuned > t).astype(int)
    f1 = f1_score(y_test, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thresh = t

final_preds = (y_proba_tuned >= best_thresh).astype(int)
best_prec = precision_score(y_test, final_preds)
best_rec = recall_score(y_test, final_preds)
opt_thresh = best_thresh
opt_f1 = best_f1
y_pred_opt = final_preds

pr_auc = average_precision_score(y_test, y_proba_tuned)
print(f"  Best Threshold: {best_thresh:.3f}")
print(f"  F1: {best_f1:.4f}")
print(f"  Precision: {best_prec:.4f}")
print(f"  Recall: {best_rec:.4f}")
print(f"  ROC-AUC: {tuned_auc:.4f}")
print(f"  PR-AUC: {pr_auc:.4f}")
print("  Confusion Matrix:")
print(confusion_matrix(y_test, final_preds))

print("=== Evaluation Summary ===")
print(f"ROC-AUC: {tuned_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")
print(f"Best Threshold: {best_thresh:.3f}")
print(f"F1: {best_f1:.4f}")
print(f"Precision: {best_prec:.4f}")
print(f"Recall: {best_rec:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SHAP analysis — Run A (full) and Run B (metabolic-only)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SHAP analysis — XGBoost, full vs metabolic-only vs age-stratified")
print("=" * 70)

shap_results = {}
for run_label, run_results, X_te in [
    ("Full",          results_A, X_test_full),
    ("Metabolic-only", results_B, X_test_meta),
]:
    clf = run_results["XGBoost"]["clf"]
    X_sample = X_te.sample(n=min(SHAP_SAMPLE, len(X_te)),
                           random_state=RANDOM_STATE)
    print(f"  Computing SHAP for {run_label} XGBoost ({len(X_sample)} samples)...")
    try:
        explainer   = shap.TreeExplainer(clf)
        shap_vals   = explainer.shap_values(X_sample)
        shap_results[run_label] = {
            "explainer": explainer,
            "values":    shap_vals,
            "X_sample":  X_sample,
        }
        mean_abs = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=X_sample.columns
        ).sort_values(ascending=False)
        print(f"  Top 5 SHAP features ({run_label}):")
        print(mean_abs.head(5).round(4).to_string())
    except Exception as e:
        print(f"  [WARN] SHAP failed for {run_label}: {e}")

for label, band_res in results_C.items():
    if "XGBoost" not in band_res:
        continue
    res = band_res["XGBoost"]
    run_label = f"Age {label}"
    X_te = res["X_te"]
    X_sample = X_te.sample(n=min(SHAP_SAMPLE, len(X_te)),
                           random_state=RANDOM_STATE)
    print(f"  Computing SHAP for {run_label} XGBoost ({len(X_sample)} samples)...")
    try:
        explainer   = shap.TreeExplainer(res["clf"])
        shap_vals   = explainer.shap_values(X_sample)
        shap_results[run_label] = {
            "explainer": explainer,
            "values":    shap_vals,
            "X_sample":  X_sample,
        }
        mean_abs = pd.Series(
            np.abs(shap_vals).mean(axis=0),
            index=X_sample.columns
        ).sort_values(ascending=False)
        print(f"  Top 5 SHAP features ({run_label}):")
        print(mean_abs.head(5).round(4).to_string())
    except Exception as e:
        print(f"  [WARN] SHAP failed for {run_label}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# Stacking ensemble (full feature set)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Stacking ensemble — full feature set")
print("=" * 70)

base_estimators = [
    ("dt",  DecisionTreeClassifier(
        max_depth=8, min_samples_split=20, min_samples_leaf=10,
        random_state=RANDOM_STATE, class_weight="balanced")),
    ("rf",  RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=20,
        min_samples_leaf=10, random_state=RANDOM_STATE,
        class_weight="balanced", n_jobs=-1)),
    ("xgb", XGBClassifier(
        **{k.replace("clf__", ""): v
           for k, v in search.best_params_.items()},
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        eval_metric="logloss", verbosity=0)),
]
stack = StackingClassifier(
    estimators=base_estimators,
    final_estimator=LogisticRegression(
        random_state=RANDOM_STATE, class_weight="balanced"),
    cv=5, n_jobs=-1
)
# Fit stacker on SMOTE-augmented training data once
from imblearn.over_sampling import SMOTE as _SMOTE
X_tr_sm, y_tr_sm = _SMOTE(sampling_strategy=SMOTE_RATIO_FULL,
                            random_state=RANDOM_STATE).fit_resample(
    X_train_full, y_train)
stack.fit(X_tr_sm, y_tr_sm)

y_pred_stack  = stack.predict(X_test_full)
y_proba_stack = stack.predict_proba(X_test_full)[:, 1]
stack_f1  = f1_score(y_test, y_pred_stack)
stack_auc = roc_auc_score(y_test, y_proba_stack)
print(f"  Stacking — Test F1: {stack_f1:.4f}  AUC: {stack_auc:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# Final model comparison table
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

final_rows = []
for name, r in results_A.items():
    final_rows.append({"Run": "A-Full",      "Model": name,
                       "TestF1": r["test_f1"], "TestAUC": r["test_auc"],
                       "CVF1": r["cv_f1_mean"]})
for name, r in results_B.items():
    final_rows.append({"Run": "B-MetaOnly",  "Model": name,
                       "TestF1": r["test_f1"], "TestAUC": r["test_auc"],
                       "CVF1": r["cv_f1_mean"]})
final_rows.append({"Run": "A-Tuned",    "Model": "XGBoost (tuned)",
                   "TestF1": tuned_f1,   "TestAUC": tuned_auc,  "CVF1": search.best_score_})
final_rows.append({"Run": "A-Tuned",    "Model": "XGBoost (opt threshold)",
                   "TestF1": opt_f1,     "TestAUC": tuned_auc,  "CVF1": search.best_score_})
final_rows.append({"Run": "A-ClassWeight", "Model": "XGBoost (class-weighted)",
                   "TestF1": cw_f1,     "TestAUC": cv_auc,   "CVF1": float("nan")})
final_rows.append({"Run": "A-Stack",    "Model": "Stacking Ensemble",
                   "TestF1": stack_f1,   "TestAUC": stack_auc,  "CVF1": float("nan")})

final_df = (pd.DataFrame(final_rows)
            .sort_values("TestF1", ascending=False)
            .reset_index(drop=True))
print(final_df.round(4).to_string(index=False))

best_row = final_df.iloc[0]
print(f"\n  Best overall: {best_row['Run']} / {best_row['Model']}  "
      f"F1={best_row['TestF1']:.4f}  AUC={best_row['TestAUC']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Visualisations
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Generating plots")
print("=" * 70)

# ── 1. ROC curves — full vs metabolic-only vs age-stratified (XGBoost) ──────
fig, ax = plt.subplots(figsize=(10, 7))
for run_label, run_results, ls in [
    ("Full (XGBoost)",          results_A, "-"),
    ("Metabolic-only (XGBoost)", results_B, "--"),
]:
    fpr, tpr, _ = roc_curve(y_test, run_results["XGBoost"]["y_proba"])
    auc = run_results["XGBoost"]["test_auc"]
    ax.plot(fpr, tpr, lw=2, ls=ls, label=f"{run_label}  AUC={auc:.3f}")

for label, band_res in results_C.items():
    res = band_res["XGBoost"]
    fpr, tpr, _ = roc_curve(res["y_true"], res["y_proba"])
    ax.plot(fpr, tpr, lw=2, ls=":", label=f"Age {label} XGBoost  AUC={res['test_auc']:.3f}")

fpr_t, tpr_t, _ = roc_curve(y_test, y_proba_tuned)
ax.plot(fpr_t, tpr_t, lw=2, ls=":" , label=f"Tuned XGBoost (full)  AUC={tuned_auc:.3f}")
fpr_s, tpr_s, _ = roc_curve(y_test, y_proba_stack)
ax.plot(fpr_s, tpr_s, lw=2, ls="-.", label=f"Stacking  AUC={stack_auc:.3f}")

ax.plot([0,1],[0,1],"k--", lw=1, label="Random")
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves — Full, Metabolic-only, and Age-Stratified")
ax.legend(loc="lower right", fontsize=8); plt.tight_layout()
plt.savefig("roc_comparison.png", dpi=120, bbox_inches="tight"); plt.close()
print("  Saved: roc_comparison.png")

# ── 2. F1 comparison bar chart ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 5))
plot_df = final_df.copy()
plot_df["label"] = plot_df["Run"] + "\n" + plot_df["Model"]
bars = ax.bar(range(len(plot_df)), plot_df["TestF1"],
              color=["#4C72B0" if "Full" in r else
                     "#DD8452" if "Meta" in r else
                     "#55A868" for r in plot_df["Run"]])
ax.set_xticks(range(len(plot_df)))
ax.set_xticklabels(plot_df["label"], fontsize=8, rotation=20, ha="right")
ax.set_ylabel("Test F1"); ax.set_title("Test F1 — All Models and Runs")
for i, v in enumerate(plot_df["TestF1"]):
    ax.text(i, v + 0.003, f"{v:.3f}", ha="center", fontsize=7)
ax.set_ylim(0, plot_df["TestF1"].max() + 0.06)
from matplotlib.patches import Patch
ax.legend(handles=[
    Patch(color="#4C72B0", label="Full features"),
    Patch(color="#DD8452", label="Metabolic-only"),
    Patch(color="#55A868", label="Tuned / Ensemble"),
], fontsize=8)
plt.tight_layout()
plt.savefig("f1_comparison.png", dpi=120, bbox_inches="tight"); plt.close()
print("  Saved: f1_comparison.png")

# ── 3. SHAP beeswarm — full ────────────────────────────────────────────────────
if "Full" in shap_results:
    sr = shap_results["Full"]
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sr["values"], sr["X_sample"], show=False)
    plt.title("SHAP Summary — Full Feature Set (XGBoost)")
    plt.tight_layout()
    plt.savefig("shap_full.png", dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved: shap_full.png")

# ── 4. SHAP beeswarm — metabolic-only ─────────────────────────────────────────
if "Metabolic-only" in shap_results:
    sr = shap_results["Metabolic-only"]
    plt.figure(figsize=(10, 7))
    shap.summary_plot(sr["values"], sr["X_sample"], show=False)
    plt.title("SHAP Summary — Metabolic-Only Feature Set (XGBoost)")
    plt.tight_layout()
    plt.savefig("shap_metabolic.png", dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved: shap_metabolic.png")

# ── 5. SHAP beeswarm — age-stratified XGBoost ─────────────────────────────────
for label, band_res in results_C.items():
    if "XGBoost" not in band_res:
        continue
    res = band_res["XGBoost"]
    safe_label = label.replace("+", "plus").replace(" ", "_")
    try:
        X_sample = res["X_te"].sample(n=min(SHAP_SAMPLE, len(res["X_te"])),
                                       random_state=RANDOM_STATE)
        explainer = shap.TreeExplainer(res["clf"])
        shap_vals = explainer.shap_values(X_sample)
        plt.figure(figsize=(10, 7))
        shap.summary_plot(shap_vals, X_sample, show=False)
        plt.title(f"SHAP Summary — Age {label} XGBoost")
        plt.tight_layout()
        plot_path = f"shap_age_{safe_label}.png"
        plt.savefig(plot_path, dpi=120, bbox_inches="tight"); plt.close()
        print(f"  Saved: {plot_path}")
    except Exception as e:
        print(f"  [WARN] Could not generate SHAP for Age {label}: {e}")

# ── 6. SHAP mean |value| side-by-side comparison ──────────────────────────────
if shap_results:
    keys = list(shap_results.keys())
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 7), squeeze=False)
    for i, run_label in enumerate(keys):
        ax = axes[0, i]
        sr   = shap_results[run_label]
        imp  = pd.Series(
            np.abs(sr["values"]).mean(axis=0),
            index=sr["X_sample"].columns
        ).sort_values(ascending=True).tail(15)

        label_map = {
    "RIDAGEYR": "Age",
    "INDHHINR": "Household Income",
    "DIQ010": "Diabetes Diagnosis",
    "DIQ050": "Insulin Use",
    "LBXTR": "Triglycerides",
    "BMXWAIST": "Waist Circumference",
    "BMXBMI": "BMI",
    "BPXSY1": "Systolic Blood Pressure",
    "BPXDI1": "Diastolic Blood Pressure",
    "LBDHDD": "HDL Cholesterol",
    "DIABETES_DURATION_YRS": "Diabetes Duration",
    "IS_FEMALE": "Female",
    "IS_MALE": "Male",
    "IS_MEXICAN_AMERICAN": "Mexican American",
    "IS_NON_HISPANIC_BLACK": "Non-Hispanic Black",
    "IS_NON_HISPANIC_WHITE": "Non-Hispanic White",
    "IS_OTHER_HISPANIC": "Other Hispanic",
    "IS_OTHER_RACE": "Other Race"
}
        imp.index = imp.index.map(lambda x: label_map.get(x, x))
        ax.barh(range(len(imp)), imp.values, color="#4C72B0")
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels(imp.index, fontsize=9)
        ax.set_xlabel("mean(|SHAP value|)")
        ax.set_title(f"SHAP Importance — {run_label}")
    plt.suptitle("SHAP Feature Rankings: Full, Metabolic-only, and Age-stratified", y=1.01)
    plt.tight_layout()
    plt.savefig("shap_comparison.png", dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved: shap_comparison.png")

# ── 6. Confusion matrix — best model ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 5))
best_cm = confusion_matrix(y_test, y_pred_opt)
sns.heatmap(best_cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Normal (0)", "Impaired (1)"],
            yticklabels=["Normal (0)", "Impaired (1)"])
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — Tuned XGBoost (threshold={opt_thresh:.2f})")
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=120, bbox_inches="tight"); plt.close()
print("  Saved: confusion_matrix_best.png")

# ── 7. Age-stratified F1 comparison ───────────────────────────────────────────
if results_C:
    band_labels = list(results_C.keys())
    model_names = list(list(results_C.values())[0].keys())
    x = np.arange(len(band_labels))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, mname in enumerate(model_names):
        f1s = [results_C[b][mname]["test_f1"] for b in band_labels]
        ax.bar(x + i * width, f1s, width, label=mname, color=colors[i])
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Age {b}" for b in band_labels])
    ax.set_ylabel("Test F1")
    ax.set_title("Age-Stratified F1 Scores (Full Feature Set)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("f1_age_stratified.png", dpi=120, bbox_inches="tight"); plt.close()
    print("  Saved: f1_age_stratified.png")

# ── 8. Precision-recall curves — full vs metabolic-only vs age-stratified ───
fig, ax = plt.subplots(figsize=(10, 7))

age_colors = {
    "40-59": "C2",
    "60+":   "C3",
}
curve_defs = [
    ("Full-feature XGBoost", y_test, results_A["XGBoost"]["y_proba"], "C0"),
    ("Metabolic-only XGBoost", y_test, results_B["XGBoost"]["y_proba"], "C1"),
]
for label, band_res in results_C.items():
    curve_defs.append((
        f"Age {label} XGBoost",
        band_res["XGBoost"]["y_true"],
        band_res["XGBoost"]["y_proba"],
        age_colors.get(label, "C2")
    ))

for label, y_true, probs, color in curve_defs:
    p, r, _ = precision_recall_curve(y_true, probs)
    ax.plot(r[:-1], p[:-1], label=label, color=color)

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves — Full, Metabolic-only, and Age-Stratified")
ax.legend(fontsize=8)
ax.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig("precision_recall_side_by_side.png", dpi=120, bbox_inches="tight")
plt.close()
print("  Saved: precision_recall_side_by_side.png")

# ── Final console summary ──────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STUDY SUMMARY")
print("=" * 70)
best_meta_xgb = results_B["XGBoost"]
print(f"""
Run A (full features) — best Test AUC : {best_A_auc:.4f}
Run B (metabolic-only)— best Test AUC : {best_B_auc:.4f}
Confounder contribution (AUC delta)   : {best_A_auc - best_B_auc:.4f}

Tuned XGBoost (full, SMOTE-in-CV)
  Test F1         : {tuned_f1:.4f}
  Test AUC        : {tuned_auc:.4f}
  PR-AUC          : {pr_auc:.4f}
  Optimal threshold : {opt_thresh:.3f}  →  F1={opt_f1:.4f}

Class-weighted XGBoost (full, no SMOTE)
  Test F1         : {cw_f1:.4f}
  Test AUC        : {cv_auc:.4f}
  Precision       : {cw_prec:.4f}
  Recall          : {cw_rec:.4f}

Metabolic-only XGBoost
  Test F1         : {best_meta_xgb['test_f1']:.4f}
  Test AUC        : {best_meta_xgb['test_auc']:.4f}

Stacking Ensemble (full, SMOTE on train)
  Test F1         : {stack_f1:.4f}
  Test AUC        : {stack_auc:.4f}

Interpretation guide
---------------------
• If Run A AUC >> Run B AUC : demographics dominate; metabolic signal is
  partially masked by age/income. Stratified analysis (Run C) will reveal
  whether metabolic features rise in importance within age bands.
• If Run A AUC ≈ Run B AUC  : metabolic features alone carry most of the
  predictive signal — a strong result for the study hypothesis.
• SHAP comparison plots (shap_comparison.png) show which metabolic features
  gain or lose rank when confounders are removed.
""")