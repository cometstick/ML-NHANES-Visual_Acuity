# NHANES Ophthalmology-Metabolic ML Project

This repository contains a Python-based NHANES analysis pipeline with data exploration, imputation, synthesis, and comprehensive modeling for vision impairment.

## Project Structure

- `nhanes_imputed.csv` - Processed NHANES dataset used by the analysis scripts.
- `nhanes_merged.csv` - Merged NHANES source data used during preprocessing.
- `programs/` - Python scripts for data exploration, imputation, synthesis, and modeling.
- `training_data/` - NHANES source datasets in XPT format.

## Key Scripts

### `programs/data_learning_comprehensive.py`
- Runs a comprehensive machine learning pipeline on `nhanes_imputed.csv`.
- Trains Decision Tree, Random Forest, and XGBoost models.
- Uses three experimental settings:
  - Run A: full feature set
  - Run B: metabolic-only features
  - Run C: age-stratified models for 40-59 and 60+
- Applies SMOTE inside cross-validation to handle class imbalance.
- Produces model evaluation metrics and saves plots to the repository root.

### `programs/data_exploration.py`
- Loads `nhanes_imputed.csv` and performs exploratory analysis.
- Computes descriptive statistics, correlations, covariance, and high-correlation feature pairs.
- Generates a correlation heatmap and BPX vs `AVG_VISUAL_ACUITY` scatterplots.

### `programs/data_imputation.py`
- Prepares NHANES source files for analysis by handling missing values and creating the imputed dataset.

### `programs/data_synthesis.py`
- Contains data synthesis utilities used to augment or inspect the prepared dataset.

## Outputs

Generated files in the repository root include model and exploration plots, such as:

- `correlation_heatmap.png`
- `BPXSY1_vs_AVG_VISUAL_ACUITY.png`
- `BPXDI1_vs_AVG_VISUAL_ACUITY.png`
- `confusion_matrix_best.png`
- `f1_comparison.png`
- `roc_comparison.png`
- `precision_recall_side_by_side.png`
- `shap_full.png`
- `shap_metabolic.png`
- `shap_comparison.png`
- `shap_age_40-59.png`
- `shap_age_60plus.png`

## Requirements

Install the required Python packages before running the scripts.

```bash
python -m pip install pandas numpy scikit-learn xgboost imbalanced-learn shap matplotlib seaborn scipy
```

## Usage

Run the main training and evaluation script from the repository root:

```bash
python programs/data_learning_comprehensive.py
```

Run the data exploration script:

```bash
python programs/data_exploration.py
```

## Notes

- `VISION_IMPAIRED` is the binary prediction target used by the training script.
- `AVG_VISUAL_ACUITY` is used by the exploration script for continuous correlation analysis.
