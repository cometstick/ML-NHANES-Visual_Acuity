"""
NHANES Ophthalmology-Metabolic Pipeline
========================================
Compiles BMX, BPX, DEMO, DIQ, HDL, TRIGLY, VIX files across
1999-2000 through 2007-2008 cycles into a single merged dataframe.

Usage
-----
Set DATA_DIR to the folder containing your .xpt files, then run:
    python data_synthesis.py
"""

import os
import pandas as pd

# ── Configuration ─────────────────────────────────────────────────────────────
DATA_DIR = "./training_data"          # <-- change to your folder path if needed
OUTPUT_CSV = "nhanes_merged.csv"

CYCLES = [
    ("1999-2000", "1999-2000"),
    ("2001-2002", "2001-2002"),
    ("2003-2004", "2003-2004"),
    ("2005-2006", "2005-2006"),
    ("2007-2008", "2007-2008"),
]

# ── Column specs (SEQN always kept; listed extras are additional keeps) ────────
DEMO_COLS  = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1", "INDHHINC", "INDHHIN2", "INDHHINR"]
DIQ_KEEP  = ["SEQN", "DIQ010", "DIQ050", "DID040G", "DID040Q", "DIQ040G", "DIQ040Q", "DID040"]
HDL_COLS   = ["SEQN", "LBDHDD", "LBXHDD"]
TRIGLY_COLS= ["SEQN", "LBXTR"]
VIX_COLS   = ["SEQN", "VIDROVA", "VIDLOVA"]
BMX_COLS   = ["SEQN", "BMXWAIST", "BMXBMI"]
BPX_COLS   = ["SEQN", "BPXSY1", "BPXDI1"]

# ── Helper: load one XPT file ─────────────────────────────────────────────────
def load_xpt(prefix: str, cycle: str) -> pd.DataFrame:
    fname = os.path.join(DATA_DIR, f"{prefix}{cycle}.xpt")
    if not os.path.exists(fname):
        print(f"  [WARN] File not found: {fname}")
        return pd.DataFrame()
    df = pd.read_sas(fname, format="xport", encoding="utf-8")
    df.columns = df.columns.str.upper()
    return df

# ── Step 2: Load, filter, and merge each cycle ────────────────────────────────
print("=" * 60)
print("Step 2 — Loading and merging each cycle")
print("=" * 60)

cycle_frames = []

for label, cycle in CYCLES:
    print(f"\n  Cycle: {cycle}")

    # --- DEMO ---
    demo = load_xpt("DEMO", cycle)
    if not demo.empty:
        # Income variable changed names across cycles — pick whichever exists
        income_col = None
        for cand in ["INDHHINR", "INDHHIN2", "INDHHINC"]:
            if cand in demo.columns:
                income_col = cand
                break
        keep = ["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH1"]
        if income_col:
            keep.append(income_col)
        demo = demo[[c for c in keep if c in demo.columns]].copy()
        # Standardise income column name
        if income_col and income_col != "INDHHINR":
            demo.rename(columns={income_col: "INDHHINR"}, inplace=True)
        print(f"    DEMO  : {len(demo)} rows, cols: {demo.columns.tolist()}")

    # --- DIQ ---
    diq = load_xpt("DIQ", cycle)
    if not diq.empty: # standardize naming of labels: DID040 = dx age, DID040Q= current age.
        if "DID040G" in diq.columns and "DID040" not in diq.columns:
            diq.rename(columns={"DID040G": "DID040"}, inplace=True)
            print(" case1,   DIQ   : Renamed DID040G → DID040")
        if "DIQ040G" in diq.columns and "DIQ040" not in diq.columns:
            diq.rename(columns={"DIQ040G": "DID040"}, inplace=True)
            print(" case2,   DIQ   : Renamed DIQ040G → DID040")
        if "DIQ040Q" in diq.columns and "DID040Q" not in diq.columns:
            diq.rename(columns={"DIQ040Q": "DID040Q"}, inplace=True)
            print(" case3,   DIQ   : Renamed DIQ040Q → DID040Q")
        keep_cols = list(dict.fromkeys([c for c in DIQ_KEEP if c in diq.columns]))
        diq = diq[keep_cols].copy()
        print(f"    DIQ   : {len(diq)} rows, cols: {diq.columns.tolist()}")

    # --- HDL ---
    hdl = load_xpt("HDL", cycle)
    if not hdl.empty:
        # 1999-2004 cycles name this LBXHDD; standardise to LBDHDD before filtering
        if "LBXHDD" in hdl.columns and "LBDHDD" not in hdl.columns:
            hdl.rename(columns={"LBXHDD": "LBDHDD"}, inplace=True)
        hdl = hdl[[c for c in HDL_COLS if c in hdl.columns]].copy()
        print(f"    HDL   : {len(hdl)} rows, cols: {hdl.columns.tolist()}")

    # --- TRIGLY ---
    trigly = load_xpt("TRIGLY", cycle)
    if not trigly.empty:
        trigly = trigly[[c for c in TRIGLY_COLS if c in trigly.columns]].copy()
        print(f"    TRIGLY: {len(trigly)} rows, cols: {trigly.columns.tolist()}")

    # --- VIX ---
    vix = load_xpt("VIX", cycle)
    if not vix.empty:
        vix = vix[[c for c in VIX_COLS if c in vix.columns]].copy()
        print(f"    VIX   : {len(vix)} rows, cols: {vix.columns.tolist()}")

    # --- BMX ---
    bmx = load_xpt("BMX", cycle)
    if not bmx.empty:
        bmx = bmx[[c for c in BMX_COLS if c in bmx.columns]].copy()
        print(f"    BMX   : {len(bmx)} rows, cols: {bmx.columns.tolist()}")

    # --- BPX ---
    bpx = load_xpt("BPX", cycle)
    if not bpx.empty:
        bpx = bpx[[c for c in BPX_COLS if c in bpx.columns]].copy()
        print(f"    BPX   : {len(bpx)} rows, cols: {bpx.columns.tolist()}")

    # --- Merge all on SEQN (left join from DEMO as anchor) ---
    frames_to_merge = [df for df in [demo, diq, hdl, trigly, vix, bmx, bpx]
                       if not df.empty and "SEQN" in df.columns]

    if not frames_to_merge:
        print(f"    [WARN] No data for cycle {cycle}, skipping.")
        continue

    merged = frames_to_merge[0]
    for df in frames_to_merge[1:]:
        merged = merged.merge(df, on="SEQN", how="left")

    merged["CYCLE"] = label
    cycle_frames.append(merged)
    print(f"    → Merged shape: {merged.shape}")

# ── Step 3: Concatenate all cycles ────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 3 — Concatenating all cycles")
print("=" * 60)

nhanes = pd.concat(cycle_frames, ignore_index=True, sort=False)
print(f"\n  Combined shape (raw after concat): {nhanes.shape}")

# Step 3a: Drop SEQN — it is a merge key only, so has no analytical value
nhanes.drop(columns=["SEQN"], inplace=True, errors="ignore")
print("  SEQN dropped.")

# Fix hidden codes across all feature columns based on the following given look-up table, replacing the placeholder/coded values with NaN.
# For VIDROVA / VIDLOVA: these are Snellen denominator values (25 = 20/25, 40 = 20/40
# etc.), Since value 666 encodes "worse than 20/200" (i.e. unmeasurable), 
# cap to 200 for better numeric stability and interpretability.
HIDDEN_CODES = {"RIAGENDR": [7, 9], "RIDAGEYR": [7, 9], "RIDRETH1": [7, 9], "INDHHINR": [77, 99], "DIQ010": [7, 9], "DIQ050": [7, 9], "DID040G": [7, 9], "DIQ040G": [7, 9], "DIQ040Q": [77777, 99999], "DID040Q": [77777, 99999], "DID040": [777, 999], "VIDROVA": [666], "VIDLOVA": [666]}
for col, codes in HIDDEN_CODES.items():
    if col not in nhanes.columns:
        continue
    for code in codes:
        n_coded = (nhanes[col] == code ).sum()
        nhanes.loc[nhanes[col] == code, col] = float("nan")
        print(f"  {col}: {n_coded} values coded as {code} replaced with NaN")

# ── Step 3b: Drop rows missing the vision outcome ─────────────────────────────
print("\n" + "=" * 60)
print("Step 3b — Dropping rows with no missing/partial vision outcome (VIDROVA or VIDLOVA are NaN)")
print("=" * 60)

before = len(nhanes)
either_missing = nhanes["VIDROVA"].isna() | nhanes["VIDLOVA"].isna()
nhanes = nhanes[~either_missing].copy()
after = len(nhanes)
print(f"  Removed {before - after} rows ({100*(before-after)/before:.1f}% of sample)")
print(f"  Shape after row drop: {nhanes.shape}")

# ── Step 3c: Cap BPXDI1 values ──────────────────────────────────────
# BPXDI1: diastolic BP of ~0 is a NHANES coding convention for unmeasurable
# diastolic (certain arrhythmias). Zero is physiologically impossible so recode
# to NaN so KNN treats these as missing rather than as a legitimate low reading.
if "BPXDI1" in nhanes.columns:
    zero_mask = nhanes["BPXDI1"] < 1
    n_zeros = zero_mask.sum()
    nhanes.loc[zero_mask, "BPXDI1"] = float("nan")
    print(f"  BPXDI1: {n_zeros} zero values recoded to NaN")
else:
    print("  BPXDI1 not found, skipping.")

# ── Step 3d: Filter out invalid RIDAGEYR values ──────────────────────────────
# RIDAGEYR: Age values less than 1 are invalid (physiologically impossible)
# Recode to NaN for proper missing data handling
if "RIDAGEYR" in nhanes.columns:
    invalid_age_mask = nhanes["RIDAGEYR"] < 1
    n_invalid = invalid_age_mask.sum()
    nhanes.loc[invalid_age_mask, "RIDAGEYR"] = float("nan")
    print(f"  RIDAGEYR: {n_invalid} values < 1 recoded to NaN")
else:
    print("  RIDAGEYR not found, skipping.")

print("\n" + "=" * 60)
print("Step 4 — Feature engineering")
print("=" * 60)
# Engineered Featured #1: Average visual acuity (mean of non-null eyes per row)
if "VIDROVA" in nhanes.columns and "VIDLOVA" in nhanes.columns:
    nhanes["AVG_VISUAL_ACUITY"] = nhanes[["VIDROVA", "VIDLOVA"]].mean(axis=1, skipna=True)
    print(f"  AVG_VISUAL_ACUITY: {nhanes['AVG_VISUAL_ACUITY'].notna().sum()} non-null values")
else:
    print("  [WARN] VIDROVA or VIDLOVA missing — AVG_VISUAL_ACUITY not created")

# Engineered Feature #2: Diabetes duration in years  =  current_age  -  age_at_diagnosis (DID040)
# current_age  : RIDAGEYR (primary), DID040Q fallback for cycles where RIDAGEYR is NaN)
# age_at_dx    : DID040
# NaN for non-diabetics is expected and correct.
# Engineered Feature #2: Diabetes duration in years
if "DID040" in nhanes.columns:

    print("\n--- Computing DIABETES_DURATION_YRS ---")

    # Step 1: Initialize column to NaN for all rows
    nhanes["DIABETES_DURATION_YRS"] = float("nan")
    print("  Initialized DIABETES_DURATION_YRS to NaN for all rows")

    # Step 2: Identify diabetics (DIQ010 == 1) and "non"-diabetics (DIQ010 != 1)
    if "DIQ010" in nhanes.columns:
        diabetic_mask = nhanes["DIQ010"] == 1
        non_diabetic_mask = nhanes["DIQ010"] != 1
        num_diabetics = diabetic_mask.sum()
        num_non_diabetics = non_diabetic_mask.sum()
        print(f"  Found {num_diabetics} diabetic rows (DIQ010 == 1)")
        print(f"  Found {num_non_diabetics} non-diabetic rows (DIQ010 != 1)")
        nhanes.loc[non_diabetic_mask, "DIABETES_DURATION_YRS"] = 0
    else:
        print("  [WARN] DIQ010 not found — cannot compute duration conditionally")
        diabetic_mask = None

    # Step 3: Find current age ONLY for diabetics
    if diabetic_mask is not None and num_diabetics > 0:

        # Start with DID040Q if available
        if "DID040Q" in nhanes.columns:
            current_age = nhanes["DID040Q"].copy()
            print("  Using DID040Q as primary current age")

            # Fill missing DID040Q with RIDAGEYR if available
            if "RIDAGEYR" in nhanes.columns:
                missing_before = current_age.isna().sum()
                current_age = current_age.fillna(nhanes["RIDAGEYR"])
                missing_after = current_age.isna().sum()
                print(f"  Filled {missing_before - missing_after} missing DID040Q values using RIDAGEYR")
            else:
                print("  RIDAGEYR not available for fallback")

        elif "RIDAGEYR" in nhanes.columns:
            current_age = nhanes["RIDAGEYR"].copy()
            print("  DID040Q missing — using RIDAGEYR as current age")

        else:
            current_age = None
            print("  [WARN] No age variable available (DID040Q or RIDAGEYR)")

        # Step 4: Compute duration ONLY for diabetics
        if current_age is not None:
            computed_duration = current_age - nhanes["DID040"]

            # Clamp negatives to NaN
            negatives = (computed_duration < 0).sum()
            computed_duration[computed_duration < 0] = float("nan")
            print(f"  Clamped {negatives} negative duration values to NaN")

            # Assign ONLY to diabetics
            nhanes.loc[diabetic_mask, "DIABETES_DURATION_YRS"] = computed_duration[diabetic_mask]

            assigned = nhanes.loc[diabetic_mask, "DIABETES_DURATION_YRS"].notna().sum()
            print(f"  Successfully computed diabetic duration for {assigned} diabetic rows")

        else:
            print("  [WARN] Could not compute duration due to missing age variables")

    # Step 5: Summary stats
    non_null = nhanes["DIABETES_DURATION_YRS"].notna().sum()
    zeros = (nhanes["DIABETES_DURATION_YRS"] == 0).sum()
    print(f"  Final: {non_null} non-null values ({100 * non_null / len(nhanes):.1f}%)")
    print(f"  Final: {zeros} zero values (expected for non-diabetics)")

    # Step 6: Drop unused columns
    nhanes.drop(columns=["DID040", "DID040Q"], inplace=True, errors="ignore")
    print("  Dropped DID040 and DID040Q")

else:
    print("  [WARN] DID040 not found — DIABETES_DURATION_YRS cannot be computed")
    nhanes["DIABETES_DURATION_YRS"] = float("nan")

# ── Step 5: Summary and output ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Step 5 — Final summary")
print("=" * 60)

print(f"\n  Final shape: {nhanes.shape}")
print(f"\n  Columns ({len(nhanes.columns)}):")
print(f"  {nhanes.columns.tolist()}")

print("\n  Missing value rates (%):")
missing = (nhanes.isna().mean() * 100).round(1).sort_values(ascending=False)
print(missing.to_string())

print("\n  Cycle distribution:")
print(nhanes["CYCLE"].value_counts().sort_index().to_string())

print("\n" + "=" * 60)
print("HEAD (first 10 rows)")
print("=" * 60)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)
print(nhanes.head(10).to_string(index=False))

# Save to CSV
nhanes.to_csv(OUTPUT_CSV, index=False)
print(f"\n  Saved to: {OUTPUT_CSV}")