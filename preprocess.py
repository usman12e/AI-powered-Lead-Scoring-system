# preprocess.py
# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE:  Raw Data → Cleaning → Transformation → Feature Engineering
#            → Splitting → (Model Training in train.py)
#
# WHY WE REMOVED "Receive More Updates About Our Courses":
#   DATA LEAK. Every single "Yes" row in the dataset is a converter (100%).
#   Zero "Yes" rows are non-converters. This means the column was recorded
#   AFTER the conversion happened — existing customers were asked if they
#   want updates, and of course they said yes. The feature does not predict
#   conversion; it IS conversion being recorded a different way.
#   Training on it is giving the model the answer sheet. The model assigns
#   nearly all weight to this column, making every other signal irrelevant.
#   Removed → model now learns real behavioral patterns.
#
# FILES SAVED TO DISK:
#   models/scaler.pkl       → StandardScaler fitted on 6 original features
#   models/encoders.pkl     → LabelEncoders for 3 categorical columns
#   models/eng_params.pkl   → mean/std of the 2 engineered features
#   data/processed_data.csv → full 8-feature scaled dataset (6 + 2 engineered)
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

# 6 features — "Receive More Updates" removed (data leak)
FEATURES = [
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
    'Lead Source',
    'Last Activity',
    'What is your current occupation',
]
TARGET = 'Converted'

CATEGORICAL = [
    'Lead Source',
    'Last Activity',
    'What is your current occupation',
]
NUMERICAL = [
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
]

# Final column names for saved CSV (6 original + 2 engineered)
ALL_FEATURES = FEATURES + ['Engagement_Score', 'Pages_per_Minute']


def step(n, total, name):
    print(f"\n{'─'*55}")
    print(f"  PIPELINE STEP {n}/{total}:  {name}")
    print(f"{'─'*55}")

def ok(msg):   print(f"    ✓  {msg}")
def info(msg): print(f"    →  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# RAW DATA
# ─────────────────────────────────────────────────────────────────────────────
def load_raw():
    print("\n" + "═"*55)
    print("   RAW DATA → MODEL PIPELINE")
    print("═"*55)
    print("\n  ┌─────────────────────────────┐")
    print("  │         RAW DATA            │")
    print("  └─────────────────────────────┘")

    df = pd.read_csv('Raw/Leads X Education.csv')
    df = df[FEATURES + [TARGET]]
    info(f"Loaded {len(df)} rows, {len(FEATURES)} features + 1 target")
    info(f"'Receive More Updates' EXCLUDED  (data leak — 100% Yes = Converted)")

    print("\n  RAW DATA ISSUES FOUND:")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        for col, cnt in missing.items():
            info(f"Missing values   │ {col}: {cnt} missing ({cnt/len(df)*100:.1f}%)")
    else:
        info("Missing values   │ None found")

    for col in NUMERICAL:
        non_num = df[col].apply(
            lambda x: not pd.isna(x) and
            not str(x).replace('.','').replace('-','').isdigit()
        ).sum()
        if non_num > 0:
            info(f"Wrong format     │ {col}: {non_num} non-numeric entries")

    for col in NUMERICAL:
        neg = (pd.to_numeric(df[col], errors='coerce') < 0).sum()
        if neg > 0:
            info(f"Impossible value │ {col}: {neg} negative values")

    for col in NUMERICAL:
        numeric  = pd.to_numeric(df[col], errors='coerce').dropna()
        Q1, Q3   = numeric.quantile(0.25), numeric.quantile(0.75)
        outliers = ((numeric < Q1 - 3*(Q3-Q1)) | (numeric > Q3 + 3*(Q3-Q1))).sum()
        if outliers > 0:
            info(f"Outlier          │ {col}: {outliers} extreme values (IQR×3)")

    for col in CATEGORICAL:
        raw = df[col].dropna().astype(str)
        if raw.nunique() != raw.str.strip().str.lower().nunique():
            info(f"Inconsistent     │ {col}: case/whitespace mismatch")

    dups = df.duplicated().sum()
    if dups > 0: info(f"Duplicates       │ {dups} exact duplicate rows")
    else:        info("Duplicates       │ None found")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def clean(df):
    step(1, 4, "CLEANING")
    print("  (wrong formats · impossible values · outliers · labels · missing · duplicates)\n")
    before = len(df)

    # Wrong formats
    for col in NUMERICAL:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    ok("Wrong formats fixed   → non-numeric strings in numeric columns → NaN")

    # Impossible values
    for col in NUMERICAL:
        neg = df[col] < 0
        if neg.any():
            df.loc[neg, col] = np.nan
            ok(f"Impossible values     → {neg.sum()} negative {col} set to NaN")
        else:
            ok(f"Impossible values     → no negatives in {col}")

    # Outliers (IQR × 3 — cap, don't delete)
    for col in NUMERICAL:
        Q1, Q3  = df[col].quantile(0.25), df[col].quantile(0.75)
        upper   = Q3 + 3 * (Q3 - Q1)
        extreme = (df[col] > upper).sum()
        if extreme > 0:
            df[col] = df[col].clip(upper=upper)
            ok(f"Outliers capped       → {col}: {extreme} values capped at {upper:.1f}")
        else:
            ok(f"Outliers              → {col}: none beyond IQR×3 threshold")

    # Inconsistent labels
    for col in CATEGORICAL:
        before_u = df[col].dropna().astype(str).nunique()
        df[col]  = df[col].astype(str).str.strip().str.title()
        df[col]  = df[col].replace('Nan', np.nan)
        after_u  = df[col].dropna().nunique()
        ok(f"Labels normalised     → {col}: {before_u} → {after_u} unique values")

    # Missing values
    for col in NUMERICAL:
        n = df[col].isna().sum()
        if n > 0:
            m = df[col].median()
            df[col] = df[col].fillna(m)
            ok(f"Missing filled        → {col}: {n} NaN → median = {m:.2f}")
        else:
            ok(f"Missing values        → {col}: none")

    for col in CATEGORICAL:
        n = df[col].isna().sum()
        if n > 0:
            m = df[col].mode()[0]
            df[col] = df[col].fillna(m)
            ok(f"Missing filled        → {col}: {n} NaN → mode = '{m}'")
        else:
            ok(f"Missing values        → {col}: none")

    # Duplicates
    dups = df.duplicated().sum()
    df   = df.drop_duplicates()
    ok(f"Duplicates removed    → {dups} exact duplicate rows dropped")

    print(f"\n    Dataset: {before} rows → {len(df)} rows after cleaning")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — TRANSFORMATION
# ─────────────────────────────────────────────────────────────────────────────
def transform(df):
    step(2, 4, "TRANSFORMATION")
    print("  (label encoding · standard scaling)\n")

    encoders = {}
    for col in CATEGORICAL:
        le        = LabelEncoder()
        df[col]   = le.fit_transform(df[col].astype(str))
        encoders[col] = le
        ok(f"Label encoded         → {col}: {len(le.classes_)} categories → 0..{len(le.classes_)-1}")

    X = df[FEATURES].values
    y = df[TARGET].values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for i, col in enumerate(NUMERICAL):
        ok(f"Standard scaled       → {col}: "
           f"[{df[FEATURES].values[:,i].min():.1f}, {df[FEATURES].values[:,i].max():.1f}]"
           f" → [{X_scaled[:,i].min():.2f}, {X_scaled[:,i].max():.2f}]")

    ok("Normalisation         → all features: mean≈0, std≈1")

    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler,   'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')
    ok("Saved                 → models/scaler.pkl")
    ok("Saved                 → models/encoders.pkl")

    df_raw_numerical = df[NUMERICAL].copy()
    return X_scaled, y, df_raw_numerical


# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
def feature_engineer(X_scaled, df_raw):
    step(3, 4, "FEATURE ENGINEERING")
    print("  (new derived variables)\n")

    visits = df_raw['TotalVisits'].values
    time   = df_raw['Total Time Spent on Website'].values
    pages  = df_raw['Page Views Per Visit'].values

    # Engagement Score = visits × time
    engagement        = visits * time
    eng_mean          = engagement.mean()
    eng_std           = engagement.std() + 1e-8
    engagement_scaled = (engagement - eng_mean) / eng_std

    ok(f"Engagement Score      = TotalVisits × Time Spent")
    ok(f"                        raw [{engagement.min():.0f}, {engagement.max():.0f}]"
       f" → scaled [{engagement_scaled.min():.2f}, {engagement_scaled.max():.2f}]")

    # Pages-per-Minute = pages / (time + 1)
    ratio        = pages / (time + 1)
    ratio_mean   = ratio.mean()
    ratio_std    = ratio.std() + 1e-8
    ratio_scaled = (ratio - ratio_mean) / ratio_std

    ok(f"Pages-per-Minute      = Page Views / (Time + 1)")
    ok(f"                        raw [{ratio.min():.4f}, {ratio.max():.4f}]"
       f" → scaled [{ratio_scaled.min():.2f}, {ratio_scaled.max():.2f}]")

    eng_params = {
        'engagement_mean': eng_mean,
        'engagement_std':  eng_std,
        'ratio_mean':      ratio_mean,
        'ratio_std':       ratio_std,
    }
    joblib.dump(eng_params, 'models/eng_params.pkl')
    ok("Saved                 → models/eng_params.pkl")

    X_final = np.column_stack([X_scaled, engagement_scaled, ratio_scaled])
    ok(f"Feature matrix        → {X_scaled.shape[1]} original + 2 derived = {X_final.shape[1]} total")

    return X_final


# ─────────────────────────────────────────────────────────────────────────────
# SAVE PROCESSED DATASET
# ─────────────────────────────────────────────────────────────────────────────
def save_processed_dataset(X_final, y):
    df_out        = pd.DataFrame(X_final, columns=ALL_FEATURES)
    df_out[TARGET] = y
    os.makedirs('data', exist_ok=True)
    df_out.to_csv('data/processed_data.csv', index=False)
    ok(f"Processed dataset     → data/processed_data.csv")
    ok(f"                        {len(df_out)} rows × {len(ALL_FEATURES)} features + target")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — SPLITTING
# ─────────────────────────────────────────────────────────────────────────────
def split(X, y):
    step(4, 4, "SPLITTING")
    print("  (80% train / 20% test  —  stratified)\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    ok(f"Train set             → {len(X_train)} rows (80%)")
    ok(f"Test set              → {len(X_test)} rows (20%)")
    ok(f"Stratified            → conversion rate preserved (~{y.mean()*100:.1f}%)")
    ok(f"Random seed           → 42 (reproducible)")

    print(f"\n  ┌──────────────────────────────────────────────────┐")
    print(f"  │  Steps 1–4 complete. Handing off → STEP 5/5      │")
    print(f"  └──────────────────────────────────────────────────┘")

    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def get_data():
    df                      = load_raw()
    df                      = clean(df)
    X_scaled, y, df_raw_num = transform(df)
    X_final                 = feature_engineer(X_scaled, df_raw_num)

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │   SAVING PROCESSED DATASET  │")
    print(f"  └─────────────────────────────┘")
    save_processed_dataset(X_final, y)

    X_train, X_test, y_train, y_test = split(X_final, y)
    return X_train, X_test, y_train, y_test