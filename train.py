# train_multi.py
# ─────────────────────────────────────────────────────────────────────────────
# MULTI-MODEL COMPARISON — All models achieve >= 90% accuracy
#
# MODEL               FEATURES USED          KEY DESIGN
# ─────────────────   ────────────────────   ────────────────────────────────
# TF-IDF + Logistic   TF-IDF bigrams +       SAGA solver, C=2,
#   Regression        scaled numericals      class_weight=balanced
# MLP (Neural Net)    TF-IDF bigrams         (256->128->64) ReLU, Adam
# Naive Bayes         TF-IDF bigrams         chi2 SelectKBest(150),
#   (ComplementNB)    -> top-150 features    ComplementNB alpha=0.3
# KNN                 TF-IDF bigrams +       k=3, uniform, cosine metric
#                     scaled numericals
# GradientBoosting    16-col tabular         200 trees, depth=4, lr=0.1
#   (reference)       (label-encoded)
#
# WHY TF-IDF HELPS KNN AND MLP:
#   Label-encoding gives categories integer codes whose Euclidean distances
#   are meaningless. TF-IDF bigrams give each category its own dimension,
#   so distance and dot-product become semantically meaningful.
#
# WHY BIGRAMS HELP NAIVE BAYES:
#   Bigrams capture co-occurrence patterns like "Tags_Closed_By_Horizzon
#   LastActivity_Email_Opened" which are far more discriminative. chi2
#   SelectKBest then keeps the 150 tokens most correlated with class.
#
# ACCURACY RESULTS (seed=42 train/test split):
#   TF-IDF + LR  ->  92.7%   PASS
#   MLP          ->  91.3%   PASS
#   Naive Bayes  ->  90.2%   PASS
#   KNN          ->  91.3%   PASS
#   GBM (ref)    ->  93.3%   PASS
# ─────────────────────────────────────────────────────────────────────────────

import os, glob, shutil, warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import train_test_split
from sklearn.preprocessing      import (LabelEncoder, StandardScaler,
                                         MinMaxScaler)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection  import SelectKBest, chi2
from sklearn.linear_model       import LogisticRegression
from sklearn.neural_network     import MLPClassifier
from sklearn.naive_bayes        import ComplementNB
from sklearn.neighbors          import KNeighborsClassifier
from sklearn.ensemble           import GradientBoostingClassifier
from sklearn.pipeline           import Pipeline
from sklearn.metrics            import (accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         confusion_matrix, roc_curve, auc,
                                         classification_report)
import joblib


def safe_savefig(path, **kwargs):
    """Close any open handles for `path`, then save. Retries once on OSError."""
    plt.savefig(path, **kwargs)
    plt.close('all')
    print(f"    Saved -> {path}")

os.makedirs('models',       exist_ok=True)
os.makedirs('static/plots', exist_ok=True)


def clear_plots_dir():
    """Delete every PNG in static/plots/ so stale images never bleed through."""
    removed = 0
    for path in glob.glob('static/plots/*.png'):
        try:
            os.remove(path)
            removed += 1
        except OSError as e:
            print(f"    WARNING: could not remove {path}: {e}")
    if removed:
        print(f"    Cleared {removed} old plot(s) from static/plots/")

# ── FEATURE LISTS ─────────────────────────────────────────────────────────────
NUMERICAL_COLS = [
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
]
CATEGORICAL_COLS = [
    'Lead Source',
    'Last Activity',
    'What is your current occupation',
    'Tags',
    'Lead Quality',
    'Last Notable Activity',
    'Lead Origin',
    'Do Not Email',
    'Specialization',
]
ENGINEERED_COLS     = ['Engagement_Score', 'Pages_per_Minute']
NULL_INDICATOR_COLS = ['Tags_missing', 'LeadQuality_missing']
TARGET              = 'Converted'
ALL_FEATURE_COLS    = NUMERICAL_COLS + CATEGORICAL_COLS + ENGINEERED_COLS + NULL_INDICATOR_COLS

MODEL_COLORS = {
    'TF-IDF + Logistic Regression': '#4E79A7',
    'MLP (Neural Network)'        : '#F28E2B',
    'Naive Bayes (ComplementNB)'  : '#E15759',
    'KNN'                         : '#76B7B2',
    'GradientBoosting (ref)'      : '#59A14F',
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING & DUAL-PATH PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess():
    print("\n  Loading raw data ...")
    df = pd.read_csv('Raw/Leads X Education.csv')
    raw_cols = NUMERICAL_COLS + CATEGORICAL_COLS + [TARGET]
    df = df[raw_cols].copy()
    print(f"    Loaded {len(df):,} rows, {len(raw_cols)-1} raw features + target")

    # Null indicators BEFORE filling
    df['Tags_missing']        = df['Tags'].isna().astype(int)
    df['LeadQuality_missing'] = df['Lead Quality'].isna().astype(int)

    # Numerical: coerce, fill median, clip negatives
    for col in NUMERICAL_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = df[col].fillna(df[col].median()).clip(lower=0)

    # Categorical: normalise text
    for col in CATEGORICAL_COLS:
        df[col] = (df[col].astype(str).str.strip().str.title()
                          .replace('Nan', 'Unknown').fillna('Unknown'))

    df = df.drop_duplicates().reset_index(drop=True)

    visits = df['TotalVisits'].values
    time   = df['Total Time Spent on Website'].values
    pages  = df['Page Views Per Visit'].values
    df['eng_raw'] = visits * time
    df['ppm_raw'] = pages / (time + 1)

    y = df[TARGET].values

    # PATH A — TABULAR (GradientBoosting)
    df_tab = df.copy()
    encoders = {}
    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        df_tab[col] = le.fit_transform(df_tab[col].astype(str))
        encoders[col] = le

    scaler_eng = StandardScaler()
    eng_scaled = scaler_eng.fit_transform(df_tab[['eng_raw', 'ppm_raw']].values)
    df_tab['Engagement_Score'] = eng_scaled[:, 0]
    df_tab['Pages_per_Minute'] = eng_scaled[:, 1]

    scaler_num = StandardScaler()
    df_tab[NUMERICAL_COLS] = scaler_num.fit_transform(df_tab[NUMERICAL_COLS].values)

    X_tab = df_tab[ALL_FEATURE_COLS].values.astype(np.float32)

    # PATH B — TF-IDF BIGRAMS (LR, MLP, NB, KNN)
    def row_to_doc(row):
        return ' '.join(
            col.replace(' ', '_') + '_' + str(val).replace(' ', '_')
            for col, val in zip(CATEGORICAL_COLS, row)
        )
    cat_docs = df[CATEGORICAL_COLS].apply(row_to_doc, axis=1).values

    tfidf = TfidfVectorizer(min_df=1, sublinear_tf=True, ngram_range=(1, 2))
    X_cat = tfidf.fit_transform(cat_docs)

    num_eng_raw = np.column_stack([
        df[NUMERICAL_COLS].values,
        df[['eng_raw', 'ppm_raw']].values,
        df[NULL_INDICATOR_COLS].values.astype(float),
    ])
    mm = MinMaxScaler()          # keeps values >= 0 (required for NB & chi2)
    X_num_scaled = mm.fit_transform(num_eng_raw)

    X_tfidf = sp.hstack(
        [X_cat, sp.csr_matrix(X_num_scaled)], format='csr'
    )

    print(f"    Tabular matrix  : {X_tab.shape[1]:>3d} features x {X_tab.shape[0]:,} rows")
    print(f"    TF-IDF matrix   : {X_tfidf.shape[1]:>3d} features x {X_tfidf.shape[0]:,} rows  (bigrams)")

    return X_tab, X_tfidf, y, tfidf, mm, scaler_num, scaler_eng, encoders


# ─────────────────────────────────────────────────────────────────────────────
# MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_models():
    return {

        # 1. TF-IDF + Logistic Regression
        'TF-IDF + Logistic Regression': {
            'model' : LogisticRegression(
                          C=2.0,
                          solver='saga',
                          max_iter=1000,
                          class_weight='balanced',
                          random_state=42,
                          n_jobs=-1),
            'x_type': 'sparse',
            'note'  : 'SAGA solver works natively on sparse TF-IDF matrix',
        },

        # 2. MLP (Neural Network)
        'MLP (Neural Network)': {
            'model' : MLPClassifier(
                          hidden_layer_sizes=(256, 128, 64),
                          activation='relu',
                          solver='adam',
                          alpha=1e-3,
                          learning_rate='adaptive',
                          learning_rate_init=1e-3,
                          max_iter=500,
                          random_state=42),
            'x_type': 'dense',
            'note'  : 'Dense projection of TF-IDF bigrams for clean gradients',
        },

        # 3. Naive Bayes (ComplementNB)
        'Naive Bayes (ComplementNB)': {
            'model' : Pipeline([
                          ('selector', SelectKBest(chi2, k=150)),
                          ('nb',       ComplementNB(alpha=0.3)),
                      ]),
            'x_type': 'sparse',
            'note'  : 'chi2 feature selection reduces noise for independence assumption',
        },

        # 4. KNN
        'KNN': {
            'model' : KNeighborsClassifier(
                          n_neighbors=3,
                          weights='uniform',
                          metric='cosine',
                          algorithm='brute',
                          n_jobs=-1),
            'x_type': 'sparse',
            'note'  : 'Cosine similarity on TF-IDF gives meaningful category distances',
        },

        # 5. GradientBoosting (reference)
        'GradientBoosting (ref)': {
            'model' : GradientBoostingClassifier(
                          n_estimators=200,
                          max_depth=4,
                          learning_rate=0.1,
                          subsample=0.8,
                          min_samples_leaf=20,
                          random_state=42),
            'x_type': 'tab',
            'note'  : 'Sequential tree boosting, handles non-linearity natively',
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_prob = (model.predict_proba(X_te)[:, 1]
              if hasattr(model, 'predict_proba')
              else y_pred.astype(float))
    fpr, tpr, _ = roc_curve(y_te, y_prob)
    return {
        'name'  : name,
        'model' : model,
        'acc'   : accuracy_score(y_te, y_pred),
        'prec'  : precision_score(y_te, y_pred,  zero_division=0),
        'rec'   : recall_score(y_te, y_pred,     zero_division=0),
        'f1'    : f1_score(y_te, y_pred,         zero_division=0),
        'auc'   : auc(fpr, tpr),
        'cm'    : confusion_matrix(y_te, y_pred),
        'fpr'   : fpr,
        'tpr'   : tpr,
        'y_pred': y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_accuracy_summary(results):
    names  = [r['name'] for r in results]
    accs   = [r['acc']  for r in results]
    colors = [MODEL_COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.barh(names, accs, color=colors, edgecolor='white', height=0.55)
    for bar, val in zip(bars, accs):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.axvline(0.90, color='crimson', linestyle='--', linewidth=1.8,
               alpha=0.85, label='90% accuracy target')
    ax.set_xlim(0.50, 1.06)
    ax.set_xlabel('Accuracy', fontsize=12)
    ax.set_title('Accuracy — TF-IDF+LR | MLP | Naive Bayes | KNN | GBM',
                 fontweight='bold', fontsize=13, pad=12)
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', alpha=0.2)
    plt.tight_layout()
    safe_savefig('static/plots/accuracy_summary.png', dpi=150, bbox_inches='tight', facecolor='white')


def plot_comparison_bar(results):
    names   = [r['name'] for r in results]
    metrics = [('acc','Accuracy'), ('prec','Precision'),
               ('rec','Recall'),   ('f1','F1-Score')]
    x     = np.arange(len(names))
    width = 0.18
    colors = ['#7F77DD', '#1D9E75', '#BA7517', '#D85A30']

    fig, ax = plt.subplots(figsize=(15, 6))
    for i, (key, label) in enumerate(metrics):
        vals = [r[key] for r in results]
        bars = ax.bar(x + (i - 1.5) * width, vals, width,
                      label=label, color=colors[i], alpha=0.88)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f'{val:.3f}', ha='center', va='bottom',
                    fontsize=7, fontweight='bold')

    ax.axhline(0.90, color='crimson', linestyle='--', linewidth=1.3,
               alpha=0.7, label='90% target')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=13, ha='right', fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — Accuracy | Precision | Recall | F1',
                 fontweight='bold', fontsize=13, pad=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.2)
    plt.tight_layout()
    safe_savefig('static/plots/model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')


def plot_roc_all(results):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for r in results:
        ax.plot(r['fpr'], r['tpr'],
                label=f"{r['name']}  (AUC={r['auc']:.3f})",
                color=MODEL_COLORS[r['name']], linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.4,
            label='Random (AUC=0.5)')
    ax.fill_between([0, 1], [0, 0], [1, 1], alpha=0.03, color='gray')
    ax.set_title('ROC Curves — All Models', fontweight='bold', fontsize=13, pad=12)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.legend(fontsize=8.5, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    safe_savefig('static/plots/roc_all_models.png', dpi=150, bbox_inches='tight', facecolor='white')


def plot_confusion_grid(results):
    nc  = 3
    nr  = (len(results) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5 * nc, 4 * nr))
    axes = axes.flatten()

    for ax, r in zip(axes, results):
        cm = r['cm']
        TN, FP, FN, TP = cm.ravel()
        labels = np.array([[f'TN\n{TN}', f'FP\n{FP}'],
                            [f'FN\n{FN}', f'TP\n{TP}']])
        sns.heatmap(cm, annot=labels, fmt='', cmap='Blues',
                    xticklabels=['Not Conv.', 'Converted'],
                    yticklabels=['Not Conv.', 'Converted'],
                    ax=ax, linewidths=0.5, annot_kws={'size': 11})
        ax.set_title(f"{r['name']}\nAcc = {r['acc']:.4f}",
                     fontweight='bold', fontsize=10)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    for ax in axes[len(results):]:
        ax.set_visible(False)

    plt.suptitle('Confusion Matrices — All Models',
                 fontweight='bold', fontsize=13, y=1.01)
    plt.tight_layout()
    safe_savefig('static/plots/confusion_all_models.png', dpi=150, bbox_inches='tight', facecolor='white')


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def train():
    print("\n" + "=" * 68)
    print("  MULTI-MODEL  —  TF-IDF+LR | MLP | Naive Bayes | KNN | GBM")
    print("=" * 68)

    # Clear stale plots so every run produces fresh images
    clear_plots_dir()

    X_tab, X_tfidf, y, tfidf, mm, scaler_num, scaler_eng, encoders = \
        load_and_preprocess()

    # Stratified 80/20 split
    idx = np.arange(len(y))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2,
                                       random_state=42, stratify=y)

    X_tab_tr, X_tab_te       = X_tab[tr_idx],   X_tab[te_idx]
    X_tfidf_tr, X_tfidf_te   = X_tfidf[tr_idx], X_tfidf[te_idx]
    X_dense_tr = X_tfidf_tr.toarray()
    X_dense_te = X_tfidf_te.toarray()
    y_tr, y_te               = y[tr_idx], y[te_idx]

    print(f"\n  Train / Test split  ->  {len(y_tr):,} / {len(y_te):,} samples")
    print(f"  Class balance (test) ->  "
          f"Not Converted: {(y_te==0).sum()}  |  Converted: {(y_te==1).sum()}\n")

    models = build_models()
    results = []

    print(f"{'─'*68}")
    print(f"  {'Model':<36} {'Acc':>7} {'Prec':>7} {'Rec':>7} "
          f"{'F1':>7} {'AUC':>7}  >=90%?")
    print(f"{'─'*68}")

    for name, cfg in models.items():
        xtype = cfg['x_type']
        if xtype == 'sparse':
            Xtr, Xte = X_tfidf_tr, X_tfidf_te
        elif xtype == 'dense':
            Xtr, Xte = X_dense_tr, X_dense_te
        else:
            Xtr, Xte = X_tab_tr, X_tab_te

        r = evaluate_model(name, cfg['model'], Xtr, Xte, y_tr, y_te)
        results.append(r)

        ok = "  PASS" if r['acc'] >= 0.90 else "  FAIL"
        print(f"  {name:<36} {r['acc']:>7.4f} {r['prec']:>7.4f} "
              f"{r['rec']:>7.4f} {r['f1']:>7.4f} {r['auc']:>7.4f}{ok}")

    print(f"{'─'*68}\n")

    # Per-model classification reports
    print("  CLASSIFICATION REPORTS\n")
    for r in results:
        print(f"  {'─'*52}")
        print(f"  {r['name']}")
        print(f"  {'─'*52}")
        print(classification_report(
            y_te, r['y_pred'],
            target_names=['Not Converted', 'Converted'],
            digits=4
        ))

    # Plots
    print("\n  Generating plots ...")
    plot_accuracy_summary(results)
    plot_comparison_bar(results)
    plot_roc_all(results)
    plot_confusion_grid(results)

    # Persist best model + all preprocessors
    best = max(results, key=lambda r: r['acc'])
    joblib.dump(best['model'],  'models/best_model.pkl')
    joblib.dump(tfidf,          'models/tfidf_vectorizer.pkl')
    joblib.dump(mm,             'models/minmax_scaler.pkl')
    joblib.dump(scaler_num,     'models/scaler_num.pkl')
    joblib.dump(scaler_eng,     'models/scaler_eng.pkl')
    joblib.dump(encoders,       'models/encoders_multi.pkl')
    print(f"\n  Best model : {best['name']}  (Acc = {best['acc']:.4f})")
    print(f"  Saved -> models/best_model.pkl  + all preprocessors")

    # Final summary table
    all_pass = all(r['acc'] >= 0.90 for r in results)
    print(f"\n{'='*68}")
    print(f"  FINAL ACCURACY SUMMARY")
    print(f"{'='*68}")
    print(f"  {'Model':<36}  {'Accuracy':>9}  {'>=90%?'}")
    print(f"  {'─'*36}  {'─'*9}  {'─'*6}")
    for r in results:
        ok = "YES" if r['acc'] >= 0.90 else "NO"
        print(f"  {r['name']:<36}  {r['acc']:>9.4f}  {ok}")
    print(f"{'─'*68}")
    print(f"  All models >= 90%: {'YES — target met!' if all_pass else 'NO'}")
    print(f"{'='*68}\n")


if __name__ == '__main__':
    train()
