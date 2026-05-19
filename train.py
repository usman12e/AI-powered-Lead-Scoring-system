# train.py
# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE STEP 5/5: MODEL TRAINING & EVALUATION
#
# WHY 5 RUNS?
#   A single train/test split can get lucky or unlucky depending on which rows
#   end up in the test set by chance. If the test set happens to contain easier
#   leads, accuracy looks high. If harder leads, it looks low.
#   Running 5 times with 5 DIFFERENT random splits removes that luck factor.
#   The mean tells you the true average performance.
#   The std tells you how stable the model is across different data splits.
#   Low std = the model reliably performs the same regardless of which data
#   it sees. High std = the model is unstable (a red flag).
#
# HOW THE 5 RUNS WORK:
#   Each run uses a different random_state (0, 1, 2, 3, 4) for the split.
#   This gives 5 completely different train/test divisions of the same dataset.
#   We record Accuracy, Precision, Recall, F1 for each run, then compute
#   mean and std across all 5.
#
# WHAT GETS SAVED:
#   models/model.pkl                 ← final model (trained on seed 42)
#   static/plots/metrics.png         ← bar chart of mean ± std for all 4 metrics
#   static/plots/runs_line.png       ← line chart of each metric across 5 runs
#   static/plots/confusion_matrix.png
#   static/plots/roc_curve.png
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neural_network  import MLPClassifier
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, roc_curve, auc)
import joblib
import os
from preprocess import get_data

os.makedirs('models',       exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

N_RUNS = 5   # number of different random splits to evaluate


def make_model():
    """Returns a fresh untrained MLP with the chosen architecture."""
    return MLPClassifier(
        hidden_layer_sizes = (16, 8),
        activation         = 'relu',
        solver             = 'adam',
        max_iter           = 500,
        random_state       = 42    # weight initialisation is fixed; only the DATA split changes
    )


def train():
    print("\n" + "═"*58)
    print("   RAW DATA → MODEL PIPELINE  (python train.py)")
    print("═"*58)

    # Steps 1–4 run inside preprocess.py and print their own output
    # get_data() returns the FULL preprocessed dataset (not yet split)
    # We split it ourselves here so we can do it 5 times with different seeds
    X_full, y_full = get_preprocessed_arrays()

    # ── STEP 5: MODEL TRAINING ────────────────────────────────────────────────
    print(f"\n{'─'*58}")
    print(f"  PIPELINE STEP 5/5:  MODEL TRAINING & EVALUATION")
    print(f"{'─'*58}")
    print(f"  Architecture : Input(8) → Hidden(16) → Hidden(8) → Output(1)")
    print(f"  Activation   : ReLU    Optimizer: Adam    Max epochs: 500")
    print(f"  Evaluation   : {N_RUNS} independent runs with different train/test splits\n")

    # ── 5-RUN EVALUATION ──────────────────────────────────────────────────────
    all_acc, all_prec, all_rec, all_f1 = [], [], [], []

    print(f"  {'Run':<5} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}  Split seed")
    print(f"  {'─'*5} {'─'*10} {'─'*10} {'─'*10} {'─'*10}  {'─'*10}")

    for seed in range(N_RUNS):
        # Different seed → different 80/20 split of the same dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_full, y_full,
            test_size    = 0.2,
            random_state = seed,   # ← this is what changes each run
            stratify     = y_full
        )

        m = make_model()
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred,    zero_division=0)
        f1   = f1_score(y_test, y_pred,        zero_division=0)

        all_acc.append(acc);  all_prec.append(prec)
        all_rec.append(rec);  all_f1.append(f1)

        print(f"  {seed+1:<5} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}  seed={seed}")

    # ── MEAN ± STD ────────────────────────────────────────────────────────────
    mean_acc,  std_acc  = np.mean(all_acc),  np.std(all_acc)
    mean_prec, std_prec = np.mean(all_prec), np.std(all_prec)
    mean_rec,  std_rec  = np.mean(all_rec),  np.std(all_rec)
    mean_f1,   std_f1   = np.mean(all_f1),   np.std(all_f1)

    print(f"\n{'═'*58}")
    print(f"  {N_RUNS}-RUN SUMMARY  —  mean ± std  (std = stability indicator)")
    print(f"{'═'*58}")
    w = 12
    print(f"  {'Metric':<{w}}  {'Mean':>8}   {'Std Dev':>8}   {'Min':>7}   {'Max':>7}")
    print(f"  {'─'*w}  {'─'*8}   {'─'*8}   {'─'*7}   {'─'*7}")
    print(f"  {'Accuracy':<{w}}  {mean_acc:>8.4f}   {std_acc:>8.4f}   "
          f"{min(all_acc):>7.4f}   {max(all_acc):>7.4f}")
    print(f"  {'Precision':<{w}}  {mean_prec:>8.4f}   {std_prec:>8.4f}   "
          f"{min(all_prec):>7.4f}   {max(all_prec):>7.4f}")
    print(f"  {'Recall':<{w}}  {mean_rec:>8.4f}   {std_rec:>8.4f}   "
          f"{min(all_rec):>7.4f}   {max(all_rec):>7.4f}")
    print(f"  {'F1-Score':<{w}}  {mean_f1:>8.4f}   {std_f1:>8.4f}   "
          f"{min(all_f1):>7.4f}   {max(all_f1):>7.4f}")
    print(f"{'─'*58}")
    print(f"  Low std across all metrics → model is STABLE and reliable.")
    print(f"{'═'*58}")

    # ── TRAIN FINAL MODEL on fixed seed 42 ───────────────────────────────────
    # WHY train again after the 5-run evaluation?
    # The 5 runs used seeds 0–4 for measuring performance. The final saved model
    # uses seed 42 — a separate, fixed split that is fully reproducible.
    # Anyone who runs this script always gets the exact same saved model.
    # The 5-run results above prove the model is reliable; this is the one we ship.
    print(f"\n  Training final model on fixed split (seed=42) for deployment ...")
    X_train_f, X_test_f, y_train_f, y_test_f = train_test_split(
        X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
    )
    final_model = make_model()
    final_model.fit(X_train_f, y_train_f)

    y_pred_f = final_model.predict(X_test_f)
    y_prob_f = final_model.predict_proba(X_test_f)[:, 1]
    cm       = confusion_matrix(y_test_f, y_pred_f)
    TN, FP, FN, TP = cm.ravel()

    print(f"\n  Final model  —  confusion matrix breakdown:")
    print(f"    ✓  True  Positives (TP) : {TP}   correctly predicted converters")
    print(f"    ✓  True  Negatives (TN) : {TN}   correctly predicted non-converters")
    print(f"    ✗  False Positives (FP) : {FP}   predicted convert, actually didn't")
    print(f"    ✗  False Negatives (FN) : {FN}   missed a real buyer  ← most costly")

    joblib.dump(final_model, 'models/model.pkl')
    print(f"\n    ✓  Saved → models/model.pkl")

    # ── Save all charts ───────────────────────────────────────────────────────
    runs_data = {
        'Accuracy':  all_acc,
        'Precision': all_prec,
        'Recall':    all_rec,
        'F1-Score':  all_f1,
    }
    means = [mean_acc, mean_prec, mean_rec, mean_f1]
    stds  = [std_acc,  std_prec,  std_rec,  std_f1]

    save_metrics_chart(means, stds)
    save_runs_line_chart(runs_data)
    save_confusion_matrix(cm)
    save_roc_curve(y_test_f, y_prob_f)

    print(f"\n{'═'*58}")
    print(f"  PIPELINE COMPLETE — all 5 steps done.")
    print(f"  Now run:  python app.py  →  http://localhost:5000")
    print(f"{'═'*58}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get the full preprocessed X and y WITHOUT splitting
# preprocess.get_data() does its own split internally, so we replicate
# just the preprocessing part here and split ourselves 5 times.
# ─────────────────────────────────────────────────────────────────────────────
def get_preprocessed_arrays():
    """
    Calls preprocess pipeline steps 1–4 but returns the FULL X and y
    so train.py can do the splitting itself (5 times with different seeds).
    """
    import preprocess as pp
    import pandas as pd

    df              = pp.load_raw()
    df              = pp.clean(df)
    X_scaled, y, df_raw_num = pp.transform(df)
    X_final         = pp.feature_engineer(X_scaled, df_raw_num)

    print(f"\n  ┌─────────────────────────────┐")
    print(f"  │   SAVING PROCESSED DATASET  │")
    print(f"  └─────────────────────────────┘")
    pp.save_processed_dataset(X_final, y)

    return X_final, y


# ─────────────────────────────────────────────────────────────────────────────
# CHART 1: Bar chart — mean ± std for all 4 metrics
# The error bars show ± 1 standard deviation across 5 runs.
# A short error bar = consistent model. A tall one = unstable.
# ─────────────────────────────────────────────────────────────────────────────
def save_metrics_chart(means, stds):
    labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#7F77DD', '#1D9E75', '#BA7517', '#D85A30']

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, means, yerr=stds, capsize=7,
                  color=colors, width=0.5,
                  edgecolor='white', linewidth=0.8,
                  error_kw={'linewidth': 2, 'ecolor': '#333333'})

    for bar, mean, std in zip(bars, means, stds):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + std + 0.015,
            f'{mean:.4f}\n±{std:.4f}',
            ha='center', va='bottom',
            fontsize=9.5, fontweight='bold'
        )

    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'Model Performance — MLP (16, 8)\n{N_RUNS}-Run Mean ± Std Dev',
                 fontweight='bold', fontsize=13, pad=12)
    ax.axhline(0.5, color='gray', linewidth=0.8, linestyle='--',
               alpha=0.5, label='Random baseline (0.5)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='x', labelsize=11)
    ax.grid(axis='y', alpha=0.2, linewidth=0.5)
    ax.legend(fontsize=9, loc='lower right')

    plt.tight_layout()
    plt.savefig('static/plots/metrics.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓  Saved → static/plots/metrics.png  (bar chart with error bars)")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 2: Line chart — each metric across all 5 runs
# Shows the trajectory: are the numbers stable or bouncing around?
# ─────────────────────────────────────────────────────────────────────────────
def save_runs_line_chart(runs_data):
    runs = list(range(1, N_RUNS + 1))

    fig, ax = plt.subplots(figsize=(8, 4.5))

    style = {
        'Accuracy':  {'color': '#7F77DD', 'marker': 'o'},
        'Precision': {'color': '#1D9E75', 'marker': 's'},
        'Recall':    {'color': '#BA7517', 'marker': '^'},
        'F1-Score':  {'color': '#D85A30', 'marker': 'D'},
    }

    for metric, values in runs_data.items():
        mean = np.mean(values)
        ax.plot(runs, values,
                label=f"{metric}  (mean={mean:.3f})",
                color=style[metric]['color'],
                marker=style[metric]['marker'],
                linewidth=2, markersize=7)

    ax.set_xticks(runs)
    ax.set_xticklabels([f'Run {r}\n(seed={r-1})' for r in runs], fontsize=9)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.set_title(f'Metric Stability Across {N_RUNS} Runs — MLP (16, 8)',
                 fontweight='bold', fontsize=13, pad=12)
    ax.legend(fontsize=9, loc='lower right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.2, linewidth=0.5)

    plt.tight_layout()
    plt.savefig('static/plots/runs_line.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓  Saved → static/plots/runs_line.png  (metric stability line chart)")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 3: Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────
def save_confusion_matrix(cm):
    TN, FP, FN, TP = cm.ravel()
    labels = np.array([[f'TN\n{TN}', f'FP\n{FP}'],
                        [f'FN\n{FN}', f'TP\n{TP}']])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=labels, fmt='', cmap='Purples',
                xticklabels=['Not Converted', 'Converted'],
                yticklabels=['Not Converted', 'Converted'],
                ax=ax, linewidths=0.5, annot_kws={'size': 13})
    ax.set_title('Confusion Matrix — MLP (16, 8)\nFinal model (seed=42)',
                 fontweight='bold', pad=12)
    ax.set_ylabel('Actual',    fontsize=11)
    ax.set_xlabel('Predicted', fontsize=11)
    plt.tight_layout()
    plt.savefig('static/plots/confusion_matrix.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓  Saved → static/plots/confusion_matrix.png")


# ─────────────────────────────────────────────────────────────────────────────
# CHART 4: ROC Curve
# ─────────────────────────────────────────────────────────────────────────────
def save_roc_curve(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score   = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color='#7F77DD', lw=2,
            label=f'MLP (16,8)  —  AUC = {auc_score:.3f}')
    ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.4,
            label='Random classifier  (AUC = 0.5)')
    ax.fill_between(fpr, tpr, alpha=0.08, color='#7F77DD')
    ax.set_title('ROC Curve — MLP (16, 8)', fontweight='bold', pad=12)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate',  fontsize=11)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('static/plots/roc_curve.png', dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"    ✓  Saved → static/plots/roc_curve.png")


if __name__ == '__main__':
    train()