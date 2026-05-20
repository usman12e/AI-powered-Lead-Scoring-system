# feature_analysis.py
# ─────────────────────────────────────────────────────────────────────────────
# ONE JOB: Study each of the 7 features individually.
# For each feature, answer two questions:
#   1. How frequently does each value appear in the dataset?
#   2. What % of leads with that value actually converted?
#
# This tells you EXACTLY which feature values push a lead toward buying.
#
# Run:  python feature_analysis.py
# Saves: static/plots/feature_analysis.png
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

os.makedirs('static/plots', exist_ok=True)

FEATURES = [
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
    'Lead Source',
    'Last Activity',
    'What is your current occupation',
    'Receive More Updates About Our Courses',
]
TARGET = 'Converted'

CATEGORICAL = [
    'Lead Source',
    'Last Activity',
    'What is your current occupation',
    'Receive More Updates About Our Courses',
]
NUMERICAL = [
    'TotalVisits',
    'Total Time Spent on Website',
    'Page Views Per Visit',
]

# Colours
C_PURPLE  = '#7F77DD'
C_GREEN   = '#1D9E75'
C_ORANGE  = '#D85A30'
C_GRAY    = '#B4B2A9'
C_LIGHT   = '#EEEDFE'


def load():
    df = pd.read_csv('Leads X Education.csv')
    df = df[FEATURES + [TARGET]]
    for col in CATEGORICAL:
        df[col] = df[col].fillna('Unknown')
    df['TotalVisits']          = df['TotalVisits'].fillna(df['TotalVisits'].median())
    df['Page Views Per Visit'] = df['Page Views Per Visit'].fillna(df['Page Views Per Visit'].median())
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: Print a full text report to the terminal
# ─────────────────────────────────────────────────────────────────────────────
def print_report(df):
    total       = len(df)
    n_converted = df[TARGET].sum()
    overall_rate = n_converted / total * 100

    print("\n" + "=" * 65)
    print("   FEATURE IMPACT REPORT")
    print(f"   Dataset: {total} leads  |  Converted: {n_converted} ({overall_rate:.1f}%)")
    print("=" * 65)

    # ── Categorical features ──────────────────────────────────────────────────
    for col in CATEGORICAL:
        print(f"\n{'─'*65}")
        print(f"  FEATURE: {col}")
        print(f"{'─'*65}")
        print(f"  {'Value':<40} {'Count':>6}  {'Share':>6}  {'Conv%':>6}")
        print(f"  {'─'*40} {'─'*6}  {'─'*6}  {'─'*6}")

        grp = (df.groupby(col)
                 .agg(count=(TARGET,'count'), converted=(TARGET,'sum'))
                 .reset_index())
        grp['share']   = grp['count']   / total * 100
        grp['conv_pct'] = grp['converted'] / grp['count'] * 100
        grp = grp.sort_values('conv_pct', ascending=False)

        for _, row in grp.iterrows():
            diff = row['conv_pct'] - overall_rate
            arrow = '↑' if diff > 5 else ('↓' if diff < -5 else '→')
            print(f"  {str(row[col]):<40} {int(row['count']):>6}  "
                  f"{row['share']:>5.1f}%  {row['conv_pct']:>5.1f}% {arrow}")

    # ── Numerical features ────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  NUMERICAL FEATURES — Conversion rate by quartile")
    print(f"{'─'*65}")

    for col in NUMERICAL:
        _, bins = pd.qcut(df[col], q=4, retbins=True, duplicates='drop')
        labels = []
        for i in range(len(bins) - 1):
            lo = bins[i]
            hi = bins[i + 1]
            labels.append(f"{lo:.2f}-{hi:.2f}")

        df['_q'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        grp = (df.groupby('_q', observed=True)
                 .agg(count=(TARGET,'count'), converted=(TARGET,'sum'))
                 .reset_index())
        grp['conv_pct'] = grp['converted'] / grp['count'] * 100

        print(f"\n  {col}")
        for _, row in grp.iterrows():
            bar = '█' * int(row['conv_pct'] / 4)
            print(f"    {str(row['_q']):<12}  {row['conv_pct']:>5.1f}%  {bar}")
        df.drop(columns='_q', inplace=True)

    # ── Correlation with TARGET ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  POINT-BISERIAL CORRELATION with Converted (numerical only)")
    print(f"{'─'*65}")
    print("  (Higher absolute value = stronger linear relationship)")
    for col in NUMERICAL:
        corr = df[col].corr(df[TARGET])
        bar  = '█' * int(abs(corr) * 30)
        print(f"  {col:<38}  r = {corr:+.4f}  {bar}")

    print("\n" + "=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: Save a visual chart for each feature
# ─────────────────────────────────────────────────────────────────────────────
def save_charts(df):
    total        = len(df)
    overall_rate = df[TARGET].mean() * 100

    # We'll make a big figure: 4 rows × 2 cols (one panel per feature)
    fig, axes = plt.subplots(4, 2, figsize=(16, 22))
    fig.suptitle(
        'Feature Impact Analysis — Conversion Rate per Value\n'
        'Orange dashed line = overall dataset conversion rate',
        fontsize=14, fontweight='bold', y=1.005
    )
    axes = axes.flatten()

    panel = 0

    # ── Categorical panels ────────────────────────────────────────────────────
    for col in CATEGORICAL:
        ax = axes[panel]; panel += 1

        grp = (df.groupby(col)
                 .agg(count=(TARGET,'count'), converted=(TARGET,'sum'))
                 .reset_index())
        grp['conv_pct'] = grp['converted'] / grp['count'] * 100
        grp['share']    = grp['count']     / total        * 100
        grp = grp.sort_values('conv_pct', ascending=False)

        # Shorten long labels
        labels = [str(v)[:28] for v in grp[col]]
        x      = np.arange(len(labels))

        # Bar: conversion rate.  Alpha proportional to how common the category is.
        colors = [C_PURPLE if p >= overall_rate else C_GRAY for p in grp['conv_pct']]
        bars   = ax.bar(x, grp['conv_pct'], color=colors, width=0.6,
                        edgecolor='white', linewidth=0.8)

        # Overlay a small secondary bar showing share of dataset (right axis)
        ax2 = ax.twinx()
        ax2.bar(x + 0.0, grp['share'], width=0.6, alpha=0.08,
                color=C_PURPLE, linewidth=0)
        ax2.set_ylabel('Share of dataset (%)', fontsize=8, color=C_GRAY)
        ax2.tick_params(axis='y', labelsize=7, colors=C_GRAY)
        ax2.set_ylim(0, grp['share'].max() * 5)

        # Overall rate reference line
        ax.axhline(overall_rate, color=C_ORANGE, linewidth=1.4,
                   linestyle='--', alpha=0.8, label=f'Overall {overall_rate:.1f}%')

        # Value labels on bars
        for bar, val in zip(bars, grp['conv_pct']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.8,
                    f'{val:.0f}%', ha='center', va='bottom',
                    fontsize=8, fontweight='bold')

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
        ax.set_title(col, fontweight='bold', fontsize=10, pad=8)
        ax.set_ylabel('Conversion Rate (%)', fontsize=9)
        ax.set_ylim(0, min(105, grp['conv_pct'].max() + 12))
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # ── Numerical panels ──────────────────────────────────────────────────────
    for col in NUMERICAL:
        ax = axes[panel]; panel += 1
        _, bins = pd.qcut(df[col], q=4, retbins=True, duplicates='drop')
        labels = []
        for i in range(len(bins) - 1):
            lo = bins[i]
            hi = bins[i + 1]
            labels.append(f"{lo:.2f}-{hi:.2f}")

        df['_q'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        grp = (df.groupby('_q', observed=True)
                 .agg(count=(TARGET,'count'), converted=(TARGET,'sum'))
                 .reset_index())
        grp['conv_pct'] = grp['converted'] / grp['count'] * 100
        df.drop(columns='_q', inplace=True)

        colors = [C_GREEN if p >= overall_rate else C_GRAY for p in grp['conv_pct']]
        bars   = ax.bar(grp['_q'].astype(str), grp['conv_pct'],
                        color=colors, width=0.5,
                        edgecolor='white', linewidth=0.8)
        ax.axhline(overall_rate, color=C_ORANGE, linewidth=1.4,
                   linestyle='--', alpha=0.8, label=f'Overall {overall_rate:.1f}%')

        for bar, val in zip(bars, grp['conv_pct']):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.8,
                    f'{val:.0f}%', ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

        ax.set_title(col, fontweight='bold', fontsize=10, pad=8)
        ax.set_ylabel('Conversion Rate (%)', fontsize=9)
        ax.set_ylim(0, min(105, grp['conv_pct'].max() + 12))
        ax.spines['top'].set_visible(False)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(axis='y', alpha=0.25, linewidth=0.5)

    # Hide unused panel (we have 7 features but 8 slots)
    axes[7].set_visible(False)

    plt.tight_layout(pad=2.5)
    path = 'static/plots/feature_analysis.png'
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\n  Chart saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: Save a dark report image (layout similar to reference)
# ─────────────────────────────────────────────────────────────────────────────
def save_report_image(df):
    overall_rate = df[TARGET].mean() * 100

    def format_num(v):
        if abs(v - round(v)) < 0.05:
            return f"{int(round(v))}"
        return f"{v:.1f}"

    def get_bins(series, unit):
        _, bins = pd.qcut(series, q=4, retbins=True, duplicates='drop')
        bins = np.unique(bins)
        labels = []
        for i in range(len(bins) - 1):
            lo = bins[i]
            hi = bins[i + 1]
            if i == 0:
                labels.append(f"Bottom 25% (<{format_num(hi)} {unit})")
            elif i == len(bins) - 2:
                labels.append(f"Top 25% (>{format_num(lo)} {unit})")
            else:
                labels.append(f"{int(25*i)}–{int(25*(i+1))}% ({format_num(lo)}–{format_num(hi)} {unit})")
        return bins, labels

    def get_categorical_rows(col, top_n=6):
        grp = (df.groupby(col)
                 .agg(count=(TARGET, 'count'), converted=(TARGET, 'sum'))
                 .reset_index())
        grp['conv_pct'] = grp['converted'] / grp['count'] * 100
        grp['share']    = grp['count'] / grp['count'].sum() * 100
        grp = grp.sort_values('conv_pct', ascending=False).head(top_n)
        rows = []
        for _, row in grp.iterrows():
            rows.append({
                'label': str(row[col]),
                'conv_pct': row['conv_pct'],
                'share': row['share'],
            })
        return rows

    def get_numerical_rows(col, unit):
        bins, labels = get_bins(df[col], unit)
        df['_q'] = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        grp = (df.groupby('_q', observed=True)
                 .agg(count=(TARGET, 'count'), converted=(TARGET, 'sum'))
                 .reset_index())
        grp['conv_pct'] = grp['converted'] / grp['count'] * 100
        grp['share']    = grp['count'] / grp['count'].sum() * 100
        rows = []
        for _, row in grp.iterrows():
            rows.append({
                'label': str(row['_q']),
                'conv_pct': row['conv_pct'],
                'share': row['share'],
            })
        df.drop(columns='_q', inplace=True)
        return rows

    def feature_spreads():
        spreads = {}
        for col in CATEGORICAL:
            grp = df.groupby(col)[TARGET].mean() * 100
            spreads[col] = grp.max() - grp.min()
        for col in NUMERICAL:
            df['_q'] = pd.qcut(df[col], q=4, duplicates='drop')
            grp = df.groupby('_q', observed=True)[TARGET].mean() * 100
            spreads[col] = grp.max() - grp.min()
            df.drop(columns='_q', inplace=True)
        ranked = sorted(spreads.items(), key=lambda x: x[1], reverse=True)
        return ranked

    title_map = {
        'Total Time Spent on Website': 'Time on site',
        'Last Activity': 'Last activity',
        'Lead Source': 'Lead source',
        'What is your current occupation': 'Occupation',
        'Receive More Updates About Our Courses': 'Wants updates',
        'TotalVisits': 'Total visits',
        'Page Views Per Visit': 'Page views/visit',
    }

    unit_map = {
        'TotalVisits': 'visits',
        'Total Time Spent on Website': 'sec',
        'Page Views Per Visit': 'pages',
    }

    ranked = feature_spreads()

    # Layout: header tiles + 7 sections
    fig = plt.figure(figsize=(16, 22), facecolor='#1f1f1f')
    grid = fig.add_gridspec(
        nrows=8,
        ncols=1,
        height_ratios=[1.3, 2.4, 2.2, 2.2, 2.2, 2.0, 2.0, 2.0],
        hspace=0.35
    )

    def draw_tiles(ax, ranked_items):
        ax.set_facecolor('#1f1f1f')
        ax.axis('off')
        n = len(ranked_items)
        pad = 0.02
        tile_w = (1 - pad * (n + 1)) / n
        for i, (col, spread) in enumerate(ranked_items):
            x0 = pad + i * (tile_w + pad)
            y0 = 0.1
            ax.add_patch(plt.Rectangle(
                (x0, y0), tile_w, 0.8,
                facecolor='#232323', edgecolor='#2e2e2e', linewidth=1, zorder=1
            ))
            label = title_map.get(col, col)
            ax.text(x0 + tile_w/2, y0 + 0.55, f"#{i+1}",
                    color='#a5a5a5', fontsize=9, ha='center', va='center')
            ax.text(x0 + tile_w/2, y0 + 0.35, label,
                    color='#e6e6e6', fontsize=9.5, fontweight='bold', ha='center')
            ax.text(x0 + tile_w/2, y0 + 0.18, f"+{spread:.0f}pp",
                    color=C_GREEN if i < 3 else C_ORANGE,
                    fontsize=9.5, fontweight='bold', ha='center')

    def draw_section(ax, title, subtitle, rows, highlight=None):
        ax.set_facecolor('#1f1f1f')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # Titles
        ax.text(0.02, 0.93, title, color='#e6e6e6', fontsize=12,
                fontweight='bold', va='top')
        if highlight:
            ax.text(0.28, 0.93, f"  {highlight}", color='#1f1f1f', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='#6bb7a8', edgecolor='none'))
        ax.text(0.02, 0.82, subtitle, color='#a8a8a8', fontsize=9, va='top')

        # Column headers
        ax.text(0.02, 0.68, "GROUP", color='#6f6f6f', fontsize=8, fontweight='bold')
        ax.text(0.33, 0.68, "CONVERSION RATE", color='#6f6f6f', fontsize=8, fontweight='bold')
        ax.text(0.86, 0.68, "CONV%", color='#6f6f6f', fontsize=8, fontweight='bold', ha='right')
        ax.text(0.96, 0.68, "SHARE", color='#6f6f6f', fontsize=8, fontweight='bold', ha='right')

        bar_x = 0.33
        bar_w = 0.48
        row_h = 0.11
        y = 0.58

        # Overall marker
        marker_x = bar_x + bar_w * (overall_rate / 100)
        ax.plot([marker_x, marker_x], [0.2, 0.62], color='#d26b4e', linewidth=1, alpha=0.9)

        for row in rows:
            label = row['label']
            conv = row['conv_pct']
            share = row['share']
            ax.text(0.02, y, label, color='#d0d0d0', fontsize=9, va='center')

            # background bar
            ax.add_patch(plt.Rectangle(
                (bar_x, y - 0.025), bar_w, 0.05,
                facecolor='#2a2a2a', edgecolor='none'
            ))
            bar_color = '#7b6bd3' if conv >= overall_rate else '#8b8882'
            ax.add_patch(plt.Rectangle(
                (bar_x, y - 0.025), bar_w * (conv / 100), 0.05,
                facecolor=bar_color, edgecolor='none'
            ))

            ax.text(0.86, y, f"{conv:.0f}%", color='#cfcfcf', fontsize=9, va='center', ha='right')
            ax.text(0.96, y, f"{share:.0f}%", color='#9a9a9a', fontsize=9, va='center', ha='right')
            y -= row_h

    # Draw tiles
    ax_tiles = fig.add_subplot(grid[0, 0])
    draw_tiles(ax_tiles, ranked)

    # Build sections in ranked order
    for idx, (col, _) in enumerate(ranked, start=1):
        ax = fig.add_subplot(grid[idx, 0])
        label = title_map.get(col, col)
        if col in NUMERICAL:
            rows = get_numerical_rows(col, unit_map[col])
            subtitle = "Numerical — split into 4 equal-size groups (quartiles)."
        else:
            rows = get_categorical_rows(col)
            subtitle = "Categorical — most recent or most common option."

        highlight = None
        if idx == 1:
            highlight = "strongest signal"
        elif idx <= 3:
            highlight = "high impact"
        elif idx <= 5:
            highlight = "medium impact"
        else:
            highlight = "weak signal"

        draw_section(ax, f"{idx}. {label}", subtitle, rows, highlight=highlight)

    path = 'static/plots/impact_report.png'
    plt.savefig(path, dpi=140, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n  Report saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: Summary ranking — which feature matters most?
# ─────────────────────────────────────────────────────────────────────────────
def print_ranking(df):
    overall = df[TARGET].mean() * 100
    print("\n  FEATURE POWER RANKING")
    print("  (max spread = difference between highest and lowest conversion")
    print("   rate across that feature's values / categories)")
    print(f"  {'─'*55}")
    print(f"  {'Rank':<5} {'Feature':<42} {'Max Spread':>10}")
    print(f"  {'─'*55}")

    spreads = {}

    for col in CATEGORICAL:
        grp = df.groupby(col)[TARGET].mean() * 100
        spreads[col] = grp.max() - grp.min()

    for col in NUMERICAL:
        df['_q'] = pd.qcut(df[col], q=4, duplicates='drop')
        grp = df.groupby('_q', observed=True)[TARGET].mean() * 100
        spreads[col] = grp.max() - grp.min()
        df.drop(columns='_q', inplace=True)

    ranked = sorted(spreads.items(), key=lambda x: x[1], reverse=True)
    for i, (col, spread) in enumerate(ranked, 1):
        bar = '▓' * int(spread / 3)
        print(f"  #{i:<4} {col:<42} {spread:>7.1f}pp  {bar}")

    print(f"\n  pp = percentage points difference")
    print(f"  Overall dataset conversion rate = {overall:.1f}%")
    print("=" * 65 + "\n")


if __name__ == '__main__':
    df = load()
    print_report(df)
    print_ranking(df)
    save_charts(df)
    save_report_image(df)