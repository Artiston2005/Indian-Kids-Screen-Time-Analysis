

import argparse
import os
import sys
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ----------------------------- Helpers -----------------------------

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def savefig(fig, path: str, dpi=150):
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# ----------------------------- Load & Clean -----------------------------

def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # basic cleaning: strip column names and string values
    df.columns = [c.strip() for c in df.columns]
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    return df


def coerce_numeric(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    for c in cols:
        df[c + '_orig'] = df[c]
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


# ----------------------------- EDA -----------------------------

def basic_eda(df: pd.DataFrame, outdir: str) -> None:
    # summary tables
    df.describe(include='all').to_csv(os.path.join(outdir, 'describe_all.csv'))
    pd.DataFrame(df.dtypes.astype(str), columns=['dtype']).to_csv(os.path.join(outdir, 'dtypes.csv'))
    pd.DataFrame(df.isnull().sum(), columns=['null_count']).to_csv(os.path.join(outdir, 'nulls.csv'))

    # numeric histograms
    numcols = df.select_dtypes(include=[np.number]).columns.tolist()
    for c in numcols:
        fig = plt.figure()
        plt.hist(df[c].dropna(), bins=30)
        plt.title(f'Histogram of {c}')
        plt.xlabel(c)
        plt.ylabel('count')
        savefig(fig, os.path.join(outdir, f'hist_{c}.png'))

    # categorical barplots (top 20)
    catcols = df.select_dtypes(include=['object']).columns.tolist()
    for c in catcols:
        vc = df[c].value_counts(dropna=False).nlargest(20)
        fig = plt.figure(figsize=(8, max(3, 0.25 * len(vc))))
        vc.plot(kind='bar')
        plt.title(f'Value counts for {c}')
        plt.ylabel('count')
        savefig(fig, os.path.join(outdir, f'bar_{c}.png'))

    # correlation heatmap for numeric columns
    if len(numcols) >= 2:
        corr = df[numcols].corr()
        fig = plt.figure(figsize=(max(6, len(numcols)), max(5, len(numcols) * 0.5)))
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')
        plt.title('Correlation matrix (numeric)')
        savefig(fig, os.path.join(outdir, 'corr_heatmap.png'))


# ----------------------------- Specific analyses -----------------------------

def screentime_by_age_plot(df: pd.DataFrame, outdir: str, age_col='age', screen_col='screen_time_minutes'):
    if age_col not in df.columns or screen_col not in df.columns:
        return
    agg = df.groupby(age_col)[screen_col].agg(['median','mean','count']).reset_index()
    fig = plt.figure()
    plt.plot(agg[age_col], agg['median'], marker='o', label='median')
    plt.plot(agg[age_col], agg['mean'], marker='s', label='mean')
    plt.title('Screen time by age')
    plt.xlabel('age')
    plt.ylabel(screen_col)
    plt.legend()
    savefig(fig, os.path.join(outdir, 'screentime_by_age.png'))


def device_vs_screentime_boxplot(df: pd.DataFrame, outdir: str, device_col='device_type', screen_col='screen_time_minutes'):
    if device_col not in df.columns or screen_col not in df.columns:
        return
    plt.figure(figsize=(8,6))
    sns.boxplot(x=device_col, y=screen_col, data=df)
    plt.title('Screen time by device type')
    savefig(plt.gcf(), os.path.join(outdir, 'box_device_screen.png'))


def parental_supervision_analysis(df: pd.DataFrame, outdir: str, sup_col='parental_supervision', screen_col='screen_time_minutes'):
    if sup_col not in df.columns or screen_col not in df.columns:
        return
    agg = df.groupby(sup_col)[screen_col].agg(['median','mean','count']).reset_index()
    agg.to_csv(os.path.join(outdir, 'parental_supervision_vs_screen.csv'), index=False)
    fig = plt.figure()
    agg.plot(x=sup_col, y=['mean','median'], kind='bar', rot=0, figsize=(6,4), legend=True)
    plt.title('Screen time by parental supervision')
    savefig(fig, os.path.join(outdir, 'parental_supervision_bar.png'))


# ----------------------------- Clustering -----------------------------

def run_kmeans_clusters(df: pd.DataFrame, outdir: str, features: list, n_clusters=3):
    sub = df[features].dropna()
    if sub.shape[0] < n_clusters:
        print('Not enough rows for clustering')
        return
    scaler = StandardScaler()
    X = scaler.fit_transform(sub.values)
    km = KMeans(n_clusters=n_clusters, random_state=42)
    labels = km.fit_predict(X)
    sub2 = sub.copy()
    sub2['cluster'] = labels
    sub2.to_csv(os.path.join(outdir, 'kmeans_clustered.csv'), index=False)
    # plot cluster centers projected on first two features
    if len(features) >= 2:
        fig = plt.figure()
        plt.scatter(X[:,0], X[:,1], c=labels, alpha=0.6)
        plt.title('KMeans clusters (first 2 scaled features)')
        savefig(fig, os.path.join(outdir, 'kmeans_scatter.png'))


# ----------------------------- Main -----------------------------

def main(args):
    ensure_outdir(args.outdir)
    if not os.path.exists(args.csv):
        print('CSV file not found:', args.csv)
        sys.exit(1)

    df = load_data(args.csv)

    # Attempt to find a screen_time column: common names
    possible_screen_cols = [c for c in df.columns if 'screen' in c.lower()]
    if len(possible_screen_cols) == 0:
        print('Warning: no column name with "screen" found. Update the script or pass explicit column names.')
    else:
        print('Detected screen-time columns:', possible_screen_cols)

    # Coerce suspected numeric columns
    suspect_nums = []
    for c in df.columns:
        low = c.lower()
        if any(x in low for x in ['time','minutes','age','hours']):
            suspect_nums.append(c)
    if suspect_nums:
        df = coerce_numeric(df, suspect_nums)

    basic_eda(df, args.outdir)

    # Try specific analyses using guessed column names
    # Prefer exact column names if present
    age_col = None
    for c in df.columns:
        if c.lower() in ['age','child_age','childage']:
            age_col = c
    screen_col = None
    for c in df.columns:
        if 'screen' in c.lower() or 'time' in c.lower() and ('minute' in c.lower() or 'hour' in c.lower()):
            screen_col = c
    device_col = None
    for c in df.columns:
        if any(x in c.lower() for x in ['device','platform']):
            device_col = c
    sup_col = None
    for c in df.columns:
        if any(x in c.lower() for x in ['parent','supervision','supervised']):
            sup_col = c

    # rename coerced numeric columns if created
    # If screen_col is numeric but has _orig counterpart, prefer numeric
    if screen_col and screen_col + '_orig' in df.columns and pd.api.types.is_numeric_dtype(df[screen_col]):
        pass

    # Run plots if columns exist
    if age_col and screen_col and pd.api.types.is_numeric_dtype(df[screen_col]):
        screentime_by_age_plot(df, args.outdir, age_col=age_col, screen_col=screen_col)
    if device_col and screen_col and pd.api.types.is_numeric_dtype(df[screen_col]):
        device_vs_screentime_boxplot(df, args.outdir, device_col=device_col, screen_col=screen_col)
    if sup_col and screen_col and pd.api.types.is_numeric_dtype(df[screen_col]):
        parental_supervision_analysis(df, args.outdir, sup_col=sup_col, screen_col=screen_col)

    # Simple clustering on numeric features: choose top numeric cols
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'screen_time_minutes' in numeric_cols:
        features = ['screen_time_minutes'] + [c for c in numeric_cols if c != 'screen_time_minutes'][:2]
    else:
        features = numeric_cols[:3]
    if len(features) >= 1:
        run_kmeans_clusters(df, args.outdir, features, n_clusters=3)

    # Save cleaned sample
    df.sample(frac=1.0).head(200).to_csv(os.path.join(args.outdir, 'sample_output.csv'), index=False)

    print('Analysis complete. Check the folder:', args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Indian Kids Screentime 2025 - EDA script')
    parser.add_argument('--csv', required=True, help='Path to the CSV file')
    parser.add_argument('--outdir', default='results', help='Output folder for plots and tables')
    args = parser.parse_args()
    main(args)
