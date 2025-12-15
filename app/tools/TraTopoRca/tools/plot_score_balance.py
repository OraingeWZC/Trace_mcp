import argparse
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    ap = argparse.ArgumentParser(description='Plot Service vs Host score balance (scatter)')
    ap.add_argument('--csv', default='dataset\dataset_demo\processed/reports_1120\score_distribution_debug.csv', help='Path to score_distribution_debug.csv')
    ap.add_argument('--out', default='dataset\dataset_demo\processed/reports_1120\score_balance_analysis.png', help='Output image filename')
    args = ap.parse_args()

    if not os.path.isfile(args.csv):
        raise FileNotFoundError(f'CSV not found: {args.csv}')

    df = pd.read_csv(args.csv)
    # 兼容列名
    svc_col = 'svc_max' if 'svc_max' in df.columns else 'svc_max_score'
    host_col = 'host_max' if 'host_max' in df.columns else 'host_max_score'
    if svc_col not in df.columns or host_col not in df.columns:
        raise ValueError('CSV missing svc_max/host_max columns')

    df[svc_col] = pd.to_numeric(df[svc_col])
    df[host_col] = pd.to_numeric(df[host_col])

    plt.figure(figsize=(9, 9))
    sns.set_style('whitegrid')
    hue_col = 'gt_type' if 'gt_type' in df.columns else None
    palette = {'svc': 'blue', 'host': 'red', 'Service': 'blue', 'Host': 'red'}
    sns.scatterplot(
        data=df,
        x=svc_col,
        y=host_col,
        hue=hue_col,
        style=hue_col,
        s=80,
        alpha=0.75,
        palette=palette if hue_col else None,
    )

    # 平衡线 x=y
    max_val = max(df[svc_col].max(), df[host_col].max())
    plt.plot([0, max_val], [0, max_val], 'k--', label='Balance (x=y)', linewidth=1.8)

    # 决策边界线（依据 lambda_host）
    lambda_col = 'lambda_host' if 'lambda_host' in df.columns else None
    if lambda_col and (df[lambda_col].notna().any()):
        try:
            lam = float(df[lambda_col].dropna().iloc[-1])
        except Exception:
            lam = 0.0
        if lam > 0:
            slope = (1.0 - lam) / lam
            x2 = max_val / max(slope, 1e-6)
            plt.plot([0, x2], [0, max_val], 'r-.', label=f'Boundary (lambda={lam:.2f})')

    plt.title('Service vs Host Max Score Scatter')
    plt.xlabel('Max Service Score (normalized)')
    plt.ylabel('Max Host Score (normalized)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f'Saved: {os.path.abspath(args.out)}')


if __name__ == '__main__':
    main()

