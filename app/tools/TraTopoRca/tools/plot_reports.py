import os
import argparse
import pandas as pd
import matplotlib
# Force non-GUI backend to avoid Qt plugin issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def plot_training_dynamics(report_dir: str):
    path = os.path.join(report_dir, 'training_dynamics.csv')
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    outs = []

    # Plot log sigmas
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in [
        ('log_sigma_struct', 'log_sigma_struct'),
        ('log_sigma_lat', 'log_sigma_lat'),
        ('log_sigma_host', 'log_sigma_host'),
    ]:
        if col in df.columns:
            ax.plot(df['epoch'], df[col], label=label)
    ax.set_title('Kendall Log Sigmas (lower => higher weight)')
    ax.set_xlabel('epoch')
    ax.set_ylabel('log sigma')
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(report_dir, 'training_sigmas.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outs.append(out)

    # Plot losses
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in [
        ('loss_total', 'loss_total'),
        ('loss_struct', 'loss_struct'),
        ('loss_lat', 'loss_lat'),
    ]:
        if col in df.columns:
            ax.plot(df['epoch'], df[col], label=label)
    ax.set_title('Training Loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(report_dir, 'training_losses.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outs.append(out)
    return outs


def plot_epoch_metrics(report_dir: str):
    path = os.path.join(report_dir, 'epoch_metrics.csv')
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    outs = []

    # Filter rows with numeric epoch for curves, keep last row separately if epoch is NaN
    df_num = df[pd.to_numeric(df['epoch'], errors='coerce').notna()].copy()
    df_num['epoch'] = df_num['epoch'].astype(int)

    # Top-K metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in [
        ('svc_top1', 'svc_top1'),
        ('host_top1', 'host_top1'),
        ('mixed_top1', 'mixed_top1'),
        ('mixed_top5', 'mixed_top5'),
    ]:
        if col in df_num.columns:
            ax.plot(df_num['epoch'], df_num[col], marker='o', label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_title('RCA Top-K over Epochs')
    ax.set_xlabel('epoch')
    ax.set_ylabel('accuracy')
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(report_dir, 'epoch_topk.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outs.append(out)

    # Detection metrics
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, label in [
        ('auc', 'auc'),
        ('best_fscore', 'best_fscore'),
        ('fscore', 'fscore'),
        ('precision', 'precision'),
        ('recall', 'recall'),
    ]:
        if col in df_num.columns:
            ax.plot(df_num['epoch'], df_num[col], marker='o', label=label)
    ax.set_ylim(0.0, 1.05)
    ax.set_title('Detection Metrics over Epochs')
    ax.set_xlabel('epoch')
    ax.set_ylabel('score')
    ax.grid(True, alpha=0.3)
    ax.legend()
    out = os.path.join(report_dir, 'epoch_scores.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outs.append(out)
    return outs


def plot_conflicts(report_dir: str):
    path = os.path.join(report_dir, 'rca_conflicts.csv')
    if not os.path.isfile(path):
        return []
    df = pd.read_csv(path)
    outs = []

    # Count per epoch
    grp = df.groupby('epoch', dropna=False)['trace_id'].count().reset_index(name='count')
    grp['epoch'] = grp['epoch'].fillna('(none)')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(grp['epoch'].astype(str), grp['count'])
    ax.set_title('Conflict Samples per Epoch')
    ax.set_xlabel('epoch')
    ax.set_ylabel('count')
    ax.grid(True, axis='y', alpha=0.3)
    out = os.path.join(report_dir, 'conflicts_counts.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outs.append(out)

    # Avg delta per epoch
    grp2 = df.groupby('epoch', dropna=False)['delta'].mean().reset_index(name='avg_delta')
    grp2['epoch'] = grp2['epoch'].fillna('(none)')
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(grp2['epoch'].astype(str), grp2['avg_delta'], marker='o')
    ax.set_title('Average Score Delta (HostTop1 - GTSvc) per Epoch')
    ax.set_xlabel('epoch')
    ax.set_ylabel('avg delta')
    ax.grid(True, alpha=0.3)
    out = os.path.join(report_dir, 'conflicts_delta.png')
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    outs.append(out)

    # Top offending hosts
    if 'top1_id' in df.columns:
        top_host = df.groupby('top1_id')['trace_id'].count().reset_index(name='count').sort_values('count', ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(top_host['top1_id'].astype(str), top_host['count'])
        ax.set_title('Top Offending Hosts (as Mixed Top-1)')
        ax.set_xlabel('host_id')
        ax.set_ylabel('count')
        ax.grid(True, axis='y', alpha=0.3)
        out = os.path.join(report_dir, 'conflicts_top_hosts.png')
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
        outs.append(out)
    return outs


def plot_dashboard(report_dir: str, outfile: str = 'report_dashboard.png'):
    """Compose a single dashboard image combining training/epoch/conflict plots.
    Returns a list containing the saved path (or empty if nothing to plot).
    """
    from matplotlib import gridspec as _gridspec

    path_td = os.path.join(report_dir, 'training_dynamics.csv')
    path_em = os.path.join(report_dir, 'epoch_metrics.csv')
    path_cf = os.path.join(report_dir, 'rca_conflicts.csv')

    if not any(os.path.isfile(p) for p in [path_td, path_em, path_cf]):
        return []

    tdf = pd.read_csv(path_td) if os.path.isfile(path_td) else None
    edf = pd.read_csv(path_em) if os.path.isfile(path_em) else None
    cdf = pd.read_csv(path_cf) if os.path.isfile(path_cf) else None

    fig = plt.figure(figsize=(14, 10))
    gs = _gridspec.GridSpec(2, 2, figure=fig, wspace=0.25, hspace=0.35)
    ax00 = fig.add_subplot(gs[0, 0])  # training sigmas
    ax01 = fig.add_subplot(gs[0, 1])  # training losses
    ax10 = fig.add_subplot(gs[1, 0])  # epoch metrics
    ax11 = fig.add_subplot(gs[1, 1])  # conflicts

    # Training sigmas
    if tdf is not None and not tdf.empty and 'epoch' in tdf.columns:
        for col, label in [
            ('log_sigma_struct', 'log_sigma_struct'),
            ('log_sigma_lat', 'log_sigma_lat'),
            ('log_sigma_host', 'log_sigma_host'),
        ]:
            if col in tdf.columns:
                ax00.plot(tdf['epoch'], tdf[col], label=label)
        ax00.set_title('Kendall Log Sigmas (lower => higher weight)')
        ax00.set_xlabel('epoch')
        ax00.set_ylabel('log sigma')
        ax00.grid(True, alpha=0.3)
        ax00.legend()
    else:
        ax00.set_title('Kendall Log Sigmas')
        ax00.text(0.5, 0.5, 'training_dynamics.csv not found', ha='center', va='center', transform=ax00.transAxes)
        ax00.axis('off')

    # Training losses
    if tdf is not None and not tdf.empty and 'epoch' in tdf.columns:
        for col, label in [
            ('loss_total', 'loss_total'),
            ('loss_struct', 'loss_struct'),
            ('loss_lat', 'loss_lat'),
        ]:
            if col in tdf.columns:
                ax01.plot(tdf['epoch'], tdf[col], label=label)
        ax01.set_title('Training Loss')
        ax01.set_xlabel('epoch')
        ax01.set_ylabel('loss')
        ax01.grid(True, alpha=0.3)
        ax01.legend()
    else:
        ax01.set_title('Training Loss')
        ax01.text(0.5, 0.5, 'training_dynamics.csv not found', ha='center', va='center', transform=ax01.transAxes)
        ax01.axis('off')

    # Epoch metrics (RCA + detection)
    if edf is not None and not edf.empty and 'epoch' in edf.columns:
        df_num = edf[pd.to_numeric(edf['epoch'], errors='coerce').notna()].copy()
        if not df_num.empty:
            df_num['epoch'] = df_num['epoch'].astype(int)
            for col, label in [
                ('svc_top1', 'svc_top1'),
                ('host_top1', 'host_top1'),
                ('mixed_top1', 'mixed_top1'),
                ('mixed_top5', 'mixed_top5'),
            ]:
                if col in df_num.columns:
                    ax10.plot(df_num['epoch'], df_num[col], marker='o', label=label)
            ax10.set_ylim(0.0, 1.05)
            ax10.set_title('RCA Top-K (solid) & Detection (dashed)')
            ax10.set_xlabel('epoch')
            ax10.set_ylabel('accuracy')
            ax10.grid(True, alpha=0.3)
            l1 = ax10.legend(loc='upper left')

            # detection on twin axis
            ax10b = ax10.twinx()
            plotted = False
            for col, label in [('auc', 'auc'), ('best_fscore', 'best_fscore'), ('fscore', 'fscore')]:
                if col in df_num.columns:
                    ax10b.plot(df_num['epoch'], df_num[col], linestyle='--', alpha=0.6, label=label)
                    plotted = True
            if plotted:
                ax10b.set_ylim(0.0, 1.05)
                ax10b.set_ylabel('score')
                l2 = ax10b.legend(loc='lower right')
        else:
            ax10.set_title('Epoch Metrics')
            ax10.text(0.5, 0.5, 'No numeric epochs', ha='center', va='center', transform=ax10.transAxes)
            ax10.axis('off')
    else:
        ax10.set_title('Epoch Metrics')
        ax10.text(0.5, 0.5, 'epoch_metrics.csv not found', ha='center', va='center', transform=ax10.transAxes)
        ax10.axis('off')

    # Conflicts (count + delta)
    if cdf is not None and not cdf.empty and 'trace_id' in cdf.columns:
        grp = cdf.groupby('epoch', dropna=False)['trace_id'].count().reset_index(name='count')
        grp['epoch'] = grp['epoch'].fillna('(none)')
        ax11.bar(grp['epoch'].astype(str), grp['count'], color='#4C72B0', alpha=0.85, label='count')
        if 'delta' in cdf.columns:
            grp2 = cdf.groupby('epoch', dropna=False)['delta'].mean().reset_index(name='avg_delta')
            grp2['epoch'] = grp2['epoch'].fillna('(none)')
            ax11b = ax11.twinx()
            ax11b.plot(grp2['epoch'].astype(str), grp2['avg_delta'], color='#DD8452', marker='o', label='avg_delta')
            ax11b.set_ylabel('avg delta')
            ax11b.grid(False)
        ax11.set_title('Conflicts per Epoch')
        ax11.set_xlabel('epoch')
        ax11.set_ylabel('count')
        ax11.grid(True, axis='y', alpha=0.3)
    else:
        ax11.set_title('Conflicts')
        ax11.text(0.5, 0.5, 'rca_conflicts.csv not found', ha='center', va='center', transform=ax11.transAxes)
        ax11.axis('off')

    out = os.path.join(report_dir, outfile)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return [out]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reports-dir', default="E:\ZJU\AIOps\Projects\TraDNN\TraTopoRca\dataset\dataset_demo\processed/reports_1120", 
                    help='Directory containing epoch_metrics.csv, training_dynamics.csv, rca_conflicts.csv')
    ap.add_argument('--dashboard', dest='dashboard', action='store_true', default=True, help='Generate single dashboard image (default: True).')
    ap.add_argument('--no-dashboard', dest='dashboard', action='store_false', help='Disable dashboard and generate separate images instead.')
    ap.add_argument('--out', default='report_dashboard.png', help='Output filename for dashboard (when --dashboard)')
    args = ap.parse_args()
    report_dir = _ensure_dir(args.reports_dir)

    outs = []
    if args.dashboard:
        # Only generate the single dashboard image by default
        outs += plot_dashboard(report_dir, args.out)
    else:
        # Generate separate images for each section
        outs += plot_training_dynamics(report_dir)
        outs += plot_epoch_metrics(report_dir)
        outs += plot_conflicts(report_dir)

    if outs:
        print('\n'.join(outs))
    else:
        print('No plots generated (required CSVs not found).')


if __name__ == '__main__':
    main()
