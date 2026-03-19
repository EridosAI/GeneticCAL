"""Generate all publication-quality figures for the CAL biology paper (v3).
Updated to λ=1.0 reporting, half-transformed scoring, corrected poster children."""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Paths
PAPER_DIR = Path(__file__).parent
FIG_DIR = PAPER_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)
GENE_AAR_RESULTS = Path("/mnt/c/Users/Jason/Desktop/Eridos/Gene CAL/Gene AAR/results")
GENE_AAR_DATA = Path("/mnt/c/Users/Jason/Desktop/Eridos/Gene CAL/Gene AAR/data")

# Style
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
})

BLUE = '#2171B5'
GREY = '#969696'
RED = '#CB181D'


def fig1_cross_boundary_overview():
    """Figure 1: Cross-Boundary AUC Across All Four Experiments (λ=1.0)."""
    experiments = ['Replogle\n(≥900)', 'DepMap\n(random neg.)', 'Drug\nFingerprints†', 'Drug\nL1000']
    cosine = [0.518, 0.570, 0.534, 0.466]
    cal = [0.908, 0.947, 0.554, 0.842]
    shuffled = [None, None, 0.636, 0.908]

    fig, ax = plt.subplots(figsize=(8, 5.5))

    x = np.arange(len(experiments))
    w = 0.22

    ax.bar(x - w, cosine, w, color=GREY, label='Cosine baseline',
           zorder=3, edgecolor='black', linewidth=0.5)
    ax.bar(x, cal, w, color=BLUE, label='CAL (assoc-only)',
           zorder=3, edgecolor='black', linewidth=0.5)

    # Shuffled bars for drug experiments only
    for i, s in enumerate(shuffled):
        if s is not None:
            ax.bar(x[i] + w, s, w, color=RED, label='Shuffled' if i == 2 else '',
                   zorder=3, edgecolor='black', linestyle='--', linewidth=1.0)

    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax.text(-0.35, 0.505, 'chance', fontsize=9, color='black', alpha=0.6)

    # "Shuffled > Real" annotation — in clear space, up and left
    ax.annotate('Shuffled > Real', xy=(2.22, 0.65), xytext=(1.65, 0.92),
                fontsize=10, ha='center', color=RED, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5,
                                connectionstyle='arc3,rad=-0.2'))
    ax.annotate('', xy=(3.00, 0.90), xytext=(2.20, 0.93),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5,
                                shrinkA=6, shrinkB=18,
                                connectionstyle='arc3,rad=0.25'))

    # "No latent signal" for drug fingerprints
    ax.annotate('No latent\nsignal', xy=(2, 0.56), xytext=(1.55, 0.70),
                fontsize=9, ha='center', color='#555555', fontstyle='italic',
                arrowprops=dict(arrowstyle='->', color='#555555', lw=1.0,
                                connectionstyle='arc3,rad=0.2'))

    ax.set_ylabel('Cross-Boundary AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(experiments)
    ax.set_ylim(0.35, 1.05)
    ax.set_xlim(-0.5, 3.7)

    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), frameon=False,
              ncol=3, fontsize=10)

    fig.text(0.12, 0.01,
             '† Drug Fingerprints: overall AUC shown (cross-boundary not reported separately)',
             fontsize=8, color='grey')

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(FIG_DIR / 'fig1_cross_boundary_overview.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig1_cross_boundary_overview.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Fig 1 saved.")


def fig2_threshold_sweep():
    """Figure 2: Confidence Threshold Sweep — 2-panel (Overall + CB), 4 points each."""
    labels = ['Low\n(≥400)', 'Medium\n(≥700)', 'High\n(≥900)', 'Exp-only\n(≥700)']
    n_pairs = [93562, 41434, 23268, 23438]

    # Overall AUC
    cos_overall = [0.570, 0.644, 0.692, 0.707]
    cal_overall = [0.800, 0.877, 0.910, 0.907]

    # Cross-boundary AUC
    cos_cb = [0.529, 0.534, 0.518, 0.517]
    cal_cb = [0.783, 0.861, 0.908, 0.902]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(labels))
    w = 0.3

    for ax, cos_vals, cal_vals, title in [
        (ax1, cos_overall, cal_overall, 'Overall AUC'),
        (ax2, cos_cb, cal_cb, 'Cross-Boundary AUC'),
    ]:
        bars_cos = ax.bar(x - w/2, cos_vals, w, color=GREY, label='Cosine baseline',
                          zorder=3, edgecolor='black', linewidth=0.5)
        bars_cal = ax.bar(x + w/2, cal_vals, w, color=BLUE, label='CAL (assoc-only)',
                          zorder=3, edgecolor='black', linewidth=0.5)

        # Exp-only bars: different hatch pattern
        bars_cos[-1].set_hatch('//')
        bars_cal[-1].set_hatch('//')

        ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
        ax.set_ylabel(title)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.4, 1.0)

        # Pair count annotations above bars
        for i, n in enumerate(n_pairs):
            ax.text(i, max(cos_vals[i], cal_vals[i]) + 0.015,
                    f'{n//1000}K', ha='center', fontsize=8, color='#555555')

    ax1.text(-0.05, 1.05, 'A', transform=ax1.transAxes, fontsize=14, fontweight='bold')
    ax2.text(-0.05, 1.05, 'B', transform=ax2.transAxes, fontsize=14, fontweight='bold')

    # Single legend on panel A only (with all 3 entries including hatch)
    hatch_patch = mpatches.Patch(facecolor='white', edgecolor='black', hatch='//',
                                  label='Experimental channel only')
    ax1.legend(loc='upper left', frameon=False, fontsize=9,
               handles=ax1.get_legend_handles_labels()[0] + [hatch_patch])

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig2_threshold_sweep.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig2_threshold_sweep.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Fig 2 saved.")


def _load_embeddings_and_model():
    """Load PCA-50 embeddings, gene names, STRING pairs, and trained model."""
    import torch, h5py

    h5_path = GENE_AAR_DATA / 'K562_essential_normalized_bulk_01.h5ad'
    with h5py.File(h5_path, 'r') as f:
        X = f['X'][:]
        gene_names_raw = f['obs']['gene_transcript'][:]
        gene_names_full = [g.decode() if isinstance(g, bytes) else str(g) for g in gene_names_raw]
        gene_names = []
        for g in gene_names_full:
            parts = g.split('_')
            gene_names.append(parts[1] if len(parts) > 1 else g)

    ckpt = torch.load(GENE_AAR_RESULTS / 'model_high.pt', map_location='cpu', weights_only=False)
    pca_components = ckpt['pca_components']
    pca_mean = ckpt['pca_mean']

    pca_comp_np = pca_components.numpy() if hasattr(pca_components, 'numpy') else pca_components
    pca_mean_np = pca_mean.numpy() if hasattr(pca_mean, 'numpy') else pca_mean
    X_centered = X - pca_mean_np
    embeddings = X_centered @ pca_comp_np.T  # (2285, 50)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normed = embeddings / (norms + 1e-8)

    import torch.nn as nn
    class PAMModel(nn.Module):
        def __init__(self, dim=50, hidden=1024):
            super().__init__()
            self.alpha_logit = nn.Parameter(torch.tensor(0.0))
            self.mlp = nn.Sequential(
                nn.Linear(dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
                nn.Linear(hidden, dim), nn.LayerNorm(dim),
            )
        def forward(self, x):
            alpha = torch.sigmoid(self.alpha_logit)
            h = self.mlp(x)
            out = alpha * x + (1 - alpha) * h
            return out / (out.norm(dim=-1, keepdim=True) + 1e-8)

    model = PAMModel()
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        emb_tensor = torch.from_numpy(embeddings).float()
        assoc_emb = model(emb_tensor).numpy()

    string_pairs = np.load(GENE_AAR_DATA / 'string_pairs_high.npy')

    return gene_names, embeddings_normed, assoc_emb, string_pairs


def _half_transformed_score(assoc_emb, emb_normed, i, j):
    """Compute half-transformed score: 0.5 * (f(a)·b + f(b)·a)."""
    return 0.5 * (np.dot(assoc_emb[i], emb_normed[j]) +
                  np.dot(assoc_emb[j], emb_normed[i]))


def fig3_distribution_and_examples():
    """Figure 3: Violin distributions + poster children with lift arrows.
    Uses half-transformed scoring to match paper evaluation."""
    import torch

    print("  Fig 3: Loading embeddings and model...")
    gene_names, emb_normed, assoc_emb, string_pairs = _load_embeddings_and_model()

    n_genes = len(gene_names)
    string_set = set()
    for i, j in string_pairs:
        string_set.add((min(i, j), max(i, j)))

    # Build gene name -> index lookup
    gene_to_idx = {name: idx for idx, name in enumerate(gene_names)}

    # Positive pairs: compute cosine and half-transformed association scores
    print(f"  Fig 3: Computing half-transformed scores for {len(string_pairs)} positive pairs...")
    pos_cos, pos_assoc = [], []
    for i, j in string_pairs:
        pos_cos.append(np.dot(emb_normed[i], emb_normed[j]))
        pos_assoc.append(_half_transformed_score(assoc_emb, emb_normed, i, j))
    pos_cos = np.array(pos_cos)
    pos_assoc = np.array(pos_assoc)

    # Negative pairs: 5000 random non-STRING pairs
    print("  Fig 3: Sampling negative pairs...")
    rng = np.random.RandomState(42)
    neg_cos, neg_assoc = [], []
    while len(neg_cos) < 5000:
        i, j = rng.randint(0, n_genes, size=2)
        if i == j:
            continue
        pair = (min(i, j), max(i, j))
        if pair in string_set:
            continue
        neg_cos.append(np.dot(emb_normed[i], emb_normed[j]))
        neg_assoc.append(_half_transformed_score(assoc_emb, emb_normed, i, j))
    neg_cos = np.array(neg_cos)
    neg_assoc = np.array(neg_assoc)

    # Verify poster child values
    poster_pairs = [
        ('C7ORF26', 'INTS1', 0.074, 0.393),
        ('SNAPC2', 'SNAPC4', 0.167, 0.471),
        ('HSPA5', 'MANF', -0.048, 0.353),
    ]
    print("\n  Poster child verification (half-transformed):")
    for ga, gb, expected_cos, expected_assoc in poster_pairs:
        ia, ib = gene_to_idx.get(ga), gene_to_idx.get(gb)
        if ia is not None and ib is not None:
            cos_v = np.dot(emb_normed[ia], emb_normed[ib])
            assoc_v = _half_transformed_score(assoc_emb, emb_normed, ia, ib)
            print(f"  {ga}–{gb}: cos={cos_v:.3f} (exp {expected_cos:.3f}), "
                  f"assoc={assoc_v:.3f} (exp {expected_assoc:.3f})")
        else:
            print(f"  {ga}–{gb}: gene not found (ia={ia}, ib={ib})")

    # ── Bins for Panel A ──
    bin_edges = [(-0.3, -0.1), (-0.1, 0.0), (0.0, 0.1), (0.1, 0.2),
                 (0.2, 0.4), (0.4, 0.8)]
    bin_labels = ['[−0.3,\n−0.1)', '[−0.1,\n0.0)', '[0.0,\n0.1)',
                  '[0.1,\n0.2)', '[0.2,\n0.4)', '[0.4,\n0.8)']

    pos_binned, neg_binned = [], []
    for lo, hi in bin_edges:
        pos_binned.append(pos_assoc[(pos_cos >= lo) & (pos_cos < hi)])
        neg_binned.append(neg_assoc[(neg_cos >= lo) & (neg_cos < hi)])

    # ── Create figure ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6),
                                    gridspec_kw={'width_ratios': [2.2, 1]})

    # ── Panel A: Violins ──
    n_bins = len(bin_edges)
    positions_neg = np.arange(n_bins) * 2 - 0.35
    positions_pos = np.arange(n_bins) * 2 + 0.35
    width = 0.6

    # Cross-boundary shading (first 4 bins)
    cb_left = positions_neg[0] - width / 2 - 0.15
    cb_right = positions_pos[3] + width / 2 + 0.15
    ax1.axvspan(cb_left, cb_right, color='#FFF9C4', alpha=0.45, zorder=0)

    for idx in range(n_bins):
        for data, positions, color in [
            (neg_binned[idx], positions_neg[idx], GREY),
            (pos_binned[idx], positions_pos[idx], BLUE),
        ]:
            if len(data) >= 10:
                vp = ax1.violinplot([data], positions=[positions],
                                    widths=width, showmedians=True, showextrema=False)
                for body in vp['bodies']:
                    body.set_facecolor(color)
                    body.set_alpha(0.5)
                    body.set_edgecolor('black')
                    body.set_linewidth(0.5)
                vp['cmedians'].set_color('black')
                vp['cmedians'].set_linewidth(1.2)
            elif len(data) > 0:
                ax1.scatter(np.full(len(data), positions) +
                           rng.uniform(-0.1, 0.1, len(data)),
                           data, s=5, c=color, alpha=0.4, zorder=2)

    ax1.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.4, zorder=1)
    ax1.set_xticks(np.arange(n_bins) * 2)
    ax1.set_xticklabels(bin_labels, fontsize=9)
    ax1.set_xlabel('Expression Cosine Similarity (binned)')
    ax1.set_ylabel('CAL Association Score')

    ylims = ax1.get_ylim()
    ax1.text((cb_left + cb_right) / 2, ylims[1] * 0.97,
             'Cross-boundary regime', ha='center', fontsize=9,
             color='#B8860B', fontstyle='italic')

    grey_patch = mpatches.Patch(color=GREY, alpha=0.5, label='Non-associated')
    blue_patch = mpatches.Patch(color=BLUE, alpha=0.5, label='STRING-positive')
    ax1.legend(handles=[blue_patch, grey_patch], loc='lower right', frameon=False, fontsize=10)
    ax1.text(-0.03, 1.05, 'A', transform=ax1.transAxes, fontsize=16, fontweight='bold')

    # ── Panel B: Poster Children with Lift Arrows ──
    # Poster children — staggered labels to avoid overlap in the cramped cosine range
    poster_children = [
        ('HSPA5', 'MANF', -0.048, 0.353,
         'Anti-correlated ER\nstress partners',
         -0.12, 0.16, 'left', 'top'),       # bottom-left, long leader up-right
        ('C7ORF26', 'INTS1', 0.074, 0.393,
         'Reclassified\nIntegrator subunit',
         0.28, 0.30, 'left', 'center'),      # mid-right, horizontal leader
        ('SNAPC2', 'SNAPC4', 0.167, 0.471,
         'Invisible 4th\ncomplex subunit',
         0.32, 0.52, 'left', 'bottom'),      # top-right
    ]

    ax2.plot([-0.15, 0.55], [-0.15, 0.55], '--', color='black',
             linewidth=0.8, alpha=0.3, zorder=1)
    ax2.text(0.30, 0.26, 'Assoc = Cosine', fontsize=7.5, color='grey', alpha=0.7,
             rotation=40, ha='left', va='bottom')
    ax2.axhline(0, color='black', linestyle='--', linewidth=0.6, alpha=0.3, zorder=1)

    for ga, gb, cos_v, assoc_v, desc, txt_x, txt_y, ha, va in poster_children:
        ax2.scatter(cos_v, cos_v, s=50, c=GREY, edgecolors='black', linewidths=0.6,
                   zorder=3, marker='o')
        ax2.scatter(cos_v, assoc_v, s=70, c=BLUE, edgecolors='black', linewidths=0.8,
                   zorder=4, marker='o')
        ax2.annotate('', xy=(cos_v, assoc_v - 0.008),
                    xytext=(cos_v, cos_v + 0.008),
                    arrowprops=dict(arrowstyle='->', color=BLUE, lw=2.0,
                                   shrinkA=2, shrinkB=2),
                    zorder=3)
        ax2.annotate(f'{ga}–{gb}\n{desc}',
                    xy=(cos_v, assoc_v),
                    xytext=(txt_x, txt_y),
                    fontsize=8, ha=ha, va=va,
                    arrowprops=dict(arrowstyle='->', color='black', lw=0.8,
                                   shrinkA=0, shrinkB=4),
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                             edgecolor='#CCCCCC', alpha=0.95),
                    zorder=6)

    ax2.set_xlabel('Expression Cosine Similarity')
    ax2.set_ylabel('CAL Association Score')
    ax2.set_xlim(-0.15, 0.55)
    ax2.set_ylim(-0.15, 0.55)

    grey_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=GREY,
                           markeredgecolor='black', markersize=7, label='If assoc = cosine')
    blue_dot = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=BLUE,
                           markeredgecolor='black', markersize=7, label='Actual CAL score')
    ax2.legend(handles=[grey_dot, blue_dot], loc='lower right', frameon=False, fontsize=8.5)
    ax2.text(-0.03, 1.05, 'B', transform=ax2.transAxes, fontsize=16, fontweight='bold')

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig3_distribution_and_examples.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig3_distribution_and_examples.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Fig 3 saved.")


def fig4_degree_dependence():
    """Figure 4: Degree-Dependence Quintile Bar Chart (cross-boundary only)."""
    csv_path = GENE_AAR_RESULTS / 'degree_vs_improvement_pairs.csv'
    df = pd.read_csv(csv_path)

    # Filter to cross-boundary pairs (|cosine| < 0.2)
    cb = df[df['cosine'].abs() < 0.2].copy()
    print(f"  Fig 4: {len(cb)} cross-boundary pairs from {len(df)} total")

    # Degree quintiles
    cb['quintile'] = pd.qcut(cb['mean_deg'], 5, labels=False)
    quintile_stats = cb.groupby('quintile').agg(
        mean_cos=('cosine', 'mean'),
        mean_assoc=('assoc', 'mean'),
        mean_improvement=('improvement', 'mean'),
        deg_min=('mean_deg', 'min'),
        deg_max=('mean_deg', 'max'),
        n=('cosine', 'count'),
    ).reset_index()

    print("\n  Cross-boundary quintile analysis:")
    print(f"  {'Q':>3} {'Deg range':>15} {'N':>6} {'Cos':>8} {'Assoc':>8} {'Δ':>8}")
    for _, row in quintile_stats.iterrows():
        q = int(row['quintile']) + 1
        print(f"  Q{q}  {row['deg_min']:.0f}–{row['deg_max']:.0f}  "
              f"{row['n']:>6.0f}  {row['mean_cos']:>8.3f}  {row['mean_assoc']:>8.3f}  "
              f"+{row['mean_improvement']:>7.3f}")

    labels = []
    values = []
    colors = ['#08519C', '#2171B5', '#4292C6', '#6BAED6', '#9ECAE1']
    for _, row in quintile_stats.iterrows():
        q = int(row['quintile']) + 1
        labels.append(f"Q{q}\n({row['deg_min']:.0f}–{row['deg_max']:.0f})")
        values.append(row['mean_improvement'])

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=0.6, width=0.65)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'+{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('Degree quintile (STRING interactions)', fontsize=12)
    ax.set_ylabel('Cross-boundary improvement (Δ)', fontsize=12)
    ax.set_ylim(0, max(values) * 1.25)
    ax.tick_params(labelsize=10)

    # Spearman correlation on CB pairs
    from scipy.stats import spearmanr
    rho, pval = spearmanr(cb['mean_deg'], cb['improvement'])
    ax.text(0.95, 0.95, f'Spearman r = {rho:.3f}\nn = {len(cb):,}',
            transform=ax.transAxes, fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', edgecolor='grey'))

    # Subtitle: data source clarification
    ax.set_title('High confidence (≥900), cross-boundary pairs (|cosine| < 0.2)',
                 fontsize=9, color='#555555', pad=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig4_degree_dependence.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig4_degree_dependence.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Fig 4 saved.")


def fig5_cb_sensitivity():
    """Figure 5 (Appendix): Cross-Boundary Sensitivity — CAL holds steady while cosine collapses."""
    thresholds = [0.30, 0.20, 0.15, 0.10, 0.05]
    cosine_auc = [0.540, 0.518, 0.516, 0.510, 0.491]
    cal_auc = [0.907, 0.908, 0.909, 0.914, 0.915]

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.fill_between(thresholds, cosine_auc, cal_auc, alpha=0.15, color=BLUE, zorder=1)
    ax.plot(thresholds, cal_auc, 'o-', color=BLUE, linewidth=2.0, markersize=7,
            label='CAL (assoc-only)', zorder=3)
    ax.plot(thresholds, cosine_auc, 's-', color=GREY, linewidth=2.0, markersize=7,
            label='Cosine baseline', zorder=3)
    ax.axhline(0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5, zorder=1)
    ax.text(0.295, 0.505, 'chance', fontsize=9, color='black', alpha=0.6)

    ax.set_xlabel('Cross-boundary threshold (|cosine| < t)')
    ax.set_ylabel('AUC')
    ax.set_ylim(0.4, 1.0)
    ax.set_xlim(0.32, 0.03)  # Reversed: tighter thresholds to the right
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f'{t:.2f}' for t in thresholds])
    ax.legend(loc='upper left', frameon=False, fontsize=10)

    # Delta annotation at tightest threshold — offset to avoid legend
    ax.annotate(f'Δ = +{cal_auc[-1] - cosine_auc[-1]:.3f}',
                xy=(0.05, (cal_auc[-1] + cosine_auc[-1]) / 2),
                xytext=(0.12, 0.62),
                fontsize=10, ha='center', color=BLUE,
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.0))

    fig.tight_layout()
    fig.savefig(FIG_DIR / 'fig5_cb_sensitivity.pdf', bbox_inches='tight')
    fig.savefig(FIG_DIR / 'fig5_cb_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("  Fig 5 saved.")


if __name__ == '__main__':
    print("Generating figures (v3, λ=1.0)...")
    fig1_cross_boundary_overview()
    fig2_threshold_sweep()
    fig3_distribution_and_examples()
    fig4_degree_dependence()
    fig5_cb_sensitivity()

    print("\nOutput files:")
    for f in sorted(FIG_DIR.iterdir()):
        if f.suffix in ('.pdf', '.png'):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name}: {size_kb:.1f} KB")
