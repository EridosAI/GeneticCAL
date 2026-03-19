#!/usr/bin/env python3
"""
PAM-Genetics Phase 4: Cluster Analysis -- Expression Space vs Association Space
================================================================================

Purpose: Run HDBSCAN clustering in both the original PCA-50 expression space
         and the PAM association space (post-model). Compare clusterings.
         Look for PAM-specific clusters: gene groups that cluster together in
         association space but NOT in expression space.

Methodology mirrors Replogle et al. 2022: Pearson correlation + HDBSCAN.
We use cosine similarity (equivalent to Pearson on Z-scored data) + HDBSCAN.

Usage:
    python scripts/04_cluster_analysis.py
"""

import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import hdbscan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PCA_DIM = 50
HDBSCAN_MIN_CLUSTER = 5
HDBSCAN_MIN_SAMPLES = 3

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
MODEL_FILE = RESULTS_DIR / "model_medium.pt"
PAIRS_CSV = DATA_DIR / "string_pairs_medium.csv"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model (must match 03_train.py exactly)
# ---------------------------------------------------------------------------
class AssociationMLP(nn.Module):
    def __init__(self, input_dim=PCA_DIM, hidden_dim=1024, n_layers=4):
        super().__init__()
        self.alpha_logit = nn.Parameter(torch.tensor(0.0))
        layers = []
        for i in range(n_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = input_dim if i == n_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_d, out_d))
            layers.append(nn.LayerNorm(out_d))
            if i < n_layers - 1:
                layers.append(nn.GELU())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha_logit)
        g = self.mlp(x)
        out = alpha * x + (1 - alpha) * g
        return F.normalize(out, dim=-1)


# ---------------------------------------------------------------------------
# Step 1: Load everything
# ---------------------------------------------------------------------------
def step1_load():
    print("=" * 70)
    print("STEP 1: LOAD DATA, MODEL, STRING PAIRS")
    print("=" * 70)

    # Expression data
    import anndata as ad
    adata = ad.read_h5ad(BULK_FILE)
    X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    n = X_raw.shape[0]
    print(f"  Expression data: {n} perturbations x {X_raw.shape[1]} genes")

    # Extract gene names from index
    gene_names = []
    for idx_str in adata.obs.index:
        parts = str(idx_str).split('_')
        gene_names.append(parts[1].upper() if len(parts) >= 2 else str(idx_str))
    print(f"  Gene names extracted: {len(gene_names)}")

    # PCA-50 + L2 normalize (use saved PCA from model checkpoint for exact match)
    ckpt = torch.load(MODEL_FILE, map_location='cpu', weights_only=False)
    pca_components = ckpt['pca_components']  # (50, 8563)
    pca_mean = ckpt['pca_mean']              # (8563,)

    X_pca = (X_raw - pca_mean) @ pca_components.T
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_expr = X_pca / (norms + 1e-8)  # L2-normalized expression embeddings
    print(f"  PCA-{PCA_DIM} embeddings: {X_expr.shape}")

    # Load model
    model = AssociationMLP().to(DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    alpha = torch.sigmoid(model.alpha_logit).item()
    print(f"  Model loaded (alpha={alpha:.3f})")

    # Compute PAM association embeddings
    X_expr_tensor = torch.tensor(X_expr, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        X_assoc_tensor = model(X_expr_tensor)
    X_assoc = X_assoc_tensor.cpu().numpy()
    print(f"  Association embeddings: {X_assoc.shape}")

    # STRING pairs
    string_df = pd.read_csv(PAIRS_CSV)
    print(f"  STRING pairs: {len(string_df)}")

    # Build STRING adjacency for gene-level lookup
    string_adj = defaultdict(set)
    for _, row in string_df.iterrows():
        string_adj[row['gene1']].add(row['gene2'])
        string_adj[row['gene2']].add(row['gene1'])

    # Check for Replogle cluster labels in metadata
    cluster_col = None
    for col in adata.obs.columns:
        cl = col.lower()
        if any(x in cl for x in ['cluster', 'group', 'module', 'category', 'class']):
            n_uniq = adata.obs[col].nunique()
            if 2 < n_uniq < 500:  # plausible cluster count
                cluster_col = col
                print(f"  Found potential cluster column: '{col}' ({n_uniq} unique)")
                break

    if cluster_col is None:
        print("  No Replogle cluster labels found in h5ad metadata.")
        print("  (Published clusters were in Supplementary Table 3, not in h5ad.)")

    replogle_labels = None
    if cluster_col:
        replogle_labels = adata.obs[cluster_col].values

    return X_expr, X_assoc, gene_names, string_adj, string_df, replogle_labels, n


# ---------------------------------------------------------------------------
# Step 2: Pairwise cosine similarity in both spaces
# ---------------------------------------------------------------------------
def step2_cosine_matrices(X_expr, X_assoc):
    print("\n" + "=" * 70)
    print("STEP 2: PAIRWISE COSINE SIMILARITY MATRICES")
    print("=" * 70)

    cos_expr = X_expr @ X_expr.T
    cos_assoc = X_assoc @ X_assoc.T

    # Correlation between the two matrices
    upper_expr = cos_expr[np.triu_indices(len(cos_expr), k=1)]
    upper_assoc = cos_assoc[np.triu_indices(len(cos_assoc), k=1)]
    corr = np.corrcoef(upper_expr, upper_assoc)[0, 1]

    print(f"  Expression cosine: mean={upper_expr.mean():.4f}, std={upper_expr.std():.4f}")
    print(f"  Association cosine: mean={upper_assoc.mean():.4f}, std={upper_assoc.std():.4f}")
    print(f"  Correlation between matrices: {corr:.4f}")

    # How much does the model rearrange similarities?
    big_change = np.abs(upper_assoc - upper_expr) > 0.3
    print(f"  Pairs with |delta cos| > 0.3: {big_change.sum():,} "
          f"({big_change.mean()*100:.1f}%)")

    return cos_expr, cos_assoc, corr


# ---------------------------------------------------------------------------
# Step 3: HDBSCAN clustering in both spaces
# ---------------------------------------------------------------------------
def step3_hdbscan(X_expr, X_assoc, cos_expr, cos_assoc):
    print("\n" + "=" * 70)
    print("STEP 3: HDBSCAN CLUSTERING")
    print("=" * 70)

    results = {}

    for name, X, cos_mat in [('expression', X_expr, cos_expr),
                              ('association', X_assoc, cos_assoc)]:
        print(f"\n  --- {name.upper()} space ---")

        # HDBSCAN on cosine distance = 1 - cosine_similarity
        dist_matrix = (1.0 - cos_mat).astype(np.float64)
        np.fill_diagonal(dist_matrix, 0)
        # Clip small negatives from floating point
        dist_matrix = np.clip(dist_matrix, 0, 2)

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=HDBSCAN_MIN_CLUSTER,
            min_samples=HDBSCAN_MIN_SAMPLES,
            metric='precomputed',
        )
        labels = clusterer.fit_predict(dist_matrix)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        n_clustered = (labels != -1).sum()

        cluster_sizes = []
        if n_clusters > 0:
            for c in range(max(labels) + 1):
                sz = (labels == c).sum()
                if sz > 0:
                    cluster_sizes.append(sz)

        print(f"    Clusters found: {n_clusters}")
        print(f"    Clustered genes: {n_clustered} ({n_clustered/len(labels)*100:.1f}%)")
        print(f"    Noise (unclustered): {n_noise} ({n_noise/len(labels)*100:.1f}%)")
        if cluster_sizes:
            print(f"    Cluster sizes: min={min(cluster_sizes)}, "
                  f"median={int(np.median(cluster_sizes))}, "
                  f"max={max(cluster_sizes)}")
            print(f"    Top 10 sizes: {sorted(cluster_sizes, reverse=True)[:10]}")

        results[name] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': int(n_noise),
            'n_clustered': int(n_clustered),
            'cluster_sizes': sorted(cluster_sizes, reverse=True),
        }

    # Compare clusterings
    labels_expr = results['expression']['labels']
    labels_assoc = results['association']['labels']

    # Only compute ARI/NMI on genes that are clustered in BOTH
    both_clustered = (labels_expr != -1) & (labels_assoc != -1)
    n_both = both_clustered.sum()
    print(f"\n  Genes clustered in both spaces: {n_both}")

    if n_both > 10:
        ari = adjusted_rand_score(labels_expr[both_clustered], labels_assoc[both_clustered])
        nmi = normalized_mutual_info_score(labels_expr[both_clustered],
                                           labels_assoc[both_clustered])
        print(f"  Adjusted Rand Index: {ari:.4f}")
        print(f"  Normalized Mutual Information: {nmi:.4f}")
        results['ari'] = float(ari)
        results['nmi'] = float(nmi)
    else:
        print("  Too few co-clustered genes for ARI/NMI.")
        results['ari'] = None
        results['nmi'] = None

    results['n_both_clustered'] = int(n_both)
    return results


# ---------------------------------------------------------------------------
# Step 4: Find PAM-specific clusters
# ---------------------------------------------------------------------------
def step4_pam_specific_clusters(cluster_results, gene_names, cos_expr, cos_assoc):
    print("\n" + "=" * 70)
    print("STEP 4: PAM-SPECIFIC CLUSTERS")
    print("(Genes grouped by association but NOT by expression)")
    print("=" * 70)

    labels_expr = cluster_results['expression']['labels']
    labels_assoc = cluster_results['association']['labels']

    # A PAM-specific cluster = an association cluster where members are spread
    # across multiple expression clusters or mostly noise in expression space
    pam_clusters = []

    if cluster_results['association']['n_clusters'] == 0:
        print("  No association clusters found.")
        return pam_clusters

    for c in range(max(labels_assoc) + 1):
        members = np.where(labels_assoc == c)[0]
        if len(members) < HDBSCAN_MIN_CLUSTER:
            continue

        # Where do these members sit in expression clustering?
        expr_labels_for_members = labels_expr[members]
        n_noise_in_expr = (expr_labels_for_members == -1).sum()
        unique_expr_clusters = set(expr_labels_for_members[expr_labels_for_members != -1])

        # Mean pairwise cosine within this cluster in both spaces
        member_cos_expr = []
        member_cos_assoc = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                member_cos_expr.append(cos_expr[members[i], members[j]])
                member_cos_assoc.append(cos_assoc[members[i], members[j]])

        mean_expr_cos = np.mean(member_cos_expr) if member_cos_expr else 0
        mean_assoc_cos = np.mean(member_cos_assoc) if member_cos_assoc else 0

        # PAM-specific criterion: low expression cohesion, high association cohesion
        # OR: most members are noise in expression space / spread across many expr clusters
        pct_noise_expr = n_noise_in_expr / len(members)
        fragmentation = len(unique_expr_clusters)  # how many expr clusters they span

        is_pam_specific = (
            (pct_noise_expr > 0.5) or  # most members unclustered in expression
            (fragmentation >= 3 and mean_expr_cos < 0.3)  # spread across many + low cohesion
        )

        info = {
            'assoc_cluster_id': int(c),
            'size': len(members),
            'member_indices': members.tolist(),
            'member_genes': [gene_names[i] for i in members],
            'mean_expr_cosine': float(mean_expr_cos),
            'mean_assoc_cosine': float(mean_assoc_cos),
            'pct_noise_in_expr': float(pct_noise_expr),
            'n_expr_clusters_spanned': int(fragmentation),
            'is_pam_specific': is_pam_specific,
        }

        if is_pam_specific:
            pam_clusters.append(info)

    # Sort by size
    pam_clusters.sort(key=lambda x: x['size'], reverse=True)

    if pam_clusters:
        print(f"  Found {len(pam_clusters)} PAM-specific clusters")
        for i, pc in enumerate(pam_clusters[:15]):
            print(f"\n  Cluster A{pc['assoc_cluster_id']} "
                  f"({pc['size']} genes):")
            print(f"    Expression cosine:   {pc['mean_expr_cosine']:.3f}")
            print(f"    Association cosine:   {pc['mean_assoc_cosine']:.3f}")
            print(f"    % noise in expr:     {pc['pct_noise_in_expr']*100:.0f}%")
            print(f"    Expr clusters spanned: {pc['n_expr_clusters_spanned']}")
            # Show gene names (first 20)
            genes_str = ', '.join(pc['member_genes'][:20])
            if len(pc['member_genes']) > 20:
                genes_str += f" ... (+{len(pc['member_genes'])-20} more)"
            print(f"    Genes: {genes_str}")
    else:
        print("  No PAM-specific clusters found.")
        print("  (All association clusters also cluster in expression space.)")

    return pam_clusters


# ---------------------------------------------------------------------------
# Step 5: Validate PAM-specific clusters against STRING
# ---------------------------------------------------------------------------
def step5_validate_with_string(pam_clusters, string_adj, string_df):
    print("\n" + "=" * 70)
    print("STEP 5: VALIDATE PAM-SPECIFIC CLUSTERS AGAINST STRING")
    print("=" * 70)

    if not pam_clusters:
        print("  No PAM-specific clusters to validate.")
        return pam_clusters

    for pc in pam_clusters:
        genes = pc['member_genes']
        n = len(genes)

        # Count STRING edges within this cluster
        internal_edges = 0
        total_possible = n * (n - 1) // 2
        edge_list = []

        for i in range(n):
            for j in range(i + 1, n):
                if genes[j] in string_adj.get(genes[i], set()):
                    internal_edges += 1
                    edge_list.append((genes[i], genes[j]))

        # Expected edges under random: (total_possible * total_string_edges / total_possible_pairs)
        # But simpler: just report density
        density = internal_edges / total_possible if total_possible > 0 else 0

        # Compare to background density
        # Background: ~41434 edges among ~1845 genes = 41434 / (1845*1844/2) = ~0.024
        bg_density = 41434 / (1845 * 1844 / 2)

        enrichment = density / bg_density if bg_density > 0 else 0

        pc['n_internal_string_edges'] = internal_edges
        pc['total_possible_edges'] = total_possible
        pc['string_density'] = float(density)
        pc['background_density'] = float(bg_density)
        pc['string_enrichment'] = float(enrichment)
        pc['top_string_edges'] = edge_list[:10]

        print(f"\n  Cluster A{pc['assoc_cluster_id']} ({n} genes):")
        print(f"    STRING edges within cluster: {internal_edges}/{total_possible}")
        print(f"    Density: {density:.4f} (background: {bg_density:.4f})")
        print(f"    Enrichment: {enrichment:.1f}x")
        if edge_list:
            print(f"    Example edges: {edge_list[:5]}")

    # Summary
    enriched = [pc for pc in pam_clusters if pc['string_enrichment'] > 2.0]
    print(f"\n  Summary: {len(enriched)}/{len(pam_clusters)} PAM-specific clusters "
          f"have STRING enrichment > 2x")

    return pam_clusters


# ---------------------------------------------------------------------------
# Step 6: Comprehensive cluster-level comparison
# ---------------------------------------------------------------------------
def step6_cluster_comparison(cluster_results, gene_names, cos_expr, cos_assoc, string_adj):
    print("\n" + "=" * 70)
    print("STEP 6: CLUSTER-LEVEL COMPARISON (all clusters)")
    print("=" * 70)

    all_cluster_info = {'expression': [], 'association': []}

    for space_name in ['expression', 'association']:
        labels = cluster_results[space_name]['labels']
        cos_mat = cos_expr if space_name == 'expression' else cos_assoc
        n_clusters = cluster_results[space_name]['n_clusters']

        if n_clusters == 0:
            continue

        for c in range(max(labels) + 1):
            members = np.where(labels == c)[0]
            if len(members) < HDBSCAN_MIN_CLUSTER:
                continue

            # Internal cohesion (mean pairwise cosine)
            pair_cos = []
            n_string = 0
            n_possible = 0
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    pair_cos.append(cos_mat[members[i], members[j]])
                    g1, g2 = gene_names[members[i]], gene_names[members[j]]
                    if g2 in string_adj.get(g1, set()):
                        n_string += 1
                    n_possible += 1

            mean_cos = np.mean(pair_cos) if pair_cos else 0
            string_frac = n_string / n_possible if n_possible > 0 else 0

            info = {
                'cluster_id': int(c),
                'size': len(members),
                'mean_internal_cosine': float(mean_cos),
                'string_fraction': float(string_frac),
                'genes_sample': [gene_names[i] for i in members[:10]],
            }
            all_cluster_info[space_name].append(info)

    for space_name in ['expression', 'association']:
        clusters = all_cluster_info[space_name]
        if not clusters:
            print(f"\n  {space_name.upper()}: no clusters")
            continue

        print(f"\n  {space_name.upper()} clusters ({len(clusters)} total):")
        print(f"    {'ID':>4s} {'Size':>5s} {'Cos':>6s} {'STRING%':>8s}  Sample genes")
        print(f"    {'-'*4} {'-'*5} {'-'*6} {'-'*8}  {'-'*40}")
        for ci in sorted(clusters, key=lambda x: x['size'], reverse=True)[:20]:
            genes_str = ', '.join(ci['genes_sample'][:5])
            print(f"    {ci['cluster_id']:>4d} {ci['size']:>5d} {ci['mean_internal_cosine']:>6.3f} "
                  f"{ci['string_fraction']*100:>7.1f}%  {genes_str}")

    return all_cluster_info


# ---------------------------------------------------------------------------
# Step 7: Plots
# ---------------------------------------------------------------------------
def step7_plots(cluster_results, cos_expr, cos_assoc, pam_clusters, all_cluster_info):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    print("\n" + "=" * 70)
    print("STEP 7: PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: t-SNE colored by expression clusters
    ax = axes[0, 0]
    labels_expr = cluster_results['expression']['labels']
    n_expr_clusters = cluster_results['expression']['n_clusters']

    # t-SNE on expression cosine distance
    dist_expr = 1.0 - cos_expr
    np.fill_diagonal(dist_expr, 0)
    dist_expr = np.clip(dist_expr, 0, 2)

    np.random.seed(42)
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, perplexity=30,
                init='random')
    coords = tsne.fit_transform(dist_expr)

    noise_mask = labels_expr == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1],
               c='lightgray', s=3, alpha=0.3, label='noise', rasterized=True)
    if n_expr_clusters > 0:
        clustered = ~noise_mask
        scatter = ax.scatter(coords[clustered, 0], coords[clustered, 1],
                             c=labels_expr[clustered], cmap='tab20', s=6, alpha=0.6,
                             rasterized=True)
    ax.set_title(f'Expression Space t-SNE\n({n_expr_clusters} clusters)')
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot 2: Same t-SNE but colored by association clusters
    ax = axes[0, 1]
    labels_assoc = cluster_results['association']['labels']
    n_assoc_clusters = cluster_results['association']['n_clusters']

    noise_mask_a = labels_assoc == -1
    ax.scatter(coords[noise_mask_a, 0], coords[noise_mask_a, 1],
               c='lightgray', s=3, alpha=0.3, label='noise', rasterized=True)
    if n_assoc_clusters > 0:
        clustered_a = ~noise_mask_a
        scatter_a = ax.scatter(coords[clustered_a, 0], coords[clustered_a, 1],
                               c=labels_assoc[clustered_a], cmap='tab20', s=6, alpha=0.6,
                               rasterized=True)
    ax.set_title(f'Association Clusters on Expression t-SNE\n({n_assoc_clusters} clusters)')
    ax.set_xticks([])
    ax.set_yticks([])

    # Plot 3: Cluster size distribution comparison
    ax = axes[1, 0]
    expr_sizes = cluster_results['expression']['cluster_sizes']
    assoc_sizes = cluster_results['association']['cluster_sizes']

    if expr_sizes and assoc_sizes:
        max_size = max(max(expr_sizes), max(assoc_sizes))
        bins = np.arange(0, min(max_size + 10, 200), 5)
        ax.hist(expr_sizes, bins=bins, alpha=0.5, color='steelblue',
                label=f'Expression ({len(expr_sizes)} clusters)')
        ax.hist(assoc_sizes, bins=bins, alpha=0.5, color='darkorange',
                label=f'Association ({len(assoc_sizes)} clusters)')
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('Count')
        ax.set_title('Cluster Size Distribution')
        ax.legend()
    elif expr_sizes:
        ax.hist(expr_sizes, bins=20, alpha=0.7, color='steelblue')
        ax.set_title(f'Expression Cluster Sizes ({len(expr_sizes)} clusters)\n(no association clusters found)')
    elif assoc_sizes:
        ax.hist(assoc_sizes, bins=20, alpha=0.7, color='darkorange')
        ax.set_title(f'Association Cluster Sizes ({len(assoc_sizes)} clusters)\n(no expression clusters found)')
    else:
        ax.text(0.5, 0.5, 'No clusters in either space', ha='center', va='center',
                transform=ax.transAxes)

    # Plot 4: PAM-specific clusters -- STRING enrichment
    ax = axes[1, 1]
    if pam_clusters:
        sizes = [pc['size'] for pc in pam_clusters]
        enrichments = [pc['string_enrichment'] for pc in pam_clusters]
        assoc_cos = [pc['mean_assoc_cosine'] for pc in pam_clusters]

        scatter = ax.scatter(sizes, enrichments, c=assoc_cos, cmap='YlOrRd',
                             s=50, alpha=0.8, edgecolors='black', linewidths=0.5)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Background rate')
        ax.axhline(2.0, color='red', linestyle='--', alpha=0.5, label='2x enrichment')
        ax.set_xlabel('Cluster size')
        ax.set_ylabel('STRING enrichment (fold over background)')
        ax.set_title(f'PAM-Specific Clusters ({len(pam_clusters)})\nSTRING Interaction Enrichment')
        ax.legend(fontsize=8)
        plt.colorbar(scatter, ax=ax, label='Mean assoc cosine')

        # Label top clusters
        for pc in pam_clusters[:5]:
            if pc['string_enrichment'] > 2:
                ax.annotate(f"A{pc['assoc_cluster_id']}\n({pc['size']}g)",
                            (pc['size'], pc['string_enrichment']),
                            fontsize=7, ha='center')
    else:
        ax.text(0.5, 0.5, 'No PAM-specific clusters found', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)

    plt.tight_layout()
    plot_file = RESULTS_DIR / "04_cluster_plots.png"
    plt.savefig(plot_file, dpi=150)
    print(f"  Plots saved to {plot_file}")
    plt.close()


# ---------------------------------------------------------------------------
# Step 8: Summary
# ---------------------------------------------------------------------------
def step8_summary(cluster_results, pam_clusters, all_cluster_info, corr):
    print("\n" + "=" * 70)
    print("STEP 8: SUMMARY")
    print("=" * 70)

    n_expr = cluster_results['expression']['n_clusters']
    n_assoc = cluster_results['association']['n_clusters']
    ari = cluster_results.get('ari')
    nmi = cluster_results.get('nmi')

    print(f"""
  Expression space:  {n_expr} clusters, {cluster_results['expression']['n_clustered']} genes clustered
  Association space: {n_assoc} clusters, {cluster_results['association']['n_clustered']} genes clustered

  Cosine matrix correlation: {corr:.4f}
  ARI (clustered in both):   {ari if ari is not None else 'N/A'}
  NMI (clustered in both):   {nmi if nmi is not None else 'N/A'}

  PAM-specific clusters:     {len(pam_clusters)}""")

    if pam_clusters:
        enriched = [pc for pc in pam_clusters if pc['string_enrichment'] > 2.0]
        total_genes = sum(pc['size'] for pc in pam_clusters)
        print(f"  Total genes in PAM-specific clusters: {total_genes}")
        print(f"  STRING-enriched (>2x): {len(enriched)}")

        if enriched:
            print(f"\n  Top STRING-enriched PAM-specific clusters:")
            for pc in sorted(enriched, key=lambda x: x['string_enrichment'], reverse=True)[:5]:
                print(f"    A{pc['assoc_cluster_id']}: {pc['size']} genes, "
                      f"{pc['string_enrichment']:.1f}x STRING enrichment, "
                      f"expr cos={pc['mean_expr_cosine']:.3f}, "
                      f"assoc cos={pc['mean_assoc_cosine']:.3f}")
                print(f"      Genes: {', '.join(pc['member_genes'][:10])}")
                if pc['top_string_edges']:
                    print(f"      STRING edges: {pc['top_string_edges'][:3]}")

    # Save results
    save_results = {
        'expression_clusters': cluster_results['expression']['n_clusters'],
        'expression_clustered': cluster_results['expression']['n_clustered'],
        'expression_noise': cluster_results['expression']['n_noise'],
        'expression_sizes': cluster_results['expression']['cluster_sizes'],
        'association_clusters': cluster_results['association']['n_clusters'],
        'association_clustered': cluster_results['association']['n_clustered'],
        'association_noise': cluster_results['association']['n_noise'],
        'association_sizes': cluster_results['association']['cluster_sizes'],
        'cosine_matrix_correlation': float(corr),
        'ari': ari,
        'nmi': nmi,
        'n_both_clustered': cluster_results['n_both_clustered'],
        'n_pam_specific_clusters': len(pam_clusters),
        'pam_specific_clusters': [
            {k: v for k, v in pc.items() if k != 'member_indices'}
            for pc in pam_clusters
        ],
        'hdbscan_params': {
            'min_cluster_size': HDBSCAN_MIN_CLUSTER,
            'min_samples': HDBSCAN_MIN_SAMPLES,
        },
    }

    results_file = RESULTS_DIR / "04_cluster_results.json"
    with open(results_file, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("PAM-Genetics Phase 4: Cluster Analysis")
    print("=" * 70)
    print()

    X_expr, X_assoc, gene_names, string_adj, string_df, replogle_labels, n = step1_load()
    cos_expr, cos_assoc, corr = step2_cosine_matrices(X_expr, X_assoc)
    cluster_results = step3_hdbscan(X_expr, X_assoc, cos_expr, cos_assoc)
    pam_clusters = step4_pam_specific_clusters(cluster_results, gene_names, cos_expr, cos_assoc)
    pam_clusters = step5_validate_with_string(pam_clusters, string_adj, string_df)
    all_cluster_info = step6_cluster_comparison(
        cluster_results, gene_names, cos_expr, cos_assoc, string_adj)
    step7_plots(cluster_results, cos_expr, cos_assoc, pam_clusters, all_cluster_info)
    step8_summary(cluster_results, pam_clusters, all_cluster_info, corr)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
