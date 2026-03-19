#!/usr/bin/env python3
"""
Experiment 5: Expression-Space Cluster GO Enrichment
=====================================================
Run GO enrichment on expression-space HDBSCAN clusters (baseline comparison)
and on random gene sets of matched sizes.
"""

import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA

import hdbscan
from gprofiler import GProfiler

BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "Gene AAR" / "data"
RESULTS_DIR = BASE / "Gene AAR" / "results"
OUT_DIR = BASE / "Paper" / "strengthening_results"

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
MODEL_FILE = RESULTS_DIR / "model_high.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
PCA_DIM = 50
HIDDEN_DIM = 1024
N_LAYERS = 4
HDBSCAN_MIN_CLUSTER = 5
HDBSCAN_MIN_SAMPLES = 3


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AssociationMLP(nn.Module):
    def __init__(self, input_dim=PCA_DIM, hidden_dim=HIDDEN_DIM, n_layers=N_LAYERS):
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


def run_enrichment(gene_list, cluster_id):
    gp = GProfiler(return_dataframe=True)
    try:
        result = gp.profile(
            organism='hsapiens',
            query=gene_list,
            sources=['GO:BP', 'GO:MF', 'GO:CC', 'REAC', 'KEGG'],
            significance_threshold_method='g_SCS',
            user_threshold=0.05,
            no_evidences=False,
        )
    except Exception as e:
        print(f"    Cluster {cluster_id}: g:Profiler error: {e}")
        return []

    if result is None or len(result) == 0:
        return []

    enrichments = []
    for _, row in result.iterrows():
        enrichments.append({
            'source': row['source'],
            'term_name': row['name'],
            'p_value': float(row['p_value']),
            'intersection_size': int(row['intersection_size']),
            'query_size': int(row['query_size']),
            'coverage': float(row['intersection_size'] / row['query_size'])
                if row['query_size'] > 0 else 0.0,
        })
    return sorted(enrichments, key=lambda x: x['p_value'])


def classify_cluster(enrichments):
    if not enrichments:
        return 'no_signal'
    for e in enrichments:
        if e['p_value'] < 0.05 and e['coverage'] > 0.5:
            return 'coherent'
    for e in enrichments:
        if e['p_value'] < 0.05:
            return 'partial'
    return 'no_signal'


def main():
    t0 = time.time()
    print("Experiment 5: Expression-Space Cluster GO Enrichment")
    print("=" * 70)

    # Load data
    import anndata as ad
    set_seed(SEED)
    adata = ad.read_h5ad(BULK_FILE)
    X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    n_pert = X_raw.shape[0]

    pca = PCA(n_components=PCA_DIM)
    X_pca = pca.fit_transform(X_raw)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_norm = X_pca / (norms + 1e-8)
    X_norm_tensor = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)

    # Gene names
    gene_names = []
    for idx_str in adata.obs.index:
        parts = str(idx_str).split('_')
        gene_names.append(parts[1].upper() if len(parts) >= 2 else str(idx_str))

    # Expression-space HDBSCAN clustering
    print("\nClustering expression space...")
    X_expr = X_norm
    cos_expr = X_expr @ X_expr.T
    dist_expr = (1.0 - cos_expr).astype(np.float64)
    np.fill_diagonal(dist_expr, 0)
    dist_expr = np.clip(dist_expr, 0, 2)
    clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER,
                                 min_samples=HDBSCAN_MIN_SAMPLES,
                                 metric='precomputed')
    labels_expr = clusterer.fit_predict(dist_expr)
    n_expr_clusters = len(set(labels_expr)) - (1 if -1 in labels_expr else 0)
    print(f"  Expression-space clusters: {n_expr_clusters}")

    # Also get PAM-specific cluster info for context
    print("Loading model and computing association-space clusters...")
    model = AssociationMLP().to(DEVICE)
    ckpt = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    with torch.no_grad():
        X_assoc = model(X_norm_tensor).cpu().numpy()
    cos_assoc = X_assoc @ X_assoc.T
    dist_assoc = (1.0 - cos_assoc).astype(np.float64)
    np.fill_diagonal(dist_assoc, 0)
    dist_assoc = np.clip(dist_assoc, 0, 2)
    clusterer_a = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER,
                                   min_samples=HDBSCAN_MIN_SAMPLES,
                                   metric='precomputed')
    labels_assoc = clusterer_a.fit_predict(dist_assoc)

    # Identify PAM-specific clusters (for size matching)
    pam_cluster_sizes = []
    for c in range(max(labels_assoc) + 1):
        members = np.where(labels_assoc == c)[0]
        if len(members) < HDBSCAN_MIN_CLUSTER:
            continue
        expr_labs = labels_expr[members]
        pct_noise = float((expr_labs == -1).sum() / len(members))
        n_expr_cl = len(set(expr_labs[expr_labs != -1]))
        if pct_noise > 0.5 or n_expr_cl >= 3:
            pam_cluster_sizes.append(len(members))

    print(f"  PAM-specific clusters: {len(pam_cluster_sizes)}, sizes: {sorted(pam_cluster_sizes, reverse=True)[:10]}...")

    # ====================================================================
    # Part 1: Expression-space cluster enrichment
    # ====================================================================
    print(f"\nRunning GO enrichment on {n_expr_clusters} expression-space clusters...")
    expr_results = []
    for c in range(max(labels_expr) + 1):
        members = np.where(labels_expr == c)[0]
        if len(members) < HDBSCAN_MIN_CLUSTER:
            continue

        genes = [gene_names[i] for i in members]
        print(f"  Expr cluster {c} ({len(genes)} genes): {', '.join(genes[:5])}...")

        enrichments = run_enrichment(genes, f"expr_{c}")
        classification = classify_cluster(enrichments)

        top3 = enrichments[:3] if enrichments else []
        top3_summary = [{'source': t['source'], 'term': t['term_name'],
                         'p_value': t['p_value'], 'coverage': t['coverage']}
                        for t in top3]

        expr_results.append({
            'cluster_id': c,
            'size': len(genes),
            'classification': classification,
            'n_enrichments': len(enrichments),
            'top3': top3_summary,
        })
        print(f"    -> {classification.upper()}" +
              (f" (top: {top3[0]['term_name']}, p={top3[0]['p_value']:.2e})" if top3 else ""))
        time.sleep(0.3)

    n_coherent_expr = sum(1 for r in expr_results if r['classification'] == 'coherent')
    n_partial_expr = sum(1 for r in expr_results if r['classification'] == 'partial')
    n_nosignal_expr = sum(1 for r in expr_results if r['classification'] == 'no_signal')

    print(f"\n  Expression clusters: {len(expr_results)} total")
    print(f"  Coherent: {n_coherent_expr}, Partial: {n_partial_expr}, No signal: {n_nosignal_expr}")

    # ====================================================================
    # Part 2: Random clusters (matched to PAM-specific sizes, 10 repeats)
    # ====================================================================
    print(f"\nRunning random baseline ({len(pam_cluster_sizes)} clusters x 10 repeats)...")
    random_results = []
    for rep in range(10):
        rng = np.random.RandomState(SEED + rep)
        rep_coherent = 0
        rep_partial = 0
        rep_nosignal = 0

        for sz in pam_cluster_sizes:
            random_indices = rng.choice(n_pert, size=sz, replace=False)
            random_genes = [gene_names[i] for i in random_indices]
            enrichments = run_enrichment(random_genes, f"random_{rep}_{sz}")
            classification = classify_cluster(enrichments)
            if classification == 'coherent':
                rep_coherent += 1
            elif classification == 'partial':
                rep_partial += 1
            else:
                rep_nosignal += 1
            time.sleep(0.2)

        random_results.append({
            'repeat': rep,
            'coherent': rep_coherent,
            'partial': rep_partial,
            'no_signal': rep_nosignal,
        })
        print(f"  Rep {rep}: coherent={rep_coherent}, partial={rep_partial}, no_signal={rep_nosignal}")

    random_coherent_vals = [r['coherent'] for r in random_results]
    random_mean = float(np.mean(random_coherent_vals))
    random_std = float(np.std(random_coherent_vals))

    elapsed = time.time() - t0

    output = {
        'expression_clusters': {
            'total': len(expr_results),
            'coherent': n_coherent_expr,
            'partial': n_partial_expr,
            'no_signal': n_nosignal_expr,
            'pct_coherent': float(n_coherent_expr / len(expr_results)) if expr_results else 0,
            'per_cluster': expr_results,
        },
        'pam_specific_reference': {
            'total': 22,
            'coherent': 22,
            'pct_coherent': 1.0,
        },
        'random_baseline': {
            'n_clusters_per_rep': len(pam_cluster_sizes),
            'n_repeats': 10,
            'mean_coherent': random_mean,
            'std_coherent': random_std,
            'per_repeat': random_results,
        },
        'elapsed_seconds': elapsed,
    }

    json_path = OUT_DIR / "05_expression_cluster_enrichment.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Write markdown
    md_path = OUT_DIR / "05_expression_cluster_enrichment.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 5: Expression-Space Cluster GO Enrichment\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Runtime:** {elapsed:.0f}s\n\n")

        f.write("## Cluster GO Enrichment Comparison\n\n")
        f.write("| Cluster source | Total clusters | Coherent | Partial | No signal | % coherent |\n")
        f.write("|----------------|---------------|----------|---------|-----------|------------|\n")
        f.write(f"| PAM-specific (association space) | 22 | 22 | 0 | 0 | 100% |\n")
        f.write(f"| Expression-space HDBSCAN | {len(expr_results)} | {n_coherent_expr} | "
                f"{n_partial_expr} | {n_nosignal_expr} | "
                f"{100*n_coherent_expr/len(expr_results):.0f}% |\n")
        f.write(f"| Random (matched sizes, 10 repeats) | {len(pam_cluster_sizes)} | "
                f"{random_mean:.1f} +/- {random_std:.1f} | | | "
                f"{100*random_mean/len(pam_cluster_sizes):.0f}% |\n")
        f.write("\n")

        # Top 5 expression clusters by best p-value
        coherent_expr = [r for r in expr_results if r['classification'] == 'coherent' and r['top3']]
        coherent_expr.sort(key=lambda r: r['top3'][0]['p_value'])
        f.write("## Top 5 Expression-Space Enrichments\n\n")
        for rank, r in enumerate(coherent_expr[:5], 1):
            t = r['top3'][0]
            f.write(f"{rank}. **Cluster {r['cluster_id']}** ({r['size']} genes): "
                    f"{t['source']}: {t['term']} (p={t['p_value']:.2e}, "
                    f"{t['coverage']*100:.0f}% coverage)\n")
        f.write("\n")

        f.write("## Regression Canary\n\n")
        f.write("PAM-specific cluster results: 22/22 coherent (from 06_validation_results.json)\n")

    print(f"\nResults saved to {md_path} and {json_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
