#!/usr/bin/env python3
"""
PAM-Genetics Phase 6: Cluster Validation via GO/Pathway Enrichment
===================================================================

For each PAM-specific cluster from the high-confidence (900) model:
  1. Extract gene names
  2. Run GO enrichment via g:Profiler API (GO:BP, GO:MF, Reactome, KEGG)
  3. Classify as Coherent / Partial / No signal
  4. Flag hub-driven clusters
  5. Produce summary table and paper-ready examples

Usage:
    python scripts/06_cluster_validation.py
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

import hdbscan
from gprofiler import GProfiler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PCA_DIM = 50
HIDDEN_DIM = 1024
N_LAYERS = 4
HDBSCAN_MIN_CLUSTER = 5
HDBSCAN_MIN_SAMPLES = 3
HUB_THRESHOLD = 50  # gene with >50 STRING interactions = hub

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
MODEL_FILE = RESULTS_DIR / "model_high.pt"
STRING_LINKS = DATA_DIR / "9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO = DATA_DIR / "9606.protein.info.v12.0.txt.gz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Model (same as 03/05)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Hub gene detection from full STRING network
# ---------------------------------------------------------------------------
def count_string_degrees(gene_set):
    """Count STRING interactions per gene across the FULL human network."""
    print("  Loading STRING protein info for ENSP->gene mapping...")
    info_df = pd.read_csv(STRING_INFO, sep='\t')
    ensp_to_gene = {}
    for _, row in info_df.iterrows():
        g = str(row['preferred_name']).upper()
        ensp_to_gene[row['#string_protein_id']] = g

    print("  Counting STRING degrees (full network, score >= 700)...")
    degree = defaultdict(int)
    gene_set_upper = {g.upper() for g in gene_set}
    chunk_iter = pd.read_csv(STRING_LINKS, sep=' ', chunksize=500000)
    for chunk in chunk_iter:
        chunk = chunk[chunk['combined_score'] >= 700]
        for _, row in chunk.iterrows():
            g1 = ensp_to_gene.get(row['protein1'], '')
            g2 = ensp_to_gene.get(row['protein2'], '')
            if g1 in gene_set_upper:
                degree[g1] += 1
            if g2 in gene_set_upper:
                degree[g2] += 1
    return degree


# ---------------------------------------------------------------------------
# GO/Pathway enrichment via g:Profiler
# ---------------------------------------------------------------------------
def run_enrichment(gene_list, cluster_id):
    """Query g:Profiler for a gene list. Returns list of enrichment results."""
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
            'term_id': row['native'],
            'term_name': row['name'],
            'p_value': float(row['p_value']),
            'term_size': int(row['term_size']),
            'intersection_size': int(row['intersection_size']),
            'query_size': int(row['query_size']),
            'coverage': float(row['intersection_size'] / row['query_size'])
                if row['query_size'] > 0 else 0.0,
            'intersections': row.get('intersections', ''),
        })
    return sorted(enrichments, key=lambda x: x['p_value'])


def classify_cluster(enrichments, n_genes):
    """Classify a cluster based on its enrichment results."""
    if not enrichments:
        return 'no_signal'
    # Check for any term covering >50% of cluster genes
    for e in enrichments:
        if e['p_value'] < 0.05 and e['coverage'] > 0.5:
            return 'coherent'
    # Has significant terms but none cover >50%
    for e in enrichments:
        if e['p_value'] < 0.05:
            return 'partial'
    return 'no_signal'


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    print("PAM-Genetics Phase 6: Cluster Validation (GO/Pathway Enrichment)")
    print("=" * 70)

    # --- Load expression data and PCA ---
    print("\nLoading expression data...")
    import anndata as ad
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
    print(f"  {n_pert} perturbations, PCA-{PCA_DIM}")

    # --- Load high-confidence model ---
    print("\nLoading high-confidence model...")
    model = AssociationMLP().to(DEVICE)
    ckpt = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Loaded {MODEL_FILE}")

    # --- Compute association embeddings + HDBSCAN ---
    print("\nComputing association embeddings and clustering...")
    with torch.no_grad():
        X_assoc = model(X_norm_tensor).cpu().numpy()
    X_expr = X_norm_tensor.cpu().numpy()

    cos_expr = X_expr @ X_expr.T
    cos_assoc = X_assoc @ X_assoc.T

    labels_dict = {}
    for name, cos_mat in [('expression', cos_expr), ('association', cos_assoc)]:
        dist = (1.0 - cos_mat).astype(np.float64)
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, 2)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER,
                                     min_samples=HDBSCAN_MIN_SAMPLES,
                                     metric='precomputed')
        labels = clusterer.fit_predict(dist)
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        labels_dict[name] = labels
        print(f"  {name}: {n_cl} clusters")

    # --- Identify PAM-specific clusters ---
    print("\nIdentifying PAM-specific clusters...")
    labels_a = labels_dict['association']
    labels_e = labels_dict['expression']

    pam_clusters = []
    for c in range(max(labels_a) + 1):
        members = np.where(labels_a == c)[0]
        if len(members) < HDBSCAN_MIN_CLUSTER:
            continue
        expr_labs = labels_e[members]
        pct_noise = float((expr_labs == -1).sum() / len(members))
        n_expr_cl = len(set(expr_labs[expr_labs != -1]))

        if pct_noise > 0.5 or n_expr_cl >= 3:
            genes = [gene_names[i] for i in members]
            pam_clusters.append({
                'cluster_id': c,
                'size': len(members),
                'genes': sorted(genes),
                'pct_noise_expr': pct_noise,
                'n_expr_clusters': n_expr_cl,
            })

    print(f"  Found {len(pam_clusters)} PAM-specific clusters")

    # --- Collect all cluster genes for hub detection ---
    all_cluster_genes = set()
    for pc in pam_clusters:
        all_cluster_genes.update(pc['genes'])

    # --- Count STRING degrees for hub detection ---
    print("\nCounting STRING degrees for hub gene detection...")
    degree = count_string_degrees(all_cluster_genes)
    hub_genes = {g for g, d in degree.items() if d > HUB_THRESHOLD}
    print(f"  {len(hub_genes)} hub genes (>{HUB_THRESHOLD} interactions) among cluster members")

    # --- Run g:Profiler enrichment for each cluster ---
    print(f"\nRunning GO/pathway enrichment for {len(pam_clusters)} clusters...")
    print("  (querying g:Profiler API -- may take a moment)")

    all_results = []
    for i, pc in enumerate(pam_clusters):
        cid = pc['cluster_id']
        genes = pc['genes']
        print(f"\n  Cluster {cid} ({pc['size']} genes): {', '.join(genes[:6])}"
              + (f"..." if len(genes) > 6 else ""))

        enrichments = run_enrichment(genes, cid)
        classification = classify_cluster(enrichments, len(genes))

        # Hub analysis
        cluster_hubs = [g for g in genes if g in hub_genes]
        pct_hub = len(cluster_hubs) / len(genes) if genes else 0
        hub_driven = pct_hub > 0.5

        # Top 3 terms
        top3 = enrichments[:3] if enrichments else []
        top3_summary = []
        for t in top3:
            top3_summary.append({
                'source': t['source'],
                'term': t['term_name'],
                'p_value': t['p_value'],
                'coverage': t['coverage'],
                'intersection': t['intersection_size'],
            })

        status_str = classification.upper()
        if hub_driven:
            status_str += " (hub-driven)"
        print(f"    Classification: {status_str}")
        if top3:
            for t in top3_summary:
                print(f"    {t['source']}: {t['term']} "
                      f"(p={t['p_value']:.2e}, {t['intersection']}/{pc['size']} genes)")
        else:
            print(f"    No significant enrichments")

        result = {
            'cluster_id': cid,
            'size': pc['size'],
            'genes': genes,
            'pct_noise_expr': pc['pct_noise_expr'],
            'classification': classification,
            'n_enrichments': len(enrichments),
            'top3': top3_summary,
            'all_enrichments': enrichments[:10],  # keep top 10 for JSON
            'hub_genes': cluster_hubs,
            'pct_hub': pct_hub,
            'hub_driven': hub_driven,
        }
        all_results.append(result)
        # Brief pause to be polite to g:Profiler API
        time.sleep(0.5)

    # ------------------------------------------------------------------
    # Summary statistics
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_coherent = sum(1 for r in all_results if r['classification'] == 'coherent')
    n_partial = sum(1 for r in all_results if r['classification'] == 'partial')
    n_no_signal = sum(1 for r in all_results if r['classification'] == 'no_signal')
    n_hub_driven = sum(1 for r in all_results if r['hub_driven'])
    n_total = len(all_results)

    print(f"\n  Total PAM-specific clusters: {n_total}")
    print(f"  Coherent (p<0.05, >50% coverage):  {n_coherent} ({100*n_coherent/n_total:.0f}%)")
    print(f"  Partial  (p<0.05, <50% coverage):  {n_partial} ({100*n_partial/n_total:.0f}%)")
    print(f"  No signal:                         {n_no_signal} ({100*n_no_signal/n_total:.0f}%)")
    print(f"\n  Hub-driven clusters (>50% hubs):   {n_hub_driven} ({100*n_hub_driven/n_total:.0f}%)")

    # Coherent but NOT hub-driven = strongest evidence
    n_coherent_non_hub = sum(1 for r in all_results
                             if r['classification'] == 'coherent' and not r['hub_driven'])
    print(f"  Coherent & non-hub-driven:         {n_coherent_non_hub}")

    # ------------------------------------------------------------------
    # Top 5 most coherent clusters for paper
    # ------------------------------------------------------------------
    coherent_clusters = [r for r in all_results if r['classification'] == 'coherent']
    # Sort by best p-value
    coherent_clusters.sort(key=lambda r: r['top3'][0]['p_value'] if r['top3'] else 1.0)
    top5 = coherent_clusters[:5]

    print(f"\n{'='*70}")
    print("TOP 5 COHERENT CLUSTERS (paper examples)")
    print("=" * 70)
    for rank, r in enumerate(top5, 1):
        print(f"\n  #{rank}: Cluster {r['cluster_id']} ({r['size']} genes)")
        print(f"  Genes: {', '.join(r['genes'])}")
        print(f"  Hub genes: {', '.join(r['hub_genes']) if r['hub_genes'] else 'none'}")
        for t in r['top3']:
            print(f"    {t['source']}: {t['term']}")
            print(f"      p={t['p_value']:.2e}, {t['intersection']}/{r['size']} genes "
                  f"({t['coverage']*100:.0f}% coverage)")

    # ------------------------------------------------------------------
    # Save full results JSON
    # ------------------------------------------------------------------
    output = {
        'model': str(MODEL_FILE),
        'n_clusters': n_total,
        'summary': {
            'coherent': n_coherent,
            'partial': n_partial,
            'no_signal': n_no_signal,
            'hub_driven': n_hub_driven,
            'coherent_non_hub': n_coherent_non_hub,
        },
        'clusters': all_results,
    }
    results_file = RESULTS_DIR / "06_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {results_file}")

    # ------------------------------------------------------------------
    # Save summary CSV
    # ------------------------------------------------------------------
    rows = []
    for r in all_results:
        top_term = r['top3'][0]['term'] if r['top3'] else ''
        top_p = r['top3'][0]['p_value'] if r['top3'] else None
        top_cov = r['top3'][0]['coverage'] if r['top3'] else None
        top_source = r['top3'][0]['source'] if r['top3'] else ''
        rows.append({
            'cluster_id': r['cluster_id'],
            'size': r['size'],
            'classification': r['classification'],
            'top_source': top_source,
            'top_term': top_term,
            'top_p_value': top_p,
            'top_coverage': top_cov,
            'n_enrichments': r['n_enrichments'],
            'n_hub_genes': len(r['hub_genes']),
            'pct_hub': r['pct_hub'],
            'hub_driven': r['hub_driven'],
            'genes': ';'.join(r['genes']),
        })
    df = pd.DataFrame(rows)
    csv_file = RESULTS_DIR / "06_validation_summary.csv"
    df.to_csv(csv_file, index=False)
    print(f"  Summary CSV saved to {csv_file}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
