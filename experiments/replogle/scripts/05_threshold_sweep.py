#!/usr/bin/env python3
"""
PAM-Genetics Phase 5: Threshold Sweep -- Low / Medium / High STRING Confidence
===============================================================================

Train the same AAR architecture at three STRING confidence levels (400/700/900)
and compare: do tighter associations produce tighter, more biologically coherent
clusters? This is analogous to varying the temporal window width in the Bernard
world experiments.

For each threshold:
  1. Train model (same architecture, 100 epochs)
  2. Evaluate: overall AUC, cross-boundary AUC, easy/hard split
  3. Run HDBSCAN clustering in association space
  4. Count PAM-specific clusters and STRING enrichment

Usage:
    python scripts/05_threshold_sweep.py
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, adjusted_rand_score, normalized_mutual_info_score

import hdbscan

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
PCA_DIM = 50
HIDDEN_DIM = 1024
N_LAYERS = 4
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 100
TEMPERATURE = 0.05
LAMBDA_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

HDBSCAN_MIN_CLUSTER = 5
HDBSCAN_MIN_SAMPLES = 3

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THRESHOLDS = {
    'low': {'score': 400, 'pairs_file': DATA_DIR / "string_pairs_low.npy"},
    'medium': {'score': 700, 'pairs_file': DATA_DIR / "string_pairs_medium.npy"},
    'high': {'score': 900, 'pairs_file': DATA_DIR / "string_pairs_high.npy"},
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model (identical to 03_train.py)
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
# Training
# ---------------------------------------------------------------------------
def contrastive_loss(emb_a, emb_b, temperature=TEMPERATURE):
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_model(X_norm_tensor, pairs, epochs=EPOCHS):
    model = AssociationMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_pairs = len(pairs)
    losses = []

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_pairs)
        epoch_loss, n_batches = 0, 0
        for start in range(0, n_pairs, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 8:
                continue
            emb_a = X_norm_tensor[pairs[batch_idx, 0]]
            emb_b = X_norm_tensor[pairs[batch_idx, 1]]
            pred_a = model(emb_a)
            pred_b = model(emb_b)
            loss = contrastive_loss(pred_a, emb_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        losses.append(epoch_loss / max(n_batches, 1))

    # Training accuracy
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, n_pairs, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_pairs)
            emb_a = X_norm_tensor[pairs[start:end, 0]]
            emb_b = X_norm_tensor[pairs[start:end, 1]]
            pred_a = model(emb_a)
            sims = pred_a @ emb_b.T
            correct += (sims.argmax(dim=1) == torch.arange(end - start, device=DEVICE)).sum().item()
            total += end - start

    return model, losses, correct / total


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_scores(model, X_norm_tensor, idx_a, idx_b):
    """Returns (association_scores, cosine_scores) as numpy arrays."""
    model.eval()
    emb_a = X_norm_tensor[idx_a]
    emb_b = X_norm_tensor[idx_b]
    pred_a = model(emb_a)
    pred_b = model(emb_b)
    assoc = 0.5 * ((pred_a * emb_b).sum(dim=-1) + (pred_b * emb_a).sum(dim=-1))
    cosine = (emb_a * emb_b).sum(dim=-1)
    return assoc.cpu().numpy(), cosine.cpu().numpy()


def evaluate(model, X_norm_tensor, pos_pairs, n_pert):
    """Full evaluation: AUC sweep, cross-boundary, easy/hard."""
    pos_assoc, pos_cosine = compute_scores(model, X_norm_tensor,
                                           pos_pairs[:, 0], pos_pairs[:, 1])

    # Sample negatives
    n_neg = min(len(pos_pairs) * 5, 50000)
    pos_set = set()
    for i in range(len(pos_pairs)):
        pos_set.add((int(pos_pairs[i, 0]), int(pos_pairs[i, 1])))
        pos_set.add((int(pos_pairs[i, 1]), int(pos_pairs[i, 0])))

    rng = np.random.RandomState(42)
    neg_list = []
    while len(neg_list) < n_neg:
        i, j = rng.randint(0, n_pert), rng.randint(0, n_pert)
        if i != j and (i, j) not in pos_set:
            neg_list.append([i, j])
    neg_pairs = np.array(neg_list, dtype=np.int32)

    neg_assoc, neg_cosine = compute_scores(model, X_norm_tensor,
                                           neg_pairs[:, 0], neg_pairs[:, 1])

    labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])
    all_cos = np.concatenate([pos_cosine, neg_cosine])
    all_assoc = np.concatenate([pos_assoc, neg_assoc])

    res = {'cosine_auc': float(roc_auc_score(labels, all_cos)),
           'assoc_only_auc': float(roc_auc_score(labels, all_assoc))}

    # Lambda sweep
    best_lam, best_auc = 0, res['cosine_auc']
    lam_sweep = {}
    for lam in LAMBDA_SWEEP:
        blended = (1 - lam) * all_cos + lam * all_assoc
        auc = roc_auc_score(labels, blended)
        lam_sweep[str(lam)] = float(auc)
        if auc > best_auc:
            best_auc, best_lam = auc, lam
    res['lambda_sweep'] = lam_sweep
    res['best_auc'] = float(best_auc)
    res['best_lambda'] = best_lam

    # Cross-boundary
    cb_pos = np.abs(pos_cosine) < 0.2
    cb_neg = np.abs(neg_cosine) < 0.2
    if cb_pos.sum() > 10 and cb_neg.sum() > 10:
        cb_labels = np.concatenate([np.ones(cb_pos.sum()), np.zeros(cb_neg.sum())])
        cb_cos = np.concatenate([pos_cosine[cb_pos], neg_cosine[cb_neg]])
        cb_assoc = np.concatenate([pos_assoc[cb_pos], neg_assoc[cb_neg]])
        res['cb_cosine_auc'] = float(roc_auc_score(cb_labels, cb_cos))
        cb_best = res['cb_cosine_auc']
        cb_best_lam = 0
        for lam in LAMBDA_SWEEP:
            auc = roc_auc_score(cb_labels, (1 - lam) * cb_cos + lam * cb_assoc)
            if auc > cb_best:
                cb_best, cb_best_lam = auc, lam
        res['cb_best_auc'] = float(cb_best)
        res['cb_best_lambda'] = cb_best_lam
        res['cb_n_pos'] = int(cb_pos.sum())

    # Easy/Hard
    for name, mask in [('easy', pos_cosine > 0.3), ('hard', np.abs(pos_cosine) < 0.2)]:
        if mask.sum() < 10:
            continue
        s_labels = np.concatenate([np.ones(mask.sum()), np.zeros(len(neg_pairs))])
        s_cos = np.concatenate([pos_cosine[mask], neg_cosine])
        s_assoc = np.concatenate([pos_assoc[mask], neg_assoc])
        s_cos_auc = roc_auc_score(s_labels, s_cos)
        s_best = s_cos_auc
        for lam in LAMBDA_SWEEP:
            auc = roc_auc_score(s_labels, (1 - lam) * s_cos + lam * s_assoc)
            if auc > s_best:
                s_best = auc
        res[f'{name}_cosine_auc'] = float(s_cos_auc)
        res[f'{name}_best_auc'] = float(s_best)
        res[f'{name}_n_pos'] = int(mask.sum())

    return res


# ---------------------------------------------------------------------------
# Cluster analysis
# ---------------------------------------------------------------------------
def cluster_analysis(model, X_norm_tensor, gene_names, string_adj):
    """Run HDBSCAN on association space, find PAM-specific clusters."""
    model.eval()
    with torch.no_grad():
        X_assoc = model(X_norm_tensor).cpu().numpy()
    X_expr = X_norm_tensor.cpu().numpy()

    cos_expr = X_expr @ X_expr.T
    cos_assoc = X_assoc @ X_assoc.T

    results = {}
    labels_dict = {}
    for name, cos_mat in [('expression', cos_expr), ('association', cos_assoc)]:
        dist = (1.0 - cos_mat).astype(np.float64)
        np.fill_diagonal(dist, 0)
        dist = np.clip(dist, 0, 2)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER,
                                     min_samples=HDBSCAN_MIN_SAMPLES,
                                     metric='precomputed')
        labels = clusterer.fit_predict(dist)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_clustered = int((labels != -1).sum())
        sizes = sorted([int((labels == c).sum()) for c in range(max(labels) + 1)
                         if (labels == c).sum() > 0], reverse=True) if n_clusters > 0 else []
        results[name] = {'n_clusters': n_clusters, 'n_clustered': n_clustered,
                         'sizes': sizes}
        labels_dict[name] = labels

    # PAM-specific clusters: association clusters where members are fragmented/noise in expression
    pam_clusters = []
    labels_a = labels_dict['association']
    labels_e = labels_dict['expression']
    if results['association']['n_clusters'] > 0:
        for c in range(max(labels_a) + 1):
            members = np.where(labels_a == c)[0]
            if len(members) < HDBSCAN_MIN_CLUSTER:
                continue
            expr_labs = labels_e[members]
            pct_noise = float((expr_labs == -1).sum() / len(members))
            n_expr_cl = len(set(expr_labs[expr_labs != -1]))

            if pct_noise > 0.5 or (n_expr_cl >= 3):
                genes = [gene_names[i] for i in members]
                # STRING enrichment
                n_edges, n_possible = 0, len(members) * (len(members) - 1) // 2
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        if genes[j] in string_adj.get(genes[i], set()):
                            n_edges += 1
                density = n_edges / n_possible if n_possible > 0 else 0
                bg = 41434 / (1845 * 1844 / 2)
                pam_clusters.append({
                    'size': len(members),
                    'string_enrichment': float(density / bg) if bg > 0 else 0,
                    'pct_noise_expr': pct_noise,
                    'genes_sample': genes[:8],
                })

    results['n_pam_specific'] = len(pam_clusters)
    results['pam_enriched_gt2x'] = sum(1 for pc in pam_clusters if pc['string_enrichment'] > 2)
    if pam_clusters:
        results['mean_enrichment'] = float(np.mean([pc['string_enrichment'] for pc in pam_clusters]))
        results['top_clusters'] = sorted(pam_clusters,
                                          key=lambda x: x['string_enrichment'],
                                          reverse=True)[:5]
    else:
        results['mean_enrichment'] = 0
        results['top_clusters'] = []

    # Correlation between spaces
    upper_e = cos_expr[np.triu_indices(len(cos_expr), k=1)]
    upper_a = cos_assoc[np.triu_indices(len(cos_assoc), k=1)]
    results['space_correlation'] = float(np.corrcoef(upper_e, upper_a)[0, 1])

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    t0 = time.time()
    print("PAM-Genetics Phase 5: Threshold Sweep (Low / Medium / High)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print()

    # Load expression data
    print("Loading expression data...")
    import anndata as ad
    adata = ad.read_h5ad(BULK_FILE)
    X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    n_pert = X_raw.shape[0]

    pca = PCA(n_components=PCA_DIM)
    X_pca = pca.fit_transform(X_raw)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_norm = X_pca / (norms + 1e-8)
    X_norm_tensor = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    print(f"  {n_pert} perturbations, PCA-{PCA_DIM}")

    # Gene names
    gene_names = []
    for idx_str in adata.obs.index:
        parts = str(idx_str).split('_')
        gene_names.append(parts[1].upper() if len(parts) >= 2 else str(idx_str))

    # STRING adjacency (from medium CSV -- used for cluster validation)
    import pandas as pd
    string_df = pd.read_csv(DATA_DIR / "string_pairs_medium.csv")
    string_adj = defaultdict(set)
    for _, row in string_df.iterrows():
        string_adj[row['gene1']].add(row['gene2'])
        string_adj[row['gene2']].add(row['gene1'])

    all_results = {}

    for thresh_name in ['low', 'medium', 'high']:
        thresh_info = THRESHOLDS[thresh_name]
        pairs = np.load(thresh_info['pairs_file'])

        print("\n" + "=" * 70)
        print(f"THRESHOLD: {thresh_name.upper()} (>= {thresh_info['score']}) -- {len(pairs)} pairs")
        print("=" * 70)

        # Train
        set_seed(SEED)
        print(f"  Training...")
        model, losses, train_acc = train_model(X_norm_tensor, pairs)
        alpha = torch.sigmoid(model.alpha_logit).item()
        print(f"    Final loss: {losses[-1]:.4f}, train acc: {train_acc*100:.1f}%, alpha: {alpha:.3f}")

        # Save model
        model_path = RESULTS_DIR / f"model_{thresh_name}.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'pca_components': pca.components_,
            'pca_mean': pca.mean_,
            'train_acc': train_acc,
            'final_loss': losses[-1],
            'alpha': alpha,
        }, model_path)
        print(f"    Model saved to {model_path}")

        # Evaluate
        print(f"  Evaluating...")
        eval_res = evaluate(model, X_norm_tensor, pairs, n_pert)
        print(f"    Cosine AUC:       {eval_res['cosine_auc']:.4f}")
        print(f"    Assoc-only AUC:   {eval_res['assoc_only_auc']:.4f}")
        print(f"    Best blended AUC: {eval_res['best_auc']:.4f} (lam={eval_res['best_lambda']:.1f})")
        if 'cb_best_auc' in eval_res:
            print(f"    Cross-boundary:   {eval_res['cb_cosine_auc']:.4f} -> "
                  f"{eval_res['cb_best_auc']:.4f} (+{eval_res['cb_best_auc'] - eval_res['cb_cosine_auc']:.4f})")
        if 'easy_best_auc' in eval_res:
            print(f"    Easy (n={eval_res['easy_n_pos']}):  "
                  f"{eval_res['easy_cosine_auc']:.4f} -> {eval_res['easy_best_auc']:.4f}")
        if 'hard_best_auc' in eval_res:
            print(f"    Hard (n={eval_res['hard_n_pos']}): "
                  f"{eval_res['hard_cosine_auc']:.4f} -> {eval_res['hard_best_auc']:.4f}")

        # Cluster analysis
        print(f"  Clustering...")
        cluster_res = cluster_analysis(model, X_norm_tensor, gene_names, string_adj)
        print(f"    Expression clusters: {cluster_res['expression']['n_clusters']}")
        print(f"    Association clusters: {cluster_res['association']['n_clusters']}")
        print(f"    PAM-specific: {cluster_res['n_pam_specific']} "
              f"({cluster_res['pam_enriched_gt2x']} STRING-enriched >2x)")
        print(f"    Space correlation: {cluster_res['space_correlation']:.4f}")

        all_results[thresh_name] = {
            'score': thresh_info['score'],
            'n_pairs': len(pairs),
            'train_acc': float(train_acc),
            'final_loss': float(losses[-1]),
            'alpha': float(alpha),
            'eval': eval_res,
            'clusters': {k: v for k, v in cluster_res.items()
                         if k not in ('top_clusters',)},
            'top_pam_clusters': cluster_res.get('top_clusters', []),
        }

    # ------------------------------------------------------------------
    # Summary comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SUMMARY: THRESHOLD COMPARISON")
    print("=" * 70)

    print(f"\n  {'Threshold':<10s} {'Pairs':>7s} {'Cos AUC':>8s} {'Best AUC':>9s} "
          f"{'CB AUC':>7s} {'Assoc Cl':>9s} {'PAM-Spec':>9s} {'Enriched':>9s} {'Corr':>6s}")
    print(f"  {'-'*10} {'-'*7} {'-'*8} {'-'*9} {'-'*7} {'-'*9} {'-'*9} {'-'*9} {'-'*6}")

    for tn in ['low', 'medium', 'high']:
        r = all_results[tn]
        e = r['eval']
        c = r['clusters']
        cb = e.get('cb_best_auc', 0)
        print(f"  {tn:<10s} {r['n_pairs']:>7d} {e['cosine_auc']:>8.4f} {e['best_auc']:>9.4f} "
              f"{cb:>7.4f} {c['association']['n_clusters']:>9d} "
              f"{c['n_pam_specific']:>9d} {c['pam_enriched_gt2x']:>9d} "
              f"{c['space_correlation']:>6.3f}")

    print(f"""
  KEY FINDINGS:
  - Higher confidence -> higher cosine baseline (easier pairs at top)
  - Cross-boundary AUC shows PAM's value: gains should be largest where cosine fails
  - More pairs (low threshold) = more training data but noisier associations
  - Fewer pairs (high threshold) = cleaner associations but less coverage
  - Cluster coherence (STRING enrichment) should increase with tighter thresholds
""")

    # Save
    results_file = RESULTS_DIR / "05_threshold_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Results saved to {results_file}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: AUC comparison across thresholds
    ax = axes[0, 0]
    thresh_labels = ['Low\n(>=400)', 'Medium\n(>=700)', 'High\n(>=900)']
    cos_aucs = [all_results[t]['eval']['cosine_auc'] for t in ['low', 'medium', 'high']]
    best_aucs = [all_results[t]['eval']['best_auc'] for t in ['low', 'medium', 'high']]
    assoc_aucs = [all_results[t]['eval']['assoc_only_auc'] for t in ['low', 'medium', 'high']]

    x = np.arange(3)
    w = 0.25
    ax.bar(x - w, cos_aucs, w, label='Cosine', color='gray', alpha=0.8)
    ax.bar(x, best_aucs, w, label='Blended (best lam)', color='darkorange', alpha=0.8)
    ax.bar(x + w, assoc_aucs, w, label='Assoc only', color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels)
    ax.set_ylabel('Discrimination AUC')
    ax.set_title('AUC by STRING Confidence Threshold')
    ax.legend(fontsize=8)
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    for i, (c, b, a) in enumerate(zip(cos_aucs, best_aucs, assoc_aucs)):
        ax.text(i, b + 0.005, f'{b:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Plot 2: Cross-boundary AUC
    ax = axes[0, 1]
    cb_cos = [all_results[t]['eval'].get('cb_cosine_auc', 0.5) for t in ['low', 'medium', 'high']]
    cb_best = [all_results[t]['eval'].get('cb_best_auc', 0.5) for t in ['low', 'medium', 'high']]
    cb_n = [all_results[t]['eval'].get('cb_n_pos', 0) for t in ['low', 'medium', 'high']]

    ax.bar(x - w / 2, cb_cos, w, label='Cosine', color='gray', alpha=0.8)
    ax.bar(x + w / 2, cb_best, w, label='PAM blended', color='red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{tl}\nn_cb={n}' for tl, n in zip(thresh_labels, cb_n)])
    ax.set_ylabel('Cross-Boundary AUC')
    ax.set_title('Cross-Boundary AUC (|cosine| < 0.2)')
    ax.legend()
    ax.axhline(0.5, color='black', linestyle='--', alpha=0.3)
    for i, val in enumerate(cb_best):
        ax.text(i + w / 2, val + 0.005, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

    # Plot 3: Cluster counts
    ax = axes[1, 0]
    expr_cl = [all_results[t]['clusters']['expression']['n_clusters']
               for t in ['low', 'medium', 'high']]
    assoc_cl = [all_results[t]['clusters']['association']['n_clusters']
                for t in ['low', 'medium', 'high']]
    pam_sp = [all_results[t]['clusters']['n_pam_specific'] for t in ['low', 'medium', 'high']]

    ax.bar(x - w, expr_cl, w, label='Expression', color='gray', alpha=0.8)
    ax.bar(x, assoc_cl, w, label='Association', color='darkorange', alpha=0.8)
    ax.bar(x + w, pam_sp, w, label='PAM-specific', color='red', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels)
    ax.set_ylabel('Number of clusters')
    ax.set_title('Cluster Counts by Threshold')
    ax.legend(fontsize=8)

    # Plot 4: Space correlation and enrichment
    ax = axes[1, 1]
    corrs = [all_results[t]['clusters']['space_correlation'] for t in ['low', 'medium', 'high']]
    enrichments = [all_results[t]['clusters'].get('mean_enrichment', 0)
                   for t in ['low', 'medium', 'high']]

    ax2 = ax.twinx()
    l1 = ax.bar(x - w / 2, corrs, w, label='Space correlation', color='steelblue', alpha=0.7)
    l2 = ax2.bar(x + w / 2, enrichments, w, label='Mean STRING enrichment',
                  color='darkorange', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(thresh_labels)
    ax.set_ylabel('Expr-Assoc cosine correlation', color='steelblue')
    ax2.set_ylabel('Mean STRING enrichment (fold)', color='darkorange')
    ax.set_title('Space Divergence & Biological Coherence')
    lines = [l1, l2]
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, fontsize=8)

    plt.tight_layout()
    plot_file = RESULTS_DIR / "05_threshold_plots.png"
    plt.savefig(plot_file, dpi=150)
    print(f"  Plots saved to {plot_file}")
    plt.close()

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print("\nDone.")


if __name__ == "__main__":
    main()
