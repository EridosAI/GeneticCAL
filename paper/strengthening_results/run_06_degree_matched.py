#!/usr/bin/env python3
"""
Experiment 6: Degree-Matched Negatives at Evaluation
======================================================
Generate negatives where replacement genes have similar STRING degree
to the positive pair genes. Compare AUC with random negatives.
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "Gene AAR" / "data"
RESULTS_DIR = BASE / "Gene AAR" / "results"
OUT_DIR = BASE / "Paper" / "strengthening_results"

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
MODEL_FILE = RESULTS_DIR / "model_high.pt"
STRING_MEDIUM = DATA_DIR / "string_pairs_medium.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
PCA_DIM = 50
HIDDEN_DIM = 1024
N_LAYERS = 4


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


@torch.no_grad()
def compute_scores(model, X_norm_tensor, idx_a, idx_b):
    model.eval()
    emb_a = X_norm_tensor[idx_a]
    emb_b = X_norm_tensor[idx_b]
    pred_a = model(emb_a)
    pred_b = model(emb_b)
    assoc = 0.5 * ((pred_a * emb_b).sum(dim=-1) + (pred_b * emb_a).sum(dim=-1))
    cosine = (emb_a * emb_b).sum(dim=-1)
    return assoc.cpu().numpy(), cosine.cpu().numpy()


def main():
    t0 = time.time()
    print("Experiment 6: Degree-Matched Negatives")
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
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}

    # Load model
    model = AssociationMLP().to(DEVICE)
    ckpt = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load high-confidence pairs
    pairs = np.load(DATA_DIR / "string_pairs_high.npy")
    print(f"  {len(pairs)} high-confidence pairs, {n_pert} genes")

    # Build positive set
    pos_set = set()
    for i in range(len(pairs)):
        pos_set.add((int(pairs[i, 0]), int(pairs[i, 1])))
        pos_set.add((int(pairs[i, 1]), int(pairs[i, 0])))

    # Compute STRING degree per gene index (using medium CSV, score >= 700)
    print("\nComputing STRING degrees from medium-confidence pairs...")
    string_df = pd.read_csv(STRING_MEDIUM)
    degree = defaultdict(int)
    for _, row in string_df.iterrows():
        g1, g2 = row['gene1'], row['gene2']
        if g1 in gene_to_idx:
            degree[gene_to_idx[g1]] += 1
        if g2 in gene_to_idx:
            degree[gene_to_idx[g2]] += 1

    # Ensure all genes have a degree entry (0 for genes not in STRING)
    degrees = np.array([degree.get(i, 0) for i in range(n_pert)])
    print(f"  Mean degree: {degrees.mean():.1f}, max: {degrees.max()}, "
          f"genes with degree 0: {(degrees==0).sum()}")

    # Compute positive scores
    pos_assoc, pos_cosine = compute_scores(model, X_norm_tensor, pairs[:, 0], pairs[:, 1])

    # Compute mean degree per positive pair
    pos_mean_degree = (degrees[pairs[:, 0]] + degrees[pairs[:, 1]]) / 2.0

    # Generate degree-matched negatives
    print("\nGenerating degree-matched negatives (±20%)...")
    rng = np.random.RandomState(SEED)
    n_neg_per_pos = 5

    # Sort genes by degree for efficient sampling
    sorted_by_degree = np.argsort(degrees)
    gene_degree_sorted = degrees[sorted_by_degree]

    neg_list = []
    n_widened = 0
    for pi in range(len(pairs)):
        target_degree = pos_mean_degree[pi]
        lo_20 = target_degree * 0.8
        hi_20 = target_degree * 1.2

        # Find candidates within ±20%
        left = np.searchsorted(gene_degree_sorted, lo_20, side='left')
        right = np.searchsorted(gene_degree_sorted, hi_20, side='right')
        candidates = sorted_by_degree[left:right]

        if len(candidates) < 20:
            # Widen to ±50%
            lo_50 = target_degree * 0.5
            hi_50 = max(target_degree * 1.5, 5)  # ensure non-zero range for low-degree genes
            left = np.searchsorted(gene_degree_sorted, lo_50, side='left')
            right = np.searchsorted(gene_degree_sorted, hi_50, side='right')
            candidates = sorted_by_degree[left:right]
            n_widened += 1

        if len(candidates) < 2:
            # Fallback: use all genes
            candidates = np.arange(n_pert)

        # Sample negative pairs from candidates
        n_sampled = 0
        max_attempts = 100
        attempts = 0
        while n_sampled < n_neg_per_pos and attempts < max_attempts:
            i = int(rng.choice(candidates))
            j = int(rng.choice(candidates))
            if i != j and (i, j) not in pos_set:
                neg_list.append([i, j])
                n_sampled += 1
            attempts += 1

    neg_pairs_dm = np.array(neg_list, dtype=np.int32)
    print(f"  Generated {len(neg_pairs_dm)} degree-matched negatives "
          f"({n_widened} pairs needed widening to ±50%)")
    print(f"  Neg mean degree: {(degrees[neg_pairs_dm[:,0]] + degrees[neg_pairs_dm[:,1]]).mean()/2:.1f}")
    print(f"  Pos mean degree: {pos_mean_degree.mean():.1f}")

    # Evaluate with degree-matched negatives
    neg_assoc_dm, neg_cosine_dm = compute_scores(model, X_norm_tensor,
                                                  neg_pairs_dm[:, 0], neg_pairs_dm[:, 1])

    labels_dm = np.concatenate([np.ones(len(pairs)), np.zeros(len(neg_pairs_dm))])
    all_cos_dm = np.concatenate([pos_cosine, neg_cosine_dm])
    all_assoc_dm = np.concatenate([pos_assoc, neg_assoc_dm])

    dm_cosine_auc = float(roc_auc_score(labels_dm, all_cos_dm))
    dm_auc_09 = float(roc_auc_score(labels_dm, 0.1 * all_cos_dm + 0.9 * all_assoc_dm))
    dm_auc_10 = float(roc_auc_score(labels_dm, all_assoc_dm))

    # Cross-boundary
    cb_pos_dm = np.abs(pos_cosine) < 0.2
    cb_neg_dm = np.abs(neg_cosine_dm) < 0.2
    dm_cb_cosine_auc = None
    dm_cb_assoc_auc = None
    if cb_pos_dm.sum() > 10 and cb_neg_dm.sum() > 10:
        cb_labels = np.concatenate([np.ones(cb_pos_dm.sum()), np.zeros(cb_neg_dm.sum())])
        cb_cos = np.concatenate([pos_cosine[cb_pos_dm], neg_cosine_dm[cb_neg_dm]])
        cb_assoc = np.concatenate([pos_assoc[cb_pos_dm], neg_assoc_dm[cb_neg_dm]])
        dm_cb_cosine_auc = float(roc_auc_score(cb_labels, cb_cos))
        dm_cb_assoc_auc = float(roc_auc_score(cb_labels, cb_assoc))

    # Reference: random negatives (reproduce for canary)
    print("\nGenerating random negatives for comparison...")
    rng2 = np.random.RandomState(42)
    n_neg_rand = min(len(pairs) * 5, 50000)
    neg_list_rand = []
    while len(neg_list_rand) < n_neg_rand:
        i, j = rng2.randint(0, n_pert), rng2.randint(0, n_pert)
        if i != j and (i, j) not in pos_set:
            neg_list_rand.append([i, j])
    neg_pairs_rand = np.array(neg_list_rand, dtype=np.int32)

    neg_assoc_rand, neg_cosine_rand = compute_scores(model, X_norm_tensor,
                                                      neg_pairs_rand[:, 0], neg_pairs_rand[:, 1])

    labels_rand = np.concatenate([np.ones(len(pairs)), np.zeros(len(neg_pairs_rand))])
    all_cos_rand = np.concatenate([pos_cosine, neg_cosine_rand])
    all_assoc_rand = np.concatenate([pos_assoc, neg_assoc_rand])

    rand_cosine_auc = float(roc_auc_score(labels_rand, all_cos_rand))
    rand_auc_09 = float(roc_auc_score(labels_rand, 0.1 * all_cos_rand + 0.9 * all_assoc_rand))
    rand_auc_10 = float(roc_auc_score(labels_rand, all_assoc_rand))

    cb_pos_r = np.abs(pos_cosine) < 0.2
    cb_neg_r = np.abs(neg_cosine_rand) < 0.2
    rand_cb_assoc_auc = None
    if cb_pos_r.sum() > 10 and cb_neg_r.sum() > 10:
        cb_labels_r = np.concatenate([np.ones(cb_pos_r.sum()), np.zeros(cb_neg_r.sum())])
        cb_assoc_r = np.concatenate([pos_assoc[cb_pos_r], neg_assoc_rand[cb_neg_r]])
        rand_cb_assoc_auc = float(roc_auc_score(cb_labels_r, cb_assoc_r))

    print(f"\n  Regression canary: random AUC (lam=0.9) = {rand_auc_09:.6f} (ref: 0.883415)")

    print(f"\n  RESULTS:")
    print(f"  {'Neg type':<25} {'Cosine AUC':>11} {'CAL 0.9':>9} {'CAL 1.0':>9} {'CB AUC':>9}")
    print(f"  {'-'*25} {'-'*11} {'-'*9} {'-'*9} {'-'*9}")
    print(f"  {'Random (existing)':<25} {rand_cosine_auc:>11.4f} {rand_auc_09:>9.4f} {rand_auc_10:>9.4f} {rand_cb_assoc_auc or 0:>9.4f}")
    print(f"  {'Degree-matched (±20%)':<25} {dm_cosine_auc:>11.4f} {dm_auc_09:>9.4f} {dm_auc_10:>9.4f} {dm_cb_assoc_auc or 0:>9.4f}")

    elapsed = time.time() - t0

    output = {
        'random_negatives': {
            'n_neg': len(neg_pairs_rand),
            'cosine_auc': rand_cosine_auc,
            'auc_lam_09': rand_auc_09,
            'auc_lam_10': rand_auc_10,
            'cb_assoc_auc': rand_cb_assoc_auc,
        },
        'degree_matched_negatives': {
            'n_neg': len(neg_pairs_dm),
            'n_widened': n_widened,
            'cosine_auc': dm_cosine_auc,
            'auc_lam_09': dm_auc_09,
            'auc_lam_10': dm_auc_10,
            'cb_cosine_auc': dm_cb_cosine_auc,
            'cb_assoc_auc': dm_cb_assoc_auc,
            'neg_mean_degree': float((degrees[neg_pairs_dm[:,0]] + degrees[neg_pairs_dm[:,1]]).mean()/2),
            'pos_mean_degree': float(pos_mean_degree.mean()),
        },
        'elapsed_seconds': elapsed,
    }

    json_path = OUT_DIR / "06_degree_matched_negatives.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    md_path = OUT_DIR / "06_degree_matched_negatives.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 6: Degree-Matched Negatives (high confidence >= 900)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Runtime:** {elapsed:.0f}s\n\n")

        f.write("## Results\n\n")
        f.write("| Negative type | N neg | Overall AUC (lam=0.9) | Assoc-only AUC | CB AUC (assoc) | Delta vs random |\n")
        f.write("|---------------|-------|-----------------------|----------------|----------------|------------------|\n")
        f.write(f"| Random (existing) | {len(neg_pairs_rand)} | {rand_auc_09:.4f} | {rand_auc_10:.4f} | "
                f"{rand_cb_assoc_auc:.4f} | — |\n")
        dm_cb_str = f"{dm_cb_assoc_auc:.4f}" if dm_cb_assoc_auc is not None else "N/A"
        f.write(f"| Degree-matched (±20%) | {len(neg_pairs_dm)} | {dm_auc_09:.4f} | {dm_auc_10:.4f} | "
                f"{dm_cb_str} | "
                f"{dm_auc_09 - rand_auc_09:+.4f} |\n")
        f.write("\n")

        f.write("## Degree Statistics\n\n")
        f.write(f"- Positive pair mean degree: {pos_mean_degree.mean():.1f}\n")
        f.write(f"- Degree-matched neg mean degree: {(degrees[neg_pairs_dm[:,0]] + degrees[neg_pairs_dm[:,1]]).mean()/2:.1f}\n")
        f.write(f"- Pairs needing ±50% widening: {n_widened}\n\n")

        f.write("## Regression Canary\n\n")
        f.write(f"- Random neg AUC (lam=0.9): {rand_auc_09:.6f} (ref: 0.883415)\n")
        f.write(f"- Random neg CB AUC: {rand_cb_assoc_auc} (ref: 0.9082)\n")

    print(f"\nResults saved to {md_path} and {json_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
