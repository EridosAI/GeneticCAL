#!/usr/bin/env python3
"""
Experiment 4: Cross-Boundary AUC at Multiple Thresholds
========================================================
Recompute AUC at |cos| < 0.30, 0.20, 0.15, 0.10, 0.05.
No retraining — load existing high-confidence model.
"""

import json, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

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

CB_THRESHOLDS = [0.30, 0.20, 0.15, 0.10, 0.05]


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
    print("Experiment 4: Cross-Boundary AUC at Multiple Thresholds")
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

    # Load model
    model = AssociationMLP().to(DEVICE)
    ckpt = torch.load(MODEL_FILE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Load high-confidence pairs
    pairs = np.load(DATA_DIR / "string_pairs_high.npy")
    print(f"  {len(pairs)} high-confidence pairs")

    # Compute all scores
    pos_assoc, pos_cosine = compute_scores(model, X_norm_tensor, pairs[:, 0], pairs[:, 1])

    # Generate negatives (same as 05_threshold_sweep.py)
    n_neg = min(len(pairs) * 5, 50000)
    pos_set = set()
    for i in range(len(pairs)):
        pos_set.add((int(pairs[i, 0]), int(pairs[i, 1])))
        pos_set.add((int(pairs[i, 1]), int(pairs[i, 0])))

    rng = np.random.RandomState(42)
    neg_list = []
    while len(neg_list) < n_neg:
        i, j = rng.randint(0, n_pert), rng.randint(0, n_pert)
        if i != j and (i, j) not in pos_set:
            neg_list.append([i, j])
    neg_pairs = np.array(neg_list, dtype=np.int32)

    neg_assoc, neg_cosine = compute_scores(model, X_norm_tensor,
                                           neg_pairs[:, 0], neg_pairs[:, 1])

    # Regression canary: overall AUC
    labels_all = np.concatenate([np.ones(len(pairs)), np.zeros(len(neg_pairs))])
    all_cos = np.concatenate([pos_cosine, neg_cosine])
    all_assoc = np.concatenate([pos_assoc, neg_assoc])
    overall_auc_09 = float(roc_auc_score(labels_all, 0.1 * all_cos + 0.9 * all_assoc))
    overall_cosine = float(roc_auc_score(labels_all, all_cos))
    print(f"\n  Regression canary: overall AUC (lam=0.9) = {overall_auc_09:.6f} (ref: 0.883415)")
    print(f"  Cosine AUC = {overall_cosine:.6f} (ref: 0.692422)")

    # Cross-boundary at multiple thresholds
    results = []
    print(f"\n  {'|cos| thresh':<14} {'Pos':>6} {'Neg':>6} {'Cos AUC':>9} {'CAL 0.9':>9} {'CAL 1.0':>9} {'Δ(1.0)':>8}")
    print(f"  {'-'*14} {'-'*6} {'-'*6} {'-'*9} {'-'*9} {'-'*9} {'-'*8}")

    for thresh in CB_THRESHOLDS:
        cb_pos = np.abs(pos_cosine) < thresh
        cb_neg = np.abs(neg_cosine) < thresh
        n_pos = int(cb_pos.sum())
        n_neg_cb = int(cb_neg.sum())

        if n_pos < 10 or n_neg_cb < 10:
            print(f"  < {thresh:.2f}         {n_pos:>6} {n_neg_cb:>6}  (too few pairs)")
            results.append({
                'threshold': thresh, 'n_pos': n_pos, 'n_neg': n_neg_cb,
                'skipped': True
            })
            continue

        cb_labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg_cb)])
        cb_cos = np.concatenate([pos_cosine[cb_pos], neg_cosine[cb_neg]])
        cb_assoc = np.concatenate([pos_assoc[cb_pos], neg_assoc[cb_neg]])

        cos_auc = float(roc_auc_score(cb_labels, cb_cos))
        auc_09 = float(roc_auc_score(cb_labels, 0.1 * cb_cos + 0.9 * cb_assoc))
        auc_10 = float(roc_auc_score(cb_labels, cb_assoc))
        delta = auc_10 - cos_auc

        print(f"  < {thresh:.2f}         {n_pos:>6} {n_neg_cb:>6} {cos_auc:>9.4f} {auc_09:>9.4f} {auc_10:>9.4f} {delta:>+8.4f}")

        results.append({
            'threshold': thresh, 'n_pos': n_pos, 'n_neg': n_neg_cb,
            'cosine_auc': cos_auc, 'cal_auc_09': auc_09, 'cal_auc_10': auc_10,
            'delta_10': delta
        })

    elapsed = time.time() - t0
    output = {
        'regression_canary': {
            'overall_auc_09': overall_auc_09,
            'overall_cosine_auc': overall_cosine,
        },
        'cross_boundary_results': results,
        'elapsed_seconds': elapsed,
    }

    json_path = OUT_DIR / "04_cross_boundary_sensitivity.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    md_path = OUT_DIR / "04_cross_boundary_sensitivity.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 4: Cross-Boundary Sensitivity (high confidence >= 900)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Runtime:** {elapsed:.0f}s\n\n")

        f.write("## Results\n\n")
        f.write("| |cos| threshold | Pos pairs | Neg pairs | Cosine AUC | CAL AUC (lam=0.9) | CAL AUC (lam=1.0) | Delta (lam=1.0) |\n")
        f.write("|-----------------|-----------|-----------|------------|--------------------|--------------------|------------------|\n")
        for r in results:
            if r.get('skipped'):
                f.write(f"| < {r['threshold']:.2f} | {r['n_pos']} | {r['n_neg']} | (too few) | | | |\n")
            else:
                f.write(f"| < {r['threshold']:.2f} | {r['n_pos']} | {r['n_neg']} | "
                        f"{r['cosine_auc']:.4f} | {r['cal_auc_09']:.4f} | {r['cal_auc_10']:.4f} | "
                        f"+{r['delta_10']:.4f} |\n")

        f.write("\n## Regression Canary\n\n")
        f.write(f"- |cos| < 0.20 cosine AUC: {[r for r in results if r['threshold']==0.2][0].get('cosine_auc', 'N/A')} (ref: 0.5182)\n")
        f.write(f"- |cos| < 0.20 CAL AUC (lam=0.9): {[r for r in results if r['threshold']==0.2][0].get('cal_auc_09', 'N/A')} (ref: 0.9018)\n")

    print(f"\nResults saved to {md_path} and {json_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
