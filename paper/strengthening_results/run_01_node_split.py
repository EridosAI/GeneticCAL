#!/usr/bin/env python3
"""
Experiment 1: Node-Split Inductive Test
========================================
Hold out 30% of GENES (not edges). Train on pairs where both genes are in
the train set. Evaluate on pairs where at least one gene is held out.
"""

import json, time, sys, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Paths  (run from Gene CAL root)
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent.parent  # Gene CAL/
DATA_DIR = BASE / "Gene AAR" / "data"
RESULTS_DIR = BASE / "Gene AAR" / "results"
OUT_DIR = BASE / "Paper" / "strengthening_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Config — identical to 05_threshold_sweep.py
SEED = 42
PCA_DIM = 50
HIDDEN_DIM = 1024
N_LAYERS = 4
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 100
TEMPERATURE = 0.05
LAMBDA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


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


def contrastive_loss(emb_a, emb_b, temperature=TEMPERATURE):
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_model(X_norm_tensor, pairs, epochs=EPOCHS):
    model = AssociationMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_pairs = len(pairs)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_pairs)
        for start in range(0, n_pairs, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 8:
                continue
            emb_a = X_norm_tensor[pairs[batch_idx, 0]]
            emb_b = X_norm_tensor[pairs[batch_idx, 1]]
            pred_a = model(emb_a)
            loss = contrastive_loss(pred_a, emb_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

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

    return model, correct / total


@torch.no_grad()
def compute_scores(model, X_norm_tensor, idx_a, idx_b):
    """Half-transformed scoring: 0.5 * (f(A)·B + f(B)·A)"""
    model.eval()
    emb_a = X_norm_tensor[idx_a]
    emb_b = X_norm_tensor[idx_b]
    pred_a = model(emb_a)
    pred_b = model(emb_b)
    assoc = 0.5 * ((pred_a * emb_b).sum(dim=-1) + (pred_b * emb_a).sum(dim=-1))
    cosine = (emb_a * emb_b).sum(dim=-1)
    return assoc.cpu().numpy(), cosine.cpu().numpy()


def evaluate(model, X_norm_tensor, pos_pairs, neg_pairs):
    """Evaluate with pre-generated negatives."""
    pos_assoc, pos_cosine = compute_scores(model, X_norm_tensor,
                                           pos_pairs[:, 0], pos_pairs[:, 1])
    neg_assoc, neg_cosine = compute_scores(model, X_norm_tensor,
                                           neg_pairs[:, 0], neg_pairs[:, 1])

    labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])
    all_cos = np.concatenate([pos_cosine, neg_cosine])
    all_assoc = np.concatenate([pos_assoc, neg_assoc])

    res = {'cosine_auc': float(roc_auc_score(labels, all_cos)),
           'assoc_only_auc': float(roc_auc_score(labels, all_assoc))}

    # Lambda sweep
    for lam in LAMBDA_SWEEP:
        blended = (1 - lam) * all_cos + lam * all_assoc
        auc = float(roc_auc_score(labels, blended))
        res[f'auc_lam_{lam}'] = auc

    # Cross-boundary |cos| < 0.2
    cb_pos = np.abs(pos_cosine) < 0.2
    cb_neg = np.abs(neg_cosine) < 0.2
    if cb_pos.sum() > 10 and cb_neg.sum() > 10:
        cb_labels = np.concatenate([np.ones(cb_pos.sum()), np.zeros(cb_neg.sum())])
        cb_cos = np.concatenate([pos_cosine[cb_pos], neg_cosine[cb_neg]])
        cb_assoc = np.concatenate([pos_assoc[cb_pos], neg_assoc[cb_neg]])
        res['cb_cosine_auc'] = float(roc_auc_score(cb_labels, cb_cos))
        res['cb_assoc_auc'] = float(roc_auc_score(cb_labels, cb_assoc))
        for lam in [0.9]:
            blended = (1 - lam) * cb_cos + lam * cb_assoc
            res[f'cb_auc_lam_{lam}'] = float(roc_auc_score(cb_labels, blended))
        res['cb_n_pos'] = int(cb_pos.sum())
        res['cb_n_neg'] = int(cb_neg.sum())

    return res


def generate_negatives(pos_pairs, all_pos_set, n_pert, n_neg, seed=42):
    """Generate random negatives excluding ALL positive pairs."""
    rng = np.random.RandomState(seed)
    neg_list = []
    while len(neg_list) < n_neg:
        i, j = rng.randint(0, n_pert), rng.randint(0, n_pert)
        if i != j and (i, j) not in all_pos_set:
            neg_list.append([i, j])
    return np.array(neg_list, dtype=np.int32)


def run_node_split(X_norm_tensor, pairs, n_pert, threshold_name, threshold_score):
    """Run node-split inductive experiment for one threshold."""
    print(f"\n{'='*70}")
    print(f"NODE SPLIT: {threshold_name} (>= {threshold_score}), {len(pairs)} pairs")
    print(f"{'='*70}")

    # Get all unique gene indices
    all_genes = sorted(set(pairs[:, 0].tolist() + pairs[:, 1].tolist()))
    n_genes = len(all_genes)
    print(f"  Unique genes in pairs: {n_genes}")

    # Split genes 70/30
    set_seed(SEED)
    gene_perm = np.random.permutation(all_genes)
    n_train_genes = int(len(gene_perm) * 0.7)
    train_genes = set(gene_perm[:n_train_genes].tolist())
    test_genes = set(gene_perm[n_train_genes:].tolist())

    # Split pairs
    train_pairs_list = []
    test_pairs_list = []
    test_both_unseen = 0
    test_one_unseen = 0

    for i in range(len(pairs)):
        g1, g2 = int(pairs[i, 0]), int(pairs[i, 1])
        g1_train = g1 in train_genes
        g2_train = g2 in train_genes
        if g1_train and g2_train:
            train_pairs_list.append([g1, g2])
        else:
            test_pairs_list.append([g1, g2])
            if not g1_train and not g2_train:
                test_both_unseen += 1
            else:
                test_one_unseen += 1

    train_pairs = np.array(train_pairs_list, dtype=np.int32)
    test_pairs = np.array(test_pairs_list, dtype=np.int32)

    print(f"  Train genes: {len(train_genes)}, Test genes: {len(test_genes)}")
    print(f"  Train pairs: {len(train_pairs)}, Test pairs: {len(test_pairs)}")
    print(f"  Test pairs (both unseen): {test_both_unseen}, (one unseen): {test_one_unseen}")

    # Build full positive set for negative generation (exclude ALL positives)
    all_pos_set = set()
    for i in range(len(pairs)):
        all_pos_set.add((int(pairs[i, 0]), int(pairs[i, 1])))
        all_pos_set.add((int(pairs[i, 1]), int(pairs[i, 0])))

    # Train
    set_seed(SEED)
    print(f"  Training on {len(train_pairs)} pairs...")
    model, train_acc = train_model(X_norm_tensor, train_pairs)
    alpha = torch.sigmoid(model.alpha_logit).item()
    print(f"  Train acc: {train_acc*100:.1f}%, alpha: {alpha:.3f}")

    # Generate negatives for test pairs
    n_neg_test = min(len(test_pairs) * 5, 50000)
    neg_test = generate_negatives(test_pairs, all_pos_set, n_pert, n_neg_test, seed=SEED)

    # Generate negatives for train pairs (transductive sanity check)
    n_neg_train = min(len(train_pairs) * 5, 50000)
    neg_train = generate_negatives(train_pairs, all_pos_set, n_pert, n_neg_train, seed=SEED)

    # Evaluate on test pairs
    print(f"  Evaluating on test pairs ({len(test_pairs)})...")
    test_res = evaluate(model, X_norm_tensor, test_pairs, neg_test)

    # Evaluate on train pairs (sanity check)
    print(f"  Evaluating on train pairs ({len(train_pairs)})...")
    train_res = evaluate(model, X_norm_tensor, train_pairs, neg_train)

    result = {
        'threshold': threshold_name,
        'threshold_score': threshold_score,
        'n_total_pairs': len(pairs),
        'n_train_genes': len(train_genes),
        'n_test_genes': len(test_genes),
        'n_train_pairs': len(train_pairs),
        'n_test_pairs': len(test_pairs),
        'test_both_unseen': test_both_unseen,
        'test_one_unseen': test_one_unseen,
        'train_acc': float(train_acc),
        'alpha': float(alpha),
        'train_eval': train_res,
        'test_eval': test_res,
    }

    print(f"\n  RESULTS ({threshold_name}):")
    print(f"  Train AUC (transductive): cosine={train_res['cosine_auc']:.4f}, "
          f"assoc={train_res['assoc_only_auc']:.4f}, lam0.9={train_res['auc_lam_0.9']:.4f}")
    print(f"  Test AUC (inductive):     cosine={test_res['cosine_auc']:.4f}, "
          f"assoc={test_res['assoc_only_auc']:.4f}, lam0.9={test_res['auc_lam_0.9']:.4f}")
    if 'cb_cosine_auc' in test_res:
        print(f"  Test CB AUC:              cosine={test_res['cb_cosine_auc']:.4f}, "
              f"assoc={test_res['cb_assoc_auc']:.4f}, lam0.9={test_res.get('cb_auc_lam_0.9','N/A')}")

    return result


def main():
    t0 = time.time()
    print("Experiment 1: Node-Split Inductive Test")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load expression data
    print("\nLoading expression data...")
    import anndata as ad
    adata = ad.read_h5ad(BULK_FILE)
    X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    n_pert = X_raw.shape[0]

    set_seed(SEED)
    pca = PCA(n_components=PCA_DIM)
    X_pca = pca.fit_transform(X_raw)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_norm = X_pca / (norms + 1e-8)
    X_norm_tensor = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)
    print(f"  {n_pert} perturbations, PCA-{PCA_DIM}")

    all_results = {}

    # Medium confidence (>=700)
    pairs_medium = np.load(DATA_DIR / "string_pairs_medium.npy")
    all_results['medium'] = run_node_split(X_norm_tensor, pairs_medium, n_pert, 'medium', 700)

    # High confidence (>=900)
    pairs_high = np.load(DATA_DIR / "string_pairs_high.npy")
    all_results['high'] = run_node_split(X_norm_tensor, pairs_high, n_pert, 'high', 900)

    elapsed = time.time() - t0
    all_results['elapsed_seconds'] = elapsed

    # Save JSON
    json_path = OUT_DIR / "01_node_split_inductive.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Write markdown report
    md_path = OUT_DIR / "01_node_split_inductive.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 1: Node-Split Inductive Test\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Runtime:** {elapsed:.0f}s\n\n")

        for thresh in ['medium', 'high']:
            r = all_results[thresh]
            te = r['test_eval']
            tr = r['train_eval']
            f.write(f"## Node-Split Results ({thresh} confidence >= {r['threshold_score']})\n\n")
            f.write(f"- Train genes: {r['n_train_genes']}, Test genes: {r['n_test_genes']}\n")
            f.write(f"- Train pairs: {r['n_train_pairs']}, Test pairs: {r['n_test_pairs']}\n")
            f.write(f"- Test pairs (both genes unseen): {r['test_both_unseen']}, (one gene unseen): {r['test_one_unseen']}\n")
            f.write(f"- Train accuracy: {r['train_acc']*100:.1f}%\n")
            f.write(f"- Learned alpha: {r['alpha']:.3f}\n\n")

            f.write("| Metric | Cosine | CAL (lam=0.9) | CAL (lam=1.0) | Delta (lam=1.0) |\n")
            f.write("|--------|--------|---------------|---------------|------------------|\n")
            f.write(f"| Train AUC (transductive) | {tr['cosine_auc']:.4f} | {tr['auc_lam_0.9']:.4f} | {tr['assoc_only_auc']:.4f} | +{tr['assoc_only_auc']-tr['cosine_auc']:.4f} |\n")
            f.write(f"| Test AUC (inductive) | {te['cosine_auc']:.4f} | {te['auc_lam_0.9']:.4f} | {te['assoc_only_auc']:.4f} | +{te['assoc_only_auc']-te['cosine_auc']:.4f} |\n")
            if 'cb_cosine_auc' in te:
                f.write(f"| Test CB AUC (|cos|<0.2) | {te['cb_cosine_auc']:.4f} | {te.get('cb_auc_lam_0.9','N/A')} | {te['cb_assoc_auc']:.4f} | +{te['cb_assoc_auc']-te['cb_cosine_auc']:.4f} |\n")
            if 'cb_cosine_auc' in tr:
                f.write(f"| Train CB AUC (|cos|<0.2) | {tr['cb_cosine_auc']:.4f} | {tr.get('cb_auc_lam_0.9','N/A')} | {tr['cb_assoc_auc']:.4f} | +{tr['cb_assoc_auc']-tr['cb_cosine_auc']:.4f} |\n")
            f.write("\n")

        # Comparison with edge-split
        f.write("## Comparison with Edge-Split Inductive\n\n")
        f.write("| Split Type | Test AUC (lam=0.9) | Test AUC (lam=1.0) |\n")
        f.write("|------------|--------------------|--------------------|  \n")
        f.write(f"| Edge-split (medium, from 03_train_results) | 0.7960 | 0.8266 |\n")
        f.write(f"| Node-split (medium) | {all_results['medium']['test_eval']['auc_lam_0.9']:.4f} | {all_results['medium']['test_eval']['assoc_only_auc']:.4f} |\n")
        f.write(f"| Node-split (high) | {all_results['high']['test_eval']['auc_lam_0.9']:.4f} | {all_results['high']['test_eval']['assoc_only_auc']:.4f} |\n")
        f.write("\n")

        # Regression canary
        f.write("## Regression Canary\n\n")
        for thresh in ['medium', 'high']:
            tr = all_results[thresh]['train_eval']
            f.write(f"- {thresh} train AUC (transductive): {tr['cosine_auc']:.4f} cosine, {tr['auc_lam_0.9']:.4f} lam=0.9\n")

    print(f"\nResults saved to {md_path} and {json_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
