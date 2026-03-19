#!/usr/bin/env python3
"""
Experiment 2: Seed Variance (3 seeds)
======================================
Train the high-confidence (>=900) model with seeds 42, 123, 456.
Report mean +/- SD for all metrics.
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
OUT_DIR.mkdir(parents=True, exist_ok=True)

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PCA_DIM = 50
HIDDEN_DIM = 1024
N_LAYERS = 4
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 100
TEMPERATURE = 0.05
SEEDS = [42, 123, 456]


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


def train_model(X_norm_tensor, pairs, seed, epochs=EPOCHS):
    set_seed(seed)
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


def evaluate(model, X_norm_tensor, pos_pairs, n_pert):
    pos_assoc, pos_cosine = compute_scores(model, X_norm_tensor,
                                           pos_pairs[:, 0], pos_pairs[:, 1])
    # Generate negatives with FIXED seed 42 for consistency across models
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

    res = {
        'cosine_auc': float(roc_auc_score(labels, all_cos)),
        'assoc_only_auc': float(roc_auc_score(labels, all_assoc)),
    }

    for lam in [0.9, 1.0]:
        blended = (1 - lam) * all_cos + lam * all_assoc
        res[f'auc_lam_{lam}'] = float(roc_auc_score(labels, blended))

    # Cross-boundary
    cb_pos = np.abs(pos_cosine) < 0.2
    cb_neg = np.abs(neg_cosine) < 0.2
    if cb_pos.sum() > 10 and cb_neg.sum() > 10:
        cb_labels = np.concatenate([np.ones(cb_pos.sum()), np.zeros(cb_neg.sum())])
        cb_cos = np.concatenate([pos_cosine[cb_pos], neg_cosine[cb_neg]])
        cb_assoc = np.concatenate([pos_assoc[cb_pos], neg_assoc[cb_neg]])
        res['cb_cosine_auc'] = float(roc_auc_score(cb_labels, cb_cos))
        for lam in [0.9, 1.0]:
            blended = (1 - lam) * cb_cos + lam * cb_assoc
            res[f'cb_auc_lam_{lam}'] = float(roc_auc_score(cb_labels, blended))

    return res


def main():
    t0 = time.time()
    print("Experiment 2: Seed Variance (3 seeds)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    import anndata as ad
    adata = ad.read_h5ad(BULK_FILE)
    X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    n_pert = X_raw.shape[0]

    # PCA with seed 42 (same as original)
    set_seed(42)
    pca = PCA(n_components=PCA_DIM)
    X_pca = pca.fit_transform(X_raw)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_norm = X_pca / (norms + 1e-8)
    X_norm_tensor = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)

    pairs = np.load(DATA_DIR / "string_pairs_high.npy")
    print(f"  {n_pert} perturbations, {len(pairs)} high-confidence pairs")

    all_seed_results = {}
    for seed in SEEDS:
        print(f"\n{'='*70}")
        print(f"SEED: {seed}")
        print(f"{'='*70}")

        model, losses, train_acc = train_model(X_norm_tensor, pairs, seed)
        alpha = torch.sigmoid(model.alpha_logit).item()
        print(f"  Final loss: {losses[-1]:.4f}, train acc: {train_acc*100:.1f}%, alpha: {alpha:.3f}")

        eval_res = evaluate(model, X_norm_tensor, pairs, n_pert)
        eval_res['train_acc'] = float(train_acc)
        eval_res['final_loss'] = float(losses[-1])
        eval_res['alpha'] = float(alpha)

        all_seed_results[seed] = eval_res

        print(f"  Overall AUC (lam=0.9): {eval_res['auc_lam_0.9']:.4f}")
        print(f"  Overall AUC (lam=1.0): {eval_res['assoc_only_auc']:.4f}")
        print(f"  CB AUC (lam=0.9): {eval_res.get('cb_auc_lam_0.9', 'N/A')}")
        print(f"  CB AUC (lam=1.0): {eval_res.get('cb_auc_lam_1.0', 'N/A')}")

    # Regression canary: seed 42 must match reference
    ref_overall = 0.8834147460030944
    ref_cb = 0.9017540253526348
    s42 = all_seed_results[42]
    print(f"\n{'='*70}")
    print("REGRESSION CANARY (seed 42):")
    print(f"  Overall AUC (lam=0.9): {s42['auc_lam_0.9']:.6f} (ref: {ref_overall:.6f})")
    print(f"  CB AUC (lam=0.9): {s42.get('cb_auc_lam_0.9', 'N/A')} (ref: {ref_cb:.6f})")

    # Compute mean/SD
    metrics = ['auc_lam_0.9', 'assoc_only_auc', 'cb_auc_lam_0.9', 'cb_auc_lam_1.0',
               'train_acc', 'final_loss', 'alpha']
    stats = {}
    for m in metrics:
        vals = [all_seed_results[s].get(m) for s in SEEDS if all_seed_results[s].get(m) is not None]
        if vals:
            stats[m] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                        'values': vals}

    elapsed = time.time() - t0

    # Save JSON
    output = {'seeds': SEEDS, 'per_seed': {str(s): r for s, r in all_seed_results.items()},
              'stats': stats, 'elapsed_seconds': elapsed}
    json_path = OUT_DIR / "02_seed_variance.json"
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Write markdown
    md_path = OUT_DIR / "02_seed_variance.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 2: Seed Variance (3 seeds)\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Runtime:** {elapsed:.0f}s\n\n")

        f.write("## Results (high confidence >= 900)\n\n")
        f.write("| Seed | Train Acc | Final Loss | Alpha | Overall AUC (lam=0.9) | Overall AUC (lam=1.0) | CB AUC (lam=0.9) | CB AUC (lam=1.0) |\n")
        f.write("|------|-----------|------------|-------|----------------------|----------------------|------------------|------------------|\n")
        for seed in SEEDS:
            r = all_seed_results[seed]
            f.write(f"| {seed} | {r['train_acc']*100:.1f}% | {r['final_loss']:.4f} | "
                    f"{r['alpha']:.3f} | {r['auc_lam_0.9']:.4f} | {r['assoc_only_auc']:.4f} | "
                    f"{r.get('cb_auc_lam_0.9', 'N/A')} | {r.get('cb_auc_lam_1.0', 'N/A')} |\n")

        f.write(f"| **Mean** | {stats['train_acc']['mean']*100:.1f}% | {stats['final_loss']['mean']:.4f} | "
                f"{stats['alpha']['mean']:.3f} | {stats['auc_lam_0.9']['mean']:.4f} | "
                f"{stats['assoc_only_auc']['mean']:.4f} | {stats['cb_auc_lam_0.9']['mean']:.4f} | "
                f"{stats['cb_auc_lam_1.0']['mean']:.4f} |\n")
        f.write(f"| **SD** | {stats['train_acc']['std']*100:.1f}% | {stats['final_loss']['std']:.4f} | "
                f"{stats['alpha']['std']:.3f} | {stats['auc_lam_0.9']['std']:.4f} | "
                f"{stats['assoc_only_auc']['std']:.4f} | {stats['cb_auc_lam_0.9']['std']:.4f} | "
                f"{stats['cb_auc_lam_1.0']['std']:.4f} |\n")
        f.write("\n")

        f.write("## Regression Canary\n\n")
        f.write(f"- Seed 42 overall AUC (lam=0.9): {s42['auc_lam_0.9']:.6f} (reference: {ref_overall:.6f})\n")
        f.write(f"- Seed 42 CB AUC (lam=0.9): {s42.get('cb_auc_lam_0.9', 'N/A')} (reference: {ref_cb:.6f})\n")

    print(f"\nResults saved to {md_path} and {json_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
