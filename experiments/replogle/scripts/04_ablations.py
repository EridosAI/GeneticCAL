#!/usr/bin/env python3
"""
DepMap PAM Phase 4: Ablation Controls
=======================================

Run at the best N from Phase 3 (reads from 03_train_results.json).

1. Shuffled pairs      — scramble pairings, retrain
2. Similar positives   — top-N highest expression cosine pairs instead
3. Random negatives    — replace in-batch negatives with random samples
4. Inductive 70/30     — train on 70%, evaluate on held-out 30%

Kill criteria:
  - Shuffled must destroy signal
  - Similar positives must not match full model

Prerequisites:
    python scripts/03_train.py

Usage:
    python scripts/04_ablations.py
"""

import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
PCA_DIM = 100
HIDDEN_DIM = 1024
N_LAYERS = 4
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 100
TEMPERATURE = 0.05

RESULTS_DIR = Path("results")
EMB_PATH = RESULTS_DIR / "gene_embeddings_pca100.npy"
GENE_LIST_PATH = RESULTS_DIR / "gene_list.json"
TRAIN_RESULTS_PATH = RESULTS_DIR / "03_train_results.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model (same as 03_train.py)
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


def contrastive_loss(emb_a, emb_b, temperature=TEMPERATURE):
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def contrastive_loss_random_negatives(emb_a, emb_b, all_emb, temperature=TEMPERATURE,
                                      n_neg=511):
    B = emb_a.shape[0]
    device = emb_a.device
    neg_idx = torch.randint(0, len(all_emb), (B, n_neg), device=device)
    neg_emb = all_emb[neg_idx]

    pos_sim = (emb_a * emb_b).sum(dim=-1, keepdim=True) / temperature
    neg_sim = torch.bmm(neg_emb, emb_a.unsqueeze(-1)).squeeze(-1) / temperature
    logits = torch.cat([pos_sim, neg_sim], dim=1)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss_a = F.cross_entropy(logits, labels)

    neg_sim_b = torch.bmm(neg_emb, emb_b.unsqueeze(-1)).squeeze(-1) / temperature
    logits_b = torch.cat([pos_sim, neg_sim_b], dim=1)
    loss_b = F.cross_entropy(logits_b, labels)
    return (loss_a + loss_b) / 2


def train_model(X_tensor, pairs, epochs=EPOCHS, verbose=True,
                use_random_negatives=False):
    model = AssociationMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_pairs = len(pairs)
    losses = []
    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_pairs)
        epoch_loss = 0
        n_batches = 0
        for start in range(0, n_pairs, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 8:
                continue
            emb_a = X_tensor[pairs[batch_idx, 0]]
            emb_b = X_tensor[pairs[batch_idx, 1]]
            pred_a = model(emb_a)
            pred_b = model(emb_b)

            if use_random_negatives:
                # For random negatives, use transformed embeddings as the pool
                with torch.no_grad():
                    all_transformed = model(X_tensor)
                loss = contrastive_loss_random_negatives(pred_a, pred_b, all_transformed)
            else:
                loss = contrastive_loss(pred_a, pred_b)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        if verbose and (epoch + 1) % 20 == 0:
            alpha = torch.sigmoid(model.alpha_logit).item()
            print(f"      Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, alpha={alpha:.3f}")

    return model, losses


@torch.no_grad()
def compute_pam_scores(model, X_tensor, idx_a, idx_b):
    model.eval()
    pred_a = model(X_tensor[idx_a])
    pred_b = model(X_tensor[idx_b])
    return (pred_a * pred_b).sum(dim=-1).cpu().numpy()


@torch.no_grad()
def compute_cosine_scores(X_tensor, idx_a, idx_b):
    return (X_tensor[idx_a] * X_tensor[idx_b]).sum(dim=-1).cpu().numpy()


def evaluate_model(model, X_tensor, pos_pairs, n_genes):
    pos_pam = compute_pam_scores(model, X_tensor, pos_pairs[:, 0], pos_pairs[:, 1])
    pos_cos = compute_cosine_scores(X_tensor, pos_pairs[:, 0], pos_pairs[:, 1])

    n_neg = min(len(pos_pairs) * 5, 200_000)
    pos_set = set()
    for i in range(len(pos_pairs)):
        pos_set.add((int(pos_pairs[i, 0]), int(pos_pairs[i, 1])))
        pos_set.add((int(pos_pairs[i, 1]), int(pos_pairs[i, 0])))

    rng = np.random.RandomState(42)
    neg_list = []
    while len(neg_list) < n_neg:
        i = rng.randint(0, n_genes)
        j = rng.randint(0, n_genes)
        if i != j and (i, j) not in pos_set:
            neg_list.append([i, j])
    neg_pairs = np.array(neg_list, dtype=np.int32)

    neg_pam = compute_pam_scores(model, X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])
    neg_cos = compute_cosine_scores(X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])

    all_labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(n_neg)])
    all_cos = np.concatenate([pos_cos, neg_cos])
    all_pam = np.concatenate([pos_pam, neg_pam])

    results = {
        "cosine_auc": float(roc_auc_score(all_labels, all_cos)),
        "pam_auc": float(roc_auc_score(all_labels, all_pam)),
    }

    cb_mask_pos = np.abs(pos_cos) < 0.2
    cb_mask_neg = np.abs(neg_cos) < 0.2
    if cb_mask_pos.sum() > 10 and cb_mask_neg.sum() > 10:
        cb_labels = np.concatenate([np.ones(cb_mask_pos.sum()),
                                    np.zeros(cb_mask_neg.sum())])
        cb_cos = np.concatenate([pos_cos[cb_mask_pos], neg_cos[cb_mask_neg]])
        cb_pam = np.concatenate([pos_pam[cb_mask_pos], neg_pam[cb_mask_neg]])
        results["cb_cosine_auc"] = float(roc_auc_score(cb_labels, cb_cos))
        results["cb_pam_auc"] = float(roc_auc_score(cb_labels, cb_pam))
        results["cb_n_pos"] = int(cb_mask_pos.sum())
        results["cb_n_neg"] = int(cb_mask_neg.sum())

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    t0 = time.time()

    print("DepMap PAM Phase 4: Ablation Controls")
    print("=" * 70)
    print(f"  Device: {DEVICE}")

    # Determine best N from Phase 3
    with open(TRAIN_RESULTS_PATH) as f:
        train_meta = json.load(f)
    best_N = train_meta.get("best_N")
    if best_N is None:
        # Fallback: pick first available
        best_N = int(list(train_meta["results"].keys())[0])
    print(f"  Best N from Phase 3: {best_N:,}")

    # Load data
    emb = np.load(EMB_PATH)
    with open(GENE_LIST_PATH) as f:
        genes = json.load(f)
    n_genes = len(genes)
    X_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)

    pair_path = RESULTS_DIR / f"pairs_{best_N}.npy"
    pairs = np.load(pair_path)
    print(f"  Pairs: {len(pairs)} from {pair_path}")

    # Reference: full model results
    full_res = train_meta["results"][str(best_N)]
    print(f"\n  Full model reference:")
    print(f"    Cosine AUC: {full_res['cosine_auc']:.4f}")
    print(f"    PAM AUC:    {full_res['pam_auc']:.4f}")
    print(f"    Delta:      {full_res['pam_auc'] - full_res['cosine_auc']:+.4f}")

    ablation_results = {}

    # ==================================================================
    # Ablation 1: Shuffled pairs
    # ==================================================================
    print(f"\n{'='*70}")
    print("ABLATION 1: SHUFFLED PAIRS")
    print(f"{'='*70}")
    set_seed(SEED)
    shuf_pairs = pairs.copy()
    perm = np.random.permutation(len(shuf_pairs))
    shuf_pairs[:, 1] = shuf_pairs[perm, 1]

    model_shuf, _ = train_model(X_tensor, shuf_pairs, verbose=True)
    # Evaluate shuffled model on REAL pairs
    res_shuf = evaluate_model(model_shuf, X_tensor, pairs, n_genes)
    ablation_results["shuffled"] = res_shuf

    delta = res_shuf["pam_auc"] - res_shuf["cosine_auc"]
    print(f"\n  Shuffled — evaluated on real pairs:")
    print(f"    PAM AUC: {res_shuf['pam_auc']:.4f}, Delta: {delta:+.4f}")
    if delta < full_res["pam_auc"] - full_res["cosine_auc"]:
        print(f"    PASS: shuffling degrades signal")
    else:
        print(f"    FAIL: shuffled model matches or beats full model")

    # ==================================================================
    # Ablation 2: Similar positives
    # ==================================================================
    print(f"\n{'='*70}")
    print("ABLATION 2: SIMILAR POSITIVES (highest expression cosine)")
    print(f"{'='*70}")
    set_seed(SEED)

    # Build set of real co-essential pairs for exclusion
    real_set = set()
    for i in range(len(pairs)):
        real_set.add((int(pairs[i, 0]), int(pairs[i, 1])))
        real_set.add((int(pairs[i, 1]), int(pairs[i, 0])))

    # Compute all pairwise cosines and find top-N non-co-essential pairs
    print(f"  Computing pairwise cosines for {n_genes} genes...")
    # For large N genes, sample to avoid memory issues
    if n_genes > 10000:
        # Sample upper triangle pairs and rank
        rng = np.random.RandomState(42)
        n_sample = min(best_N * 20, 5_000_000)
        sample_i = rng.randint(0, n_genes, size=n_sample)
        sample_j = rng.randint(0, n_genes, size=n_sample)
        mask = sample_i < sample_j  # upper triangle
        sample_i, sample_j = sample_i[mask], sample_j[mask]
        # Filter out real pairs
        not_real = np.array([(int(a), int(b)) not in real_set
                             for a, b in zip(sample_i, sample_j)])
        sample_i, sample_j = sample_i[not_real], sample_j[not_real]
        cos_vals = np.sum(emb[sample_i] * emb[sample_j], axis=1)
        top_k_idx = np.argsort(cos_vals)[-best_N:]
        sim_pairs = np.stack([sample_i[top_k_idx], sample_j[top_k_idx]],
                             axis=1).astype(np.int32)
    else:
        cos_matrix = emb @ emb.T
        triu_i, triu_j = np.triu_indices(n_genes, k=1)
        cos_flat = cos_matrix[triu_i, triu_j]
        not_real = np.array([(int(triu_i[k]), int(triu_j[k])) not in real_set
                             for k in range(len(triu_i))])
        cos_flat_nr = cos_flat.copy()
        cos_flat_nr[~not_real] = -2  # exclude real pairs
        top_k_idx = np.argsort(cos_flat_nr)[-best_N:]
        sim_pairs = np.stack([triu_i[top_k_idx], triu_j[top_k_idx]],
                             axis=1).astype(np.int32)

    sim_cos_mean = np.sum(emb[sim_pairs[:, 0]] * emb[sim_pairs[:, 1]], axis=1).mean()
    print(f"  Similar pairs: {len(sim_pairs)}, mean cosine: {sim_cos_mean:.3f}")

    model_sim, _ = train_model(X_tensor, sim_pairs, verbose=True)
    # Evaluate on REAL pairs
    res_sim = evaluate_model(model_sim, X_tensor, pairs, n_genes)
    ablation_results["similar_positives"] = res_sim

    delta = res_sim["pam_auc"] - res_sim["cosine_auc"]
    print(f"\n  Similar positives — evaluated on real pairs:")
    print(f"    PAM AUC: {res_sim['pam_auc']:.4f}, Delta: {delta:+.4f}")
    if delta < full_res["pam_auc"] - full_res["cosine_auc"]:
        print(f"    PASS: similar positives do not match full model")
    else:
        print(f"    FAIL: similar positives match or beat full model")

    # ==================================================================
    # Ablation 3: Random negatives
    # ==================================================================
    print(f"\n{'='*70}")
    print("ABLATION 3: RANDOM NEGATIVES")
    print(f"{'='*70}")
    set_seed(SEED)

    model_rn, _ = train_model(X_tensor, pairs, verbose=True,
                               use_random_negatives=True)
    res_rn = evaluate_model(model_rn, X_tensor, pairs, n_genes)
    ablation_results["random_negatives"] = res_rn

    delta = res_rn["pam_auc"] - res_rn["cosine_auc"]
    print(f"\n  Random negatives:")
    print(f"    PAM AUC: {res_rn['pam_auc']:.4f}, Delta: {delta:+.4f}")

    # ==================================================================
    # Ablation 4: Inductive 70/30
    # ==================================================================
    print(f"\n{'='*70}")
    print("ABLATION 4: INDUCTIVE SPLIT (70/30)")
    print(f"{'='*70}")
    set_seed(SEED)

    n_train = int(len(pairs) * 0.7)
    perm_ind = np.random.permutation(len(pairs))
    train_pairs = pairs[perm_ind[:n_train]]
    test_pairs = pairs[perm_ind[n_train:]]
    print(f"  Train: {len(train_pairs)}, Test: {len(test_pairs)}")

    model_ind, _ = train_model(X_tensor, train_pairs, verbose=True)

    # Evaluate on held-out test pairs
    res_ind = evaluate_model(model_ind, X_tensor, test_pairs, n_genes)
    res_ind["n_train"] = len(train_pairs)
    res_ind["n_test"] = len(test_pairs)
    ablation_results["inductive_70_30"] = res_ind

    delta = res_ind["pam_auc"] - res_ind["cosine_auc"]
    print(f"\n  Inductive (evaluated on held-out 30%):")
    print(f"    Cosine AUC: {res_ind['cosine_auc']:.4f}")
    print(f"    PAM AUC:    {res_ind['pam_auc']:.4f}")
    print(f"    Delta:      {delta:+.4f}")

    # ==================================================================
    # Summary
    # ==================================================================
    print(f"\n{'='*70}")
    print("ABLATION SUMMARY")
    print(f"{'='*70}")

    full_delta = full_res["pam_auc"] - full_res["cosine_auc"]
    print(f"\n  {'Model':<22s}  {'PAM AUC':>8s}  {'Delta':>8s}  {'vs Full':>8s}  Status")
    print(f"  {'-'*22}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")

    print(f"  {'Full model':<22s}  {full_res['pam_auc']:>8.4f}  {full_delta:>+8.4f}  "
          f"{'---':>8s}  reference")

    for name, res in ablation_results.items():
        delta = res["pam_auc"] - res["cosine_auc"]
        vs_full = delta - full_delta
        status = ""
        if name == "shuffled":
            status = "PASS" if delta < full_delta else "FAIL"
        elif name == "similar_positives":
            status = "PASS" if delta < full_delta else "FAIL"
        elif name == "random_negatives":
            status = "expected modest effect"
        elif name == "inductive_70_30":
            status = f"{'generalizes' if delta > 0 else 'fails to generalize'}"

        label = name.replace("_", " ")
        print(f"  {label:<22s}  {res['pam_auc']:>8.4f}  {delta:>+8.4f}  "
              f"{vs_full:>+8.4f}  {status}")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    all_results = {
        "best_N": best_N,
        "full_model": full_res,
        "ablations": ablation_results,
        "elapsed_seconds": elapsed,
    }
    results_path = RESULTS_DIR / "04_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results: {results_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: PAM AUC
    ax = axes[0]
    names = ["Full model"] + [k.replace("_", "\n") for k in ablation_results.keys()]
    aucs = [full_res["pam_auc"]] + [r["pam_auc"] for r in ablation_results.values()]
    colors = ["darkorange"] + ["steelblue"] * len(ablation_results)
    bars = ax.bar(range(len(names)), aucs, color=colors, alpha=0.8)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("PAM Discrimination AUC")
    ax.set_title(f"Ablation Comparison (N={best_N:,})")
    ax.axhline(full_res["cosine_auc"], color="gray", linestyle="--",
               label=f"Cosine baseline ({full_res['cosine_auc']:.3f})")
    ax.legend()
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Bar chart: Cross-boundary AUC
    ax = axes[1]
    cb_names = ["Full model"]
    cb_vals = [full_res.get("cb_pam_auc", 0.5)]
    for k, r in ablation_results.items():
        cb_names.append(k.replace("_", "\n"))
        cb_vals.append(r.get("cb_pam_auc", 0.5))
    bars = ax.bar(range(len(cb_names)), cb_vals,
                  color=["darkorange"] + ["steelblue"] * len(ablation_results), alpha=0.8)
    ax.set_xticks(range(len(cb_names)))
    ax.set_xticklabels(cb_names, fontsize=8)
    ax.set_ylabel("Cross-Boundary AUC")
    ax.set_title("Cross-Boundary AUC (|cos| < 0.2)")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    for bar, val in zip(bars, cb_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "04_ablation_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plots: {plot_path}")
    plt.close()

    print(f"\n  Next: python scripts/05_validate.py")


if __name__ == "__main__":
    main()
