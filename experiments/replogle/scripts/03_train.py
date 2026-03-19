#!/usr/bin/env python3
"""
DepMap PAM Phase 3: Train AssociationMLP
=========================================

Train the PAM contrastive MLP on co-essentiality pairs at each N level
(25K, 50K, 100K, 200K) using PCA-100 expression embeddings.

Architecture (AAR):
    f(x) = normalize(alpha * x + (1-alpha) * g(x))
    g = 4-layer MLP, 1024 hidden, GELU + LayerNorm
    Symmetric InfoNCE, tau=0.05, batch 512, AdamW 3e-4, 100 epochs

Scoring — both-transformed (both items are corpus genes):
    assoc(A, B) = f(emb_A) . f(emb_B)

Evaluate: overall discrimination AUC, cross-boundary AUC (|cos| < 0.2),
          compare to expression cosine baseline.

Prerequisites:
    python scripts/02_pairs.py

Usage:
    python scripts/03_train.py
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

PAIR_COUNTS = [25_000, 50_000, 100_000, 200_000]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model: 4-layer MLP with learned sigmoid residual, L2-normalized output
# ---------------------------------------------------------------------------
class AssociationMLP(nn.Module):
    """
    f(x) = normalize(alpha * x + (1-alpha) * g(x))
    where alpha = sigmoid(learned_param), g = 4-layer MLP with GELU + LayerNorm
    """

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
# Contrastive loss: CLIP-style symmetric InfoNCE
# ---------------------------------------------------------------------------
def contrastive_loss(emb_a, emb_b, temperature=TEMPERATURE):
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_model(X_tensor, pairs, epochs=EPOCHS, verbose=True):
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
            print(f"    Epoch {epoch+1:3d}/{epochs}: loss={avg_loss:.4f}, alpha={alpha:.3f}")

    return model, losses


# ---------------------------------------------------------------------------
# Both-transformed scoring: assoc(A, B) = f(emb_A) . f(emb_B)
# ---------------------------------------------------------------------------
@torch.no_grad()
def compute_pam_scores(model, X_tensor, idx_a, idx_b):
    model.eval()
    pred_a = model(X_tensor[idx_a])
    pred_b = model(X_tensor[idx_b])
    return (pred_a * pred_b).sum(dim=-1).cpu().numpy()


@torch.no_grad()
def compute_cosine_scores(X_tensor, idx_a, idx_b):
    return (X_tensor[idx_a] * X_tensor[idx_b]).sum(dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate_model(model, X_tensor, pos_pairs, n_genes):
    pos_idx_a = pos_pairs[:, 0]
    pos_idx_b = pos_pairs[:, 1]
    pos_pam = compute_pam_scores(model, X_tensor, pos_idx_a, pos_idx_b)
    pos_cos = compute_cosine_scores(X_tensor, pos_idx_a, pos_idx_b)

    # Negative pairs: 5x, excluding positives
    n_neg = min(len(pos_pairs) * 5, 200_000)
    pos_set = set()
    for i in range(len(pos_pairs)):
        pos_set.add((int(pos_pairs[i, 0]), int(pos_pairs[i, 1])))
        pos_set.add((int(pos_pairs[i, 1]), int(pos_pairs[i, 0])))

    rng = np.random.RandomState(42)
    neg_pairs_list = []
    while len(neg_pairs_list) < n_neg:
        i = rng.randint(0, n_genes)
        j = rng.randint(0, n_genes)
        if i != j and (i, j) not in pos_set:
            neg_pairs_list.append([i, j])
    neg_pairs = np.array(neg_pairs_list, dtype=np.int32)

    neg_pam = compute_pam_scores(model, X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])
    neg_cos = compute_cosine_scores(X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])

    all_labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(n_neg)])
    all_cos = np.concatenate([pos_cos, neg_cos])
    all_pam = np.concatenate([pos_pam, neg_pam])

    results = {}
    results["cosine_auc"] = float(roc_auc_score(all_labels, all_cos))
    results["pam_auc"] = float(roc_auc_score(all_labels, all_pam))

    # Cross-boundary: restrict to pairs where |expression cosine| < 0.2
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

    print("DepMap PAM Phase 3: Train AssociationMLP")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print()

    # Load embeddings
    emb = np.load(EMB_PATH)
    with open(GENE_LIST_PATH) as f:
        genes = json.load(f)
    n_genes = len(genes)
    print(f"  Embeddings: {emb.shape} ({n_genes} genes, PCA-{emb.shape[1]})")

    X_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)

    all_results = {}

    for N in PAIR_COUNTS:
        pair_path = RESULTS_DIR / f"pairs_{N}.npy"
        if not pair_path.exists():
            print(f"\n  Skipping N={N:,}: {pair_path} not found")
            continue

        pairs = np.load(pair_path)
        print(f"\n{'='*70}")
        print(f"TRAINING N={N:,} PAIRS")
        print(f"{'='*70}")

        set_seed(SEED)
        model, losses = train_model(X_tensor, pairs, verbose=True)
        alpha = torch.sigmoid(model.alpha_logit).item()
        print(f"  Final loss: {losses[-1]:.4f}, alpha: {alpha:.3f}")

        # Evaluate
        results = evaluate_model(model, X_tensor, pairs, n_genes)
        results["final_loss"] = float(losses[-1])
        results["alpha"] = float(alpha)
        results["losses"] = [float(l) for l in losses]

        print(f"\n  Cosine AUC:  {results['cosine_auc']:.4f}")
        print(f"  PAM AUC:     {results['pam_auc']:.4f}")
        delta = results["pam_auc"] - results["cosine_auc"]
        print(f"  Delta:       {delta:+.4f}")

        if "cb_pam_auc" in results:
            cb_delta = results["cb_pam_auc"] - results["cb_cosine_auc"]
            print(f"\n  Cross-boundary (|cos| < 0.2):")
            print(f"    Cosine AUC: {results['cb_cosine_auc']:.4f} "
                  f"(n_pos={results['cb_n_pos']}, n_neg={results['cb_n_neg']})")
            print(f"    PAM AUC:    {results['cb_pam_auc']:.4f}")
            print(f"    Delta:      {cb_delta:+.4f}")

        # Save model
        model_path = RESULTS_DIR / f"model_{N}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "n_pairs": N,
            "alpha": alpha,
        }, model_path)
        print(f"  Model saved: {model_path}")

        all_results[str(N)] = results

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")

    print(f"\n  {'N':>8s}  {'Cos AUC':>8s}  {'PAM AUC':>8s}  {'Delta':>8s}  "
          f"{'CB Cos':>7s}  {'CB PAM':>7s}  {'CB Delta':>9s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*7}  {'-'*9}")

    best_N = None
    best_delta = -1
    for N_str, res in all_results.items():
        N = int(N_str)
        delta = res["pam_auc"] - res["cosine_auc"]
        cb_cos = res.get("cb_cosine_auc", float("nan"))
        cb_pam = res.get("cb_pam_auc", float("nan"))
        cb_delta = cb_pam - cb_cos if not np.isnan(cb_pam) else float("nan")

        marker = ""
        if delta > best_delta:
            best_delta = delta
            best_N = N

        print(f"  {N:>8,}  {res['cosine_auc']:>8.4f}  {res['pam_auc']:>8.4f}  "
              f"{delta:>+8.4f}  {cb_cos:>7.4f}  {cb_pam:>7.4f}  {cb_delta:>+9.4f}")

    if best_N:
        print(f"\n  Best N: {best_N:,} (largest PAM-over-cosine delta)")

    elapsed = time.time() - t0
    print(f"  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # Save
    meta = {
        "config": {
            "seed": SEED, "pca_dim": PCA_DIM, "hidden_dim": HIDDEN_DIM,
            "n_layers": N_LAYERS, "batch_size": BATCH_SIZE, "lr": LR,
            "epochs": EPOCHS, "temperature": TEMPERATURE,
            "n_genes": n_genes, "device": str(DEVICE),
        },
        "results": all_results,
        "best_N": best_N,
        "elapsed_seconds": elapsed,
    }
    results_path = RESULTS_DIR / "03_train_results.json"
    with open(results_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Results: {results_path}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: AUC comparison across N
    ax = axes[0, 0]
    Ns = sorted(int(k) for k in all_results.keys())
    cos_aucs = [all_results[str(n)]["cosine_auc"] for n in Ns]
    pam_aucs = [all_results[str(n)]["pam_auc"] for n in Ns]
    x = np.arange(len(Ns))
    w = 0.35
    ax.bar(x - w/2, cos_aucs, w, label="Cosine", color="gray", alpha=0.8)
    ax.bar(x + w/2, pam_aucs, w, label="PAM", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n//1000}K" for n in Ns])
    ax.set_ylabel("Discrimination AUC")
    ax.set_xlabel("N (co-essential pairs)")
    ax.set_title("Overall AUC: Cosine vs PAM")
    ax.legend()
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3)

    # Plot 2: Cross-boundary AUC
    ax = axes[0, 1]
    cb_cos = [all_results[str(n)].get("cb_cosine_auc", 0.5) for n in Ns]
    cb_pam = [all_results[str(n)].get("cb_pam_auc", 0.5) for n in Ns]
    ax.bar(x - w/2, cb_cos, w, label="Cosine", color="gray", alpha=0.8)
    ax.bar(x + w/2, cb_pam, w, label="PAM", color="red", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{n//1000}K" for n in Ns])
    ax.set_ylabel("Cross-Boundary AUC")
    ax.set_xlabel("N")
    ax.set_title("Cross-Boundary AUC (|cos| < 0.2)")
    ax.legend()
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3)

    # Plot 3: Delta (PAM - Cosine)
    ax = axes[1, 0]
    deltas = [all_results[str(n)]["pam_auc"] - all_results[str(n)]["cosine_auc"] for n in Ns]
    colors = ["green" if d > 0 else "red" for d in deltas]
    ax.bar([f"{n//1000}K" for n in Ns], deltas, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linestyle="-", alpha=0.5)
    ax.set_ylabel("AUC Delta (PAM - Cosine)")
    ax.set_xlabel("N")
    ax.set_title("PAM Improvement Over Cosine")

    # Plot 4: Training loss curves
    ax = axes[1, 1]
    for n in Ns:
        losses = all_results[str(n)].get("losses", [])
        if losses:
            ax.plot(losses, label=f"N={n//1000}K", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title("Training Loss Curves")
    ax.legend()

    plt.tight_layout()
    plot_path = RESULTS_DIR / "03_train_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plots: {plot_path}")
    plt.close()

    print(f"\n  Next: python scripts/04_ablations.py")


if __name__ == "__main__":
    main()
