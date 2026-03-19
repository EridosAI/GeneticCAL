#!/usr/bin/env python3
"""
DepMap PAM Phase 3b: Retrain with Random Negatives
====================================================

The in-batch negative strategy fails on co-essentiality data because
positive pairs cluster tightly in expression space, making in-batch
negatives effectively other co-essential genes. Random negatives fix
this by providing truly unrelated genes as contrast.

This script trains at N=200K with random negatives and evaluates:
  - Overall discrimination AUC (co-essential vs random pairs)
  - Cross-boundary AUC (restricted to |expression cosine| < 0.2)
  - STRING external validation at 400/700/900

Usage:
    python scripts/03b_train_random_neg.py
"""

import gzip
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
N_RANDOM_NEG = 1023  # negatives per anchor
N_PAIRS = 200_000

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

EMB_PATH = RESULTS_DIR / "gene_embeddings_pca100.npy"
GENE_LIST_PATH = RESULTS_DIR / "gene_list.json"
PAIR_PATH = RESULTS_DIR / f"pairs_{N_PAIRS}.npy"

STRING_INFO_FILE = DATA_DIR / "9606.protein.info.v12.0.txt.gz"
STRING_LINKS_FILE = DATA_DIR / "9606.protein.links.detailed.v12.0.txt.gz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Model
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
# Random-negative contrastive loss
# ---------------------------------------------------------------------------
def contrastive_loss_random_neg(pred_a, pred_b, all_pred, temperature=TEMPERATURE,
                                n_neg=N_RANDOM_NEG):
    """
    For each anchor pred_a[i], positive is pred_b[i],
    negatives are n_neg random samples from all_pred.
    Symmetric: also compute loss from pred_b's perspective.
    """
    B = pred_a.shape[0]
    device = pred_a.device

    neg_idx = torch.randint(0, len(all_pred), (B, n_neg), device=device)
    neg_emb = all_pred[neg_idx]  # B x n_neg x D

    # Forward: anchor = pred_a, positive = pred_b
    pos_sim = (pred_a * pred_b).sum(dim=-1, keepdim=True) / temperature  # B x 1
    neg_sim = torch.bmm(neg_emb, pred_a.unsqueeze(-1)).squeeze(-1) / temperature  # B x n_neg
    logits_a = torch.cat([pos_sim, neg_sim], dim=1)  # B x (1+n_neg)
    labels = torch.zeros(B, dtype=torch.long, device=device)
    loss_a = F.cross_entropy(logits_a, labels)

    # Backward: anchor = pred_b, positive = pred_a
    neg_sim_b = torch.bmm(neg_emb, pred_b.unsqueeze(-1)).squeeze(-1) / temperature
    logits_b = torch.cat([pos_sim, neg_sim_b], dim=1)
    loss_b = F.cross_entropy(logits_b, labels)

    return (loss_a + loss_b) / 2


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train_model(X_tensor, pairs):
    model = AssociationMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    n_pairs = len(pairs)
    losses = []

    for epoch in range(EPOCHS):
        model.train()
        perm = np.random.permutation(n_pairs)
        epoch_loss = 0
        n_batches = 0

        # Pre-compute all transformed embeddings for negative sampling
        with torch.no_grad():
            all_pred = model(X_tensor)

        for start in range(0, n_pairs, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 8:
                continue

            emb_a = X_tensor[pairs[batch_idx, 0]]
            emb_b = X_tensor[pairs[batch_idx, 1]]
            pred_a = model(emb_a)
            pred_b = model(emb_b)

            loss = contrastive_loss_random_neg(pred_a, pred_b, all_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)

        if (epoch + 1) % 10 == 0:
            alpha = torch.sigmoid(model.alpha_logit).item()
            print(f"    Epoch {epoch+1:3d}/{EPOCHS}: loss={avg_loss:.4f}, alpha={alpha:.3f}")

    return model, losses


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def score_pairs(model, X_tensor, idx_a, idx_b):
    """Both-transformed: f(emb_A) . f(emb_B)"""
    model.eval()
    pred_a = model(X_tensor[idx_a])
    pred_b = model(X_tensor[idx_b])
    return (pred_a * pred_b).sum(dim=-1).cpu().numpy()


@torch.no_grad()
def cosine_pairs(X_tensor, idx_a, idx_b):
    return (X_tensor[idx_a] * X_tensor[idx_b]).sum(dim=-1).cpu().numpy()


# ---------------------------------------------------------------------------
# Evaluation with full breakdown
# ---------------------------------------------------------------------------
def full_evaluation(model, X_tensor, pos_pairs, n_genes, label=""):
    pos_pam = score_pairs(model, X_tensor, pos_pairs[:, 0], pos_pairs[:, 1])
    pos_cos = cosine_pairs(X_tensor, pos_pairs[:, 0], pos_pairs[:, 1])

    # Negatives
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

    neg_pam = score_pairs(model, X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])
    neg_cos = cosine_pairs(X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])

    all_labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(n_neg)])
    all_cos = np.concatenate([pos_cos, neg_cos])
    all_pam = np.concatenate([pos_pam, neg_pam])

    results = {
        "n_pos": len(pos_pairs),
        "n_neg": n_neg,
        "cosine_auc": float(roc_auc_score(all_labels, all_cos)),
        "pam_auc": float(roc_auc_score(all_labels, all_pam)),
        "pos_cos_mean": float(pos_cos.mean()),
        "pos_pam_mean": float(pos_pam.mean()),
        "neg_cos_mean": float(neg_cos.mean()),
        "neg_pam_mean": float(neg_pam.mean()),
    }

    # Cross-boundary: |expression cosine| < 0.2
    for threshold in [0.1, 0.2, 0.3]:
        cb_mask_pos = np.abs(pos_cos) < threshold
        cb_mask_neg = np.abs(neg_cos) < threshold
        n_cb_pos = cb_mask_pos.sum()
        n_cb_neg = cb_mask_neg.sum()

        if n_cb_pos > 10 and n_cb_neg > 10:
            cb_labels = np.concatenate([np.ones(n_cb_pos), np.zeros(n_cb_neg)])
            cb_cos = np.concatenate([pos_cos[cb_mask_pos], neg_cos[cb_mask_neg]])
            cb_pam = np.concatenate([pos_pam[cb_mask_pos], neg_pam[cb_mask_neg]])

            cb_cos_auc = roc_auc_score(cb_labels, cb_cos)
            cb_pam_auc = roc_auc_score(cb_labels, cb_pam)

            results[f"cb{threshold}_cosine_auc"] = float(cb_cos_auc)
            results[f"cb{threshold}_pam_auc"] = float(cb_pam_auc)
            results[f"cb{threshold}_delta"] = float(cb_pam_auc - cb_cos_auc)
            results[f"cb{threshold}_n_pos"] = int(n_cb_pos)
            results[f"cb{threshold}_n_neg"] = int(n_cb_neg)

    return results, all_labels, all_cos, all_pam, pos_cos, neg_cos, pos_pam, neg_pam


# ---------------------------------------------------------------------------
# STRING validation
# ---------------------------------------------------------------------------
def string_validation(model, X_tensor, genes, gene_to_idx, n_genes):
    print("\n" + "=" * 70)
    print("STRING EXTERNAL VALIDATION")
    print("=" * 70)

    if not STRING_INFO_FILE.exists() or not STRING_LINKS_FILE.exists():
        print("  STRING files not found — skipping.")
        return {}

    # Parse STRING
    ensp_to_gene = {}
    with gzip.open(STRING_INFO_FILE, "rt") as f:
        f.readline()
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ensp_to_gene[parts[0]] = parts[1].upper()

    relevant_ensp = {ensp for ensp, gene in ensp_to_gene.items()
                     if gene in gene_to_idx}

    pairs_by_conf = {400: set(), 700: set(), 900: set()}
    with gzip.open(STRING_LINKS_FILE, "rt") as f:
        header = f.readline().strip().split()
        col_map = {name: i for i, name in enumerate(header)}
        for line in f:
            parts = line.strip().split()
            p1, p2 = parts[0], parts[1]
            if p1 not in relevant_ensp or p2 not in relevant_ensp:
                continue
            gene1 = ensp_to_gene[p1]
            gene2 = ensp_to_gene[p2]
            if gene1 == gene2 or gene1 not in gene_to_idx or gene2 not in gene_to_idx:
                continue
            combined = int(parts[col_map["combined_score"]])
            pair_key = (min(gene1, gene2), max(gene1, gene2))
            for t in [400, 700, 900]:
                if combined >= t:
                    pairs_by_conf[t].add(pair_key)

    rng = np.random.RandomState(42)
    string_results = {}

    for threshold in [400, 700, 900]:
        pair_set = pairs_by_conf[threshold]
        if len(pair_set) < 20:
            continue

        pos_pairs = np.array([[gene_to_idx[g1], gene_to_idx[g2]]
                              for g1, g2 in pair_set], dtype=np.int32)

        n_neg = min(len(pos_pairs) * 5, 200_000)
        pos_set_idx = set()
        for p in pos_pairs:
            pos_set_idx.add((p[0], p[1]))
            pos_set_idx.add((p[1], p[0]))

        neg_list = []
        while len(neg_list) < n_neg:
            i = rng.randint(0, n_genes)
            j = rng.randint(0, n_genes)
            if i != j and (i, j) not in pos_set_idx:
                neg_list.append([i, j])
        neg_pairs = np.array(neg_list, dtype=np.int32)

        pos_pam = score_pairs(model, X_tensor, pos_pairs[:, 0], pos_pairs[:, 1])
        neg_pam = score_pairs(model, X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])
        pos_cos = cosine_pairs(X_tensor, pos_pairs[:, 0], pos_pairs[:, 1])
        neg_cos = cosine_pairs(X_tensor, neg_pairs[:, 0], neg_pairs[:, 1])

        labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(n_neg)])
        all_cos = np.concatenate([pos_cos, neg_cos])
        all_pam = np.concatenate([pos_pam, neg_pam])

        cos_auc = roc_auc_score(labels, all_cos)
        pam_auc = roc_auc_score(labels, all_pam)

        # Cross-boundary on STRING
        cb_mask_pos = np.abs(pos_cos) < 0.2
        cb_mask_neg = np.abs(neg_cos) < 0.2
        cb_info = {}
        if cb_mask_pos.sum() > 10 and cb_mask_neg.sum() > 10:
            cb_labels = np.concatenate([np.ones(cb_mask_pos.sum()),
                                        np.zeros(cb_mask_neg.sum())])
            cb_cos_auc = roc_auc_score(
                cb_labels,
                np.concatenate([pos_cos[cb_mask_pos], neg_cos[cb_mask_neg]]))
            cb_pam_auc = roc_auc_score(
                cb_labels,
                np.concatenate([pos_pam[cb_mask_pos], neg_pam[cb_mask_neg]]))
            cb_info = {
                "cb_cosine_auc": float(cb_cos_auc),
                "cb_pam_auc": float(cb_pam_auc),
                "cb_delta": float(cb_pam_auc - cb_cos_auc),
                "cb_n_pos": int(cb_mask_pos.sum()),
            }

        string_results[threshold] = {
            "n_pairs": len(pos_pairs),
            "cosine_auc": float(cos_auc),
            "pam_auc": float(pam_auc),
            "delta": float(pam_auc - cos_auc),
            **cb_info,
        }

        print(f"\n  STRING >= {threshold} ({len(pos_pairs):,} pairs):")
        print(f"    Cosine AUC:  {cos_auc:.4f}")
        print(f"    PAM AUC:     {pam_auc:.4f}")
        print(f"    Delta:       {pam_auc - cos_auc:+.4f}")
        if cb_info:
            print(f"    Cross-boundary (|cos|<0.2, n_pos={cb_info['cb_n_pos']}):")
            print(f"      Cosine AUC: {cb_info['cb_cosine_auc']:.4f}")
            print(f"      PAM AUC:    {cb_info['cb_pam_auc']:.4f}")
            print(f"      Delta:      {cb_info['cb_delta']:+.4f}")

    return string_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)
    t0 = time.time()

    print("DepMap PAM 3b: Retrain with Random Negatives (N=200K)")
    print("=" * 70)
    print(f"  Device: {DEVICE}")
    print(f"  Random negatives per anchor: {N_RANDOM_NEG}")
    print()

    # Load
    emb = np.load(EMB_PATH)
    with open(GENE_LIST_PATH) as f:
        genes = json.load(f)
    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    pairs = np.load(PAIR_PATH)
    X_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
    print(f"  Genes: {n_genes}, Pairs: {len(pairs):,}")
    print(f"  Embeddings: {emb.shape}")

    # ======================================================================
    # Train
    # ======================================================================
    print(f"\n{'='*70}")
    print("TRAINING (random negatives)")
    print(f"{'='*70}")

    model, losses = train_model(X_tensor, pairs)
    alpha = torch.sigmoid(model.alpha_logit).item()
    print(f"\n  Final loss: {losses[-1]:.4f}, alpha: {alpha:.3f}")

    # Save model
    model_path = RESULTS_DIR / "model_200000_randneg.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "n_pairs": N_PAIRS,
        "alpha": alpha,
        "strategy": "random_negatives",
        "n_random_neg": N_RANDOM_NEG,
    }, model_path)
    print(f"  Model: {model_path}")

    # ======================================================================
    # Evaluate on co-essentiality pairs
    # ======================================================================
    print(f"\n{'='*70}")
    print("EVALUATION: CO-ESSENTIALITY DISCRIMINATION")
    print(f"{'='*70}")

    eval_res, all_labels, all_cos, all_pam, pos_cos, neg_cos, pos_pam, neg_pam = \
        full_evaluation(model, X_tensor, pairs, n_genes)

    print(f"\n  Overall:")
    print(f"    Cosine AUC: {eval_res['cosine_auc']:.4f}")
    print(f"    PAM AUC:    {eval_res['pam_auc']:.4f}")
    print(f"    Delta:      {eval_res['pam_auc'] - eval_res['cosine_auc']:+.4f}")
    print(f"    Pos PAM mean: {eval_res['pos_pam_mean']:.4f}  "
          f"Neg PAM mean: {eval_res['neg_pam_mean']:.4f}")

    for t in [0.1, 0.2, 0.3]:
        k = f"cb{t}"
        if f"{k}_pam_auc" in eval_res:
            print(f"\n  Cross-boundary |cos| < {t} "
                  f"(n_pos={eval_res[f'{k}_n_pos']}, n_neg={eval_res[f'{k}_n_neg']}):")
            print(f"    Cosine AUC: {eval_res[f'{k}_cosine_auc']:.4f}")
            print(f"    PAM AUC:    {eval_res[f'{k}_pam_auc']:.4f}")
            print(f"    Delta:      {eval_res[f'{k}_delta']:+.4f}")

    # ======================================================================
    # STRING validation
    # ======================================================================
    string_results = string_validation(model, X_tensor, genes, gene_to_idx, n_genes)

    # ======================================================================
    # Save everything
    # ======================================================================
    elapsed = time.time() - t0

    all_results = {
        "config": {
            "seed": SEED, "pca_dim": PCA_DIM, "hidden_dim": HIDDEN_DIM,
            "n_layers": N_LAYERS, "batch_size": BATCH_SIZE, "lr": LR,
            "epochs": EPOCHS, "temperature": TEMPERATURE,
            "n_random_neg": N_RANDOM_NEG, "n_pairs": N_PAIRS,
            "n_genes": n_genes, "device": str(DEVICE),
            "strategy": "random_negatives",
        },
        "training": {
            "final_loss": float(losses[-1]),
            "alpha": float(alpha),
            "losses": [float(l) for l in losses],
        },
        "coessentiality_eval": eval_res,
        "string_validation": {str(k): v for k, v in string_results.items()},
        "elapsed_seconds": elapsed,
    }

    results_path = RESULTS_DIR / "03b_randneg_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results: {results_path}")

    # ======================================================================
    # Plots
    # ======================================================================
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed — skipping plots.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Training loss
    ax = axes[0, 0]
    ax.plot(losses, color="darkorange", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Contrastive Loss")
    ax.set_title(f"Training Loss (random negatives, final={losses[-1]:.4f})")

    # Plot 2: Overall + cross-boundary AUC comparison
    ax = axes[0, 1]
    categories = ["Overall"]
    cos_vals = [eval_res["cosine_auc"]]
    pam_vals = [eval_res["pam_auc"]]
    for t in [0.3, 0.2, 0.1]:
        k = f"cb{t}"
        if f"{k}_pam_auc" in eval_res:
            categories.append(f"|cos|<{t}\n(n={eval_res[f'{k}_n_pos']})")
            cos_vals.append(eval_res[f"{k}_cosine_auc"])
            pam_vals.append(eval_res[f"{k}_pam_auc"])

    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w/2, cos_vals, w, label="Cosine", color="gray", alpha=0.8)
    ax.bar(x + w/2, pam_vals, w, label="PAM (rand neg)", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylabel("Discrimination AUC")
    ax.set_title("Co-Essentiality: Cosine vs PAM")
    ax.legend()
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3)
    for i, (c, p) in enumerate(zip(cos_vals, pam_vals)):
        ax.text(i - w/2, c + 0.005, f"{c:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=8)

    # Plot 3: STRING validation
    ax = axes[1, 0]
    if string_results:
        thresholds = sorted(string_results.keys())
        x = np.arange(len(thresholds))
        s_cos = [string_results[t]["cosine_auc"] for t in thresholds]
        s_pam = [string_results[t]["pam_auc"] for t in thresholds]
        ax.bar(x - w/2, s_cos, w, label="Cosine", color="gray", alpha=0.8)
        ax.bar(x + w/2, s_pam, w, label="PAM", color="darkorange", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f">={t}\n(n={string_results[t]['n_pairs']})"
                            for t in thresholds])
        ax.set_ylabel("AUC")
        ax.set_xlabel("STRING confidence")
        ax.set_title("STRING Validation")
        ax.legend()
        ax.axhline(0.5, color="black", linestyle="--", alpha=0.3)
        for i, (c, p) in enumerate(zip(s_cos, s_pam)):
            ax.text(i - w/2, c + 0.005, f"{c:.3f}", ha="center", fontsize=8)
            ax.text(i + w/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "No STRING data", ha="center", va="center",
                transform=ax.transAxes)

    # Plot 4: PAM score distributions
    ax = axes[1, 1]
    ax.hist(pos_pam, bins=80, alpha=0.5, color="darkorange", density=True,
            label=f"Co-essential (n={len(pos_pam)})")
    ax.hist(neg_pam, bins=80, alpha=0.5, color="gray", density=True,
            label=f"Random (n={len(neg_pam)})")
    ax.set_xlabel("PAM score (f(A).f(B))")
    ax.set_ylabel("Density")
    ax.set_title("PAM Score Distribution")
    ax.legend()

    plt.tight_layout()
    plot_path = RESULTS_DIR / "03b_randneg_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plots: {plot_path}")
    plt.close()

    # ======================================================================
    # Final verdict
    # ======================================================================
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")
    overall_delta = eval_res["pam_auc"] - eval_res["cosine_auc"]
    print(f"\n  Overall:  PAM {eval_res['pam_auc']:.4f} vs Cosine {eval_res['cosine_auc']:.4f} "
          f"(delta {overall_delta:+.4f})")

    if "cb0.2_delta" in eval_res:
        cb_delta = eval_res["cb0.2_delta"]
        print(f"\n  Cross-boundary (|cos|<0.2):")
        print(f"            PAM {eval_res['cb0.2_pam_auc']:.4f} vs "
              f"Cosine {eval_res['cb0.2_cosine_auc']:.4f} "
              f"(delta {cb_delta:+.4f})")

    print(f"\n  Total time: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
