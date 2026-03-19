"""Phase 4: Ablations at best N.

1. Shuffled pairs — must destroy signal (if not, model learned FP geometry)
2. Similar positives — top-N highest cosine pairs (must not match co-lethality)
3. Random negatives — replace in-batch negatives with explicit random negatives
4. Inductive 70/30 — train on 70% of drugs, evaluate on held-out 30%
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset

# Import model and training from Phase 3
import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_train = importlib.import_module("03_train")
AssociationMLP = _train.AssociationMLP
PairDataset = _train.PairDataset
clip_loss = _train.clip_loss
evaluate_model = _train.evaluate_model

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters (same as Phase 3)
HIDDEN_DIM = 1024
NUM_LAYERS = 4
BATCH_SIZE = 512
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.05
EPOCHS = 100
GRAD_CLIP = 1.0
N_EVAL_NEG = 50_000


# ── Random negatives training ─────────────────────────────────────────────

class PairDatasetWithRandomNeg(Dataset):
    """Dataset that supplies explicit random negatives instead of relying on
    in-batch negatives."""

    def __init__(self, pairs: np.ndarray, embeddings: torch.Tensor,
                 n_neg: int = 15):
        self.pairs = pairs
        self.embeddings = embeddings
        self.n_drugs = len(embeddings)
        self.n_neg = n_neg
        self.pair_set = set(map(tuple, pairs)) | set(map(tuple, pairs[:, ::-1]))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        anchor = self.embeddings[i]
        positive = self.embeddings[j]
        # Sample random negatives
        negs = []
        while len(negs) < self.n_neg:
            k = torch.randint(0, self.n_drugs, (1,)).item()
            if k != i and k != j:
                negs.append(self.embeddings[k])
        negatives = torch.stack(negs)
        return anchor, positive, negatives


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor,
                  negatives: torch.Tensor, temperature: float = TEMPERATURE):
    """InfoNCE with explicit negatives. anchor/positive: [B, D], negatives: [B, N, D]."""
    pos_sim = (anchor * positive).sum(dim=-1, keepdim=True)  # [B, 1]
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1)  # [B, N]
    logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


def train_random_neg(pairs: np.ndarray, embeddings: np.ndarray,
                     epochs: int = EPOCHS, n_neg: int = 15) -> tuple:
    """Train with explicit random negatives."""
    embedding_dim = embeddings.shape[1]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = PairDatasetWithRandomNeg(pairs, emb_tensor, n_neg=n_neg)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    model = AssociationMLP(embedding_dim=embedding_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0

        for anchors, positives, negatives in loader:
            anchors = anchors.to(DEVICE)
            positives = positives.to(DEVICE)
            negatives = negatives.to(DEVICE)

            pred_a = model(anchors)
            pred_p = model(positives)
            # Transform negatives too (both-transformed)
            B, N, D = negatives.shape
            pred_n = model(negatives.view(B * N, D)).view(B, N, -1)

            loss = info_nce_loss(pred_a, pred_p, pred_n)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * anchors.size(0)
            total += anchors.size(0)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            alpha = torch.sigmoid(model.alpha_logit).item()
            print(f"    Epoch {epoch + 1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"alpha={alpha:.3f}")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.0f}s")
    return model, loss_history


# ── In-batch training (reuse from Phase 3) ─────────────────────────────────

def train_inbatch(pairs: np.ndarray, embeddings: np.ndarray,
                  epochs: int = EPOCHS) -> tuple:
    """Train with in-batch negatives (CLIP-style)."""
    embedding_dim = embeddings.shape[1]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = PairDataset(pairs, emb_tensor)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    model = AssociationMLP(embedding_dim=embedding_dim).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total = 0

        for anchors, positives in loader:
            anchors = anchors.to(DEVICE)
            positives = positives.to(DEVICE)

            pred_a = model(anchors)
            pred_b = model(positives)
            loss = clip_loss(pred_a, pred_b)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item() * anchors.size(0)
            total += anchors.size(0)

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        loss_history.append(avg_loss)

        if (epoch + 1) % 20 == 0 or epoch == 0:
            alpha = torch.sigmoid(model.alpha_logit).item()
            print(f"    Epoch {epoch + 1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"alpha={alpha:.3f}")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.0f}s")
    return model, loss_history


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4: Ablations")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    embeddings = np.load(DATA / "embeddings_raw.npy")
    co_lethality = np.load(DATA / "co_lethality.npy")
    n_drugs = len(embeddings)
    print(f"  Drugs: {n_drugs}, Embedding dim: {embeddings.shape[1]}")

    # Determine best N from Phase 3 results
    with open(RESULTS / "03_train.json") as f:
        train_results = json.load(f)
    best_n = max(train_results.keys(), key=lambda k: train_results[k]["auc_pam"])
    best_n = int(best_n)
    print(f"  Best N from Phase 3: {best_n:,}")

    pairs = np.load(DATA / f"pairs_{best_n}.npy")

    # Cosine similarity for all pairs
    emb_norm = normalize(embeddings, axis=1)
    cos_matrix = emb_norm @ emb_norm.T

    results_all = {}

    # ── Ablation 1: Shuffled pairs ─────────────────────────────────────────

    print(f"\n{'─' * 60}")
    print("Ablation 1: Shuffled pairs")
    print(f"{'─' * 60}")
    # Shuffle second column of pairs (break association, keep same drugs)
    rng = np.random.default_rng(42)
    shuffled_pairs = pairs.copy()
    shuffled_pairs[:, 1] = rng.permutation(shuffled_pairs[:, 1])

    model_shuf, _ = train_inbatch(shuffled_pairs, embeddings)
    result_shuf = evaluate_model(model_shuf, embeddings, pairs, co_lethality,
                                 tag="Shuffled: ")
    results_all["shuffled"] = result_shuf

    # ── Ablation 2: Similar positives ──────────────────────────────────────

    print(f"\n{'─' * 60}")
    print("Ablation 2: Similar positives (top-N highest cosine pairs)")
    print(f"{'─' * 60}")
    # Extract top-N pairs by cosine similarity
    triu_idx = np.triu_indices(n_drugs, k=1)
    cos_values = cos_matrix[triu_idx]
    cos_rank = np.argsort(-cos_values)
    sim_top_idx = cos_rank[:best_n]
    similar_pairs = np.stack([triu_idx[0][sim_top_idx], triu_idx[1][sim_top_idx]], axis=1)

    model_sim, _ = train_inbatch(similar_pairs, embeddings)
    result_sim = evaluate_model(model_sim, embeddings, pairs, co_lethality,
                                tag="Similar: ")
    results_all["similar_positives"] = result_sim

    # ── Ablation 3: Random negatives ───────────────────────────────────────

    print(f"\n{'─' * 60}")
    print("Ablation 3: Random negatives (explicit, not in-batch)")
    print(f"{'─' * 60}")
    model_rn, _ = train_random_neg(pairs, embeddings, n_neg=15)
    result_rn = evaluate_model(model_rn, embeddings, pairs, co_lethality,
                               tag="RandomNeg: ")
    results_all["random_negatives"] = result_rn

    # ── Ablation 4: Inductive 70/30 ───────────────────────────────────────

    print(f"\n{'─' * 60}")
    print("Ablation 4: Inductive 70/30 (train on 70% drugs, eval on held-out 30%)")
    print(f"{'─' * 60}")

    # Split drugs 70/30
    drug_perm = rng.permutation(n_drugs)
    n_train = int(0.7 * n_drugs)
    train_drugs = set(drug_perm[:n_train].tolist())
    test_drugs = set(drug_perm[n_train:].tolist())
    print(f"  Train drugs: {len(train_drugs)}, Test drugs: {len(test_drugs)}")

    # Filter pairs: train pairs = both drugs in train set
    train_pairs = np.array([p for p in pairs if p[0] in train_drugs and p[1] in train_drugs])
    # Test pairs: at least one drug in test set
    test_pairs_pos = np.array([p for p in pairs if p[0] in test_drugs or p[1] in test_drugs])
    print(f"  Train pairs: {len(train_pairs)}, Test positive pairs: {len(test_pairs_pos)}")

    if len(train_pairs) < 1000:
        print("  ⚠ Too few train pairs for inductive split. Skipping.")
        results_all["inductive"] = {"skipped": True, "reason": "too_few_train_pairs"}
    else:
        model_ind, _ = train_inbatch(train_pairs, embeddings)

        # Evaluate on held-out test pairs
        if len(test_pairs_pos) > 50:
            result_ind = evaluate_model(model_ind, embeddings, test_pairs_pos,
                                        co_lethality, tag="Inductive: ")
            results_all["inductive"] = result_ind
        else:
            print("  ⚠ Too few test pairs for evaluation.")
            results_all["inductive"] = {"skipped": True, "reason": "too_few_test_pairs"}

    # ── Reference: Phase 3 best model ──────────────────────────────────────

    results_all["reference"] = train_results[str(best_n)]
    results_all["best_n"] = best_n

    # ── Figures ────────────────────────────────────────────────────────────

    print("\n--- Generating figures ---")

    ablation_names = ["reference", "shuffled", "similar_positives", "random_negatives"]
    if not results_all.get("inductive", {}).get("skipped"):
        ablation_names.append("inductive")

    display_names = {
        "reference": f"Co-lethal\n(N={best_n//1000}K)",
        "shuffled": "Shuffled",
        "similar_positives": "Similar\nPositives",
        "random_negatives": "Random\nNegatives",
        "inductive": "Inductive\n70/30",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall AUC
    aucs = [results_all[name]["auc_pam"] for name in ablation_names]
    cos_aucs = [results_all[name]["auc_cos"] for name in ablation_names]
    x = range(len(ablation_names))
    width = 0.35

    axes[0].bar([xi - width / 2 for xi in x], aucs, width,
                label="PAM", color="steelblue")
    axes[0].bar([xi + width / 2 for xi in x], cos_aucs, width,
                label="Cosine", color="orange")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([display_names[n] for n in ablation_names], fontsize=9)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Overall AUC: Ablation Comparison")
    axes[0].legend()
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.3)

    # Cross-boundary AUC
    cb_aucs = [results_all[name].get("auc_pam_cb") for name in ablation_names]
    cb_cos = [results_all[name].get("auc_cos_cb") for name in ablation_names]
    valid_idx = [i for i, v in enumerate(cb_aucs) if v is not None]

    if valid_idx:
        valid_names = [ablation_names[i] for i in valid_idx]
        valid_pam = [cb_aucs[i] for i in valid_idx]
        valid_cos = [cb_cos[i] for i in valid_idx]
        x2 = range(len(valid_names))
        axes[1].bar([xi - width / 2 for xi in x2], valid_pam, width,
                    label="PAM", color="steelblue")
        axes[1].bar([xi + width / 2 for xi in x2], valid_cos, width,
                    label="Cosine", color="orange")
        axes[1].set_xticks(list(x2))
        axes[1].set_xticklabels([display_names[n] for n in valid_names], fontsize=9)
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Cross-boundary AUC (|cos| < 0.2)")
        axes[1].legend()
        axes[1].axhline(0.5, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "04_ablations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/04_ablations.png")

    # ── Save results ───────────────────────────────────────────────────────

    # Convert any non-serializable values
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(RESULTS / "04_ablations.json", "w") as f:
        json.dump(clean_for_json(results_all), f, indent=2)
    print(f"  Saved: results/04_ablations.json")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)

    for name in ablation_names:
        r = results_all[name]
        if isinstance(r, dict) and "auc_pam" in r:
            cb_str = ""
            if r.get("auc_pam_cb") is not None:
                cb_str = f", CB={r['auc_pam_cb']:.4f}"
            print(f"  {display_names[name].replace(chr(10), ' ')}: "
                  f"PAM={r['auc_pam']:.4f}, Cos={r['auc_cos']:.4f}{cb_str}")

    # Kill criteria
    ref = results_all["reference"]
    shuf = results_all["shuffled"]
    sim = results_all["similar_positives"]

    if shuf["auc_pam"] >= ref["auc_pam"] * 0.95:
        print("\n  ⚠ KILL: Shuffled ablation doesn't destroy signal.")
        print("  Model may have learned fingerprint geometry, not association.")

    if sim["auc_pam"] >= ref["auc_pam"] * 0.95:
        print("\n  ⚠ KILL: Similar-positives matches co-lethality performance.")
        print("  Chemical similarity already captures co-lethality.")

    print("\nPhase 4 complete.")
