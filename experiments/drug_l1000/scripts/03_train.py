"""Phase 3: Train PAM (AssociationMLP) with both in-batch and random negatives.

Trains contrastive model on co-lethal drug pairs using L1000 transcriptional
embeddings. Both negative sampling strategies are run at each N level.
Evaluates overall AUC and cross-boundary AUC vs cosine baseline.
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

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
MODELS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Hyperparameters ────────────────────────────────────────────────────────

HIDDEN_DIM = 1024
NUM_LAYERS = 4
BATCH_SIZE = 512
LR = 3e-4
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.05
EPOCHS = 100
GRAD_CLIP = 1.0
N_EVAL_NEG = 50_000
N_RANDOM_NEG = 15  # explicit negatives per positive for random neg training


# ── Model ──────────────────────────────────────────────────────────────────

class AssociationMLP(nn.Module):
    """PAM predictor: transforms embeddings into association-query space.

    f(x) = normalize(alpha * x + (1-alpha) * g(x))
    g = multi-layer MLP with GELU + LayerNorm
    alpha = sigmoid(learned_param)
    """

    def __init__(self, embedding_dim: int, hidden_dim: int = HIDDEN_DIM,
                 num_layers: int = NUM_LAYERS):
        super().__init__()
        layers = []
        layers.append(nn.Linear(embedding_dim, hidden_dim))
        layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, embedding_dim))
        self.net = nn.Sequential(*layers)
        self.alpha_logit = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.net(x)
        alpha = torch.sigmoid(self.alpha_logit)
        out = alpha * x + (1 - alpha) * transformed
        return F.normalize(out, dim=-1)


# ── Datasets ──────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, pairs: np.ndarray, embeddings: torch.Tensor):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.embeddings[i], self.embeddings[j]


class PairDatasetWithRandomNeg(Dataset):
    """Dataset that supplies explicit random negatives."""

    def __init__(self, pairs: np.ndarray, embeddings: torch.Tensor,
                 n_neg: int = N_RANDOM_NEG):
        self.pairs = pairs
        self.embeddings = embeddings
        self.n_drugs = len(embeddings)
        self.n_neg = n_neg

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        anchor = self.embeddings[i]
        positive = self.embeddings[j]
        negs = []
        while len(negs) < self.n_neg:
            k = torch.randint(0, self.n_drugs, (1,)).item()
            if k != i and k != j:
                negs.append(self.embeddings[k])
        negatives = torch.stack(negs)
        return anchor, positive, negatives


# ── Loss functions ─────────────────────────────────────────────────────────

def clip_loss(emb_a: torch.Tensor, emb_b: torch.Tensor,
              temperature: float = TEMPERATURE) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style) with in-batch negatives."""
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor,
                  negatives: torch.Tensor, temperature: float = TEMPERATURE):
    """InfoNCE with explicit negatives. anchor/positive: [B, D], negatives: [B, N, D]."""
    pos_sim = (anchor * positive).sum(dim=-1, keepdim=True)  # [B, 1]
    neg_sim = torch.bmm(negatives, anchor.unsqueeze(-1)).squeeze(-1)  # [B, N]
    logits = torch.cat([pos_sim, neg_sim], dim=-1) / temperature
    labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, labels)


# ── Training functions ─────────────────────────────────────────────────────

def train_inbatch(pairs: np.ndarray, embeddings: np.ndarray,
                  epochs: int = EPOCHS, tag: str = "") -> tuple:
    """Train AssociationMLP with in-batch negatives."""
    embedding_dim = embeddings.shape[1]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = PairDataset(pairs, emb_tensor)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    model = AssociationMLP(embedding_dim=embedding_dim).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [{tag}] Model: {embedding_dim}d -> {HIDDEN_DIM}h x {NUM_LAYERS}L, "
          f"{param_count:,} params")
    print(f"  [{tag}] In-batch negatives: {BATCH_SIZE - 1} per positive")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_history = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
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

            B = anchors.size(0)
            total_loss += loss.item() * B

            with torch.no_grad():
                sim = pred_a @ pred_b.T
                preds = sim.argmax(dim=1)
                labels = torch.arange(B, device=DEVICE)
                correct += (preds == labels).sum().item()
                total += B

        scheduler.step()
        avg_loss = total_loss / len(dataset)
        accuracy = correct / total if total > 0 else 0.0
        loss_history.append(avg_loss)

        marker = ""
        if accuracy > best_acc:
            best_acc = accuracy
            marker = " *"

        if (epoch + 1) % 20 == 0 or epoch == 0:
            alpha = torch.sigmoid(model.alpha_logit).item()
            print(f"    [{tag}] Epoch {epoch + 1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"acc={accuracy:.4f}, alpha={alpha:.3f}{marker}")

    elapsed = time.time() - t0
    print(f"  [{tag}] Training complete in {elapsed:.0f}s. Best accuracy: {best_acc:.4f}")
    return model, loss_history


def train_random_neg(pairs: np.ndarray, embeddings: np.ndarray,
                     epochs: int = EPOCHS, n_neg: int = N_RANDOM_NEG,
                     tag: str = "") -> tuple:
    """Train AssociationMLP with explicit random negatives."""
    embedding_dim = embeddings.shape[1]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = PairDatasetWithRandomNeg(pairs, emb_tensor, n_neg=n_neg)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    model = AssociationMLP(embedding_dim=embedding_dim).to(DEVICE)
    print(f"  [{tag}] Random negatives: {n_neg} per positive")

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
            print(f"    [{tag}] Epoch {epoch + 1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"alpha={alpha:.3f}")

    elapsed = time.time() - t0
    print(f"  [{tag}] Training complete in {elapsed:.0f}s")
    return model, loss_history


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, embeddings: np.ndarray, pairs: np.ndarray,
                   co_lethality: np.ndarray, tag: str = "") -> dict:
    """Evaluate PAM vs cosine on overall and cross-boundary discrimination."""
    model.eval()
    n_drugs = len(embeddings)

    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = model(emb_tensor).cpu().numpy()

    pam_matrix = z @ z.T

    from sklearn.preprocessing import normalize as sk_normalize
    emb_norm = sk_normalize(embeddings, axis=1)
    cos_matrix = emb_norm @ emb_norm.T

    n_pos = len(pairs)
    pos_pam = np.array([pam_matrix[i, j] for i, j in pairs])
    pos_cos = np.array([cos_matrix[i, j] for i, j in pairs])

    pos_set = set(map(tuple, pairs)) | set(map(tuple, pairs[:, ::-1]))
    rng = np.random.default_rng(42)
    neg_pairs = []
    while len(neg_pairs) < min(N_EVAL_NEG, n_pos):
        i = rng.integers(0, n_drugs)
        j = rng.integers(0, n_drugs)
        if i != j and (i, j) not in pos_set:
            neg_pairs.append((i, j))
    neg_pairs = np.array(neg_pairs)

    neg_pam = np.array([pam_matrix[i, j] for i, j in neg_pairs])
    neg_cos = np.array([cos_matrix[i, j] for i, j in neg_pairs])

    labels = np.concatenate([np.ones(len(pos_pam)), np.zeros(len(neg_pam))])
    auc_pam = roc_auc_score(labels, np.concatenate([pos_pam, neg_pam]))
    auc_cos = roc_auc_score(labels, np.concatenate([pos_cos, neg_cos]))

    print(f"  {tag}Overall AUC: PAM={auc_pam:.4f}, Cosine={auc_cos:.4f}, "
          f"Δ={auc_pam - auc_cos:+.4f}")

    cb_pos_mask = np.abs(pos_cos) < 0.2
    cb_neg_mask = np.abs(neg_cos) < 0.2

    auc_pam_cb = None
    auc_cos_cb = None
    if cb_pos_mask.sum() > 50 and cb_neg_mask.sum() > 50:
        cb_labels = np.concatenate([
            np.ones(cb_pos_mask.sum()), np.zeros(cb_neg_mask.sum())
        ])
        auc_pam_cb = roc_auc_score(
            cb_labels,
            np.concatenate([pos_pam[cb_pos_mask], neg_pam[cb_neg_mask]])
        )
        auc_cos_cb = roc_auc_score(
            cb_labels,
            np.concatenate([pos_cos[cb_pos_mask], neg_cos[cb_neg_mask]])
        )
        print(f"  {tag}Cross-boundary AUC (|cos|<0.2): PAM={auc_pam_cb:.4f}, "
              f"Cosine={auc_cos_cb:.4f}, Δ={auc_pam_cb - auc_cos_cb:+.4f}")
        print(f"    (pos={cb_pos_mask.sum()}, neg={cb_neg_mask.sum()})")
    else:
        print(f"  {tag}Cross-boundary: insufficient pairs "
              f"(pos={cb_pos_mask.sum()}, neg={cb_neg_mask.sum()})")

    alpha = torch.sigmoid(model.alpha_logit).item()

    return {
        "auc_pam": float(auc_pam),
        "auc_cos": float(auc_cos),
        "auc_delta": float(auc_pam - auc_cos),
        "auc_pam_cb": float(auc_pam_cb) if auc_pam_cb is not None else None,
        "auc_cos_cb": float(auc_cos_cb) if auc_cos_cb is not None else None,
        "auc_delta_cb": float(auc_pam_cb - auc_cos_cb) if auc_pam_cb is not None else None,
        "cb_pos_count": int(cb_pos_mask.sum()),
        "cb_neg_count": int(cb_neg_mask.sum()),
        "alpha": float(alpha),
        "n_pos": int(n_pos),
        "n_neg": int(len(neg_pairs)),
    }


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 3: Train (In-batch + Random Negatives)")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    embeddings = np.load(DATA / "embeddings_raw.npy")
    co_lethality = np.load(DATA / "co_lethality.npy")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Co-lethality: {co_lethality.shape}")
    print(f"  Device: {DEVICE}")

    # Find all pair files
    pair_files = sorted(DATA.glob("pairs_*.npy"))
    N_LEVELS = []
    for pf in pair_files:
        n = int(pf.stem.split("_")[1])
        N_LEVELS.append(n)
    print(f"  N levels: {N_LEVELS}")

    results_all = {}
    all_losses = {}

    for N in N_LEVELS:
        pairs = np.load(DATA / f"pairs_{N}.npy")

        # ── In-batch negatives ──────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print(f"Training with N = {N:,} pairs — IN-BATCH negatives")
        print(f"{'─' * 60}")

        model_ib, losses_ib = train_inbatch(pairs, embeddings, tag=f"IB-{N//1000}K")
        all_losses[f"ib_{N}"] = losses_ib

        torch.save({
            "model_state_dict": model_ib.state_dict(),
            "embedding_dim": embeddings.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "n_pairs": N,
            "neg_type": "inbatch",
        }, MODELS / f"pam_ib_{N}.pt")
        print(f"  Saved: models/pam_ib_{N}.pt")

        result_ib = evaluate_model(model_ib, embeddings, pairs, co_lethality,
                                   tag=f"IB-{N//1000}K: ")
        result_ib["n_pairs"] = N
        result_ib["neg_type"] = "inbatch"
        results_all[f"ib_{N}"] = result_ib

        # ── Random negatives ────────────────────────────────────────────
        print(f"\n{'─' * 60}")
        print(f"Training with N = {N:,} pairs — RANDOM negatives")
        print(f"{'─' * 60}")

        model_rn, losses_rn = train_random_neg(pairs, embeddings,
                                                tag=f"RN-{N//1000}K")
        all_losses[f"rn_{N}"] = losses_rn

        torch.save({
            "model_state_dict": model_rn.state_dict(),
            "embedding_dim": embeddings.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "n_pairs": N,
            "neg_type": "random",
        }, MODELS / f"pam_rn_{N}.pt")
        print(f"  Saved: models/pam_rn_{N}.pt")

        result_rn = evaluate_model(model_rn, embeddings, pairs, co_lethality,
                                   tag=f"RN-{N//1000}K: ")
        result_rn["n_pairs"] = N
        result_rn["neg_type"] = "random"
        results_all[f"rn_{N}"] = result_rn

    # ── Determine best configuration ──────────────────────────────────────

    best_key = max(results_all.keys(), key=lambda k: results_all[k]["auc_pam"])
    best_result = results_all[best_key]
    best_neg_type = best_result["neg_type"]
    best_n = best_result["n_pairs"]
    print(f"\n  Best config: {best_key} (PAM AUC = {best_result['auc_pam']:.4f})")

    # Copy best model as the canonical "pam_best.pt"
    src = MODELS / f"pam_{best_key}.pt"
    torch.save(torch.load(src, map_location="cpu", weights_only=True),
               MODELS / "pam_best.pt")
    # Also save as pam_{N}.pt for backward compatibility with ablation scripts
    torch.save(torch.load(src, map_location="cpu", weights_only=True),
               MODELS / f"pam_{best_n}.pt")

    results_all["best_config"] = {
        "key": best_key,
        "neg_type": best_neg_type,
        "n_pairs": best_n,
        "auc_pam": best_result["auc_pam"],
        "auc_cos": best_result["auc_cos"],
    }

    # ── Figures ────────────────────────────────────────────────────────────

    print("\n--- Generating figures ---")

    # Training curves: in-batch vs random for each N
    n_subplots = len(N_LEVELS)
    fig, axes = plt.subplots(1, max(n_subplots, 1) + 1, figsize=(5 * (n_subplots + 1), 5))
    if n_subplots == 0:
        axes = [axes] if not hasattr(axes, '__len__') else axes

    for i, N in enumerate(N_LEVELS):
        ax = axes[i]
        ib_key = f"ib_{N}"
        rn_key = f"rn_{N}"
        if ib_key in all_losses:
            ax.plot(range(1, len(all_losses[ib_key]) + 1), all_losses[ib_key],
                    label="In-batch", color="steelblue")
        if rn_key in all_losses:
            ax.plot(range(1, len(all_losses[rn_key]) + 1), all_losses[rn_key],
                    label="Random", color="coral")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"N={N//1000}K")
        ax.legend()

    # AUC comparison bar chart
    ax = axes[-1]
    ib_aucs = [results_all.get(f"ib_{N}", {}).get("auc_pam", 0) for N in N_LEVELS]
    rn_aucs = [results_all.get(f"rn_{N}", {}).get("auc_pam", 0) for N in N_LEVELS]
    cos_aucs = [results_all.get(f"ib_{N}", {}).get("auc_cos", 0) for N in N_LEVELS]
    x = np.arange(len(N_LEVELS))
    width = 0.25
    ax.bar(x - width, ib_aucs, width, label="PAM (in-batch)", color="steelblue")
    ax.bar(x, rn_aucs, width, label="PAM (random)", color="coral")
    ax.bar(x + width, cos_aucs, width, label="Cosine", color="orange")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{N//1000}K" for N in N_LEVELS])
    ax.set_xlabel("N pairs")
    ax.set_ylabel("AUC")
    ax.set_title("Overall AUC Comparison")
    ax.legend(fontsize=8)
    ax.axhline(0.5, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "03_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/03_training_curves.png")

    # Cross-boundary AUC comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    ib_cb = [results_all.get(f"ib_{N}", {}).get("auc_pam_cb") for N in N_LEVELS]
    rn_cb = [results_all.get(f"rn_{N}", {}).get("auc_pam_cb") for N in N_LEVELS]
    cos_cb = [results_all.get(f"ib_{N}", {}).get("auc_cos_cb") for N in N_LEVELS]

    valid_idx = [i for i in range(len(N_LEVELS))
                 if ib_cb[i] is not None or rn_cb[i] is not None]
    if valid_idx:
        valid_n = [N_LEVELS[i] for i in valid_idx]
        x = np.arange(len(valid_n))
        vals_ib = [ib_cb[i] if ib_cb[i] is not None else 0 for i in valid_idx]
        vals_rn = [rn_cb[i] if rn_cb[i] is not None else 0 for i in valid_idx]
        vals_cos = [cos_cb[i] if cos_cb[i] is not None else 0 for i in valid_idx]
        ax.bar(x - width, vals_ib, width, label="PAM (in-batch)", color="steelblue")
        ax.bar(x, vals_rn, width, label="PAM (random)", color="coral")
        ax.bar(x + width, vals_cos, width, label="Cosine", color="orange")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{n//1000}K" for n in valid_n])
        ax.set_xlabel("N pairs")
        ax.set_ylabel("AUC")
        ax.set_title("Cross-boundary AUC (|cos| < 0.2)")
        ax.legend()
        ax.axhline(0.5, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "03_cross_boundary_auc.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/03_cross_boundary_auc.png")

    # ── Save results ───────────────────────────────────────────────────────

    with open(RESULTS / "03_train.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"\n  Saved: results/03_train.json")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 3 SUMMARY")
    print("=" * 60)

    for N in N_LEVELS:
        ib = results_all.get(f"ib_{N}", {})
        rn = results_all.get(f"rn_{N}", {})
        print(f"\n  N = {N:,}:")
        print(f"    In-batch:  PAM={ib.get('auc_pam', 0):.4f}, "
              f"Cos={ib.get('auc_cos', 0):.4f}, Δ={ib.get('auc_delta', 0):+.4f}")
        print(f"    Random:    PAM={rn.get('auc_pam', 0):.4f}, "
              f"Cos={rn.get('auc_cos', 0):.4f}, Δ={rn.get('auc_delta', 0):+.4f}")
        if ib.get("auc_pam_cb") is not None:
            print(f"    IB cross-boundary: PAM={ib['auc_pam_cb']:.4f}")
        if rn.get("auc_pam_cb") is not None:
            print(f"    RN cross-boundary: PAM={rn['auc_pam_cb']:.4f}")

    print(f"\n  Best: {best_key} (PAM AUC = {best_result['auc_pam']:.4f})")

    all_below = all(
        results_all.get(k, {}).get("auc_delta", 0) < 0
        for k in results_all if k != "best_config"
    )
    if all_below:
        print("\n  ⚠ PAM AUC below cosine at all configurations.")
        print("  Check if embeddings are confounded with associations.")
    else:
        print(f"\n  ✓ PAM outperforms cosine in at least one configuration.")

    print("\nPhase 3 complete.")
