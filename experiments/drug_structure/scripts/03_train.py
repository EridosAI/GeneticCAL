"""Phase 3: Train PAM (AssociationMLP) at multiple N levels.

Trains contrastive model on co-lethal drug pairs using Morgan fingerprint
embeddings. Evaluates overall AUC and cross-boundary AUC vs cosine baseline.
Both drugs are transformed (both-transformed scoring).
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
N_EVAL_NEG = 50_000  # negatives for evaluation


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


# ── Dataset ────────────────────────────────────────────────────────────────

class PairDataset(Dataset):
    def __init__(self, pairs: np.ndarray, embeddings: torch.Tensor):
        self.pairs = pairs
        self.embeddings = embeddings

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i, j = self.pairs[idx]
        return self.embeddings[i], self.embeddings[j]


# ── Loss ───────────────────────────────────────────────────────────────────

def clip_loss(emb_a: torch.Tensor, emb_b: torch.Tensor,
              temperature: float = TEMPERATURE) -> torch.Tensor:
    """Symmetric InfoNCE (CLIP-style) with in-batch negatives."""
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2


# ── Training ───────────────────────────────────────────────────────────────

def train_model(pairs: np.ndarray, embeddings: np.ndarray,
                epochs: int = EPOCHS, tag: str = "") -> tuple:
    """Train AssociationMLP and return (model, loss_history)."""
    embedding_dim = embeddings.shape[1]
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32)
    dataset = PairDataset(pairs, emb_tensor)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )

    model = AssociationMLP(embedding_dim=embedding_dim).to(DEVICE)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Model: {embedding_dim}d -> {HIDDEN_DIM}h x {NUM_LAYERS}L, "
          f"{param_count:,} params")
    print(f"  Device: {DEVICE}, batch={BATCH_SIZE}, "
          f"{BATCH_SIZE - 1} in-batch negatives")

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
            print(f"    Epoch {epoch + 1:3d}/{epochs}: loss={avg_loss:.4f}, "
                  f"acc={accuracy:.4f}, alpha={alpha:.3f}{marker}")

    elapsed = time.time() - t0
    print(f"  Training complete in {elapsed:.0f}s. Best accuracy: {best_acc:.4f}")
    return model, loss_history


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_model(model, embeddings: np.ndarray, pairs: np.ndarray,
                   co_lethality: np.ndarray, tag: str = "") -> dict:
    """Evaluate PAM vs cosine on overall and cross-boundary discrimination."""
    model.eval()
    n_drugs = len(embeddings)

    # Transform all embeddings
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = model(emb_tensor).cpu().numpy()

    # PAM similarity matrix (both-transformed)
    pam_matrix = z @ z.T

    # Cosine similarity matrix
    from sklearn.preprocessing import normalize as sk_normalize
    emb_norm = sk_normalize(embeddings, axis=1)
    cos_matrix = emb_norm @ emb_norm.T

    # Positive pairs: the training pairs
    n_pos = len(pairs)
    pos_pam = np.array([pam_matrix[i, j] for i, j in pairs])
    pos_cos = np.array([cos_matrix[i, j] for i, j in pairs])

    # Negative pairs: random pairs not in positive set
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

    # Overall AUC
    labels = np.concatenate([np.ones(len(pos_pam)), np.zeros(len(neg_pam))])
    auc_pam = roc_auc_score(labels, np.concatenate([pos_pam, neg_pam]))
    auc_cos = roc_auc_score(labels, np.concatenate([pos_cos, neg_cos]))

    print(f"  {tag}Overall AUC: PAM={auc_pam:.4f}, Cosine={auc_cos:.4f}, "
          f"Δ={auc_pam - auc_cos:+.4f}")

    # Cross-boundary AUC (|cosine| < 0.2)
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
    print("Phase 3: Train")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    embeddings = np.load(DATA / "embeddings_raw.npy")
    co_lethality = np.load(DATA / "co_lethality.npy")
    print(f"  Embeddings: {embeddings.shape}")
    print(f"  Co-lethality: {co_lethality.shape}")

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
        print(f"\n{'─' * 60}")
        print(f"Training with N = {N:,} pairs")
        print(f"{'─' * 60}")

        pairs = np.load(DATA / f"pairs_{N}.npy")

        # Train
        model, losses = train_model(pairs, embeddings, tag=f"N={N}: ")
        all_losses[N] = losses

        # Save model
        torch.save({
            "model_state_dict": model.state_dict(),
            "embedding_dim": embeddings.shape[1],
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "n_pairs": N,
        }, MODELS / f"pam_{N}.pt")
        print(f"  Saved: models/pam_{N}.pt")

        # Evaluate
        result = evaluate_model(model, embeddings, pairs, co_lethality, tag=f"N={N}: ")
        result["n_pairs"] = N
        results_all[str(N)] = result

    # ── Figures ────────────────────────────────────────────────────────────

    print("\n--- Generating figures ---")

    # Training curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for N, losses in all_losses.items():
        axes[0].plot(range(1, len(losses) + 1), losses, label=f"N={N//1000}K")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss Curves")
    axes[0].legend()

    # AUC by N
    n_vals = sorted([int(k) for k in results_all.keys()])
    auc_pam = [results_all[str(n)]["auc_pam"] for n in n_vals]
    auc_cos = [results_all[str(n)]["auc_cos"] for n in n_vals]

    x = range(len(n_vals))
    width = 0.35
    axes[1].bar([xi - width / 2 for xi in x], auc_pam, width, label="PAM", color="steelblue")
    axes[1].bar([xi + width / 2 for xi in x], auc_cos, width, label="Cosine", color="orange")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels([f"{n // 1000}K" for n in n_vals])
    axes[1].set_xlabel("Number of Pairs (N)")
    axes[1].set_ylabel("AUC")
    axes[1].set_title("Overall AUC: PAM vs Cosine")
    axes[1].legend()
    axes[1].axhline(0.5, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "03_training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/03_training_curves.png")

    # Cross-boundary AUC
    fig, ax = plt.subplots(figsize=(8, 5))
    auc_pam_cb = [results_all[str(n)].get("auc_pam_cb") for n in n_vals]
    auc_cos_cb = [results_all[str(n)].get("auc_cos_cb") for n in n_vals]

    valid_idx = [i for i, v in enumerate(auc_pam_cb) if v is not None]
    if valid_idx:
        valid_n = [n_vals[i] for i in valid_idx]
        valid_pam = [auc_pam_cb[i] for i in valid_idx]
        valid_cos = [auc_cos_cb[i] for i in valid_idx]
        x = range(len(valid_n))
        ax.bar([xi - width / 2 for xi in x], valid_pam, width,
               label="PAM", color="steelblue")
        ax.bar([xi + width / 2 for xi in x], valid_cos, width,
               label="Cosine", color="orange")
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{n // 1000}K" for n in valid_n])
        ax.set_xlabel("Number of Pairs (N)")
        ax.set_ylabel("AUC")
        ax.set_title("Cross-boundary AUC (|cos| < 0.2): PAM vs Cosine")
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

    for n in n_vals:
        r = results_all[str(n)]
        print(f"\n  N = {n:,}:")
        print(f"    Overall AUC: PAM={r['auc_pam']:.4f}, "
              f"Cosine={r['auc_cos']:.4f}, Δ={r['auc_delta']:+.4f}")
        if r.get("auc_pam_cb") is not None:
            print(f"    Cross-boundary AUC: PAM={r['auc_pam_cb']:.4f}, "
                  f"Cosine={r['auc_cos_cb']:.4f}, Δ={r['auc_delta_cb']:+.4f}")
        print(f"    Alpha: {r['alpha']:.3f}")

    # Best N
    best_n = max(n_vals, key=lambda n: results_all[str(n)]["auc_pam"])
    best_r = results_all[str(best_n)]
    print(f"\n  Best N: {best_n:,} (PAM AUC = {best_r['auc_pam']:.4f})")

    # Kill criteria check
    all_below = all(results_all[str(n)]["auc_delta"] < 0 for n in n_vals)
    if all_below:
        print("\n  ⚠ PAM AUC below cosine at all N levels.")
        print("  Phase 4 will test random negatives before declaring negative result.")
    else:
        print(f"\n  ✓ PAM outperforms cosine at some N levels.")

    print("\nPhase 3 complete.")
