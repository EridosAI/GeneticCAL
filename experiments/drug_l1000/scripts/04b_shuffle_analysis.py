"""Phase 4b: Analyze shuffled failure mode.

Diagnoses WHY the shuffled model beats the reference:
1. Degree-PAM correlation for reference vs shuffled
2. Stratified AUC by degree quintile
3. Degree-controlled AUC (negatives matched on degree)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

import importlib
sys.path.insert(0, str(Path(__file__).resolve().parent))
_train = importlib.import_module("03_train")
AssociationMLP = _train.AssociationMLP

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HIDDEN_DIM = 1024
NUM_LAYERS = 4


def load_model(path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=True)
    model = AssociationMLP(
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        num_layers=checkpoint.get("num_layers", NUM_LAYERS),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_pam_matrix(model, embeddings):
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = model(emb_tensor).cpu().numpy()
    return z @ z.T


# ── Main ───────────────────────────────────────────────────────────────────

print("=" * 60)
print("Phase 4b: Shuffled Failure Analysis")
print("=" * 60)

# Load data
embeddings = np.load(DATA / "embeddings_raw.npy")
co_lethality = np.load(DATA / "co_lethality.npy")
n_drugs = len(embeddings)
print(f"  Drugs: {n_drugs}")

# Load Phase 3 best config
with open(RESULTS / "03_train.json") as f:
    train_results = json.load(f)
best_config = train_results["best_config"]
best_n = best_config["n_pairs"]
print(f"  Best config: {best_config['key']} (N={best_n:,})")

pairs = np.load(DATA / f"pairs_{best_n}.npy")

# Load reference and shuffled models
model_ref = load_model(MODELS / f"pam_{best_config['key']}.pt")
# Reconstruct shuffled pairs to load shuffled model — but we don't have it saved.
# Instead, train a quick shuffled model? No — the ablations script didn't save the model.
# Let's retrain the shuffled model and save it.

# Actually, let's check if we can just re-derive the shuffled model's behavior.
# The ablation result was AUC=0.909. We need the actual model.
# Let's retrain shuffled (this takes ~29 min with RN-200K).

print("\n--- Retraining shuffled model (needed for analysis) ---")
rng = np.random.default_rng(42)
shuffled_pairs = pairs.copy()
shuffled_pairs[:, 1] = rng.permutation(shuffled_pairs[:, 1])

model_shuf, _ = _train.train_random_neg(shuffled_pairs, embeddings, tag="Shuf")
torch.save({
    "model_state_dict": model_shuf.state_dict(),
    "embedding_dim": embeddings.shape[1],
    "hidden_dim": HIDDEN_DIM,
    "num_layers": NUM_LAYERS,
}, MODELS / "pam_shuffled.pt")
print("  Saved: models/pam_shuffled.pt")

# Compute PAM matrices
print("\n--- Computing PAM score matrices ---")
pam_ref = compute_pam_matrix(model_ref, embeddings)
pam_shuf = compute_pam_matrix(model_shuf, embeddings)

emb_norm = normalize(embeddings, axis=1)
cos_matrix = emb_norm @ emb_norm.T

# ── 1. Degree-PAM correlation ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("1. DEGREE vs MEAN PAM SCORE")
print("=" * 60)

# Degree from real pairs
degree = np.zeros(n_drugs, dtype=int)
for i, j in pairs:
    degree[i] += 1
    degree[j] += 1

# Degree from shuffled pairs
degree_shuf = np.zeros(n_drugs, dtype=int)
for i, j in shuffled_pairs:
    degree_shuf[i] += 1
    degree_shuf[j] += 1

# Mean PAM score per drug
mean_pam_ref = (pam_ref.sum(axis=1) - np.diag(pam_ref)) / (n_drugs - 1)
mean_pam_shuf = (pam_shuf.sum(axis=1) - np.diag(pam_shuf)) / (n_drugs - 1)
mean_cos = (cos_matrix.sum(axis=1) - np.diag(cos_matrix)) / (n_drugs - 1)

# Spearman correlations
rho_ref, p_ref = stats.spearmanr(degree, mean_pam_ref)
rho_shuf, p_shuf = stats.spearmanr(degree_shuf, mean_pam_shuf)
rho_ref_cos, p_ref_cos = stats.spearmanr(degree, mean_cos)

# Also: does degree in REAL pairs predict shuffled PAM?
rho_real_deg_shuf_pam, p_rdsp = stats.spearmanr(degree, mean_pam_shuf)
# Does degree in SHUFFLED pairs predict shuffled PAM?
rho_shuf_deg_shuf_pam, p_sdsp = stats.spearmanr(degree_shuf, mean_pam_shuf)

print(f"  Reference model:")
print(f"    degree(real) vs mean_PAM_ref:  rho={rho_ref:.4f}  (p={p_ref:.2e})")
print(f"    degree(real) vs mean_cosine:   rho={rho_ref_cos:.4f}  (p={p_ref_cos:.2e})")
print(f"  Shuffled model:")
print(f"    degree(shuf) vs mean_PAM_shuf: rho={rho_shuf:.4f}  (p={p_shuf:.2e})")
print(f"    degree(real) vs mean_PAM_shuf: rho={rho_real_deg_shuf_pam:.4f}  (p={p_rdsp:.2e})")
print(f"  Note: shuffled preserves column-1 degree but randomizes column-0 partners")

# Degree correlation between real and shuffled
rho_deg_corr, _ = stats.spearmanr(degree, degree_shuf)
print(f"\n  Degree correlation (real vs shuffled): rho={rho_deg_corr:.4f}")
print(f"  Degree stats (real):     mean={degree.mean():.1f}, std={degree.std():.1f}, "
      f"max={degree.max()}")
print(f"  Degree stats (shuffled): mean={degree_shuf.mean():.1f}, std={degree_shuf.std():.1f}, "
      f"max={degree_shuf.max()}")

# ── 2. Stratified AUC by degree quintile ──────────────────────────────────

print("\n" + "=" * 60)
print("2. STRATIFIED AUC BY DEGREE QUINTILE")
print("=" * 60)

# For evaluation: positive pairs = real co-lethal pairs, negatives = random
pos_pam_ref = np.array([pam_ref[i, j] for i, j in pairs])
pos_pam_shuf = np.array([pam_shuf[i, j] for i, j in pairs])
pos_cos = np.array([cos_matrix[i, j] for i, j in pairs])

# Min degree of each positive pair
pos_min_degree = np.array([min(degree[i], degree[j]) for i, j in pairs])

# Sample negatives
n_neg = len(pairs)
neg_rng = np.random.default_rng(123)
pos_set = set(map(tuple, pairs)) | set(map(tuple, pairs[:, ::-1]))
neg_pairs = []
while len(neg_pairs) < n_neg:
    i = neg_rng.integers(0, n_drugs)
    j = neg_rng.integers(0, n_drugs)
    if i != j and (i, j) not in pos_set:
        neg_pairs.append((i, j))
neg_pairs = np.array(neg_pairs)

neg_pam_ref = np.array([pam_ref[i, j] for i, j in neg_pairs])
neg_pam_shuf = np.array([pam_shuf[i, j] for i, j in neg_pairs])
neg_cos = np.array([cos_matrix[i, j] for i, j in neg_pairs])
neg_min_degree = np.array([min(degree[i], degree[j]) for i, j in neg_pairs])

# Combine
all_pam_ref = np.concatenate([pos_pam_ref, neg_pam_ref])
all_pam_shuf = np.concatenate([pos_pam_shuf, neg_pam_shuf])
all_cos = np.concatenate([pos_cos, neg_cos])
all_labels = np.concatenate([np.ones(len(pairs)), np.zeros(n_neg)])
all_min_degree = np.concatenate([pos_min_degree, neg_min_degree])

# Quintile bins on min_degree
try:
    quintiles = pd.qcut(all_min_degree, 5, labels=False, duplicates="drop")
except ValueError:
    quintiles = pd.cut(all_min_degree, 5, labels=False)

n_bins = len(np.unique(quintiles))
print(f"  Bins: {n_bins}")

strat_results = []
for q in range(n_bins):
    mask = quintiles == q
    n_total = mask.sum()
    n_pos = all_labels[mask].sum()
    n_neg_q = n_total - n_pos
    deg_min = all_min_degree[mask].min()
    deg_max = all_min_degree[mask].max()

    if n_pos < 20 or n_neg_q < 20:
        print(f"  Q{q} (degree {deg_min}-{deg_max}): n={n_total}, "
              f"pos={int(n_pos)}, neg={int(n_neg_q)} — SKIPPED (too few)")
        continue

    auc_ref = roc_auc_score(all_labels[mask], all_pam_ref[mask])
    auc_shuf = roc_auc_score(all_labels[mask], all_pam_shuf[mask])
    auc_cos = roc_auc_score(all_labels[mask], all_cos[mask])

    print(f"  Q{q} (degree {deg_min:3d}-{deg_max:3d}): n={n_total:6d}, "
          f"pos={int(n_pos):6d}, neg={int(n_neg_q):6d} | "
          f"Ref={auc_ref:.4f}  Shuf={auc_shuf:.4f}  Cos={auc_cos:.4f}  "
          f"Ref>Shuf={'✓' if auc_ref > auc_shuf else '✗'}")

    strat_results.append({
        "quintile": int(q),
        "degree_min": int(deg_min),
        "degree_max": int(deg_max),
        "n_total": int(n_total),
        "n_pos": int(n_pos),
        "n_neg": int(n_neg_q),
        "auc_ref": float(auc_ref),
        "auc_shuf": float(auc_shuf),
        "auc_cos": float(auc_cos),
        "ref_beats_shuf": bool(auc_ref > auc_shuf),
    })

# ── 3. Degree-controlled evaluation ──────────────────────────────────────

print("\n" + "=" * 60)
print("3. DEGREE-CONTROLLED AUC")
print("=" * 60)
print("  For each positive pair (A,B), sample negative (A,C) where")
print("  degree(C) ≈ degree(B) but C is not co-lethal with A.")

# Build degree bins for efficient sampling
degree_to_drugs = {}
for d_idx in range(n_drugs):
    deg = degree[d_idx]
    if deg not in degree_to_drugs:
        degree_to_drugs[deg] = []
    degree_to_drugs[deg].append(d_idx)

# For degree matching, allow ±10% tolerance
def find_degree_matched(target_degree, exclude_set, rng, tolerance=0.1):
    """Find a drug with similar degree not in exclude_set."""
    low = max(0, int(target_degree * (1 - tolerance)))
    high = int(target_degree * (1 + tolerance)) + 1
    candidates = []
    for d in range(low, high + 1):
        if d in degree_to_drugs:
            candidates.extend(degree_to_drugs[d])
    candidates = [c for c in candidates if c not in exclude_set]
    if not candidates:
        # Widen tolerance
        for d in range(max(0, low - 20), high + 20):
            if d in degree_to_drugs:
                candidates.extend(degree_to_drugs[d])
        candidates = [c for c in candidates if c not in exclude_set]
    if not candidates:
        return None
    return rng.choice(candidates)

# Build co-lethality adjacency for fast lookup
co_lethal_adj = {}
for i, j in pairs:
    co_lethal_adj.setdefault(i, set()).add(j)
    co_lethal_adj.setdefault(j, set()).add(i)

ctrl_rng = np.random.default_rng(999)
ctrl_pos_ref = []
ctrl_pos_shuf = []
ctrl_pos_cos = []
ctrl_neg_ref = []
ctrl_neg_shuf = []
ctrl_neg_cos = []
skipped = 0

# Sample degree-matched negatives for a subsample of positive pairs
n_sample = min(50000, len(pairs))
sample_idx = ctrl_rng.choice(len(pairs), n_sample, replace=False)

for idx in sample_idx:
    i, j = pairs[idx]
    # Positive scores
    ctrl_pos_ref.append(pam_ref[i, j])
    ctrl_pos_shuf.append(pam_shuf[i, j])
    ctrl_pos_cos.append(cos_matrix[i, j])

    # Find degree-matched negative for drug i
    exclude = {i, j} | co_lethal_adj.get(i, set())
    c = find_degree_matched(degree[j], exclude, ctrl_rng)
    if c is None:
        skipped += 1
        ctrl_neg_ref.append(pam_ref[i, ctrl_rng.integers(0, n_drugs)])
        ctrl_neg_shuf.append(pam_shuf[i, ctrl_rng.integers(0, n_drugs)])
        ctrl_neg_cos.append(cos_matrix[i, ctrl_rng.integers(0, n_drugs)])
    else:
        ctrl_neg_ref.append(pam_ref[i, c])
        ctrl_neg_shuf.append(pam_shuf[i, c])
        ctrl_neg_cos.append(cos_matrix[i, c])

ctrl_pos_ref = np.array(ctrl_pos_ref)
ctrl_pos_shuf = np.array(ctrl_pos_shuf)
ctrl_pos_cos = np.array(ctrl_pos_cos)
ctrl_neg_ref = np.array(ctrl_neg_ref)
ctrl_neg_shuf = np.array(ctrl_neg_shuf)
ctrl_neg_cos = np.array(ctrl_neg_cos)

labels_ctrl = np.concatenate([np.ones(n_sample), np.zeros(n_sample)])

auc_ctrl_ref = roc_auc_score(labels_ctrl,
                              np.concatenate([ctrl_pos_ref, ctrl_neg_ref]))
auc_ctrl_shuf = roc_auc_score(labels_ctrl,
                               np.concatenate([ctrl_pos_shuf, ctrl_neg_shuf]))
auc_ctrl_cos = roc_auc_score(labels_ctrl,
                              np.concatenate([ctrl_pos_cos, ctrl_neg_cos]))

print(f"\n  Degree-controlled AUC ({n_sample:,} pairs, {skipped} fallbacks):")
print(f"    Reference PAM:  {auc_ctrl_ref:.4f}")
print(f"    Shuffled PAM:   {auc_ctrl_shuf:.4f}")
print(f"    Cosine:         {auc_ctrl_cos:.4f}")
print(f"    Ref - Shuf:     {auc_ctrl_ref - auc_ctrl_shuf:+.4f}")

if auc_ctrl_ref > auc_ctrl_shuf + 0.02:
    print(f"\n  ✓ After degree control, reference beats shuffled by "
          f"{auc_ctrl_ref - auc_ctrl_shuf:.3f}")
    print("  There IS real co-lethality signal beyond degree confounding.")
elif auc_ctrl_ref > auc_ctrl_shuf:
    print(f"\n  ~ Marginal advantage for reference ({auc_ctrl_ref - auc_ctrl_shuf:+.4f})")
else:
    print(f"\n  ✗ Shuffled still matches/beats reference after degree control.")
    print("  The model may be learning pure embedding structure.")

# ── Figures ────────────────────────────────────────────────────────────────

print("\n--- Generating figures ---")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Fig 1: Degree vs mean PAM (reference)
axes[0, 0].scatter(degree, mean_pam_ref, s=3, alpha=0.3, color="steelblue",
                    rasterized=True)
axes[0, 0].set_xlabel("Co-lethal Degree")
axes[0, 0].set_ylabel("Mean PAM Score (Reference)")
axes[0, 0].set_title(f"Reference: Degree vs PAM (ρ={rho_ref:.3f})")

# Fig 2: Degree vs mean PAM (shuffled)
axes[0, 1].scatter(degree_shuf, mean_pam_shuf, s=3, alpha=0.3, color="coral",
                    rasterized=True)
axes[0, 1].set_xlabel("Shuffled Degree")
axes[0, 1].set_ylabel("Mean PAM Score (Shuffled)")
axes[0, 1].set_title(f"Shuffled: Degree vs PAM (ρ={rho_shuf:.3f})")

# Fig 3: Stratified AUC
if strat_results:
    q_labels = [f"Q{r['quintile']}\n({r['degree_min']}-{r['degree_max']})"
                for r in strat_results]
    x = np.arange(len(strat_results))
    width = 0.25
    axes[1, 0].bar(x - width, [r["auc_ref"] for r in strat_results], width,
                    label="Reference", color="steelblue")
    axes[1, 0].bar(x, [r["auc_shuf"] for r in strat_results], width,
                    label="Shuffled", color="coral")
    axes[1, 0].bar(x + width, [r["auc_cos"] for r in strat_results], width,
                    label="Cosine", color="orange")
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(q_labels, fontsize=8)
    axes[1, 0].set_xlabel("Min-Degree Quintile")
    axes[1, 0].set_ylabel("AUC")
    axes[1, 0].set_title("Stratified AUC by Pair Degree")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].axhline(0.5, color="gray", ls="--", alpha=0.3)

# Fig 4: Degree-controlled AUC comparison
ctrl_names = ["Reference\nPAM", "Shuffled\nPAM", "Cosine"]
ctrl_aucs = [auc_ctrl_ref, auc_ctrl_shuf, auc_ctrl_cos]
colors = ["steelblue", "coral", "orange"]
axes[1, 1].bar(range(3), ctrl_aucs, color=colors)
axes[1, 1].set_xticks(range(3))
axes[1, 1].set_xticklabels(ctrl_names)
axes[1, 1].set_ylabel("AUC")
axes[1, 1].set_title("Degree-Controlled AUC")
axes[1, 1].axhline(0.5, color="gray", ls="--", alpha=0.3)
for i, v in enumerate(ctrl_aucs):
    axes[1, 1].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)

plt.tight_layout()
plt.savefig(FIGURES / "04b_shuffle_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/04b_shuffle_analysis.png")

# ── Save results ───────────────────────────────────────────────────────────

results = {
    "degree_correlations": {
        "ref_degree_vs_pam": {"rho": float(rho_ref), "p": float(p_ref)},
        "shuf_degree_vs_pam": {"rho": float(rho_shuf), "p": float(p_shuf)},
        "real_degree_vs_shuf_pam": {"rho": float(rho_real_deg_shuf_pam), "p": float(p_rdsp)},
        "cosine_vs_degree": {"rho": float(rho_ref_cos), "p": float(p_ref_cos)},
        "degree_real_vs_shuf": float(rho_deg_corr),
    },
    "stratified_auc": strat_results,
    "degree_controlled_auc": {
        "n_pairs": int(n_sample),
        "n_fallbacks": int(skipped),
        "auc_ref": float(auc_ctrl_ref),
        "auc_shuf": float(auc_ctrl_shuf),
        "auc_cos": float(auc_ctrl_cos),
        "ref_minus_shuf": float(auc_ctrl_ref - auc_ctrl_shuf),
    },
}

with open(RESULTS / "04b_shuffle_analysis.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"  Saved: results/04b_shuffle_analysis.json")

# ── Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 4b SUMMARY")
print("=" * 60)

print(f"\n  Degree-PAM correlations:")
print(f"    Reference:  ρ={rho_ref:.4f}")
print(f"    Shuffled:   ρ={rho_shuf:.4f}")

print(f"\n  Stratified AUC (Ref > Shuf in low-degree bins?):")
for r in strat_results:
    marker = "✓" if r["ref_beats_shuf"] else "✗"
    print(f"    Q{r['quintile']} (deg {r['degree_min']:3d}-{r['degree_max']:3d}): "
          f"Ref={r['auc_ref']:.4f} vs Shuf={r['auc_shuf']:.4f}  {marker}")

print(f"\n  Degree-controlled AUC:")
print(f"    Reference: {auc_ctrl_ref:.4f}")
print(f"    Shuffled:  {auc_ctrl_shuf:.4f}")
print(f"    Delta:     {auc_ctrl_ref - auc_ctrl_shuf:+.4f}")

print("\nPhase 4b complete.")
