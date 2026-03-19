"""Phase 2: Compute co-lethality matrix and extract association pairs.

Builds co-lethality matrix from PRISM sensitivity data (G = max(0, -sens)),
ranks all drug pairs, extracts top-N at multiple levels, and analyzes
cross-boundary fraction (transcriptionally dissimilar but co-lethal drugs).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

# ── Load Phase 1 outputs ──────────────────────────────────────────────────

print("=" * 60)
print("Phase 2: Pairs")
print("=" * 60)

print("\n--- Loading Phase 1 outputs ---")
embeddings = np.load(DATA / "embeddings_raw.npy")
sens_matrix = np.load(DATA / "sensitivity_matrix.npy")  # cell_lines x drugs
drug_df = pd.read_csv(DATA / "drug_info.csv")

n_drugs = len(drug_df)
n_cells = sens_matrix.shape[0]
print(f"  Drugs: {n_drugs}")
print(f"  Cell lines: {n_cells}")
print(f"  Embedding dim: {embeddings.shape[1]}")

# ── Compute co-lethality matrix ───────────────────────────────────────────

print("\n--- Computing co-lethality matrix ---")
# G = max(0, -sensitivity), NaN → 0
G = np.nan_to_num(-sens_matrix, nan=0.0).clip(min=0).astype(np.float32)
print(f"  G matrix: {G.shape} (cell_lines x drugs)")
print(f"  G nonzero fraction: {(G > 0).sum() / G.size:.4f}")
print(f"  G mean (nonzero): {G[G > 0].mean():.4f}")

# Co-lethality = G.T @ G (drugs x drugs)
co_lethality = (G.T @ G).astype(np.float32)
np.fill_diagonal(co_lethality, 0)
print(f"  Co-lethality matrix: {co_lethality.shape}")

# Distribution of co-lethality scores
triu_idx = np.triu_indices(n_drugs, k=1)
co_values = co_lethality[triu_idx]
print(f"  Total pairs: {len(co_values):,}")
print(f"  Co-lethality: mean={co_values.mean():.4f}, std={co_values.std():.4f}")
print(f"  Co-lethality: median={np.median(co_values):.4f}, max={co_values.max():.4f}")
print(f"  Nonzero pairs: {(co_values > 0).sum():,} ({(co_values > 0).mean():.4f})")

# ── Pre-compute cosine similarity for all pairs ───────────────────────────

print("\n--- Computing pairwise cosine similarity ---")
emb_norm = normalize(embeddings, axis=1)
cos_matrix = (emb_norm @ emb_norm.T).astype(np.float32)
cos_values = cos_matrix[triu_idx]

# ── Extract top-N pairs at multiple levels ─────────────────────────────────

print("\n--- Extracting top-N co-lethal pairs ---")
rank_order = np.argsort(-co_values)

total_pairs = len(co_values)
N_LEVELS = []
for n in [10_000, 25_000, 50_000, 100_000, 200_000]:
    if n <= total_pairs * 0.25:
        N_LEVELS.append(n)
    else:
        print(f"  Skipping N={n:,} (>{total_pairs * 0.25:,.0f} = 25% of total pairs)")

if not N_LEVELS:
    for pct in [1, 2, 5, 10]:
        n = int(total_pairs * pct / 100)
        if n >= 1000:
            N_LEVELS.append(n)
    print(f"  Using percentile-based N levels: {N_LEVELS}")

print(f"  N levels: {N_LEVELS}")

pair_i = triu_idx[0]
pair_j = triu_idx[1]

results_all = {}
all_pair_data = {}

for N in N_LEVELS:
    print(f"\n  --- N = {N:,} ---")
    top_idx = rank_order[:N]

    pairs = np.stack([pair_i[top_idx], pair_j[top_idx]], axis=1)
    co_scores = co_values[top_idx]
    cos_scores = cos_values[top_idx]

    print(f"    Co-lethality: min={co_scores.min():.4f}, "
          f"mean={co_scores.mean():.4f}, max={co_scores.max():.4f}")
    print(f"    Cosine: mean={cos_scores.mean():.4f}, std={cos_scores.std():.4f}")

    # Cross-boundary fraction: |cosine| < 0.2
    cross_boundary = np.abs(cos_scores) < 0.2
    cb_frac = cross_boundary.mean()
    print(f"    Cross-boundary (|cos| < 0.2): {cross_boundary.sum():,} "
          f"({cb_frac:.4f} = {cb_frac * 100:.1f}%)")

    for thresh in [0.1, 0.3, 0.5]:
        frac = (np.abs(cos_scores) < thresh).mean()
        print(f"    |cos| < {thresh}: {frac:.4f}")

    # Baseline AUC: cosine similarity for discriminating top-N vs random
    rng = np.random.default_rng(42)
    neg_idx = rng.choice(len(co_values), size=N, replace=False)
    neg_mask = np.isin(neg_idx, top_idx)
    while neg_mask.any():
        neg_idx[neg_mask] = rng.choice(len(co_values), size=neg_mask.sum())
        neg_mask = np.isin(neg_idx, top_idx)

    neg_cos = cos_values[neg_idx]
    labels = np.concatenate([np.ones(N), np.zeros(N)])
    scores_cos = np.concatenate([cos_scores, neg_cos])
    auc_cos = roc_auc_score(labels, scores_cos)
    print(f"    Cosine baseline AUC: {auc_cos:.4f}")

    # Cross-boundary AUC
    cb_pos_mask = np.abs(cos_scores) < 0.2
    cb_neg_mask = np.abs(neg_cos) < 0.2
    if cb_pos_mask.sum() > 50 and cb_neg_mask.sum() > 50:
        cb_labels = np.concatenate([
            np.ones(cb_pos_mask.sum()), np.zeros(cb_neg_mask.sum())
        ])
        cb_cos_scores = np.concatenate([cos_scores[cb_pos_mask], neg_cos[cb_neg_mask]])
        cb_auc_cos = roc_auc_score(cb_labels, cb_cos_scores)
        print(f"    Cross-boundary cosine AUC: {cb_auc_cos:.4f} "
              f"(pos={cb_pos_mask.sum()}, neg={cb_neg_mask.sum()})")
    else:
        cb_auc_cos = None
        print(f"    Cross-boundary cosine AUC: insufficient pairs")

    np.save(DATA / f"pairs_{N}.npy", pairs)
    print(f"    Saved: data/pairs_{N}.npy")

    results_all[str(N)] = {
        "n_pairs": int(N),
        "co_lethality_min": float(co_scores.min()),
        "co_lethality_mean": float(co_scores.mean()),
        "co_lethality_max": float(co_scores.max()),
        "cosine_mean": float(cos_scores.mean()),
        "cosine_std": float(cos_scores.std()),
        "cross_boundary_frac_0.2": float(cb_frac),
        "cross_boundary_frac_0.1": float((np.abs(cos_scores) < 0.1).mean()),
        "cross_boundary_frac_0.3": float((np.abs(cos_scores) < 0.3).mean()),
        "cosine_baseline_auc": float(auc_cos),
        "cross_boundary_cosine_auc": float(cb_auc_cos) if cb_auc_cos is not None else None,
    }
    all_pair_data[N] = {
        "cos_scores": cos_scores,
        "co_scores": co_scores,
    }

# ── Save co-lethality matrix ──────────────────────────────────────────────

np.save(DATA / "co_lethality.npy", co_lethality)
print(f"\n  Saved: data/co_lethality.npy {co_lethality.shape}")

# ── Figures ────────────────────────────────────────────────────────────────

print("\n--- Generating figures ---")

# Fig 1: Co-lethality distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

nonzero_co = co_values[co_values > 0]
axes[0].hist(nonzero_co, bins=200, density=True, alpha=0.7, color="steelblue")
axes[0].set_xlabel("Co-lethality Score")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Co-lethality Distribution (nonzero, N={len(nonzero_co):,})")
axes[0].set_yscale("log")

for N in N_LEVELS:
    threshold = co_values[rank_order[N - 1]]
    axes[0].axvline(threshold, color="red", ls="--", alpha=0.5,
                    label=f"Top-{N//1000}K: {threshold:.2f}")
axes[0].legend(fontsize=8)

for N in N_LEVELS[:3]:
    top_cos = all_pair_data[N]["cos_scores"]
    axes[1].hist(top_cos, bins=100, density=True, alpha=0.5,
                 label=f"Top-{N//1000}K co-lethal")
axes[1].hist(cos_values, bins=100, density=True, alpha=0.3,
             color="gray", label="All pairs")
axes[1].axvline(0.2, color="red", ls="--", alpha=0.7, label="|cos|=0.2")
axes[1].axvline(-0.2, color="red", ls="--", alpha=0.7)
axes[1].set_xlabel("L1000 Cosine Similarity")
axes[1].set_ylabel("Density")
axes[1].set_title("Cosine Distribution: Co-lethal vs All Pairs")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIGURES / "02_co_lethality_dist.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/02_co_lethality_dist.png")

# Fig 2: Cosine vs co-lethality scatter
fig, ax = plt.subplots(figsize=(8, 6))
n_sample = min(500_000, len(co_values))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(co_values), n_sample, replace=False)
ax.scatter(cos_values[sample_idx], co_values[sample_idx],
           s=0.5, alpha=0.05, color="steelblue", rasterized=True)
ax.set_xlabel("L1000 Cosine Similarity")
ax.set_ylabel("Co-lethality Score")
ax.set_title(f"Transcriptional Similarity vs Co-lethality ({n_sample:,} sampled pairs)")
ax.axvline(0.2, color="red", ls="--", alpha=0.5)
ax.axvline(-0.2, color="red", ls="--", alpha=0.5)
plt.tight_layout()
plt.savefig(FIGURES / "02_cosine_vs_colethality.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/02_cosine_vs_colethality.png")

# Fig 3: Cross-boundary fraction by N
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
n_vals = [N for N in N_LEVELS]
cb_fracs = [results_all[str(N)]["cross_boundary_frac_0.2"] for N in N_LEVELS]
cos_aucs = [results_all[str(N)]["cosine_baseline_auc"] for N in N_LEVELS]

axes[0].bar(range(len(n_vals)), cb_fracs, color="steelblue", alpha=0.7)
axes[0].set_xticks(range(len(n_vals)))
axes[0].set_xticklabels([f"{n//1000}K" for n in n_vals])
axes[0].set_xlabel("Number of Pairs (N)")
axes[0].set_ylabel("Cross-boundary Fraction (|cos| < 0.2)")
axes[0].set_title("Cross-boundary Fraction by N")
axes[0].axhline(0.02, color="red", ls="--", alpha=0.5, label="2% kill threshold")
axes[0].axhline(0.10, color="green", ls="--", alpha=0.5, label="10% exciting threshold")
axes[0].legend()

axes[1].bar(range(len(n_vals)), cos_aucs, color="orange", alpha=0.7)
axes[1].set_xticks(range(len(n_vals)))
axes[1].set_xticklabels([f"{n//1000}K" for n in n_vals])
axes[1].set_xlabel("Number of Pairs (N)")
axes[1].set_ylabel("Cosine Baseline AUC")
axes[1].set_title("Cosine Baseline AUC by N")
axes[1].axhline(0.5, color="gray", ls="--", alpha=0.5)

plt.tight_layout()
plt.savefig(FIGURES / "02_cross_boundary.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/02_cross_boundary.png")

# ── Save results ───────────────────────────────────────────────────────────

with open(RESULTS / "02_pairs_stats.json", "w") as f:
    json.dump(results_all, f, indent=2)
print(f"\n  Saved: results/02_pairs_stats.json")

# ── Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 2 SUMMARY")
print("=" * 60)
print(f"  Drugs: {n_drugs}, Cell lines: {n_cells}")
print(f"  Total pairs: {len(co_values):,}")
print(f"  N levels: {N_LEVELS}")

for N in N_LEVELS:
    r = results_all[str(N)]
    print(f"\n  N = {N:,}:")
    print(f"    Cross-boundary (|cos| < 0.2): {r['cross_boundary_frac_0.2']:.4f} "
          f"({r['cross_boundary_frac_0.2'] * 100:.1f}%)")
    print(f"    Cosine baseline AUC: {r['cosine_baseline_auc']:.4f}")

best_cb = max(results_all.values(), key=lambda x: x["cross_boundary_frac_0.2"])
best_cb_frac = best_cb["cross_boundary_frac_0.2"]

if best_cb_frac < 0.02:
    print(f"\n  ⚠ KILL CRITERION: Cross-boundary < 2% at all N levels.")
    print("  Cosine already captures co-lethality. Consider aborting.")
elif best_cb_frac >= 0.10:
    print(f"\n  ✓ EXCITING: Cross-boundary >= 10% — transcriptional scaffold hopping!")
else:
    print(f"\n  ~ Moderate cross-boundary signal ({best_cb_frac:.1%}). Proceed with caution.")

# Kill criterion: cosine baseline already very high
best_auc = max(r["cosine_baseline_auc"] for r in results_all.values())
if best_auc > 0.85:
    print(f"\n  ⚠ WARNING: Cosine baseline AUC > 0.85 ({best_auc:.4f}).")
    print("  L1000 signatures may already predict co-lethality (confounding).")

print("\nPhase 2 complete.")
