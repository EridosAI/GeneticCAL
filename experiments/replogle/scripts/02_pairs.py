#!/usr/bin/env python3
"""
DepMap PAM Phase 2: Construct Co-Essentiality Pairs
=====================================================

From the CRISPR gene-effect matrix, compute co-essentiality:
    G = max(0, -score)        # clamp negatives to zero
    coess = G @ G.T           # genes x genes
    coess[diag] = 0

Rank all gene pairs by co-essentiality score.
Extract top-N at N = 25K, 50K, 100K, 200K.

For each N: report co-essentiality distribution, expression cosine
of selected pairs, number of unique genes, baseline cosine AUC.

Prerequisites:
    python scripts/01_explore.py   (produces results/crispr_matrix.npy,
                                     gene_embeddings_pca100.npy, gene_list.json)

Usage:
    python scripts/02_pairs.py
"""

import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

CRISPR_PATH = RESULTS_DIR / "crispr_matrix.npy"
EMB_PATH = RESULTS_DIR / "gene_embeddings_pca100.npy"
GENE_LIST_PATH = RESULTS_DIR / "gene_list.json"

PAIR_COUNTS = [25_000, 50_000, 100_000, 200_000]


# ---------------------------------------------------------------------------
# Step 1 — Load data
# ---------------------------------------------------------------------------
def step1_load():
    print("=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    crispr = np.load(CRISPR_PATH)  # genes x cell_lines
    emb = np.load(EMB_PATH)        # genes x pca_dim
    with open(GENE_LIST_PATH) as f:
        genes = json.load(f)

    print(f"  CRISPR matrix: {crispr.shape}  (genes x cell_lines)")
    print(f"  Gene embeddings: {emb.shape}  (genes x pca_dim)")
    print(f"  Genes: {len(genes)}")

    assert crispr.shape[0] == len(genes) == emb.shape[0]
    return crispr, emb, genes


# ---------------------------------------------------------------------------
# Step 2 — Compute co-essentiality matrix
# ---------------------------------------------------------------------------
def step2_coessentiality(crispr):
    print("\n" + "=" * 70)
    print("STEP 2: COMPUTE CO-ESSENTIALITY MATRIX")
    print("=" * 70)

    # G = max(0, -score): scores near -1 (essential) become ~1
    G = np.maximum(0, -crispr)  # genes x cell_lines
    print(f"  G matrix shape: {G.shape}")
    print(f"  G nonzero fraction: {(G > 0).mean()*100:.1f}%")
    print(f"  G mean (nonzero): {G[G > 0].mean():.4f}")

    # Co-essentiality = G @ G.T  (genes x genes)
    print(f"  Computing G @ G.T ({G.shape[0]}x{G.shape[1]} @ {G.shape[1]}x{G.shape[0]})...")
    coess = G @ G.T
    np.fill_diagonal(coess, 0)

    print(f"  Co-essentiality matrix shape: {coess.shape}")
    print(f"  Max co-essentiality: {coess.max():.4f}")
    print(f"  Mean (off-diagonal): {coess[coess > 0].mean():.4f}")

    return coess


# ---------------------------------------------------------------------------
# Step 3 — Rank pairs and extract top-N
# ---------------------------------------------------------------------------
def step3_rank_pairs(coess, genes):
    print("\n" + "=" * 70)
    print("STEP 3: RANK PAIRS BY CO-ESSENTIALITY")
    print("=" * 70)

    n_genes = len(genes)
    # Upper triangle indices
    triu_i, triu_j = np.triu_indices(n_genes, k=1)
    scores = coess[triu_i, triu_j]
    total_pairs = len(scores)
    print(f"  Total unique pairs: {total_pairs:,}")
    print(f"  Score distribution:")
    print(f"    Mean:   {scores.mean():.4f}")
    print(f"    Median: {np.median(scores):.4f}")
    print(f"    Max:    {scores.max():.4f}")
    print(f"    >0:     {(scores > 0).sum():,} ({(scores > 0).mean()*100:.1f}%)")

    # Sort descending
    sort_idx = np.argsort(scores)[::-1]

    pair_sets = {}
    for N in PAIR_COUNTS:
        if N > total_pairs:
            print(f"\n  N={N:,}: only {total_pairs:,} pairs available, using all")
            N = total_pairs
        top_idx = sort_idx[:N]
        pairs = np.stack([triu_i[top_idx], triu_j[top_idx]], axis=1).astype(np.int32)
        top_scores = scores[top_idx]

        n_unique_genes = len(set(pairs[:, 0]) | set(pairs[:, 1]))
        print(f"\n  N={N:,}:")
        print(f"    Min co-essentiality in set: {top_scores[-1]:.4f}")
        print(f"    Max co-essentiality in set: {top_scores[0]:.4f}")
        print(f"    Mean:                       {top_scores.mean():.4f}")
        print(f"    Unique genes:               {n_unique_genes:,} / {n_genes:,}")

        pair_sets[N] = {"pairs": pairs, "scores": top_scores}

    return pair_sets


# ---------------------------------------------------------------------------
# Step 4 — Expression cosine analysis for each pair set
# ---------------------------------------------------------------------------
def step4_expression_cosine(pair_sets, emb, genes):
    print("\n" + "=" * 70)
    print("STEP 4: EXPRESSION COSINE OF SELECTED PAIRS")
    print("=" * 70)

    n_genes = len(genes)

    for N, data in pair_sets.items():
        pairs = data["pairs"]
        # Cosine similarity in expression (PCA) space
        cos_vals = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)

        n_cross = (np.abs(cos_vals) < 0.2).sum()
        n_high = (cos_vals > 0.5).sum()

        print(f"\n  N={N:,} pairs:")
        print(f"    Expression cosine — mean:   {cos_vals.mean():.4f}")
        print(f"    Expression cosine — median:  {np.median(cos_vals):.4f}")
        print(f"    Expression cosine — std:     {cos_vals.std():.4f}")
        print(f"    Cross-boundary (|cos| < 0.2): {n_cross:,} ({n_cross/N*100:.1f}%)")
        print(f"    High cosine (> 0.5):          {n_high:,} ({n_high/N*100:.1f}%)")

        data["cosines"] = cos_vals
        data["n_cross_boundary"] = n_cross


# ---------------------------------------------------------------------------
# Step 5 — Baseline cosine AUC for each pair set
# ---------------------------------------------------------------------------
def step5_baseline_auc(pair_sets, emb, genes):
    print("\n" + "=" * 70)
    print("STEP 5: BASELINE COSINE AUC")
    print("=" * 70)

    n_genes = len(genes)
    rng = np.random.RandomState(42)

    for N, data in pair_sets.items():
        pairs = data["pairs"]
        pos_cos = data["cosines"]

        # Build positive set for filtering
        pos_set = set()
        for i in range(len(pairs)):
            pos_set.add((int(pairs[i, 0]), int(pairs[i, 1])))
            pos_set.add((int(pairs[i, 1]), int(pairs[i, 0])))

        # Sample 5x negatives
        n_neg = min(N * 5, 200_000)
        neg_cos = []
        while len(neg_cos) < n_neg:
            i = rng.randint(0, n_genes)
            j = rng.randint(0, n_genes)
            if i != j and (i, j) not in pos_set:
                neg_cos.append(float(np.sum(emb[i] * emb[j])))
        neg_cos = np.array(neg_cos)

        labels = np.concatenate([np.ones(len(pos_cos)), np.zeros(len(neg_cos))])
        scores = np.concatenate([pos_cos, neg_cos])
        auc = roc_auc_score(labels, scores)

        print(f"\n  N={N:,}:")
        print(f"    Cosine AUC: {auc:.4f}")
        print(f"    Pos mean:   {pos_cos.mean():.4f}   Neg mean: {neg_cos.mean():.4f}")

        data["cosine_auc"] = auc


# ---------------------------------------------------------------------------
# Step 6 — Save pairs and results
# ---------------------------------------------------------------------------
def step6_save(pair_sets, genes):
    print("\n" + "=" * 70)
    print("STEP 6: SAVE PAIRS AND RESULTS")
    print("=" * 70)

    summary = {}
    for N, data in pair_sets.items():
        pairs = data["pairs"]
        scores = data["scores"]
        cosines = data["cosines"]

        # Save pair indices
        pair_path = RESULTS_DIR / f"pairs_{N}.npy"
        np.save(pair_path, pairs)
        print(f"  {pair_path}: {len(pairs)} pairs")

        # Save scores alongside
        score_path = RESULTS_DIR / f"pairs_{N}_coess_scores.npy"
        np.save(score_path, scores)

        n_unique = len(set(pairs[:, 0]) | set(pairs[:, 1]))
        summary[str(N)] = {
            "n_pairs": int(N),
            "n_unique_genes": n_unique,
            "coess_min": float(scores[-1]),
            "coess_max": float(scores[0]),
            "coess_mean": float(scores.mean()),
            "cos_mean": float(cosines.mean()),
            "cos_std": float(cosines.std()),
            "n_cross_boundary": int(data["n_cross_boundary"]),
            "pct_cross_boundary": float(data["n_cross_boundary"] / N * 100),
            "cosine_auc": data["cosine_auc"],
        }

    results_path = RESULTS_DIR / "02_pairs_results.json"
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results: {results_path}")

    return summary


# ---------------------------------------------------------------------------
# Step 7 — Plots
# ---------------------------------------------------------------------------
def step7_plots(pair_sets, genes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed — skipping plots.")
        return

    print("\n" + "=" * 70)
    print("STEP 7: DIAGNOSTIC PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Co-essentiality score distribution by N
    ax = axes[0, 0]
    for N, data in pair_sets.items():
        ax.hist(data["scores"], bins=80, alpha=0.5, density=True,
                label=f"N={N//1000}K")
    ax.set_xlabel("Co-essentiality score")
    ax.set_ylabel("Density")
    ax.set_title("Co-Essentiality Distribution by N")
    ax.legend()

    # Plot 2: Expression cosine distribution by N
    ax = axes[0, 1]
    for N, data in pair_sets.items():
        ax.hist(data["cosines"], bins=80, alpha=0.5, density=True,
                label=f"N={N//1000}K")
    ax.axvline(0.2, color="black", linestyle="--", alpha=0.5)
    ax.axvline(-0.2, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Expression cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Expression Cosine of Co-Essential Pairs")
    ax.legend()

    # Plot 3: Cross-boundary fraction vs N
    ax = axes[1, 0]
    Ns = sorted(pair_sets.keys())
    fracs = [pair_sets[n]["n_cross_boundary"] / n * 100 for n in Ns]
    ax.bar([f"{n//1000}K" for n in Ns], fracs, color="darkorange", alpha=0.8)
    ax.set_ylabel("% pairs with |cosine| < 0.2")
    ax.set_xlabel("N (top co-essential pairs)")
    ax.set_title("Cross-Boundary Fraction")

    # Plot 4: Cosine AUC vs N
    ax = axes[1, 1]
    aucs = [pair_sets[n]["cosine_auc"] for n in Ns]
    ax.bar([f"{n//1000}K" for n in Ns], aucs, color="steelblue", alpha=0.8)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="Chance")
    ax.set_ylabel("Cosine Discrimination AUC")
    ax.set_xlabel("N (top co-essential pairs)")
    ax.set_title("Baseline Cosine AUC\n(PAM must beat this)")
    ax.legend()
    for i, (n, a) in enumerate(zip(Ns, aucs)):
        ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "02_pairs_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plots saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("DepMap PAM Phase 2: Construct Co-Essentiality Pairs")
    print("=" * 70)
    print()

    crispr, emb, genes = step1_load()
    coess = step2_coessentiality(crispr)
    pair_sets = step3_rank_pairs(coess, genes)
    step4_expression_cosine(pair_sets, emb, genes)
    step5_baseline_auc(pair_sets, emb, genes)
    summary = step6_save(pair_sets, genes)
    step7_plots(pair_sets, genes)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  {'N':>8s}  {'Cos AUC':>8s}  {'Cross-Bnd %':>12s}  {'Unique Genes':>12s}")
    print(f"  {'-'*8}  {'-'*8}  {'-'*12}  {'-'*12}")
    for N in sorted(pair_sets.keys()):
        s = summary[str(N)]
        print(f"  {N:>8,}  {s['cosine_auc']:>8.4f}  {s['pct_cross_boundary']:>11.1f}%  {s['n_unique_genes']:>12,}")

    print(f"\n  Next: python scripts/03_train.py")


if __name__ == "__main__":
    main()
