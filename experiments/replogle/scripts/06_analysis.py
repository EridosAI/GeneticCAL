#!/usr/bin/env python3
"""
DepMap PAM Phase 6: Analysis
==============================

1. Degree dependence:
   - Count co-essential partners per gene
   - Bin by quintile
   - Spearman correlation between degree and PAM improvement

2. Poster child examples:
   - High PAM score + low expression cosine + known STRING interaction

Prerequisites:
    python scripts/03_train.py
    python scripts/05_validate.py  (for STRING pairs)

Usage:
    python scripts/06_analysis.py
"""

import gzip
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy import stats as sp_stats
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SEED = 42
PCA_DIM = 100
HIDDEN_DIM = 1024
N_LAYERS = 4

RESULTS_DIR = Path("results")
DATA_DIR = Path("data")

EMB_PATH = RESULTS_DIR / "gene_embeddings_pca100.npy"
GENE_LIST_PATH = RESULTS_DIR / "gene_list.json"
TRAIN_RESULTS_PATH = RESULTS_DIR / "03_train_results.json"

STRING_INFO_FILE = DATA_DIR / "9606.protein.info.v12.0.txt.gz"
STRING_LINKS_FILE = DATA_DIR / "9606.protein.links.detailed.v12.0.txt.gz"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENE_COL_RE = re.compile(r"^(.+?)\s+\((\d+)\)$")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


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
# Step 1 — Load everything
# ---------------------------------------------------------------------------
def step1_load():
    print("=" * 70)
    print("STEP 1: LOAD DATA")
    print("=" * 70)

    emb = np.load(EMB_PATH)
    with open(GENE_LIST_PATH) as f:
        genes = json.load(f)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    with open(TRAIN_RESULTS_PATH) as f:
        train_meta = json.load(f)
    best_N = train_meta.get("best_N")
    if best_N is None:
        best_N = int(list(train_meta["results"].keys())[0])

    # Load model
    model_path = RESULTS_DIR / f"model_{best_N}.pt"
    model = AssociationMLP().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load pairs
    pair_path = RESULTS_DIR / f"pairs_{best_N}.npy"
    pairs = np.load(pair_path)

    X_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)

    print(f"  Genes: {n_genes}, Best N: {best_N:,}")
    print(f"  Model: {model_path}")
    return model, X_tensor, emb, genes, gene_to_idx, n_genes, pairs, best_N


# ---------------------------------------------------------------------------
# Step 2 — Degree dependence analysis
# ---------------------------------------------------------------------------
def step2_degree_dependence(model, X_tensor, emb, genes, pairs, n_genes):
    print("\n" + "=" * 70)
    print("STEP 2: DEGREE DEPENDENCE")
    print("=" * 70)

    # Count co-essential partners per gene
    degree = np.zeros(n_genes, dtype=np.int32)
    for i in range(len(pairs)):
        degree[pairs[i, 0]] += 1
        degree[pairs[i, 1]] += 1

    print(f"  Degree distribution:")
    print(f"    Mean:   {degree.mean():.1f}")
    print(f"    Median: {np.median(degree):.0f}")
    print(f"    Max:    {degree.max()}")
    print(f"    Min:    {degree.min()}")
    print(f"    Genes with degree > 0: {(degree > 0).sum()}")

    # For each pair, compute PAM and cosine scores
    with torch.no_grad():
        model.eval()
        pred = model(X_tensor)  # n_genes x dim
        pred_np = pred.cpu().numpy()

    pair_pam = np.sum(pred_np[pairs[:, 0]] * pred_np[pairs[:, 1]], axis=1)
    pair_cos = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)
    pair_improvement = pair_pam - pair_cos

    # For each pair, compute average degree of the two genes
    pair_avg_degree = (degree[pairs[:, 0]] + degree[pairs[:, 1]]) / 2.0

    # Quintile analysis
    quintile_edges = np.percentile(pair_avg_degree, [0, 20, 40, 60, 80, 100])
    print(f"\n  Quintile analysis (by average degree of pair):")
    print(f"  {'Quintile':<10s}  {'Degree Range':<18s}  {'N Pairs':<10s}  "
          f"{'Mean PAM':>9s}  {'Mean Cos':>9s}  {'Mean Impr':>10s}")
    print(f"  {'-'*10}  {'-'*18}  {'-'*10}  {'-'*9}  {'-'*9}  {'-'*10}")

    quintile_data = []
    for q in range(5):
        lo = quintile_edges[q]
        hi = quintile_edges[q + 1]
        if q == 4:
            mask = (pair_avg_degree >= lo) & (pair_avg_degree <= hi)
        else:
            mask = (pair_avg_degree >= lo) & (pair_avg_degree < hi)
        n = mask.sum()
        if n == 0:
            continue
        mean_pam = pair_pam[mask].mean()
        mean_cos = pair_cos[mask].mean()
        mean_imp = pair_improvement[mask].mean()
        quintile_data.append({
            "quintile": q + 1, "lo": float(lo), "hi": float(hi),
            "n_pairs": int(n), "mean_pam": float(mean_pam),
            "mean_cos": float(mean_cos), "mean_improvement": float(mean_imp),
        })
        print(f"  Q{q+1:<9d}  [{lo:6.0f}, {hi:6.0f}]    {n:<10d}  "
              f"{mean_pam:>9.4f}  {mean_cos:>9.4f}  {mean_imp:>+10.4f}")

    # Spearman correlation
    rho, pval = sp_stats.spearmanr(pair_avg_degree, pair_improvement)
    print(f"\n  Spearman correlation (degree vs PAM improvement):")
    print(f"    rho = {rho:.4f}, p = {pval:.2e}")

    return {
        "degree_stats": {
            "mean": float(degree.mean()),
            "median": float(np.median(degree)),
            "max": int(degree.max()),
        },
        "quintiles": quintile_data,
        "spearman_rho": float(rho),
        "spearman_pval": float(pval),
    }


# ---------------------------------------------------------------------------
# Step 3 — Poster child examples
# ---------------------------------------------------------------------------
def step3_poster_children(model, X_tensor, emb, genes, gene_to_idx, n_genes,
                          pairs, best_N):
    print("\n" + "=" * 70)
    print("STEP 3: POSTER CHILD EXAMPLES")
    print("High PAM score + low expression cosine + known STRING interaction")
    print("=" * 70)

    # Load STRING pairs at confidence >= 700 for validation
    string_pairs_700 = set()
    if STRING_INFO_FILE.exists() and STRING_LINKS_FILE.exists():
        ensp_to_gene = {}
        with gzip.open(STRING_INFO_FILE, "rt") as f:
            f.readline()
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 2:
                    ensp_to_gene[parts[0]] = parts[1].upper()

        relevant_ensp = {ensp for ensp, gene in ensp_to_gene.items()
                         if gene in gene_to_idx}

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
                if combined >= 700:
                    string_pairs_700.add(
                        (min(gene1, gene2), max(gene1, gene2)))
        print(f"  STRING pairs (>= 700): {len(string_pairs_700):,}")
    else:
        print(f"  STRING files not found — skipping STRING annotation")

    # Compute PAM and cosine for all co-essential pairs
    with torch.no_grad():
        model.eval()
        pred = model(X_tensor)
        pred_np = pred.cpu().numpy()

    pair_pam = np.sum(pred_np[pairs[:, 0]] * pred_np[pairs[:, 1]], axis=1)
    pair_cos = np.sum(emb[pairs[:, 0]] * emb[pairs[:, 1]], axis=1)

    # Filter: low cosine, high PAM
    cross_boundary_mask = np.abs(pair_cos) < 0.2
    high_pam_mask = pair_pam > np.percentile(pair_pam, 90)
    candidate_mask = cross_boundary_mask & high_pam_mask

    candidates = np.where(candidate_mask)[0]
    print(f"\n  Candidates (|cos| < 0.2 AND PAM > 90th pct): {len(candidates)}")

    if len(candidates) == 0:
        print("  No poster child candidates found.")
        return []

    # Sort by PAM score descending
    sorted_candidates = candidates[np.argsort(pair_pam[candidates])[::-1]]

    # Report top 20
    examples = []
    print(f"\n  {'Gene A':<12s} {'Gene B':<12s} {'PAM':>7s} {'Cosine':>7s} "
          f"{'Improvement':>12s} {'STRING?':>8s}")
    print(f"  {'-'*12} {'-'*12} {'-'*7} {'-'*7} {'-'*12} {'-'*8}")

    for rank, idx in enumerate(sorted_candidates[:30]):
        g_a = genes[pairs[idx, 0]]
        g_b = genes[pairs[idx, 1]]
        pam_score = pair_pam[idx]
        cos_score = pair_cos[idx]
        improvement = pam_score - cos_score
        pair_key = (min(g_a, g_b), max(g_a, g_b))
        in_string = pair_key in string_pairs_700

        examples.append({
            "rank": rank + 1,
            "gene_a": g_a,
            "gene_b": g_b,
            "pam_score": float(pam_score),
            "cosine": float(cos_score),
            "improvement": float(improvement),
            "in_string_700": in_string,
        })

        marker = "YES" if in_string else ""
        print(f"  {g_a:<12s} {g_b:<12s} {pam_score:>7.3f} {cos_score:>+7.3f} "
              f"{improvement:>+12.3f} {marker:>8s}")

    n_in_string = sum(1 for e in examples if e["in_string_700"])
    print(f"\n  Of top {len(examples)} poster children: "
          f"{n_in_string} are in STRING >= 700 ({n_in_string/max(len(examples),1)*100:.0f}%)")

    return examples


# ---------------------------------------------------------------------------
# Step 4 — Save
# ---------------------------------------------------------------------------
def step4_save(degree_results, examples, best_N):
    print("\n" + "=" * 70)
    print("STEP 4: SAVE RESULTS")
    print("=" * 70)

    all_results = {
        "best_N": best_N,
        "degree_dependence": degree_results,
        "poster_children": examples,
    }
    results_path = RESULTS_DIR / "06_analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Results: {results_path}")

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Quintile bar chart
    ax = axes[0]
    quintiles = degree_results["quintiles"]
    if quintiles:
        q_labels = [f"Q{q['quintile']}\n[{q['lo']:.0f}-{q['hi']:.0f}]" for q in quintiles]
        q_imp = [q["mean_improvement"] for q in quintiles]
        colors = ["green" if v > 0 else "red" for v in q_imp]
        ax.bar(q_labels, q_imp, color=colors, alpha=0.8)
        ax.axhline(0, color="black", linestyle="-", alpha=0.5)
        ax.set_ylabel("Mean PAM Improvement over Cosine")
        ax.set_xlabel("Degree Quintile")
        rho = degree_results["spearman_rho"]
        ax.set_title(f"Degree Dependence (Spearman rho={rho:.3f})")

    # Plot 2: PAM vs Cosine scatter for poster children
    ax = axes[1]
    if examples:
        pam_vals = [e["pam_score"] for e in examples]
        cos_vals = [e["cosine"] for e in examples]
        in_string = [e["in_string_700"] for e in examples]
        colors_sc = ["red" if s else "steelblue" for s in in_string]
        ax.scatter(cos_vals, pam_vals, c=colors_sc, alpha=0.7, s=40)
        ax.set_xlabel("Expression Cosine")
        ax.set_ylabel("PAM Score")
        ax.set_title("Poster Children: PAM vs Cosine")
        # Legend
        ax.scatter([], [], c="red", label="In STRING >= 700")
        ax.scatter([], [], c="steelblue", label="Not in STRING")
        ax.legend()
        ax.axvline(0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plot_path = RESULTS_DIR / "06_analysis_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plots: {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)

    print("DepMap PAM Phase 6: Analysis")
    print("=" * 70)
    print()

    model, X_tensor, emb, genes, gene_to_idx, n_genes, pairs, best_N = step1_load()
    degree_results = step2_degree_dependence(model, X_tensor, emb, genes, pairs, n_genes)
    examples = step3_poster_children(model, X_tensor, emb, genes, gene_to_idx,
                                      n_genes, pairs, best_N)
    step4_save(degree_results, examples, best_N)

    print("\nDone. All 6 phases complete.")


if __name__ == "__main__":
    main()
