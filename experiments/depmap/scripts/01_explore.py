#!/usr/bin/env python3
"""
DepMap PAM Phase 1: Explore
============================

Load DepMap expression (RNA-seq TPM) and CRISPR gene-effect matrices.
Intersect genes and cell lines, handle missing values, PCA-reduce
expression profiles, report cosine similarity landscape.

Embedding source : OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
Association source: CRISPRGeneEffect.csv
These are DIFFERENT assays — never mix them.

Usage:
    python scripts/01_explore.py
"""

import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

EXPR_FILE = DATA_DIR / "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
CRISPR_FILE = DATA_DIR / "CRISPRGeneEffect.csv"

PCA_DIMS = [50, 100]
MAX_MISSING_FRAC = 0.20  # drop genes with >20% missing in either matrix

GENE_COL_RE = re.compile(r"^(.+?)\s+\((\d+)\)$")  # "GeneName (EntrezID)"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_gene_columns(columns):
    """Parse 'GeneName (EntrezID)' columns.  Returns dict {col_name: gene_name}."""
    mapping = {}
    for col in columns:
        m = GENE_COL_RE.match(col.strip())
        if m:
            mapping[col] = m.group(1)
    return mapping


# ---------------------------------------------------------------------------
# Step 1 — Load expression matrix
# ---------------------------------------------------------------------------
def step1_load_expression():
    print("=" * 70)
    print("STEP 1: LOAD EXPRESSION MATRIX")
    print("=" * 70)

    df = pd.read_csv(EXPR_FILE, index_col=0)
    print(f"  Raw shape: {df.shape}")
    print(f"  Metadata columns: {[c for c in df.columns if not GENE_COL_RE.match(c)]}")

    # Keep only default entries per model (one row per cell line)
    if "IsDefaultEntryForModel" in df.columns:
        before = len(df)
        df = df[df["IsDefaultEntryForModel"].astype(str).str.strip().str.lower() == "yes"]
        print(f"  Filtered to default entries: {before} -> {len(df)}")

    # Extract ModelID for matching to CRISPR
    model_ids = df["ModelID"].values if "ModelID" in df.columns else df.index.values
    meta_cols = [c for c in df.columns if not GENE_COL_RE.match(c)]
    gene_cols = [c for c in df.columns if GENE_COL_RE.match(c)]

    expr_df = df[gene_cols].copy()
    expr_df.index = model_ids
    expr_df.index.name = "ModelID"

    col_map = parse_gene_columns(gene_cols)
    print(f"  Gene columns: {len(gene_cols)}")
    print(f"  Cell lines:   {len(expr_df)}")
    print(f"  Sample genes:  {list(col_map.values())[:8]}")

    return expr_df, col_map


# ---------------------------------------------------------------------------
# Step 2 — Load CRISPR matrix
# ---------------------------------------------------------------------------
def step2_load_crispr():
    print("\n" + "=" * 70)
    print("STEP 2: LOAD CRISPR GENE-EFFECT MATRIX")
    print("=" * 70)

    df = pd.read_csv(CRISPR_FILE, index_col=0)
    print(f"  Raw shape: {df.shape}")

    col_map = parse_gene_columns(df.columns)
    gene_cols = [c for c in df.columns if c in col_map]
    crispr_df = df[gene_cols].copy()
    crispr_df.index.name = "ModelID"

    print(f"  Gene columns: {len(gene_cols)}")
    print(f"  Cell lines:   {len(crispr_df)}")
    print(f"  Sample genes:  {list(col_map.values())[:8]}")

    return crispr_df, col_map


# ---------------------------------------------------------------------------
# Step 3 — Intersect genes and cell lines
# ---------------------------------------------------------------------------
def step3_intersect(expr_df, expr_col_map, crispr_df, crispr_col_map):
    print("\n" + "=" * 70)
    print("STEP 3: INTERSECT GENES AND CELL LINES")
    print("=" * 70)

    # Shared cell lines
    shared_cl = sorted(set(expr_df.index) & set(crispr_df.index))
    print(f"  Expression cell lines:  {len(expr_df)}")
    print(f"  CRISPR cell lines:      {len(crispr_df)}")
    print(f"  Shared cell lines:      {len(shared_cl)}")

    # Shared genes (by gene name)
    expr_gene_to_col = {v: k for k, v in expr_col_map.items()}
    crispr_gene_to_col = {v: k for k, v in crispr_col_map.items()}
    shared_genes = sorted(set(expr_gene_to_col.keys()) & set(crispr_gene_to_col.keys()))
    print(f"  Expression genes:       {len(expr_gene_to_col)}")
    print(f"  CRISPR genes:           {len(crispr_gene_to_col)}")
    print(f"  Shared genes:           {len(shared_genes)}")

    # Subset both matrices
    expr_cols = [expr_gene_to_col[g] for g in shared_genes]
    crispr_cols = [crispr_gene_to_col[g] for g in shared_genes]

    expr_sub = expr_df.loc[shared_cl, expr_cols].copy()
    crispr_sub = crispr_df.loc[shared_cl, crispr_cols].copy()

    # Rename to gene names for consistency
    expr_sub.columns = shared_genes
    crispr_sub.columns = shared_genes

    return expr_sub, crispr_sub, shared_cl, shared_genes


# ---------------------------------------------------------------------------
# Step 4 — Handle missing values
# ---------------------------------------------------------------------------
def step4_handle_missing(expr_sub, crispr_sub, shared_genes):
    print("\n" + "=" * 70)
    print("STEP 4: HANDLE MISSING VALUES")
    print("=" * 70)

    n_cl = len(expr_sub)
    expr_missing = expr_sub.isnull().sum()
    crispr_missing = crispr_sub.isnull().sum()

    expr_frac = expr_missing / n_cl
    crispr_frac = crispr_missing / n_cl

    keep_mask = (expr_frac <= MAX_MISSING_FRAC) & (crispr_frac <= MAX_MISSING_FRAC)
    keep_genes = [g for g, k in zip(shared_genes, keep_mask) if k]

    dropped = len(shared_genes) - len(keep_genes)
    print(f"  Genes before filter: {len(shared_genes)}")
    print(f"  Dropped (>{MAX_MISSING_FRAC*100:.0f}% missing in either matrix): {dropped}")
    print(f"  Genes after filter:  {len(keep_genes)}")

    expr_clean = expr_sub[keep_genes].copy()
    crispr_clean = crispr_sub[keep_genes].copy()

    # Fill remaining NaN with 0 (small residual)
    n_expr_nan = int(expr_clean.isnull().sum().sum())
    n_crispr_nan = int(crispr_clean.isnull().sum().sum())
    print(f"  Remaining NaN (expression): {n_expr_nan} -> filling with 0")
    print(f"  Remaining NaN (CRISPR):     {n_crispr_nan} -> filling with 0")
    expr_clean = expr_clean.fillna(0.0)
    crispr_clean = crispr_clean.fillna(0.0)

    return expr_clean, crispr_clean, keep_genes


# ---------------------------------------------------------------------------
# Step 5 — PCA on expression profiles (genes as features, cell lines as obs)
# ---------------------------------------------------------------------------
def step5_pca(expr_clean, keep_genes):
    print("\n" + "=" * 70)
    print("STEP 5: PCA ON EXPRESSION PROFILES")
    print("Rows = genes, features = cell-line expression values")
    print("=" * 70)

    # Expression matrix: (cell_lines x genes).  We want gene embeddings.
    # Transpose: each gene is a sample, each cell line is a feature.
    X = expr_clean.values.T  # (n_genes x n_cell_lines)
    print(f"  Gene matrix shape: {X.shape}  (genes x cell_lines)")

    pca_results = {}
    for n_dim in PCA_DIMS:
        n_actual = min(n_dim, min(X.shape) - 1)
        pca = PCA(n_components=n_actual)
        X_pca = pca.fit_transform(X)
        cumvar = np.cumsum(pca.explained_variance_ratio_)

        # L2 normalize
        norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
        X_norm = X_pca / (norms + 1e-8)

        print(f"\n  PCA-{n_actual}:")
        print(f"    Variance explained: {cumvar[-1]*100:.1f}%")
        print(f"    First 5 PCs: {cumvar[min(4,len(cumvar)-1)]*100:.1f}%")
        print(f"    First 10 PCs: {cumvar[min(9,len(cumvar)-1)]*100:.1f}%")

        # Save embeddings
        emb_path = RESULTS_DIR / f"gene_embeddings_pca{n_actual}.npy"
        np.save(emb_path, X_norm)
        print(f"    Saved: {emb_path}")

        pca_results[n_actual] = {
            "X_norm": X_norm,
            "pca": pca,
            "cumvar": cumvar,
        }

    # Save gene list (ordered — index into embedding arrays)
    gene_list_path = RESULTS_DIR / "gene_list.json"
    with open(gene_list_path, "w") as f:
        json.dump(keep_genes, f)
    print(f"\n  Gene list saved: {gene_list_path} ({len(keep_genes)} genes)")

    return pca_results


# ---------------------------------------------------------------------------
# Step 6 — Cosine similarity landscape in expression space
# ---------------------------------------------------------------------------
def step6_cosine_landscape(pca_results, keep_genes):
    print("\n" + "=" * 70)
    print("STEP 6: COSINE SIMILARITY LANDSCAPE")
    print("=" * 70)

    cos_results = {}
    for n_dim, data in pca_results.items():
        X_norm = data["X_norm"]
        n_genes = len(X_norm)

        # Full pairwise cosine (feasible: n_genes ~17K -> ~150M pairs, manageable)
        # Actually 17K x 17K = 289M entries x 4 bytes = ~1.1GB. Let's sample instead.
        if n_genes > 5000:
            print(f"\n  PCA-{n_dim}: {n_genes} genes — sampling 2M random pairs")
            rng = np.random.RandomState(42)
            n_sample = 2_000_000
            idx_i = rng.randint(0, n_genes, size=n_sample)
            idx_j = rng.randint(0, n_genes, size=n_sample)
            # Avoid self-pairs
            mask = idx_i != idx_j
            idx_i, idx_j = idx_i[mask], idx_j[mask]
            cosines = np.sum(X_norm[idx_i] * X_norm[idx_j], axis=1)
        else:
            cos_matrix = X_norm @ X_norm.T
            cosines = cos_matrix[np.triu_indices(n_genes, k=1)]

        print(f"  PCA-{n_dim}: {len(cosines):,} pairs sampled")
        print(f"    Mean:   {cosines.mean():.4f}")
        print(f"    Std:    {cosines.std():.4f}")
        print(f"    Median: {np.median(cosines):.4f}")

        # Distribution
        bins = [-1, -0.5, -0.3, -0.1, 0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        hist, _ = np.histogram(cosines, bins=bins)
        print(f"\n    Distribution:")
        for i in range(len(bins) - 1):
            pct = hist[i] / len(cosines) * 100
            bar = "#" * int(pct * 0.5)
            print(f"      [{bins[i]:+5.2f}, {bins[i+1]:+5.2f}): {pct:5.1f}% {bar}")

        cos_results[n_dim] = cosines

    return cos_results


# ---------------------------------------------------------------------------
# Step 7 — Save data for downstream scripts
# ---------------------------------------------------------------------------
def step7_save(expr_clean, crispr_clean, keep_genes, pca_results, cos_results):
    print("\n" + "=" * 70)
    print("STEP 7: SAVE PROCESSED DATA")
    print("=" * 70)

    # Save expression and CRISPR matrices (gene-oriented: genes x cell_lines)
    expr_path = RESULTS_DIR / "expr_matrix.npy"
    crispr_path = RESULTS_DIR / "crispr_matrix.npy"
    np.save(expr_path, expr_clean.values.T)  # genes x cell_lines
    np.save(crispr_path, crispr_clean.values.T)  # genes x cell_lines
    print(f"  Expression matrix:  {expr_path}  shape={expr_clean.values.T.shape}")
    print(f"  CRISPR matrix:      {crispr_path}  shape={crispr_clean.values.T.shape}")

    # Save cell line list
    cl_path = RESULTS_DIR / "cell_line_list.json"
    with open(cl_path, "w") as f:
        json.dump(list(expr_clean.index), f)
    print(f"  Cell line list:     {cl_path} ({len(expr_clean)} cell lines)")

    # Summary stats
    stats = {
        "n_genes": len(keep_genes),
        "n_cell_lines": len(expr_clean),
        "max_missing_frac": MAX_MISSING_FRAC,
    }
    for n_dim, data in pca_results.items():
        stats[f"pca{n_dim}_var_explained"] = float(data["cumvar"][-1])
    for n_dim, cosines in cos_results.items():
        stats[f"pca{n_dim}_cos_mean"] = float(cosines.mean())
        stats[f"pca{n_dim}_cos_std"] = float(cosines.std())

    stats_path = RESULTS_DIR / "01_explore_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats_path}")

    return stats


# ---------------------------------------------------------------------------
# Step 8 — Diagnostic plots
# ---------------------------------------------------------------------------
def step8_plots(pca_results, cos_results, expr_clean):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n  matplotlib not installed — skipping plots.")
        return

    print("\n" + "=" * 70)
    print("STEP 8: DIAGNOSTIC PLOTS")
    print("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: PCA variance explained
    ax = axes[0, 0]
    for n_dim, data in pca_results.items():
        cumvar = data["cumvar"]
        ax.plot(range(1, len(cumvar) + 1), cumvar * 100, linewidth=2, label=f"PCA-{n_dim}")
    ax.axhline(90, color="red", linestyle="--", alpha=0.5, label="90%")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("PCA Variance — Gene Expression Profiles")
    ax.legend()

    # Plot 2: Cosine distribution overlay
    ax = axes[0, 1]
    for n_dim, cosines in cos_results.items():
        ax.hist(cosines, bins=100, alpha=0.5, density=True,
                label=f"PCA-{n_dim} (mean={cosines.mean():.3f})")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Density")
    ax.set_title("Pairwise Cosine — Expression Space")
    ax.legend()

    # Plot 3: Gene expression variance across cell lines
    ax = axes[1, 0]
    X = expr_clean.values.T  # genes x cell_lines
    gene_vars = X.var(axis=1)
    ax.hist(gene_vars, bins=100, color="steelblue", edgecolor="white", alpha=0.8)
    ax.set_xlabel("Variance across cell lines")
    ax.set_ylabel("Number of genes")
    ax.set_title(f"Gene Expression Variance ({len(gene_vars)} genes)")
    ax.axvline(np.median(gene_vars), color="red", linestyle="--",
               label=f"median={np.median(gene_vars):.2f}")
    ax.legend()

    # Plot 4: CRISPR cell line coverage
    ax = axes[1, 1]
    ax.text(0.5, 0.6, f"Shared genes: {X.shape[0]:,}", ha="center", va="center",
            fontsize=16, transform=ax.transAxes)
    ax.text(0.5, 0.4, f"Shared cell lines: {X.shape[1]:,}", ha="center", va="center",
            fontsize=16, transform=ax.transAxes)
    ax.text(0.5, 0.2, f"Expression embedding dim: PCA-{max(cos_results.keys())}",
            ha="center", va="center", fontsize=12, transform=ax.transAxes)
    ax.set_title("Dataset Summary")
    ax.axis("off")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "01_explore_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plots saved to {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("DepMap PAM Phase 1: Explore")
    print("=" * 70)
    print()

    expr_df, expr_col_map = step1_load_expression()
    crispr_df, crispr_col_map = step2_load_crispr()
    expr_sub, crispr_sub, shared_cl, shared_genes = step3_intersect(
        expr_df, expr_col_map, crispr_df, crispr_col_map
    )
    expr_clean, crispr_clean, keep_genes = step4_handle_missing(
        expr_sub, crispr_sub, shared_genes
    )
    pca_results = step5_pca(expr_clean, keep_genes)
    cos_results = step6_cosine_landscape(pca_results, keep_genes)
    stats = step7_save(expr_clean, crispr_clean, keep_genes, pca_results, cos_results)
    step8_plots(pca_results, cos_results, expr_clean)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Shared genes:      {stats['n_genes']:,}")
    print(f"  Shared cell lines: {stats['n_cell_lines']}")
    for n_dim in PCA_DIMS:
        n_actual = min(n_dim, stats["n_cell_lines"] - 1)
        k = f"pca{n_actual}_cos_mean"
        if k in stats:
            print(f"  PCA-{n_actual} cosine mean: {stats[k]:.4f}")

    print(f"\n  Next: python scripts/02_pairs.py")


if __name__ == "__main__":
    main()
