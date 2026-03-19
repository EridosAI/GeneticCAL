#!/usr/bin/env python3
"""Quick analysis: STRING degree vs PAM improvement."""

import json, numpy as np, torch, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
from scipy import stats
import anndata as ad

# Load data
adata = ad.read_h5ad("data/K562_essential_normalized_bulk_01.h5ad")
X_raw = adata.X.toarray() if hasattr(adata.X, "toarray") else np.array(adata.X)
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_raw)
norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
X_norm = X_pca / (norms + 1e-8)

gene_names = []
for idx_str in adata.obs.index:
    parts = str(idx_str).split("_")
    gene_names.append(parts[1].upper() if len(parts) >= 2 else str(idx_str))
name_to_idx = {g: i for i, g in enumerate(gene_names)}
gene_set = set(g.upper() for g in gene_names)


class AssociationMLP(nn.Module):
    def __init__(self, input_dim=50, hidden_dim=1024, n_layers=4):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AssociationMLP().to(device)
ckpt = torch.load("results/model_high.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

X_tensor = torch.tensor(X_norm, dtype=torch.float32, device=device)
with torch.no_grad():
    X_assoc = model(X_tensor).cpu().numpy()

# Count STRING degrees (score >= 700) for perturbation genes
print("Loading STRING degrees...")
info_df = pd.read_csv("data/9606.protein.info.v12.0.txt.gz", sep="\t")
ensp_to_gene = {}
for _, row in info_df.iterrows():
    ensp_to_gene[row["#string_protein_id"]] = str(row["preferred_name"]).upper()

degree = defaultdict(int)
chunk_iter = pd.read_csv(
    "data/9606.protein.links.detailed.v12.0.txt.gz", sep=" ", chunksize=500000
)
for chunk in chunk_iter:
    chunk = chunk[chunk["combined_score"] >= 700]
    for _, row in chunk.iterrows():
        g1 = ensp_to_gene.get(row["protein1"], "")
        g2 = ensp_to_gene.get(row["protein2"], "")
        if g1 in gene_set:
            degree[g1] += 1
        if g2 in gene_set:
            degree[g2] += 1
print(f"  Degrees computed for {len(degree)} genes")

# Load positive pairs (high confidence)
pairs = np.load("data/string_pairs_high.npy")
print(f"  {len(pairs)} high-confidence pairs")

# For each positive pair: degrees, cosine, association, improvement
print("Computing per-pair scores...")
pair_data = []
for i in range(len(pairs)):
    a, b = int(pairs[i, 0]), int(pairs[i, 1])
    ga, gb = gene_names[a], gene_names[b]
    deg_a, deg_b = degree.get(ga, 0), degree.get(gb, 0)
    mean_deg = (deg_a + deg_b) / 2
    max_deg = max(deg_a, deg_b)
    min_deg = min(deg_a, deg_b)

    cos = float(X_norm[a] @ X_norm[b])
    assoc = float(0.5 * (X_assoc[a] @ X_norm[b] + X_assoc[b] @ X_norm[a]))
    improvement = assoc - cos

    pair_data.append(dict(
        idx_a=a, idx_b=b, gene_a=ga, gene_b=gb,
        deg_a=deg_a, deg_b=deg_b,
        mean_deg=mean_deg, max_deg=max_deg, min_deg=min_deg,
        cosine=cos, assoc=assoc, improvement=improvement,
    ))

df = pd.DataFrame(pair_data)
print(f"  {len(df)} pairs analyzed")

# ------------------------------------------------------------------
print()
print("=" * 70)
print("CORRELATION: STRING DEGREE vs PAM IMPROVEMENT (assoc - cosine)")
print("=" * 70)

for deg_col in ["mean_deg", "max_deg", "min_deg"]:
    r, p = stats.spearmanr(df[deg_col], df["improvement"])
    print(f"  {deg_col:>10s}: Spearman r = {r:.4f}, p = {p:.2e}")

print()
print("CORRELATION: STRING DEGREE vs COSINE SIMILARITY")
print("-" * 70)
for deg_col in ["mean_deg", "max_deg", "min_deg"]:
    r, p = stats.spearmanr(df[deg_col], df["cosine"])
    print(f"  {deg_col:>10s}: Spearman r = {r:.4f}, p = {p:.2e}")

print()
print("CORRELATION: STRING DEGREE vs ASSOCIATION SCORE")
print("-" * 70)
for deg_col in ["mean_deg", "max_deg", "min_deg"]:
    r, p = stats.spearmanr(df[deg_col], df["assoc"])
    print(f"  {deg_col:>10s}: Spearman r = {r:.4f}, p = {p:.2e}")

# ------------------------------------------------------------------
# Binned: degree quintiles (pair-level)
print()
print("=" * 70)
print("BINNED ANALYSIS: DEGREE QUINTILES (pair-level)")
print("=" * 70)
df["deg_qnum"] = pd.qcut(df["mean_deg"], q=5, labels=False, duplicates="drop") + 1

header = f"  {'Quintile':>10s} {'Deg range':>15s} {'N pairs':>8s} {'Cosine':>8s} {'Assoc':>8s} {'Improv':>8s} {'Ratio':>7s}"
print(header)
print("  " + "-" * (len(header) - 2))

for q in sorted(df["deg_qnum"].unique()):
    sub = df[df["deg_qnum"] == q]
    deg_lo, deg_hi = sub["mean_deg"].min(), sub["mean_deg"].max()
    mc = sub["cosine"].mean()
    ma = sub["assoc"].mean()
    mi = sub["improvement"].mean()
    ratio = ma / mc if abs(mc) > 0.01 else float("inf")
    print(f"  Q{int(q):>8d} {deg_lo:>6.0f}-{deg_hi:>6.0f} {len(sub):>8d} "
          f"{mc:>8.4f} {ma:>8.4f} {mi:>+8.4f} {ratio:>7.2f}x")

# ------------------------------------------------------------------
# Per-gene analysis
print()
print("=" * 70)
print("PER-GENE ANALYSIS: DEGREE vs MEAN IMPROVEMENT")
print("=" * 70)

gene_stats = defaultdict(lambda: {"improvements": [], "cosines": [], "assocs": []})
for _, row in df.iterrows():
    for g in [row["gene_a"], row["gene_b"]]:
        gene_stats[g]["improvements"].append(row["improvement"])
        gene_stats[g]["cosines"].append(row["cosine"])
        gene_stats[g]["assocs"].append(row["assoc"])

gene_rows = []
for g, s in gene_stats.items():
    gene_rows.append(dict(
        gene=g,
        degree=degree.get(g, 0),
        n_pairs=len(s["improvements"]),
        mean_cosine=np.mean(s["cosines"]),
        mean_assoc=np.mean(s["assocs"]),
        mean_improvement=np.mean(s["improvements"]),
    ))
gdf = pd.DataFrame(gene_rows)

r_imp, p_imp = stats.spearmanr(gdf["degree"], gdf["mean_improvement"])
r_cos, p_cos = stats.spearmanr(gdf["degree"], gdf["mean_cosine"])
r_asc, p_asc = stats.spearmanr(gdf["degree"], gdf["mean_assoc"])
print(f"  {len(gdf)} unique genes")
print(f"  Degree vs mean improvement: Spearman r = {r_imp:.4f}, p = {p_imp:.2e}")
print(f"  Degree vs mean cosine:      Spearman r = {r_cos:.4f}, p = {p_cos:.2e}")
print(f"  Degree vs mean assoc:       Spearman r = {r_asc:.4f}, p = {p_asc:.2e}")

# Gene-level quintiles
gdf["deg_q"] = pd.qcut(gdf["degree"].clip(lower=1), q=5, labels=False, duplicates="drop") + 1
print(f"\n  {'Quintile':>10s} {'Deg range':>15s} {'N genes':>8s} {'Cosine':>8s} {'Assoc':>8s} {'Improv':>8s}")
print("  " + "-" * 65)
for q in sorted(gdf["deg_q"].unique()):
    sub = gdf[gdf["deg_q"] == q]
    print(f"  Q{int(q):>8d} {sub['degree'].min():>6.0f}-{sub['degree'].max():>6.0f} "
          f"{len(sub):>8d} {sub['mean_cosine'].mean():>8.4f} "
          f"{sub['mean_assoc'].mean():>8.4f} {sub['mean_improvement'].mean():>+8.4f}")

# ------------------------------------------------------------------
# Cross-boundary pairs
print()
print("=" * 70)
print("CROSS-BOUNDARY PAIRS ONLY (|cosine| < 0.2)")
print("=" * 70)
cb = df[df["cosine"].abs() < 0.2].copy()
print(f"  {len(cb)} cross-boundary pairs")
if len(cb) > 100:
    r, p = stats.spearmanr(cb["mean_deg"], cb["improvement"])
    print(f"  Degree vs improvement: Spearman r = {r:.4f}, p = {p:.2e}")
    cb["deg_q"] = pd.qcut(cb["mean_deg"], q=5, labels=False, duplicates="drop") + 1
    print(f"\n  {'Quintile':>10s} {'N pairs':>8s} {'Cosine':>8s} {'Assoc':>8s} {'Improv':>8s}")
    print("  " + "-" * 50)
    for q in sorted(cb["deg_q"].unique()):
        sub = cb[cb["deg_q"] == q]
        print(f"  Q{int(q):>8d} {len(sub):>8d} {sub['cosine'].mean():>8.4f} "
              f"{sub['assoc'].mean():>8.4f} {sub['improvement'].mean():>+8.4f}")

# ------------------------------------------------------------------
# Monotonicity test: is improvement strictly increasing across quintiles?
print()
print("=" * 70)
print("MONOTONICITY TEST")
print("=" * 70)
q_means = []
for q in sorted(df["deg_qnum"].unique()):
    q_means.append(df[df["deg_qnum"] == q]["improvement"].mean())
diffs = [q_means[i+1] - q_means[i] for i in range(len(q_means)-1)]
monotonic = all(d > 0 for d in diffs)
print(f"  Quintile improvements: {[f'{m:+.4f}' for m in q_means]}")
print(f"  Successive diffs:      {[f'{d:+.4f}' for d in diffs]}")
print(f"  Strictly monotonic:    {monotonic}")

# Save for plotting
gdf.to_csv("results/degree_vs_improvement_genes.csv", index=False)
df[["gene_a", "gene_b", "mean_deg", "cosine", "assoc", "improvement"]].to_csv(
    "results/degree_vs_improvement_pairs.csv", index=False)
print("\n  Saved results/degree_vs_improvement_{genes,pairs}.csv")
print("Done.")
