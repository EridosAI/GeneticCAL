#!/usr/bin/env python3
"""Find specific compelling gene pair examples from Q1 cross-boundary pairs."""

import numpy as np, torch, pandas as pd
import torch.nn as nn, torch.nn.functional as F
from pathlib import Path
from collections import defaultdict
from sklearn.decomposition import PCA
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

# STRING degrees
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

# Load STRING pairs with evidence channels
pairs_df = pd.read_csv("data/string_pairs_high.csv")
print(f"  {len(pairs_df)} high-confidence pairs with evidence channels")

# Add scores
print("Computing scores for all high-confidence pairs...")
rows = []
for _, row in pairs_df.iterrows():
    g1, g2 = row["gene1"], row["gene2"]
    i1, i2 = int(row["idx1"]), int(row["idx2"])
    cos = float(X_norm[i1] @ X_norm[i2])
    assoc = float(0.5 * (X_assoc[i1] @ X_norm[i2] + X_assoc[i2] @ X_norm[i1]))
    d1, d2 = degree.get(g1, 0), degree.get(g2, 0)
    rows.append(dict(
        gene1=g1, gene2=g2, idx1=i1, idx2=i2,
        cosine=cos, assoc=assoc, improvement=assoc - cos,
        deg1=d1, deg2=d2, mean_deg=(d1+d2)/2,
        min_deg=min(d1, d2), max_deg=max(d1, d2),
        combined_score=row["combined_score"],
        experimental=row["experimental"],
        database=row["database"],
        textmining=row["textmining"],
        coexpression=row["coexpression"],
    ))
df = pd.DataFrame(rows)

# Q1 cross-boundary: low degree, |cosine| < 0.2
# Use the same quintile cutoff as before
q_cuts = pd.qcut(df["mean_deg"], q=5, labels=False, duplicates="drop")
q1_cutoff = df.loc[q_cuts == 0, "mean_deg"].max()
print(f"\n  Q1 degree cutoff: mean_deg <= {q1_cutoff}")

cb_q1 = df[(df["cosine"].abs() < 0.2) & (df["mean_deg"] <= q1_cutoff)].copy()
print(f"  Q1 cross-boundary pairs: {len(cb_q1)}")

# Filter for experimental evidence (not just text-mining)
cb_q1_exp = cb_q1[cb_q1["experimental"] > 0].copy()
print(f"  ...with experimental evidence: {len(cb_q1_exp)}")

# Sort by association score (highest = PAM most confident)
cb_q1_exp = cb_q1_exp.sort_values("assoc", ascending=False)

# Also identify which genes have the fewest GO annotations
# We can proxy "poorly characterized" by low degree
print(f"\n{'='*80}")
print("TOP 20 Q1 CROSS-BOUNDARY PAIRS WITH EXPERIMENTAL EVIDENCE")
print(f"  (sorted by association score, |cosine| < 0.2, mean_deg <= {q1_cutoff})")
print(f"{'='*80}")
print(f"  {'Gene1':>10s} {'Gene2':>10s} {'Cos':>7s} {'Assoc':>7s} {'Impr':>7s} "
      f"{'Deg1':>5s} {'Deg2':>5s} {'Exper':>6s} {'DB':>5s} {'TxtMn':>6s} {'Score':>6s}")
print("  " + "-" * 85)

for i, (_, row) in enumerate(cb_q1_exp.head(20).iterrows()):
    marker = ""
    if row["min_deg"] < 30:
        marker = " <-- low-degree"
    if row["min_deg"] < 15:
        marker = " <-- VERY low-degree"
    print(f"  {row['gene1']:>10s} {row['gene2']:>10s} {row['cosine']:>7.3f} "
          f"{row['assoc']:>7.3f} {row['improvement']:>+7.3f} "
          f"{row['deg1']:>5d} {row['deg2']:>5d} {row['experimental']:>6.0f} "
          f"{row['database']:>5.0f} {row['textmining']:>6.0f} {row['combined_score']:>6.0f}"
          f"{marker}")

# Now look for the "poster child" cases: high assoc, near-zero cosine,
# experimental evidence, at least one gene with very few interactions
print(f"\n{'='*80}")
print("POSTER CHILD CANDIDATES")
print("  Criteria: |cosine| < 0.1, assoc > 0.3, experimental > 0,")
print("            at least one gene with degree < 30")
print(f"{'='*80}")

poster = cb_q1_exp[
    (cb_q1_exp["cosine"].abs() < 0.1) &
    (cb_q1_exp["assoc"] > 0.3) &
    (cb_q1_exp["min_deg"] < 30)
].sort_values("assoc", ascending=False)

if len(poster) == 0:
    print("  None found with strict criteria. Relaxing...")
    poster = cb_q1_exp[
        (cb_q1_exp["cosine"].abs() < 0.15) &
        (cb_q1_exp["assoc"] > 0.2) &
        (cb_q1_exp["min_deg"] < 50)
    ].sort_values("assoc", ascending=False)

if len(poster) == 0:
    print("  Still none. Relaxing further...")
    poster = cb_q1_exp[
        (cb_q1_exp["cosine"].abs() < 0.2) &
        (cb_q1_exp["assoc"] > 0.15) &
        (cb_q1_exp["min_deg"] < 50)
    ].sort_values("assoc", ascending=False)

print(f"  Found {len(poster)} candidates")
for _, row in poster.head(10).iterrows():
    print(f"\n  {row['gene1']} -- {row['gene2']}")
    print(f"    Cosine: {row['cosine']:.4f}   Assoc: {row['assoc']:.4f}   "
          f"Improvement: {row['improvement']:+.4f}")
    print(f"    Degree: {row['gene1']}={row['deg1']}, {row['gene2']}={row['deg2']}")
    print(f"    STRING: combined={row['combined_score']:.0f}, "
          f"experimental={row['experimental']:.0f}, "
          f"database={row['database']:.0f}, "
          f"textmining={row['textmining']:.0f}, "
          f"coexpression={row['coexpression']:.0f}")

# Also: across ALL pairs (not just Q1), find the biggest assoc scores
# for near-zero cosine with experimental evidence and low min_deg
print(f"\n{'='*80}")
print("GLOBAL SEARCH: ALL PAIRS (not just Q1)")
print("  |cosine| < 0.1, experimental > 0, min_deg < 50")
print(f"{'='*80}")

global_poster = df[
    (df["cosine"].abs() < 0.1) &
    (df["experimental"] > 0) &
    (df["min_deg"] < 50)
].sort_values("assoc", ascending=False)

print(f"  Found {len(global_poster)} candidates")
for _, row in global_poster.head(15).iterrows():
    print(f"\n  {row['gene1']} -- {row['gene2']}")
    print(f"    Cosine: {row['cosine']:.4f}   Assoc: {row['assoc']:.4f}   "
          f"Improvement: {row['improvement']:+.4f}")
    print(f"    Degree: {row['gene1']}={row['deg1']}, {row['gene2']}={row['deg2']}")
    print(f"    STRING: combined={row['combined_score']:.0f}, "
          f"experimental={row['experimental']:.0f}, "
          f"database={row['database']:.0f}, "
          f"textmining={row['textmining']:.0f}, "
          f"coexpression={row['coexpression']:.0f}")

# Summary stats on Q1 cross-boundary
print(f"\n{'='*80}")
print("Q1 CROSS-BOUNDARY STATISTICS")
print(f"{'='*80}")
print(f"  Total Q1 CB pairs: {len(cb_q1)}")
print(f"  With experimental evidence: {len(cb_q1_exp)} ({100*len(cb_q1_exp)/len(cb_q1):.0f}%)")
print(f"  Mean assoc score: {cb_q1['assoc'].mean():.4f}")
print(f"  Mean improvement: {cb_q1['improvement'].mean():+.4f}")
print(f"  Assoc > 0.3: {(cb_q1['assoc'] > 0.3).sum()} pairs")
print(f"  Assoc > 0.5: {(cb_q1['assoc'] > 0.5).sum()} pairs")

# Degree distribution of Q1 CB genes
all_degs = list(cb_q1["deg1"]) + list(cb_q1["deg2"])
print(f"\n  Degree distribution of Q1 CB genes:")
print(f"    Min: {min(all_degs)}, Median: {np.median(all_degs):.0f}, "
      f"Max: {max(all_degs)}, Mean: {np.mean(all_degs):.0f}")
print(f"    Genes with deg < 20: {sum(1 for d in all_degs if d < 20)}")
print(f"    Genes with deg < 10: {sum(1 for d in all_degs if d < 10)}")

print("\nDone.")
