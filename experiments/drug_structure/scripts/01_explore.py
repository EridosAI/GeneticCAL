"""Phase 1: Explore PRISM drug data and compute Morgan fingerprint embeddings.

Loads PRISM sensitivity matrix and treatment info, matches drugs,
computes Morgan fingerprints from SMILES, analyzes cosine distributions,
runs PCA, and saves processed data for downstream phases.
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = ROOT / "data"
FIGURES = ROOT / "figures"
DATA_OUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

FP_BITS = 2048
FP_RADIUS = 2

# ── Load data ──────────────────────────────────────────────────────────────

print("=" * 60)
print("Phase 1: Explore")
print("=" * 60)

print("\n--- Loading treatment info ---")
tinfo = pd.read_csv(ROOT / "primary-screen-replicate-collapsed-treatment-info.csv")
print(f"  Rows: {len(tinfo)}")
print(f"  Columns: {list(tinfo.columns)}")
print(f"  SMILES available: {tinfo['smiles'].notna().sum()} / {len(tinfo)}")

# Count MOA/target (treat 'NA' string as missing)
has_moa = ((tinfo["moa"].notna()) & (tinfo["moa"] != "NA")).sum()
has_target = ((tinfo["target"].notna()) & (tinfo["target"] != "NA")).sum()
print(f"  MOA available: {has_moa} / {len(tinfo)}")
print(f"  Target available: {has_target} / {len(tinfo)}")

print("\n--- Loading sensitivity matrix ---")
sens = pd.read_csv(
    ROOT / "primary-screen-replicate-collapsed-logfold-change.csv",
    index_col=0,
)
print(f"  Shape: {sens.shape[0]} cell lines x {sens.shape[1]} treatments")
print(f"  NaN fraction: {sens.isna().sum().sum() / sens.size:.3f}")

# ── Match treatments ───────────────────────────────────────────────────────

print("\n--- Matching treatments ---")
matched_ids = set(sens.columns) & set(tinfo["column_name"])
print(f"  Sensitivity columns: {len(sens.columns)}")
print(f"  Treatment info rows: {len(tinfo)}")
print(f"  Matched: {len(matched_ids)}")

tinfo_matched = tinfo[tinfo["column_name"].isin(matched_ids)].copy()

# Deduplicate by broad_id: keep dose closest to 2.5 (standard PRISM dose)
tinfo_matched["dose_diff"] = abs(tinfo_matched["dose"] - 2.5)
tinfo_dedup = (
    tinfo_matched.sort_values("dose_diff")
    .drop_duplicates(subset="broad_id", keep="first")
    .drop(columns=["dose_diff"])
    .reset_index(drop=True)
)
print(f"  Unique drugs (by broad_id): {len(tinfo_dedup)}")

# Show dose/screen distribution
dose_counts = tinfo_matched["dose"].value_counts().head(5)
screen_counts = tinfo_matched["screen_id"].value_counts().head(5)
print(f"  Top doses: {dict(dose_counts)}")
print(f"  Top screens: {dict(screen_counts)}")

# ── Compute Morgan fingerprints ────────────────────────────────────────────

print("\n--- Computing Morgan fingerprints ---")
try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem

    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    print("ERROR: RDKit not installed. Run: pip install rdkit-pypi")
    sys.exit(1)

valid_rows = []
embeddings_list = []
failed_smiles = []

for _, row in tqdm(tinfo_dedup.iterrows(), total=len(tinfo_dedup), desc="Fingerprints"):
    smiles = row["smiles"]
    if pd.isna(smiles) or not str(smiles).strip():
        failed_smiles.append((row["broad_id"], "empty"))
        continue
    smiles = str(smiles).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        # CSV artifact: commas in SMILES field cause extra text; take first token
        smiles = smiles.split(",")[0].strip()
        mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        failed_smiles.append((row["broad_id"], smiles[:50]))
        continue
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, FP_RADIUS, nBits=FP_BITS)
    embeddings_list.append(np.array(fp, dtype=np.float32))
    valid_rows.append(row)

embeddings = np.array(embeddings_list)
drug_df = pd.DataFrame(valid_rows).reset_index(drop=True)

print(f"  Valid drugs: {len(drug_df)}")
print(f"  Failed SMILES: {len(failed_smiles)}")
print(f"  Embedding shape: {embeddings.shape}")

# Fingerprint sparsity
bits_on = embeddings.sum(axis=1)
print(f"  Bits ON per drug: mean={bits_on.mean():.1f}, std={bits_on.std():.1f}, "
      f"min={bits_on.min():.0f}, max={bits_on.max():.0f}")

# ── Extract sensitivity matrix for valid drugs ─────────────────────────────

print("\n--- Building matched sensitivity matrix ---")
valid_col_names = drug_df["column_name"].tolist()
sens_matched = sens[valid_col_names].values.astype(np.float32)  # cell_lines x drugs
print(f"  Sensitivity matrix: {sens_matched.shape}")
print(f"  NaN count: {np.isnan(sens_matched).sum()}")

# ── Cosine similarity distribution ─────────────────────────────────────────

print("\n--- Cosine similarity distribution ---")
emb_norm = normalize(embeddings, axis=1)

# Sample pairs for distribution (all pairs if <5000 drugs)
n_drugs = len(embeddings)
if n_drugs <= 5000:
    # Compute full pairwise cosine
    cos_matrix = emb_norm @ emb_norm.T
    # Extract upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n_drugs, k=1)
    cos_values = cos_matrix[triu_idx]
else:
    # Sample 5000 drugs
    sample_idx = np.random.default_rng(42).choice(n_drugs, 5000, replace=False)
    emb_sample = emb_norm[sample_idx]
    cos_matrix = emb_sample @ emb_sample.T
    triu_idx = np.triu_indices(5000, k=1)
    cos_values = cos_matrix[triu_idx]

print(f"  Pairs sampled: {len(cos_values):,}")
print(f"  Cosine: mean={cos_values.mean():.4f}, std={cos_values.std():.4f}")
print(f"  Cosine: median={np.median(cos_values):.4f}")
print(f"  Cosine: min={cos_values.min():.4f}, max={cos_values.max():.4f}")

# Percentiles
for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{pct}: {np.percentile(cos_values, pct):.4f}")

# Fraction at various thresholds
for thresh in [0.1, 0.2, 0.3, 0.5, 0.7]:
    frac = (cos_values > thresh).mean()
    print(f"    Fraction > {thresh}: {frac:.4f}")

# Plot cosine distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(cos_values, bins=200, density=True, alpha=0.7, color="steelblue")
axes[0].axvline(0.2, color="red", ls="--", alpha=0.7, label="|cos|=0.2")
axes[0].axvline(0.5, color="orange", ls="--", alpha=0.7, label="|cos|=0.5")
axes[0].set_xlabel("Cosine Similarity")
axes[0].set_ylabel("Density")
axes[0].set_title(f"Morgan FP Cosine Distribution (N={n_drugs})")
axes[0].legend()

# CDF
cos_sorted = np.sort(cos_values)
cdf = np.arange(1, len(cos_sorted) + 1) / len(cos_sorted)
axes[1].plot(cos_sorted, cdf, color="steelblue", linewidth=1)
axes[1].axhline(0.5, color="gray", ls=":", alpha=0.5)
axes[1].axvline(0.2, color="red", ls="--", alpha=0.7)
axes[1].set_xlabel("Cosine Similarity")
axes[1].set_ylabel("CDF")
axes[1].set_title("Cumulative Distribution")

plt.tight_layout()
plt.savefig(FIGURES / "01_cosine_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/01_cosine_distribution.png")

# ── PCA analysis ───────────────────────────────────────────────────────────

print("\n--- PCA analysis ---")
# Fit PCA on full embeddings
pca_full = PCA(n_components=min(200, n_drugs, FP_BITS))
pca_full.fit(embeddings)

var_explained = pca_full.explained_variance_ratio_
cum_var = np.cumsum(var_explained)

print(f"  Variance explained:")
for n in [10, 20, 50, 100, 150, 200]:
    if n <= len(cum_var):
        print(f"    PCA-{n}: {cum_var[n-1]:.4f}")

# Save PCA-reduced versions
pca50 = PCA(n_components=50)
emb_pca50 = pca50.fit_transform(embeddings).astype(np.float32)

pca100 = PCA(n_components=100)
emb_pca100 = pca100.fit_transform(embeddings).astype(np.float32)

print(f"  PCA-50 shape: {emb_pca50.shape}")
print(f"  PCA-100 shape: {emb_pca100.shape}")

# Plot variance explained
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cum_var) + 1), cum_var, "b-", linewidth=1.5)
ax.axhline(0.8, color="red", ls="--", alpha=0.5, label="80%")
ax.axhline(0.9, color="orange", ls="--", alpha=0.5, label="90%")
ax.axhline(0.95, color="green", ls="--", alpha=0.5, label="95%")
for n in [50, 100]:
    if n <= len(cum_var):
        ax.axvline(n, color="gray", ls=":", alpha=0.5)
        ax.text(n + 2, cum_var[n - 1] - 0.03, f"PCA-{n}: {cum_var[n-1]:.2f}")
ax.set_xlabel("Number of Components")
ax.set_ylabel("Cumulative Variance Explained")
ax.set_title("PCA on Morgan Fingerprints (2048-bit)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "01_pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/01_pca_variance.png")

# ── Save outputs ───────────────────────────────────────────────────────────

print("\n--- Saving outputs ---")

# Drug info
drug_df.to_csv(DATA_OUT / "drug_info.csv", index=False)
print(f"  data/drug_info.csv ({len(drug_df)} drugs)")

# Embeddings
np.save(DATA_OUT / "embeddings_raw.npy", embeddings)
np.save(DATA_OUT / "embeddings_pca50.npy", emb_pca50)
np.save(DATA_OUT / "embeddings_pca100.npy", emb_pca100)
print(f"  data/embeddings_raw.npy {embeddings.shape}")
print(f"  data/embeddings_pca50.npy {emb_pca50.shape}")
print(f"  data/embeddings_pca100.npy {emb_pca100.shape}")

# Sensitivity matrix
np.save(DATA_OUT / "sensitivity_matrix.npy", sens_matched)
print(f"  data/sensitivity_matrix.npy {sens_matched.shape}")

# Cell line IDs
cell_lines = list(sens.index)
with open(DATA_OUT / "cell_line_ids.json", "w") as f:
    json.dump(cell_lines, f)

# Stats
stats = {
    "n_drugs": int(len(drug_df)),
    "n_cell_lines": int(sens_matched.shape[0]),
    "n_treatments_raw": int(len(tinfo)),
    "n_matched": int(len(matched_ids)),
    "n_unique_broad_id": int(len(tinfo_dedup)),
    "n_failed_smiles": int(len(failed_smiles)),
    "fp_bits": FP_BITS,
    "fp_radius": FP_RADIUS,
    "nan_fraction": float(np.isnan(sens_matched).sum() / sens_matched.size),
    "cosine_mean": float(cos_values.mean()),
    "cosine_std": float(cos_values.std()),
    "cosine_median": float(np.median(cos_values)),
    "cosine_frac_gt_0.2": float((cos_values > 0.2).mean()),
    "cosine_frac_gt_0.5": float((cos_values > 0.5).mean()),
    "pca50_var_explained": float(cum_var[49]) if len(cum_var) >= 50 else None,
    "pca100_var_explained": float(cum_var[99]) if len(cum_var) >= 100 else None,
    "bits_on_mean": float(bits_on.mean()),
}
with open(DATA_OUT / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"  data/stats.json")

# ── Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1 SUMMARY")
print("=" * 60)
print(f"  Drugs: {len(drug_df)}")
print(f"  Cell lines: {sens_matched.shape[0]}")
print(f"  Embedding dim: {FP_BITS}")
print(f"  Cosine mean: {cos_values.mean():.4f} (std={cos_values.std():.4f})")
print(f"  Fraction cosine > 0.5: {(cos_values > 0.5).mean():.4f}")
print(f"  Fraction cosine > 0.2: {(cos_values > 0.2).mean():.4f}")

if (cos_values > 0.5).mean() > 0.3:
    print("\n  ⚠ WARNING: High cosine clustering — potential confounding!")
    print("  Many drug pairs have similar fingerprints.")
else:
    print("\n  ✓ Cosine distribution looks broad — good for cross-boundary signal.")

print("\nPhase 1 complete.")
