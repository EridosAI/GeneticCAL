"""Phase 1b: Replace Morgan fingerprints with RDKit molecular descriptors.

Computes ~210 physicochemical descriptors per drug, cleans/imputes/standardizes,
then saves in the same format as Phase 1 so Phases 2-3 can rerun unchanged.
Backs up Morgan fingerprint embeddings first.
"""

import json
import shutil
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors

    RDLogger.logger().setLevel(RDLogger.ERROR)
except ImportError:
    print("ERROR: RDKit not installed.")
    sys.exit(1)

# ── Load drugs from Phase 1 ───────────────────────────────────────────────

print("=" * 60)
print("Phase 1b: RDKit Molecular Descriptors")
print("=" * 60)

print("\n--- Loading drug info ---")
drug_df = pd.read_csv(DATA / "drug_info.csv")
print(f"  Drugs: {len(drug_df)}")

# ── Backup Morgan fingerprint embeddings ───────────────────────────────────

morgan_path = DATA / "embeddings_raw.npy"
morgan_backup = DATA / "embeddings_morgan_backup.npy"
if morgan_path.exists() and not morgan_backup.exists():
    shutil.copy2(morgan_path, morgan_backup)
    print(f"  Backed up Morgan FPs to data/embeddings_morgan_backup.npy")

# ── Compute RDKit descriptors ──────────────────────────────────────────────

print("\n--- Computing RDKit molecular descriptors ---")
desc_names = [d[0] for d in Descriptors._descList]
calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
print(f"  Available descriptors: {len(desc_names)}")

all_descs = []
valid_mask = []

for _, row in tqdm(drug_df.iterrows(), total=len(drug_df), desc="Descriptors"):
    smiles = str(row["smiles"]).strip()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        smiles = smiles.split(",")[0].strip()
        mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        all_descs.append([np.nan] * len(desc_names))
        valid_mask.append(False)
        continue

    try:
        desc_values = list(calculator.CalcDescriptors(mol))
        all_descs.append(desc_values)
        valid_mask.append(True)
    except Exception:
        all_descs.append([np.nan] * len(desc_names))
        valid_mask.append(False)

desc_matrix = np.array(all_descs, dtype=np.float64)
valid_mask = np.array(valid_mask)
print(f"  Valid molecules: {valid_mask.sum()} / {len(drug_df)}")
print(f"  Raw descriptor matrix: {desc_matrix.shape}")

# ── Clean descriptors ──────────────────────────────────────────────────────

print("\n--- Cleaning descriptors ---")

# Replace inf with NaN
desc_matrix[~np.isfinite(desc_matrix)] = np.nan

# Drop descriptors with >20% NaN
nan_frac = np.isnan(desc_matrix).mean(axis=0)
keep_cols = nan_frac < 0.2
print(f"  Descriptors with <20% NaN: {keep_cols.sum()} / {len(desc_names)}")

# Drop constant descriptors
col_std = np.nanstd(desc_matrix[:, keep_cols], axis=0)
nonconstant = col_std > 1e-10
print(f"  Non-constant descriptors: {nonconstant.sum()} / {keep_cols.sum()}")

# Apply filters
keep_idx = np.where(keep_cols)[0][nonconstant]
desc_clean = desc_matrix[:, keep_idx]
kept_names = [desc_names[i] for i in keep_idx]
print(f"  Final descriptor count: {desc_clean.shape[1]}")

# List some kept descriptors
print(f"  Sample descriptors: {kept_names[:15]}")

# Impute remaining NaN with column median
for col in range(desc_clean.shape[1]):
    nan_mask = np.isnan(desc_clean[:, col])
    if nan_mask.any():
        median_val = np.nanmedian(desc_clean[:, col])
        desc_clean[nan_mask, col] = median_val

# Verify no NaN remains
assert not np.isnan(desc_clean).any(), "NaN still present after imputation"
print(f"  NaN after imputation: 0")

# ── Standardize ────────────────────────────────────────────────────────────

print("\n--- Standardizing ---")
scaler = StandardScaler()
desc_scaled = scaler.fit_transform(desc_clean).astype(np.float32)
print(f"  Scaled descriptor matrix: {desc_scaled.shape}")
print(f"  Mean range: [{desc_scaled.mean(axis=0).min():.4f}, {desc_scaled.mean(axis=0).max():.4f}]")
print(f"  Std range: [{desc_scaled.std(axis=0).min():.4f}, {desc_scaled.std(axis=0).max():.4f}]")

# ── Cosine similarity distribution ─────────────────────────────────────────

print("\n--- Cosine similarity distribution ---")
emb_norm = normalize(desc_scaled, axis=1)

n_drugs = len(desc_scaled)
cos_matrix = emb_norm @ emb_norm.T
triu_idx = np.triu_indices(n_drugs, k=1)
cos_values = cos_matrix[triu_idx]

print(f"  Pairs: {len(cos_values):,}")
print(f"  Cosine: mean={cos_values.mean():.4f}, std={cos_values.std():.4f}")
print(f"  Cosine: median={np.median(cos_values):.4f}")
print(f"  Cosine: min={cos_values.min():.4f}, max={cos_values.max():.4f}")

for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{pct}: {np.percentile(cos_values, pct):.4f}")

for thresh in [0.1, 0.2, 0.3, 0.5, 0.7]:
    frac = (np.abs(cos_values) > thresh).mean()
    print(f"    Fraction |cos| > {thresh}: {frac:.4f}")

# ── PCA ────────────────────────────────────────────────────────────────────

print("\n--- PCA analysis ---")
pca_full = PCA(n_components=min(200, desc_scaled.shape[1], n_drugs))
pca_full.fit(desc_scaled)
cum_var = np.cumsum(pca_full.explained_variance_ratio_)

for n in [10, 20, 50, 100, 150, 200]:
    if n <= len(cum_var):
        print(f"  PCA-{n}: {cum_var[n-1]:.4f}")

pca50 = PCA(n_components=50)
emb_pca50 = pca50.fit_transform(desc_scaled).astype(np.float32)
pca100 = PCA(n_components=min(100, desc_scaled.shape[1]))
emb_pca100 = pca100.fit_transform(desc_scaled).astype(np.float32)

# ── Compare to Morgan FPs ─────────────────────────────────────────────────

print("\n--- Comparison with Morgan fingerprints ---")
if morgan_backup.exists():
    morgan_emb = np.load(morgan_backup)
    morgan_norm = normalize(morgan_emb, axis=1)
    morgan_cos = morgan_norm @ morgan_norm.T
    morgan_cos_values = morgan_cos[triu_idx]
    print(f"  Morgan FP cosine: mean={morgan_cos_values.mean():.4f}, "
          f"std={morgan_cos_values.std():.4f}")
    print(f"  RDKit desc cosine: mean={cos_values.mean():.4f}, "
          f"std={cos_values.std():.4f}")
    # Correlation between Morgan and RDKit cosine
    from scipy.stats import spearmanr
    # Sample for speed
    rng = np.random.default_rng(42)
    sample = rng.choice(len(cos_values), min(500_000, len(cos_values)), replace=False)
    rho, _ = spearmanr(morgan_cos_values[sample], cos_values[sample])
    print(f"  Spearman(Morgan cos, RDKit cos): {rho:.4f}")

# ── Figures ────────────────────────────────────────────────────────────────

print("\n--- Generating figures ---")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Cosine distribution comparison
axes[0].hist(cos_values, bins=200, density=True, alpha=0.6, color="steelblue",
             label="RDKit descriptors")
if morgan_backup.exists():
    axes[0].hist(morgan_cos_values, bins=200, density=True, alpha=0.4,
                 color="orange", label="Morgan FP")
axes[0].axvline(0.2, color="red", ls="--", alpha=0.7)
axes[0].axvline(-0.2, color="red", ls="--", alpha=0.7)
axes[0].set_xlabel("Cosine Similarity")
axes[0].set_ylabel("Density")
axes[0].set_title("Cosine Distribution: RDKit Descriptors vs Morgan FP")
axes[0].legend()

# PCA variance explained
axes[1].plot(range(1, len(cum_var) + 1), cum_var, "b-", linewidth=1.5)
axes[1].axhline(0.8, color="red", ls="--", alpha=0.5, label="80%")
axes[1].axhline(0.9, color="orange", ls="--", alpha=0.5, label="90%")
axes[1].axhline(0.95, color="green", ls="--", alpha=0.5, label="95%")
axes[1].set_xlabel("Number of Components")
axes[1].set_ylabel("Cumulative Variance Explained")
axes[1].set_title("PCA on RDKit Descriptors")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGURES / "01b_rdkit_descriptors.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/01b_rdkit_descriptors.png")

# ── Save embeddings (overwrite raw) ───────────────────────────────────────

print("\n--- Saving embeddings ---")
np.save(DATA / "embeddings_raw.npy", desc_scaled)
np.save(DATA / "embeddings_pca50.npy", emb_pca50)
np.save(DATA / "embeddings_pca100.npy", emb_pca100)
print(f"  data/embeddings_raw.npy {desc_scaled.shape} (RDKit descriptors, standardized)")
print(f"  data/embeddings_pca50.npy {emb_pca50.shape}")
print(f"  data/embeddings_pca100.npy {emb_pca100.shape}")

# Save descriptor names
with open(DATA / "descriptor_names.json", "w") as f:
    json.dump(kept_names, f)

# Update stats
stats = {
    "embedding_type": "rdkit_descriptors",
    "n_drugs": int(n_drugs),
    "n_descriptors_raw": int(len(desc_names)),
    "n_descriptors_clean": int(desc_clean.shape[1]),
    "cosine_mean": float(cos_values.mean()),
    "cosine_std": float(cos_values.std()),
    "cosine_median": float(np.median(cos_values)),
    "cosine_frac_gt_0.2": float((np.abs(cos_values) > 0.2).mean()),
    "cosine_frac_gt_0.5": float((np.abs(cos_values) > 0.5).mean()),
    "pca50_var_explained": float(cum_var[49]) if len(cum_var) >= 50 else None,
    "pca100_var_explained": float(cum_var[99]) if len(cum_var) >= 100 else None,
}
with open(DATA / "stats_rdkit.json", "w") as f:
    json.dump(stats, f, indent=2)

# ── Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1b SUMMARY")
print("=" * 60)
print(f"  Drugs: {n_drugs}")
print(f"  Descriptors: {desc_clean.shape[1]} (from {len(desc_names)} raw)")
print(f"  Embedding dim: {desc_scaled.shape[1]}")
print(f"  Cosine mean: {cos_values.mean():.4f} (std={cos_values.std():.4f})")
print(f"  Fraction |cos| > 0.5: {(np.abs(cos_values) > 0.5).mean():.4f}")
print(f"  Fraction |cos| > 0.2: {(np.abs(cos_values) > 0.2).mean():.4f}")
print(f"\n  embeddings_raw.npy now contains RDKit descriptors.")
print(f"  Morgan FPs backed up to embeddings_morgan_backup.npy.")
print(f"\n  Rerun Phase 2 (02_pairs.py) and Phase 3 (03_train.py) next.")
print("\nPhase 1b complete.")
