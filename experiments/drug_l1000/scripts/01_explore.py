"""Phase 1: Explore LINCS L1000 data, match to PRISM, build embeddings.

Uses Level5 MODZ consensus signatures (replicate-collapsed). Selectively
loads only needed signatures (one per matched drug) and 978 landmark genes
from the GCTx file to avoid loading the full 12GB into RAM.

Pipeline:
  1. Load sig_info metadata, filter to trt_cp drugs
  2. Pick one best signature per drug (prefer 10µM, 24h)
  3. Load PRISM, match drugs by name and BRD-ID prefix
  4. Selectively load matched signatures from GCTx (978 landmark genes only)
  5. StandardScaler normalize, PCA, cosine analysis
"""

import gzip
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

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent.parent
DATA_IN = ROOT / "data"
DATA_OUT = ROOT / "data"
FIGURES = ROOT / "figures"
DATA_OUT.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)

print("=" * 60)
print("Phase 1: Explore")
print("=" * 60)

# ── Step 1: Find and decompress GCTx ──────────────────────────────────────

# Try Level5 first, fall back to Level4/Level3
LEVEL5_GZ = DATA_IN / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz"
LEVEL5 = DATA_IN / "GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx"

gctx_file = None
if LEVEL5.exists():
    gctx_file = LEVEL5
    print(f"\n  Level5 GCTx found: {LEVEL5.name}")
elif LEVEL5_GZ.exists():
    print(f"\n--- Decompressing Level5 GCTx ({LEVEL5_GZ.stat().st_size / 1e9:.1f} GB) ---")
    print("  This will take several minutes...")
    with gzip.open(LEVEL5_GZ, "rb") as f_in, open(LEVEL5, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out, length=64 * 1024 * 1024)
    print(f"  Decompressed: {LEVEL5.name} ({LEVEL5.stat().st_size / 1e9:.1f} GB)")
    gctx_file = LEVEL5
else:
    # Check for any gctx file
    gctx_files = list(DATA_IN.glob("*.gctx")) + list(DATA_IN.glob("*.gctx.gz"))
    if gctx_files:
        candidate = gctx_files[0]
        if candidate.suffix == ".gz":
            decompressed = candidate.with_suffix("")
            if not decompressed.exists():
                print(f"\n--- Decompressing {candidate.name} ---")
                with gzip.open(candidate, "rb") as f_in, open(decompressed, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out, length=64 * 1024 * 1024)
            gctx_file = decompressed
        else:
            gctx_file = candidate
        print(f"\n  Using GCTx: {gctx_file.name}")

if gctx_file is None:
    print("\nERROR: No GCTx file found in data/")
    print("Download GSE92742_Broad_LINCS_Level5_COMPZ.MODZ_n473647x12328.gctx.gz")
    print("from https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE92742")
    sys.exit(1)

# ── Step 2: Load LINCS metadata ───────────────────────────────────────────

print("\n--- Loading LINCS metadata ---")

sig_info = pd.read_csv(DATA_IN / "GSE92742_Broad_LINCS_sig_info.txt.gz", sep="\t",
                        low_memory=False)
print(f"  sig_info: {len(sig_info)} signatures")

gene_info = pd.read_csv(DATA_IN / "GSE92742_Broad_LINCS_gene_info.txt.gz", sep="\t")
landmark_genes = gene_info[gene_info["pr_is_lm"] == 1]["pr_gene_id"].astype(str).values
print(f"  Landmark genes: {len(landmark_genes)}")

pert_info = pd.read_csv(DATA_IN / "GSE92742_Broad_LINCS_pert_info.txt.gz", sep="\t")
pert_drugs = pert_info[pert_info["pert_type"] == "trt_cp"].copy()
print(f"  pert_info drugs: {len(pert_drugs)}")

# Filter to drug treatments
drug_sigs = sig_info[sig_info["pert_type"] == "trt_cp"].copy()
print(f"\n  Drug signatures (trt_cp): {len(drug_sigs)}")
print(f"  Unique drugs (pert_iname): {drug_sigs['pert_iname'].nunique()}")
print(f"  Unique pert_ids: {drug_sigs['pert_id'].nunique()}")

# Dose/time/cell distributions
print(f"\n  Dose distribution (top 5):")
for dose, count in drug_sigs["pert_dose"].value_counts().head(5).items():
    print(f"    {dose} µM: {count:,}")

print(f"\n  Timepoint distribution:")
for time, count in drug_sigs["pert_time"].value_counts().items():
    print(f"    {time}h: {count:,}")

print(f"\n  Cell line distribution (top 10):")
for cell, count in drug_sigs["cell_id"].value_counts().head(10).items():
    print(f"    {cell}: {count:,}")

# ── Step 3: Pick one best signature per drug ──────────────────────────────

print("\n--- Picking best signature per drug ---")

# Coerce pert_dose to numeric
drug_sigs["pert_dose_num"] = pd.to_numeric(drug_sigs["pert_dose"], errors="coerce")

# Score each signature: prefer 10µM, 24h, common cell lines
PREFERRED_CELLS = ["VCAP", "MCF7", "PC3", "A549", "A375", "HT29", "HA1E", "HCC515"]
cell_rank = {c: i for i, c in enumerate(PREFERRED_CELLS)}

def sig_score(row):
    """Lower score = better. Combines dose proximity, timepoint, cell line."""
    dose = row["pert_dose_num"] if pd.notna(row["pert_dose_num"]) else 999
    dose_penalty = abs(dose - 10.0)  # prefer 10µM
    time_penalty = 0 if row["pert_time"] == 24 else 100  # prefer 24h
    cell_penalty = cell_rank.get(row["cell_id"], len(PREFERRED_CELLS))
    return dose_penalty + time_penalty + cell_penalty

drug_sigs["_score"] = drug_sigs.apply(sig_score, axis=1)

# Pick best sig per pert_id (lowest score)
best_sigs = (
    drug_sigs.sort_values("_score")
    .drop_duplicates(subset="pert_id", keep="first")
    .drop(columns=["_score", "pert_dose_num"])
    .reset_index(drop=True)
)
print(f"  Best signatures selected: {len(best_sigs)}")

# Stats on selected conditions
dose_dist = best_sigs["pert_dose"].value_counts().head(5)
time_dist = best_sigs["pert_time"].value_counts()
cell_dist = best_sigs["cell_id"].value_counts().head(5)
print(f"  Selected dose distribution: {dict(dose_dist)}")
print(f"  Selected time distribution: {dict(time_dist)}")
print(f"  Selected cell distribution: {dict(cell_dist)}")

# ── Step 4: Load PRISM data and match ─────────────────────────────────────

print("\n--- Loading PRISM data ---")
tinfo = pd.read_csv(ROOT / "primary-screen-replicate-collapsed-treatment-info.csv")
print(f"  Treatment info: {len(tinfo)} rows")

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

# Match treatments to sensitivity columns
matched_ids = set(sens.columns) & set(tinfo["column_name"])
tinfo_matched = tinfo[tinfo["column_name"].isin(matched_ids)].copy()

# Deduplicate by broad_id: keep dose closest to 2.5 (standard PRISM dose)
tinfo_matched["dose_diff"] = abs(tinfo_matched["dose"] - 2.5)
tinfo_dedup = (
    tinfo_matched.sort_values("dose_diff")
    .drop_duplicates(subset="broad_id", keep="first")
    .drop(columns=["dose_diff"])
    .reset_index(drop=True)
)
print(f"  PRISM unique drugs (by broad_id): {len(tinfo_dedup)}")

# ── Step 5: Match drugs between LINCS and PRISM ──────────────────────────

print("\n--- Matching drugs between LINCS and PRISM ---")

def normalize_name(name):
    if pd.isna(name):
        return ""
    return str(name).lower().strip().replace("-", "").replace(" ", "")

# Build LINCS lookup from pert_info (has all 20K drug names)
# Map: normalized_name -> pert_id, and pert_id -> pert_id
lincs_name_to_pid = {}
lincs_pid_set = set()
for _, row in pert_drugs.iterrows():
    pid = str(row["pert_id"]).strip()
    name = str(row["pert_iname"]).strip()
    lincs_pid_set.add(pid)
    lincs_name_to_pid[normalize_name(name)] = pid
    lincs_name_to_pid[normalize_name(pid)] = pid

# Also use sig_info drug names (may differ from pert_info)
for _, row in best_sigs.iterrows():
    pid = str(row["pert_id"]).strip()
    name = str(row["pert_iname"]).strip()
    lincs_pid_set.add(pid)
    lincs_name_to_pid[normalize_name(name)] = pid

print(f"  LINCS unique drugs (pert_info): {len(pert_drugs)}")
print(f"  LINCS unique drugs (with sigs): {len(best_sigs)}")
print(f"  PRISM unique drugs: {len(tinfo_dedup)}")

# Build set of pert_ids that actually have Level5 signatures
sigs_pid_set = set(best_sigs["pert_id"].astype(str).str.strip())

# Match PRISM drugs to LINCS
name_matches = {}  # prism_broad_id -> lincs_pert_id
for _, row in tinfo_dedup.iterrows():
    prism_name = normalize_name(row["name"])
    prism_bid = str(row["broad_id"]).strip()

    # Strategy 1: Name match
    if prism_name and prism_name in lincs_name_to_pid:
        candidate_pid = lincs_name_to_pid[prism_name]
        if candidate_pid in sigs_pid_set:
            name_matches[prism_bid] = candidate_pid
            continue

    # Strategy 2: BRD-ID prefix match (PRISM BRD-Axxxxxxxx-xxx-xx-x -> BRD-Axxxxxxxx)
    parts = prism_bid.split("-")
    if len(parts) >= 2:
        short_bid = parts[0] + "-" + parts[1]
        if short_bid in sigs_pid_set:
            name_matches[prism_bid] = short_bid
            continue

    # Strategy 3: Name match even if no sig (for extended matching report)
    if prism_name and prism_name in lincs_name_to_pid:
        name_matches[prism_bid] = lincs_name_to_pid[prism_name]

print(f"\n  Matched drugs (name + BRD-ID): {len(name_matches)}")
# How many of those have actual signatures?
matched_with_sigs = sum(1 for pid in name_matches.values() if pid in sigs_pid_set)
print(f"  Matched with Level5 signatures: {matched_with_sigs}")

# ── Step 6: Load signatures from GCTx ─────────────────────────────────────

print("\n--- Loading signatures from GCTx ---")

# Collect sig_ids for matched drugs
sig_id_to_prism_bid = {}  # sig_id -> prism_broad_id
for prism_bid, lincs_pid in name_matches.items():
    # Find the best_sig for this pert_id
    match = best_sigs[best_sigs["pert_id"] == lincs_pid]
    if len(match) > 0:
        sid = match.iloc[0]["sig_id"]
        sig_id_to_prism_bid[sid] = prism_bid

sig_ids_to_load = list(sig_id_to_prism_bid.keys())
print(f"  Signatures to load: {len(sig_ids_to_load)}")

if len(sig_ids_to_load) < 100:
    print(f"\n  ⚠ KILL CRITERION: Only {len(sig_ids_to_load)} matched drug signatures.")
    print("  Insufficient data for PAM experiment.")
    sys.exit(1)

# Selectively load only needed columns (sig_ids) and rows (landmark genes)
from cmapPy.pandasGEXpress import parse as gctx_parse

print(f"  Loading {len(sig_ids_to_load)} signatures x {len(landmark_genes)} genes "
      f"from {gctx_file.name}...")
gctoo = gctx_parse.parse(str(gctx_file),
                          cid=sig_ids_to_load,
                          rid=landmark_genes)
print(f"  Loaded data shape: {gctoo.data_df.shape} (genes x signatures)")

# Verify what we got
actual_sigs = list(gctoo.data_df.columns)
actual_genes = list(gctoo.data_df.index)
print(f"  Actual signatures loaded: {len(actual_sigs)}")
print(f"  Actual genes loaded: {len(actual_genes)}")

# ── Step 7: Build embedding matrix ────────────────────────────────────────

print("\n--- Building embedding matrix ---")

# Map sig_id -> PRISM row
prism_bid_to_row = tinfo_dedup.set_index("broad_id")

embeddings_list = []
valid_prism_rows = []
skipped = 0

for sid in actual_sigs:
    prism_bid = sig_id_to_prism_bid.get(sid)
    if prism_bid is None or prism_bid not in prism_bid_to_row.index:
        skipped += 1
        continue

    # Extract 978-dim expression vector for this drug
    emb = gctoo.data_df[sid].values.astype(np.float32)
    embeddings_list.append(emb)
    valid_prism_rows.append(prism_bid_to_row.loc[prism_bid])

embeddings_raw = np.array(embeddings_list)
drug_df = pd.DataFrame(valid_prism_rows).reset_index()
# The reset_index moves broad_id from index to column
if "broad_id" not in drug_df.columns and "index" in drug_df.columns:
    drug_df = drug_df.rename(columns={"index": "broad_id"})
drug_df = drug_df.reset_index(drop=True)

print(f"  Valid drugs with both L1000 + PRISM: {len(drug_df)}")
print(f"  Raw embedding shape: {embeddings_raw.shape}")
print(f"  Skipped (no PRISM match): {skipped}")

if len(drug_df) < 500:
    print(f"\n  ⚠ WARNING: Only {len(drug_df)} drugs. May be insufficient.")

# ── Step 8: Normalize embeddings ──────────────────────────────────────────

print("\n--- Normalizing embeddings ---")

# StandardScaler normalize
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings_raw).astype(np.float32)
print(f"  StandardScaler applied: mean~{embeddings_scaled.mean():.4f}, "
      f"std~{embeddings_scaled.std():.4f}")

# Extract sensitivity matrix for valid drugs
print("\n--- Building matched sensitivity matrix ---")
valid_col_names = drug_df["column_name"].tolist()
sens_matched = sens[valid_col_names].values.astype(np.float32)
print(f"  Sensitivity matrix: {sens_matched.shape}")
print(f"  NaN count: {np.isnan(sens_matched).sum()}")

# ── Step 9: Cosine similarity distribution ────────────────────────────────

print("\n--- Cosine similarity distribution ---")
emb_norm = normalize(embeddings_scaled, axis=1)

n_drugs = len(embeddings_scaled)
if n_drugs <= 5000:
    cos_matrix = emb_norm @ emb_norm.T
    triu_idx = np.triu_indices(n_drugs, k=1)
    cos_values = cos_matrix[triu_idx]
else:
    sample_idx = np.random.default_rng(42).choice(n_drugs, 5000, replace=False)
    emb_sample = emb_norm[sample_idx]
    cos_matrix = emb_sample @ emb_sample.T
    triu_idx = np.triu_indices(5000, k=1)
    cos_values = cos_matrix[triu_idx]

print(f"  Pairs sampled: {len(cos_values):,}")
print(f"  Cosine: mean={cos_values.mean():.4f}, std={cos_values.std():.4f}")
print(f"  Cosine: median={np.median(cos_values):.4f}")
print(f"  Cosine: min={cos_values.min():.4f}, max={cos_values.max():.4f}")

for pct in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"    P{pct}: {np.percentile(cos_values, pct):.4f}")

for thresh in [0.1, 0.2, 0.3, 0.5, 0.7]:
    frac = (np.abs(cos_values) > thresh).mean()
    print(f"    Fraction |cos| > {thresh}: {frac:.4f}")

# Plot cosine distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(cos_values, bins=200, density=True, alpha=0.7, color="steelblue")
axes[0].axvline(0.2, color="red", ls="--", alpha=0.7, label="|cos|=0.2")
axes[0].axvline(-0.2, color="red", ls="--", alpha=0.7)
axes[0].axvline(0.5, color="orange", ls="--", alpha=0.7, label="|cos|=0.5")
axes[0].set_xlabel("Cosine Similarity")
axes[0].set_ylabel("Density")
axes[0].set_title(f"L1000 Cosine Distribution (N={n_drugs})")
axes[0].legend()

cos_sorted = np.sort(cos_values)
cdf = np.arange(1, len(cos_sorted) + 1) / len(cos_sorted)
axes[1].plot(cos_sorted, cdf, color="steelblue", linewidth=1)
axes[1].axhline(0.5, color="gray", ls=":", alpha=0.5)
axes[1].axvline(0.2, color="red", ls="--", alpha=0.7)
axes[1].axvline(-0.2, color="red", ls="--", alpha=0.7)
axes[1].set_xlabel("Cosine Similarity")
axes[1].set_ylabel("CDF")
axes[1].set_title("Cumulative Distribution")

plt.tight_layout()
plt.savefig(FIGURES / "01_cosine_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/01_cosine_distribution.png")

# ── Step 10: PCA analysis ─────────────────────────────────────────────────

print("\n--- PCA analysis ---")
max_components = min(200, n_drugs - 1, 978)
pca_full = PCA(n_components=max_components)
pca_full.fit(embeddings_scaled)

var_explained = pca_full.explained_variance_ratio_
cum_var = np.cumsum(var_explained)

print(f"  Variance explained:")
for n in [10, 20, 50, 100, 150, 200]:
    if n <= len(cum_var):
        print(f"    PCA-{n}: {cum_var[n-1]:.4f}")

# PCA-reduced versions (guard against small drug counts)
pca50_dim = min(50, n_drugs - 1)
pca100_dim = min(100, n_drugs - 1)

pca50 = PCA(n_components=pca50_dim)
emb_pca50 = pca50.fit_transform(embeddings_scaled).astype(np.float32)

pca100 = PCA(n_components=pca100_dim)
emb_pca100 = pca100.fit_transform(embeddings_scaled).astype(np.float32)

print(f"  PCA-{pca50_dim} shape: {emb_pca50.shape}")
print(f"  PCA-{pca100_dim} shape: {emb_pca100.shape}")

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
ax.set_title("PCA on L1000 Signatures (978 landmark genes)")
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES / "01_pca_variance.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: figures/01_pca_variance.png")

# ── Step 11: Save outputs ─────────────────────────────────────────────────

print("\n--- Saving outputs ---")

drug_df.to_csv(DATA_OUT / "drug_info.csv", index=False)
print(f"  data/drug_info.csv ({len(drug_df)} drugs)")

np.save(DATA_OUT / "embeddings_raw.npy", embeddings_scaled)
np.save(DATA_OUT / "embeddings_pca50.npy", emb_pca50)
np.save(DATA_OUT / "embeddings_pca100.npy", emb_pca100)
print(f"  data/embeddings_raw.npy {embeddings_scaled.shape}")
print(f"  data/embeddings_pca50.npy {emb_pca50.shape}")
print(f"  data/embeddings_pca100.npy {emb_pca100.shape}")

np.save(DATA_OUT / "sensitivity_matrix.npy", sens_matched)
print(f"  data/sensitivity_matrix.npy {sens_matched.shape}")

cell_lines = list(sens.index)
with open(DATA_OUT / "cell_line_ids.json", "w") as f:
    json.dump(cell_lines, f)

stats = {
    "n_drugs": int(len(drug_df)),
    "n_cell_lines": int(sens_matched.shape[0]),
    "n_treatments_raw_prism": int(len(tinfo)),
    "n_lincs_drug_sigs": int(len(drug_sigs)),
    "n_lincs_unique_drugs": int(drug_sigs["pert_id"].nunique()),
    "n_matched": int(len(name_matches)),
    "n_matched_with_sigs": int(matched_with_sigs),
    "n_valid_with_embeddings": int(len(drug_df)),
    "embedding_dim_raw": 978,
    "nan_fraction": float(np.isnan(sens_matched).sum() / sens_matched.size),
    "cosine_mean": float(cos_values.mean()),
    "cosine_std": float(cos_values.std()),
    "cosine_median": float(np.median(cos_values)),
    "cosine_frac_abs_gt_0.2": float((np.abs(cos_values) > 0.2).mean()),
    "cosine_frac_abs_gt_0.5": float((np.abs(cos_values) > 0.5).mean()),
    "pca50_var_explained": float(cum_var[pca50_dim - 1]),
    "pca100_var_explained": float(cum_var[pca100_dim - 1]) if pca100_dim <= len(cum_var) else None,
}
with open(DATA_OUT / "stats.json", "w") as f:
    json.dump(stats, f, indent=2)
print(f"  data/stats.json")

# ── Summary ───────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("PHASE 1 SUMMARY")
print("=" * 60)
print(f"  Drugs: {len(drug_df)}")
print(f"  Cell lines: {sens_matched.shape[0]}")
print(f"  Embedding dim: 978 (L1000 landmark genes, StandardScaled)")
print(f"  Cosine mean: {cos_values.mean():.4f} (std={cos_values.std():.4f})")
print(f"  Fraction |cos| > 0.5: {(np.abs(cos_values) > 0.5).mean():.4f}")
print(f"  Fraction |cos| > 0.2: {(np.abs(cos_values) > 0.2).mean():.4f}")
if pca50_dim <= len(cum_var):
    print(f"  PCA-{pca50_dim} variance: {cum_var[pca50_dim - 1]:.4f}")
if pca100_dim <= len(cum_var):
    print(f"  PCA-{pca100_dim} variance: {cum_var[pca100_dim - 1]:.4f}")

if (np.abs(cos_values) > 0.5).mean() > 0.3:
    print("\n  ⚠ WARNING: High cosine clustering — potential confounding!")
elif len(drug_df) < 500:
    print(f"\n  ⚠ WARNING: Only {len(drug_df)} drugs. May be insufficient.")
else:
    print("\n  ✓ Good: Cosine distribution looks reasonable for cross-boundary signal.")

# Kill criteria check
if len(drug_df) < 500:
    print(f"\n  ⚠ KILL CRITERION: Fewer than 500 matched drugs ({len(drug_df)}).")
if (np.abs(cos_values) > 0.5).mean() > 0.85:
    print("\n  ⚠ KILL CRITERION: Cosine baseline > 0.85 — confounding likely.")

print("\nPhase 1 complete.")
