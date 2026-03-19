#!/usr/bin/env python3
"""
DepMap PAM Phase 5: External Validation
=========================================

Validate PAM scores against external ground truth that was NEVER used
in training (co-essentiality pairs are the training signal; STRING and
CORUM are independent).

STRING validation:
    - Download 9606.protein.links.v12.0.txt.gz and
      9606.protein.info.v12.0.txt.gz from stringdb-downloads.org
    - Map STRING protein IDs to gene names
    - Compute AUC: PAM scores discriminating STRING pairs vs non-pairs
      at confidence thresholds 400 / 700 / 900

CORUM complexes (optional):
    - Mean pairwise PAM vs cosine within known protein complexes

Prerequisites:
    python scripts/03_train.py  (or 04_ablations.py — needs model + embeddings)

Usage:
    python scripts/05_validate.py
"""

import gzip
import json
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
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

# STRING files (already downloaded in data/)
STRING_LINKS_FILE = DATA_DIR / "9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO_FILE = DATA_DIR / "9606.protein.info.v12.0.txt.gz"

STRING_LINKS_URL = "https://stringdb-downloads.org/download/protein.links.detailed.v12.0/9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO_URL = "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz"

CONFIDENCE_THRESHOLDS = [400, 700, 900]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GENE_COL_RE = re.compile(r"^(.+?)\s+\((\d+)\)$")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Model (same architecture)
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
# Download helper
# ---------------------------------------------------------------------------
def download_file(url, dest, description=""):
    import requests
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  Already downloaded: {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return
    print(f"  Downloading {description or url}...")
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=65536):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (1024 * 1024) < 65536:
                print(f"\r  {downloaded/total*100:.0f}%", end="")
    print(f"\n  Saved: {dest}")


# ---------------------------------------------------------------------------
# Step 1 — Load model and embeddings
# ---------------------------------------------------------------------------
def step1_load():
    print("=" * 70)
    print("STEP 1: LOAD MODEL AND EMBEDDINGS")
    print("=" * 70)

    emb = np.load(EMB_PATH)
    with open(GENE_LIST_PATH) as f:
        genes = json.load(f)
    n_genes = len(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    print(f"  Genes: {n_genes}, Embedding dim: {emb.shape[1]}")

    # Find best N
    with open(TRAIN_RESULTS_PATH) as f:
        train_meta = json.load(f)
    best_N = train_meta.get("best_N")
    if best_N is None:
        best_N = int(list(train_meta["results"].keys())[0])
    print(f"  Best N: {best_N:,}")

    # Load model
    model_path = RESULTS_DIR / f"model_{best_N}.pt"
    model = AssociationMLP().to(DEVICE)
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"  Model loaded: {model_path}")

    X_tensor = torch.tensor(emb, dtype=torch.float32, device=DEVICE)
    return model, X_tensor, genes, gene_to_idx, n_genes, best_N


# ---------------------------------------------------------------------------
# Step 2 — Download/parse STRING
# ---------------------------------------------------------------------------
def step2_download_string():
    print("\n" + "=" * 70)
    print("STEP 2: DOWNLOAD STRING DATA")
    print("=" * 70)
    download_file(STRING_INFO_URL, STRING_INFO_FILE, "STRING protein info")
    download_file(STRING_LINKS_URL, STRING_LINKS_FILE, "STRING detailed links")


def step3_parse_string(gene_to_idx):
    print("\n" + "=" * 70)
    print("STEP 3: PARSE STRING AND MAP TO GENES")
    print("=" * 70)

    # Build ENSP -> gene name mapping
    ensp_to_gene = {}
    with gzip.open(STRING_INFO_FILE, "rt") as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ensp_to_gene[parts[0]] = parts[1].upper()

    # Filter to genes in our dataset
    relevant_ensp = {ensp for ensp, gene in ensp_to_gene.items()
                     if gene in gene_to_idx}
    print(f"  STRING proteins: {len(ensp_to_gene):,}")
    print(f"  Matching our genes: {len(relevant_ensp):,}")

    # Parse links
    pairs_by_conf = {t: set() for t in CONFIDENCE_THRESHOLDS}
    n_lines = 0

    print(f"  Parsing STRING links...")
    with gzip.open(STRING_LINKS_FILE, "rt") as f:
        header = f.readline().strip().split()
        col_map = {name: i for i, name in enumerate(header)}

        for line in f:
            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f"    ...{n_lines/1e6:.0f}M lines")

            parts = line.strip().split()
            p1, p2 = parts[0], parts[1]
            if p1 not in relevant_ensp or p2 not in relevant_ensp:
                continue

            gene1 = ensp_to_gene[p1]
            gene2 = ensp_to_gene[p2]
            if gene1 == gene2:
                continue
            if gene1 not in gene_to_idx or gene2 not in gene_to_idx:
                continue

            combined = int(parts[col_map["combined_score"]])
            pair_key = (min(gene1, gene2), max(gene1, gene2))

            for t in CONFIDENCE_THRESHOLDS:
                if combined >= t:
                    pairs_by_conf[t].add(pair_key)

    print(f"  Total lines parsed: {n_lines:,}")
    for t in CONFIDENCE_THRESHOLDS:
        print(f"  Confidence >= {t}: {len(pairs_by_conf[t]):,} pairs")

    return pairs_by_conf


# ---------------------------------------------------------------------------
# Step 4 — STRING validation AUC
# ---------------------------------------------------------------------------
def step4_string_auc(model, X_tensor, gene_to_idx, n_genes, pairs_by_conf):
    print("\n" + "=" * 70)
    print("STEP 4: STRING VALIDATION AUC")
    print("=" * 70)

    results = {}
    rng = np.random.RandomState(42)

    for threshold in CONFIDENCE_THRESHOLDS:
        pair_set = pairs_by_conf[threshold]
        if len(pair_set) < 20:
            print(f"\n  Confidence >= {threshold}: too few pairs ({len(pair_set)})")
            continue

        # Convert to index pairs
        pos_pairs = []
        for g1, g2 in pair_set:
            pos_pairs.append([gene_to_idx[g1], gene_to_idx[g2]])
        pos_pairs = np.array(pos_pairs, dtype=np.int32)

        # Negative pairs
        n_neg = min(len(pos_pairs) * 5, 200_000)
        pos_set_idx = set()
        for p in pos_pairs:
            pos_set_idx.add((p[0], p[1]))
            pos_set_idx.add((p[1], p[0]))

        neg_list = []
        while len(neg_list) < n_neg:
            i = rng.randint(0, n_genes)
            j = rng.randint(0, n_genes)
            if i != j and (i, j) not in pos_set_idx:
                neg_list.append([i, j])
        neg_pairs = np.array(neg_list, dtype=np.int32)

        # Compute scores
        with torch.no_grad():
            model.eval()
            # PAM scores (both-transformed)
            pred_pos_a = model(X_tensor[pos_pairs[:, 0]])
            pred_pos_b = model(X_tensor[pos_pairs[:, 1]])
            pos_pam = (pred_pos_a * pred_pos_b).sum(dim=-1).cpu().numpy()

            pred_neg_a = model(X_tensor[neg_pairs[:, 0]])
            pred_neg_b = model(X_tensor[neg_pairs[:, 1]])
            neg_pam = (pred_neg_a * pred_neg_b).sum(dim=-1).cpu().numpy()

            # Cosine scores
            pos_cos = (X_tensor[pos_pairs[:, 0]] * X_tensor[pos_pairs[:, 1]]).sum(
                dim=-1).cpu().numpy()
            neg_cos = (X_tensor[neg_pairs[:, 0]] * X_tensor[neg_pairs[:, 1]]).sum(
                dim=-1).cpu().numpy()

        labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(n_neg)])
        all_cos = np.concatenate([pos_cos, neg_cos])
        all_pam = np.concatenate([pos_pam, neg_pam])

        cos_auc = roc_auc_score(labels, all_cos)
        pam_auc = roc_auc_score(labels, all_pam)

        # Cross-boundary subset
        cb_cos_auc = None
        cb_pam_auc = None
        cb_n_pos = 0
        cb_mask_pos = np.abs(pos_cos) < 0.2
        cb_mask_neg = np.abs(neg_cos) < 0.2
        if cb_mask_pos.sum() > 10 and cb_mask_neg.sum() > 10:
            cb_labels = np.concatenate([np.ones(cb_mask_pos.sum()),
                                        np.zeros(cb_mask_neg.sum())])
            cb_cos_auc = float(roc_auc_score(
                cb_labels,
                np.concatenate([pos_cos[cb_mask_pos], neg_cos[cb_mask_neg]])))
            cb_pam_auc = float(roc_auc_score(
                cb_labels,
                np.concatenate([pos_pam[cb_mask_pos], neg_pam[cb_mask_neg]])))
            cb_n_pos = int(cb_mask_pos.sum())

        results[threshold] = {
            "n_pairs": len(pos_pairs),
            "n_neg": n_neg,
            "cosine_auc": float(cos_auc),
            "pam_auc": float(pam_auc),
            "delta": float(pam_auc - cos_auc),
            "pos_cos_mean": float(pos_cos.mean()),
            "neg_cos_mean": float(neg_cos.mean()),
            "cb_cosine_auc": cb_cos_auc,
            "cb_pam_auc": cb_pam_auc,
            "cb_n_pos": cb_n_pos,
        }

        print(f"\n  STRING confidence >= {threshold} ({len(pos_pairs):,} pairs):")
        print(f"    Cosine AUC: {cos_auc:.4f}")
        print(f"    PAM AUC:    {pam_auc:.4f}")
        print(f"    Delta:      {pam_auc - cos_auc:+.4f}")
        if cb_pam_auc is not None:
            print(f"    Cross-boundary: cos={cb_cos_auc:.4f}, pam={cb_pam_auc:.4f} "
                  f"(n_pos={cb_n_pos})")

    return results


# ---------------------------------------------------------------------------
# Step 5 — Save results + plots
# ---------------------------------------------------------------------------
def step5_save_and_plot(string_results, best_N):
    print("\n" + "=" * 70)
    print("STEP 5: SAVE RESULTS")
    print("=" * 70)

    all_results = {
        "best_N": best_N,
        "string_validation": {str(k): v for k, v in string_results.items()},
    }
    results_path = RESULTS_DIR / "05_validate_results.json"
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

    thresholds = sorted(string_results.keys())
    if not thresholds:
        plt.close()
        return

    # Plot 1: AUC at each threshold
    ax = axes[0]
    x = np.arange(len(thresholds))
    w = 0.35
    cos_aucs = [string_results[t]["cosine_auc"] for t in thresholds]
    pam_aucs = [string_results[t]["pam_auc"] for t in thresholds]
    ax.bar(x - w/2, cos_aucs, w, label="Cosine", color="gray", alpha=0.8)
    ax.bar(x + w/2, pam_aucs, w, label="PAM", color="darkorange", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f">={t}\n(n={string_results[t]['n_pairs']})" for t in thresholds])
    ax.set_ylabel("AUC")
    ax.set_xlabel("STRING confidence threshold")
    ax.set_title("STRING Validation AUC")
    ax.legend()
    ax.axhline(0.5, color="black", linestyle="--", alpha=0.3)
    for i, (c, p) in enumerate(zip(cos_aucs, pam_aucs)):
        ax.text(i - w/2, c + 0.005, f"{c:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, p + 0.005, f"{p:.3f}", ha="center", fontsize=8)

    # Plot 2: Delta by threshold
    ax = axes[1]
    deltas = [string_results[t]["delta"] for t in thresholds]
    colors = ["green" if d > 0 else "red" for d in deltas]
    bars = ax.bar([f">={t}" for t in thresholds], deltas, color=colors, alpha=0.8)
    ax.axhline(0, color="black", linestyle="-", alpha=0.5)
    ax.set_ylabel("AUC Delta (PAM - Cosine)")
    ax.set_xlabel("STRING confidence threshold")
    ax.set_title("PAM Improvement on STRING Validation")
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{d:+.3f}", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    plot_path = RESULTS_DIR / "05_validate_plots.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Plots: {plot_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    set_seed(SEED)

    print("DepMap PAM Phase 5: External Validation")
    print("=" * 70)
    print()

    model, X_tensor, genes, gene_to_idx, n_genes, best_N = step1_load()
    step2_download_string()
    pairs_by_conf = step3_parse_string(gene_to_idx)
    string_results = step4_string_auc(model, X_tensor, gene_to_idx, n_genes,
                                       pairs_by_conf)
    step5_save_and_plot(string_results, best_N)

    print(f"\n  Next: python scripts/06_analysis.py")


if __name__ == "__main__":
    main()
