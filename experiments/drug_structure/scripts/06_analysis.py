"""Phase 6: Analysis — degree dependence and poster child examples.

Degree dependence: do drugs with more co-lethal partners get higher PAM scores?
Poster children: high PAM score + low chemical cosine + shared MOA/target.
"""

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import model class
import importlib, sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_train = importlib.import_module("03_train")
AssociationMLP = _train.AssociationMLP

HIDDEN_DIM = 1024
NUM_LAYERS = 4


def load_best_model(embeddings):
    """Load the best model from Phase 3."""
    with open(RESULTS / "03_train.json") as f:
        train_results = json.load(f)
    best_n = max(train_results.keys(), key=lambda k: train_results[k]["auc_pam"])
    best_n = int(best_n)

    checkpoint = torch.load(MODELS / f"pam_{best_n}.pt", map_location=DEVICE,
                            weights_only=True)
    model = AssociationMLP(
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint.get("hidden_dim", HIDDEN_DIM),
        num_layers=checkpoint.get("num_layers", NUM_LAYERS),
    ).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, best_n


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 6: Analysis")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    embeddings = np.load(DATA / "embeddings_raw.npy")
    co_lethality = np.load(DATA / "co_lethality.npy")
    drug_df = pd.read_csv(DATA / "drug_info.csv")
    n_drugs = len(drug_df)
    print(f"  Drugs: {n_drugs}")

    # Load best model and compute scores
    model, best_n = load_best_model(embeddings)
    pairs = np.load(DATA / f"pairs_{best_n}.npy")
    print(f"  Best N: {best_n:,}")

    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = model(emb_tensor).cpu().numpy()
    pam_matrix = z @ z.T

    emb_norm = normalize(embeddings, axis=1)
    cos_matrix = emb_norm @ emb_norm.T

    # ── Degree dependence ──────────────────────────────────────────────────

    print("\n--- Degree dependence ---")

    # Count co-lethal partners per drug (using top-N pairs)
    degree = np.zeros(n_drugs, dtype=int)
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1

    # Mean PAM score per drug (to all other drugs)
    mean_pam = (pam_matrix.sum(axis=1) - np.diag(pam_matrix)) / (n_drugs - 1)
    mean_cos = (cos_matrix.sum(axis=1) - np.diag(cos_matrix)) / (n_drugs - 1)

    # Spearman correlations
    rho_pam, p_pam = stats.spearmanr(degree, mean_pam)
    rho_cos, p_cos = stats.spearmanr(degree, mean_cos)
    print(f"  Degree vs mean PAM: rho={rho_pam:.4f}, p={p_pam:.2e}")
    print(f"  Degree vs mean Cos: rho={rho_cos:.4f}, p={p_cos:.2e}")

    # Bin by quintile
    quintiles = pd.qcut(degree, 5, labels=False, duplicates="drop")
    n_bins = len(np.unique(quintiles))
    print(f"\n  Degree quintile analysis ({n_bins} bins):")
    for q in range(n_bins):
        mask = quintiles == q
        n_q = mask.sum()
        deg_range = f"{degree[mask].min()}-{degree[mask].max()}"
        pam_mean = mean_pam[mask].mean()
        cos_mean = mean_cos[mask].mean()
        print(f"    Q{q}: degree={deg_range}, n={n_q}, "
              f"PAM={pam_mean:.4f}, Cos={cos_mean:.4f}")

    degree_results = {
        "spearman_pam": float(rho_pam),
        "spearman_pam_p": float(p_pam),
        "spearman_cos": float(rho_cos),
        "spearman_cos_p": float(p_cos),
        "n_bins": n_bins,
    }

    # ── Poster child examples ──────────────────────────────────────────────

    print("\n--- Poster child examples ---")
    print("  (High PAM score + low chemical cosine + shared MOA/target)")

    # Build MOA and target lookup
    moa_map = {}
    target_map = {}
    for idx, row in drug_df.iterrows():
        if pd.notna(row.get("moa")) and row["moa"] != "NA":
            moa_map[idx] = set(str(row["moa"]).split("|"))
        if pd.notna(row.get("target")) and row["target"] != "NA":
            target_map[idx] = set(str(row["target"]).split(", "))

    # Find poster children: top PAM, low cosine, shared annotation
    triu_idx = np.triu_indices(n_drugs, k=1)
    pam_values = pam_matrix[triu_idx]
    cos_values = cos_matrix[triu_idx]

    # Cross-boundary pairs with high PAM
    cross_boundary = np.abs(cos_values) < 0.2
    cb_pam_order = np.argsort(-pam_values)

    poster_children = []
    seen_pairs = set()

    for rank_idx in cb_pam_order:
        if not cross_boundary[rank_idx]:
            continue
        i = triu_idx[0][rank_idx]
        j = triu_idx[1][rank_idx]

        # Check shared MOA
        shared_moa = ""
        if i in moa_map and j in moa_map:
            overlap = moa_map[i] & moa_map[j]
            if overlap:
                shared_moa = "|".join(sorted(overlap))

        # Check shared target
        shared_target = ""
        if i in target_map and j in target_map:
            overlap = target_map[i] & target_map[j]
            if overlap:
                shared_target = ", ".join(sorted(overlap))

        # Prioritize pairs with shared annotations
        has_shared = bool(shared_moa or shared_target)

        pair_key = (min(i, j), max(i, j))
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)

        pc = {
            "drug_a": str(drug_df.iloc[i].get("name", drug_df.iloc[i]["broad_id"])),
            "drug_b": str(drug_df.iloc[j].get("name", drug_df.iloc[j]["broad_id"])),
            "broad_id_a": str(drug_df.iloc[i]["broad_id"]),
            "broad_id_b": str(drug_df.iloc[j]["broad_id"]),
            "pam_score": float(pam_values[rank_idx]),
            "cosine": float(cos_values[rank_idx]),
            "co_lethality": float(co_lethality[i, j]),
            "shared_moa": shared_moa,
            "shared_target": shared_target,
            "has_shared_annotation": has_shared,
        }
        poster_children.append(pc)

        if len(poster_children) >= 200:
            break

    # Separate into annotated and unannotated
    annotated = [p for p in poster_children if p["has_shared_annotation"]]
    unannotated = [p for p in poster_children if not p["has_shared_annotation"]]

    print(f"\n  Top cross-boundary pairs (|cos| < 0.2, ranked by PAM):")
    print(f"    Total found: {len(poster_children)}")
    print(f"    With shared annotation: {len(annotated)}")

    print(f"\n  Top 20 poster children WITH shared annotation:")
    for i, pc in enumerate(annotated[:20]):
        print(f"    {i+1:2d}. {pc['drug_a']:25s} <-> {pc['drug_b']:25s}  "
              f"PAM={pc['pam_score']:.3f}  cos={pc['cosine']:.3f}  "
              f"MOA={pc['shared_moa'][:30] if pc['shared_moa'] else '-':30s}  "
              f"target={pc['shared_target'][:25] if pc['shared_target'] else '-'}")

    print(f"\n  Top 10 poster children WITHOUT shared annotation (novel predictions):")
    for i, pc in enumerate(unannotated[:10]):
        moa_a = str(drug_df.iloc[drug_df[drug_df["broad_id"] == pc["broad_id_a"]].index[0]].get("moa", "NA"))[:20] if len(drug_df[drug_df["broad_id"] == pc["broad_id_a"]]) > 0 else "?"
        moa_b = str(drug_df.iloc[drug_df[drug_df["broad_id"] == pc["broad_id_b"]].index[0]].get("moa", "NA"))[:20] if len(drug_df[drug_df["broad_id"] == pc["broad_id_b"]]) > 0 else "?"
        print(f"    {i+1:2d}. {pc['drug_a']:25s} ({moa_a}) <-> "
              f"{pc['drug_b']:25s} ({moa_b})  "
              f"PAM={pc['pam_score']:.3f}  cos={pc['cosine']:.3f}")

    # ── Figures ────────────────────────────────────────────────────────────

    print("\n--- Generating figures ---")

    # Fig 1: Degree dependence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(degree, mean_pam, s=3, alpha=0.3, color="steelblue", rasterized=True)
    axes[0].set_xlabel("Co-lethal Degree (# partners)")
    axes[0].set_ylabel("Mean PAM Score")
    axes[0].set_title(f"Degree vs PAM (ρ={rho_pam:.3f})")

    axes[1].scatter(degree, mean_cos, s=3, alpha=0.3, color="orange", rasterized=True)
    axes[1].set_xlabel("Co-lethal Degree (# partners)")
    axes[1].set_ylabel("Mean Cosine")
    axes[1].set_title(f"Degree vs Cosine (ρ={rho_cos:.3f})")

    plt.tight_layout()
    plt.savefig(FIGURES / "06_degree_dependence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/06_degree_dependence.png")

    # Fig 2: PAM vs cosine for all pairs (with poster children highlighted)
    fig, ax = plt.subplots(figsize=(8, 8))

    # Sample background pairs
    n_sample = min(200_000, len(pam_values))
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(len(pam_values), n_sample, replace=False)
    ax.scatter(cos_values[sample_idx], pam_values[sample_idx],
               s=0.5, alpha=0.03, color="gray", rasterized=True, label="All pairs")

    # Highlight poster children with shared annotations
    if annotated:
        pc_cos = [p["cosine"] for p in annotated[:50]]
        pc_pam = [p["pam_score"] for p in annotated[:50]]
        ax.scatter(pc_cos, pc_pam, s=30, color="red", alpha=0.7, zorder=5,
                   edgecolors="darkred", linewidths=0.5,
                   label=f"Poster children (n={len(annotated[:50])})")

    ax.axvline(0.2, color="red", ls="--", alpha=0.3)
    ax.axvline(-0.2, color="red", ls="--", alpha=0.3)
    ax.set_xlabel("Morgan FP Cosine Similarity")
    ax.set_ylabel("PAM Score")
    ax.set_title("PAM vs Chemical Similarity")
    ax.legend(markerscale=2)

    plt.tight_layout()
    plt.savefig(FIGURES / "06_poster_children.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/06_poster_children.png")

    # Fig 3: PAM score distribution by quintile
    fig, ax = plt.subplots(figsize=(8, 5))
    bp_data = []
    bp_labels = []
    for q in range(n_bins):
        mask = quintiles == q
        bp_data.append(mean_pam[mask])
        bp_labels.append(f"Q{q}\n({degree[mask].min()}-{degree[mask].max()})")

    ax.boxplot(bp_data, labels=bp_labels, showfliers=False)
    ax.set_xlabel("Degree Quintile (co-lethal partners)")
    ax.set_ylabel("Mean PAM Score")
    ax.set_title("PAM Score by Degree Quintile")

    plt.tight_layout()
    plt.savefig(FIGURES / "06_degree_quintiles.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/06_degree_quintiles.png")

    # ── Save results ───────────────────────────────────────────────────────

    results = {
        "degree_dependence": degree_results,
        "poster_children_annotated": annotated[:50],
        "poster_children_novel": unannotated[:50],
        "n_poster_children_total": len(poster_children),
        "n_poster_children_annotated": len(annotated),
        "best_n": best_n,
    }

    with open(RESULTS / "06_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: results/06_analysis.json")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 6 SUMMARY")
    print("=" * 60)

    print(f"\n  Degree dependence:")
    print(f"    PAM vs degree: ρ={rho_pam:.4f} (p={p_pam:.2e})")
    print(f"    Cosine vs degree: ρ={rho_cos:.4f} (p={p_cos:.2e})")

    print(f"\n  Poster children:")
    print(f"    Cross-boundary pairs found: {len(poster_children)}")
    print(f"    With shared MOA/target: {len(annotated)}")

    if annotated:
        best = annotated[0]
        print(f"\n  Best poster child:")
        print(f"    {best['drug_a']} <-> {best['drug_b']}")
        print(f"    PAM={best['pam_score']:.3f}, cosine={best['cosine']:.3f}")
        print(f"    Shared MOA: {best['shared_moa'] or 'none'}")
        print(f"    Shared target: {best['shared_target'] or 'none'}")

    print("\nPhase 6 complete.")
    print("\n" + "=" * 60)
    print("ALL PHASES COMPLETE")
    print("=" * 60)
