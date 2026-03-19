"""Phase 5: External validation against MOA, target, and drug category.

PAM was trained with NO knowledge of drug targets or mechanisms.
Tests whether PAM scores recover known functional relationships.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import model class
import importlib
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


def compute_pam_scores(model, embeddings):
    """Compute all-pairs PAM scores (both-transformed)."""
    emb_tensor = torch.tensor(embeddings, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        z = model(emb_tensor).cpu().numpy()
    return z @ z.T


def validate_annotation(pam_matrix, cos_matrix, drug_df, annotation_col,
                        label, max_neg=100_000):
    """Validate PAM vs cosine for same-annotation vs different-annotation pairs."""
    # Get valid annotations (not NA/empty)
    valid_mask = (
        drug_df[annotation_col].notna() &
        (drug_df[annotation_col] != "NA") &
        (drug_df[annotation_col] != "")
    )
    valid_idx = drug_df[valid_mask].index.tolist()
    annotations = drug_df.loc[valid_idx, annotation_col].tolist()

    print(f"\n  --- {label} ({annotation_col}) ---")
    print(f"  Drugs with annotation: {len(valid_idx)}")

    # Count unique annotations
    ann_counts = Counter(annotations)
    print(f"  Unique annotations: {len(ann_counts)}")
    print(f"  Top-10: {ann_counts.most_common(10)}")

    # Build same-annotation pairs (positives)
    ann_to_drugs = defaultdict(list)
    for idx, ann in zip(valid_idx, annotations):
        # Some drugs have multiple annotations separated by |
        for a in str(ann).split("|"):
            a = a.strip()
            if a and a != "NA":
                ann_to_drugs[a].append(idx)

    # Only use annotations with ≥2 drugs
    same_pairs = []
    for ann, drugs in ann_to_drugs.items():
        if len(drugs) >= 2:
            for ii in range(len(drugs)):
                for jj in range(ii + 1, len(drugs)):
                    same_pairs.append((drugs[ii], drugs[jj]))

    if len(same_pairs) < 50:
        print(f"  ⚠ Too few same-{label} pairs ({len(same_pairs)}). Skipping.")
        return None

    same_pairs = np.array(same_pairs)
    print(f"  Same-{label} pairs: {len(same_pairs):,}")

    # Different-annotation pairs (negatives) — sample
    rng = np.random.default_rng(42)
    same_set = set(map(tuple, same_pairs)) | set(map(tuple, same_pairs[:, ::-1]))
    diff_pairs = []
    n_neg = min(max_neg, len(same_pairs) * 2)
    while len(diff_pairs) < n_neg:
        i = rng.choice(valid_idx)
        j = rng.choice(valid_idx)
        if i != j and (i, j) not in same_set:
            diff_pairs.append((i, j))
    diff_pairs = np.array(diff_pairs)
    print(f"  Different-{label} pairs (sampled): {len(diff_pairs):,}")

    # Scores
    same_pam = np.array([pam_matrix[i, j] for i, j in same_pairs])
    same_cos = np.array([cos_matrix[i, j] for i, j in same_pairs])
    diff_pam = np.array([pam_matrix[i, j] for i, j in diff_pairs])
    diff_cos = np.array([cos_matrix[i, j] for i, j in diff_pairs])

    # AUC
    labels = np.concatenate([np.ones(len(same_pairs)), np.zeros(len(diff_pairs))])
    auc_pam = roc_auc_score(labels, np.concatenate([same_pam, diff_pam]))
    auc_cos = roc_auc_score(labels, np.concatenate([same_cos, diff_cos]))

    print(f"  AUC: PAM={auc_pam:.4f}, Cosine={auc_cos:.4f}, "
          f"Δ={auc_pam - auc_cos:+.4f}")

    # Cross-boundary subset
    all_cos = np.concatenate([same_cos, diff_cos])
    cb_mask = np.abs(all_cos) < 0.2
    if cb_mask.sum() > 100:
        cb_labels = labels[cb_mask]
        if cb_labels.sum() > 20 and (1 - cb_labels).sum() > 20:
            all_pam = np.concatenate([same_pam, diff_pam])
            auc_pam_cb = roc_auc_score(cb_labels, all_pam[cb_mask])
            auc_cos_cb = roc_auc_score(cb_labels, all_cos[cb_mask])
            print(f"  Cross-boundary AUC: PAM={auc_pam_cb:.4f}, "
                  f"Cosine={auc_cos_cb:.4f}")
        else:
            auc_pam_cb = None
            auc_cos_cb = None
    else:
        auc_pam_cb = None
        auc_cos_cb = None

    # Score distributions
    print(f"  Same-{label} PAM: mean={same_pam.mean():.4f}, std={same_pam.std():.4f}")
    print(f"  Diff-{label} PAM: mean={diff_pam.mean():.4f}, std={diff_pam.std():.4f}")
    print(f"  Same-{label} cos: mean={same_cos.mean():.4f}, std={same_cos.std():.4f}")
    print(f"  Diff-{label} cos: mean={diff_cos.mean():.4f}, std={diff_cos.std():.4f}")

    return {
        "label": label,
        "annotation_col": annotation_col,
        "n_drugs_annotated": len(valid_idx),
        "n_unique_annotations": len(ann_counts),
        "n_same_pairs": len(same_pairs),
        "n_diff_pairs": len(diff_pairs),
        "auc_pam": float(auc_pam),
        "auc_cos": float(auc_cos),
        "auc_delta": float(auc_pam - auc_cos),
        "auc_pam_cb": float(auc_pam_cb) if auc_pam_cb is not None else None,
        "auc_cos_cb": float(auc_cos_cb) if auc_cos_cb is not None else None,
        "same_pam_mean": float(same_pam.mean()),
        "diff_pam_mean": float(diff_pam.mean()),
        "same_cos_mean": float(same_cos.mean()),
        "diff_cos_mean": float(diff_cos.mean()),
    }


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 5: Validation")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    embeddings = np.load(DATA / "embeddings_raw.npy")
    drug_df = pd.read_csv(DATA / "drug_info.csv")
    print(f"  Drugs: {len(drug_df)}, Embedding dim: {embeddings.shape[1]}")

    # Load best model
    model, best_n = load_best_model(embeddings)
    print(f"  Best model: N={best_n:,}")

    # Compute score matrices
    print("\n--- Computing score matrices ---")
    pam_matrix = compute_pam_scores(model, embeddings)

    emb_norm = normalize(embeddings, axis=1)
    cos_matrix = emb_norm @ emb_norm.T

    # ── 5a. MOA validation ─────────────────────────────────────────────────

    results_all = {}

    result_moa = validate_annotation(pam_matrix, cos_matrix, drug_df,
                                     "moa", "MOA")
    if result_moa is not None:
        results_all["moa"] = result_moa

    # ── 5b. Target validation ──────────────────────────────────────────────

    result_target = validate_annotation(pam_matrix, cos_matrix, drug_df,
                                        "target", "Target")
    if result_target is not None:
        results_all["target"] = result_target

    # ── 5c. Disease area validation ────────────────────────────────────────

    if "disease.area" in drug_df.columns:
        result_disease = validate_annotation(pam_matrix, cos_matrix, drug_df,
                                             "disease.area", "Disease Area")
        if result_disease is not None:
            results_all["disease_area"] = result_disease

    # ── 5d. Phase validation (drug development phase) ──────────────────────

    if "phase" in drug_df.columns:
        result_phase = validate_annotation(pam_matrix, cos_matrix, drug_df,
                                           "phase", "Phase")
        if result_phase is not None:
            results_all["phase"] = result_phase

    # ── Figures ────────────────────────────────────────────────────────────

    print("\n--- Generating figures ---")

    valid_results = {k: v for k, v in results_all.items()
                     if isinstance(v, dict) and "auc_pam" in v}

    if valid_results:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Overall AUC comparison
        names = list(valid_results.keys())
        aucs_pam = [valid_results[n]["auc_pam"] for n in names]
        aucs_cos = [valid_results[n]["auc_cos"] for n in names]

        x = range(len(names))
        width = 0.35
        axes[0].bar([xi - width / 2 for xi in x], aucs_pam, width,
                    label="PAM", color="steelblue")
        axes[0].bar([xi + width / 2 for xi in x], aucs_cos, width,
                    label="Cosine", color="orange")
        axes[0].set_xticks(list(x))
        axes[0].set_xticklabels([n.replace("_", " ").title() for n in names])
        axes[0].set_ylabel("AUC")
        axes[0].set_title("Validation AUC: Same vs Different Annotation")
        axes[0].legend()
        axes[0].axhline(0.5, color="gray", ls="--", alpha=0.3)

        # Score distributions for MOA (if available)
        if "moa" in valid_results:
            # Re-compute for visualization
            valid_mask = (
                drug_df["moa"].notna() &
                (drug_df["moa"] != "NA") &
                (drug_df["moa"] != "")
            )
            valid_idx = drug_df[valid_mask].index.tolist()
            annotations = drug_df.loc[valid_idx, "moa"].tolist()

            ann_to_drugs = defaultdict(list)
            for idx, ann in zip(valid_idx, annotations):
                for a in str(ann).split("|"):
                    a = a.strip()
                    if a and a != "NA":
                        ann_to_drugs[a].append(idx)

            same_pam_scores = []
            diff_pam_scores = []
            rng = np.random.default_rng(42)

            for ann, drugs in ann_to_drugs.items():
                if len(drugs) >= 2:
                    for ii in range(len(drugs)):
                        for jj in range(ii + 1, len(drugs)):
                            same_pam_scores.append(pam_matrix[drugs[ii], drugs[jj]])

            # Sample diff pairs
            for _ in range(min(len(same_pam_scores) * 2, 50_000)):
                i = rng.choice(valid_idx)
                j = rng.choice(valid_idx)
                if i != j:
                    diff_pam_scores.append(pam_matrix[i, j])

            axes[1].hist(same_pam_scores, bins=100, density=True, alpha=0.6,
                         color="steelblue", label="Same MOA")
            axes[1].hist(diff_pam_scores, bins=100, density=True, alpha=0.6,
                         color="orange", label="Different MOA")
            axes[1].set_xlabel("PAM Score")
            axes[1].set_ylabel("Density")
            axes[1].set_title("PAM Score Distribution: Same vs Different MOA")
            axes[1].legend()

        plt.tight_layout()
        plt.savefig(FIGURES / "05_validation.png", dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: figures/05_validation.png")

    # ── Save results ───────────────────────────────────────────────────────

    with open(RESULTS / "05_validation.json", "w") as f:
        json.dump(results_all, f, indent=2)
    print(f"  Saved: results/05_validation.json")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 5 SUMMARY")
    print("=" * 60)

    for name, r in valid_results.items():
        print(f"\n  {r['label']}:")
        print(f"    Annotated drugs: {r['n_drugs_annotated']}, "
              f"Unique annotations: {r['n_unique_annotations']}")
        print(f"    Same-{r['label']} pairs: {r['n_same_pairs']:,}")
        print(f"    AUC: PAM={r['auc_pam']:.4f}, Cosine={r['auc_cos']:.4f}, "
              f"Δ={r['auc_delta']:+.4f}")
        if r.get("auc_pam_cb") is not None:
            print(f"    Cross-boundary AUC: PAM={r['auc_pam_cb']:.4f}, "
                  f"Cosine={r['auc_cos_cb']:.4f}")

    # Highlight results
    if "moa" in valid_results:
        moa = valid_results["moa"]
        if moa["auc_delta"] > 0.02:
            print(f"\n  ✓ PAM recovers MOA better than structure alone (+{moa['auc_delta']:.3f})")
        elif moa["auc_delta"] < -0.02:
            print(f"\n  ~ Cosine better than PAM for MOA ({moa['auc_delta']:+.3f})")
        else:
            print(f"\n  ~ PAM ≈ Cosine for MOA (Δ={moa['auc_delta']:+.3f})")

    print("\nPhase 5 complete.")
