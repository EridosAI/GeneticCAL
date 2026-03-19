"""Phase 4: Ablations at best configuration.

1. Shuffled pairs — must destroy signal (if not, model learned embedding geometry)
2. Similar positives — top-N highest cosine pairs (must not match co-lethality)
3. Inductive 70/30 — train on 70% of drugs, evaluate on held-out 30%
"""

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset

# Import model and training from Phase 3
import importlib
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
_train = importlib.import_module("03_train")
AssociationMLP = _train.AssociationMLP
PairDataset = _train.PairDataset
clip_loss = _train.clip_loss
evaluate_model = _train.evaluate_model
train_inbatch = _train.train_inbatch
train_random_neg = _train.train_random_neg

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HIDDEN_DIM = 1024
NUM_LAYERS = 4
N_EVAL_NEG = 50_000


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Phase 4: Ablations")
    print("=" * 60)

    # Load data
    print("\n--- Loading data ---")
    embeddings = np.load(DATA / "embeddings_raw.npy")
    co_lethality = np.load(DATA / "co_lethality.npy")
    n_drugs = len(embeddings)
    print(f"  Drugs: {n_drugs}, Embedding dim: {embeddings.shape[1]}")

    # Determine best config from Phase 3
    with open(RESULTS / "03_train.json") as f:
        train_results = json.load(f)

    best_config = train_results.get("best_config", {})
    best_key = best_config.get("key")
    best_n = best_config.get("n_pairs")
    best_neg_type = best_config.get("neg_type", "inbatch")

    if best_key is None:
        # Fallback: find best across all results
        score_keys = [k for k in train_results if k != "best_config"]
        best_key = max(score_keys, key=lambda k: train_results[k].get("auc_pam", 0))
        best_n = train_results[best_key].get("n_pairs", int(best_key.split("_")[-1]))
        best_neg_type = train_results[best_key].get("neg_type", "inbatch")

    print(f"  Best config: {best_key} (N={best_n:,}, neg={best_neg_type})")

    # Choose training function based on best neg type
    train_fn = train_random_neg if best_neg_type == "random" else train_inbatch

    pairs = np.load(DATA / f"pairs_{best_n}.npy")

    # Cosine similarity for all pairs
    emb_norm = normalize(embeddings, axis=1)
    cos_matrix = emb_norm @ emb_norm.T

    results_all = {}
    rng = np.random.default_rng(42)

    # ── Ablation 1: Shuffled pairs ─────────────────────────────────────────

    print(f"\n{'─' * 60}")
    print("Ablation 1: Shuffled pairs")
    print(f"{'─' * 60}")
    shuffled_pairs = pairs.copy()
    shuffled_pairs[:, 1] = rng.permutation(shuffled_pairs[:, 1])

    model_shuf, _ = train_fn(shuffled_pairs, embeddings, tag="Shuffled")
    result_shuf = evaluate_model(model_shuf, embeddings, pairs, co_lethality,
                                 tag="Shuffled: ")
    results_all["shuffled"] = result_shuf

    # ── Ablation 2: Similar positives ──────────────────────────────────────

    print(f"\n{'─' * 60}")
    print("Ablation 2: Similar positives (top-N highest cosine pairs)")
    print(f"{'─' * 60}")
    triu_idx = np.triu_indices(n_drugs, k=1)
    cos_values = cos_matrix[triu_idx]
    cos_rank = np.argsort(-cos_values)
    sim_top_idx = cos_rank[:best_n]
    similar_pairs = np.stack([triu_idx[0][sim_top_idx], triu_idx[1][sim_top_idx]], axis=1)

    model_sim, _ = train_fn(similar_pairs, embeddings, tag="Similar")
    result_sim = evaluate_model(model_sim, embeddings, pairs, co_lethality,
                                tag="Similar: ")
    results_all["similar_positives"] = result_sim

    # ── Ablation 3: Inductive 70/30 ───────────────────────────────────────

    print(f"\n{'─' * 60}")
    print("Ablation 3: Inductive 70/30 (train on 70% drugs, eval on held-out 30%)")
    print(f"{'─' * 60}")

    drug_perm = rng.permutation(n_drugs)
    n_train = int(0.7 * n_drugs)
    train_drugs = set(drug_perm[:n_train].tolist())
    test_drugs = set(drug_perm[n_train:].tolist())
    print(f"  Train drugs: {len(train_drugs)}, Test drugs: {len(test_drugs)}")

    train_pairs = np.array([p for p in pairs if p[0] in train_drugs and p[1] in train_drugs])
    test_pairs_pos = np.array([p for p in pairs if p[0] in test_drugs or p[1] in test_drugs])
    print(f"  Train pairs: {len(train_pairs)}, Test positive pairs: {len(test_pairs_pos)}")

    if len(train_pairs) < 1000:
        print("  ⚠ Too few train pairs for inductive split. Skipping.")
        results_all["inductive"] = {"skipped": True, "reason": "too_few_train_pairs"}
    else:
        model_ind, _ = train_fn(train_pairs, embeddings, tag="Inductive")

        if len(test_pairs_pos) > 50:
            result_ind = evaluate_model(model_ind, embeddings, test_pairs_pos,
                                        co_lethality, tag="Inductive: ")
            results_all["inductive"] = result_ind
        else:
            print("  ⚠ Too few test pairs for evaluation.")
            results_all["inductive"] = {"skipped": True, "reason": "too_few_test_pairs"}

    # ── Reference: Phase 3 best model ──────────────────────────────────────

    results_all["reference"] = train_results[best_key]
    results_all["best_n"] = best_n
    results_all["best_neg_type"] = best_neg_type

    # ── Figures ────────────────────────────────────────────────────────────

    print("\n--- Generating figures ---")

    ablation_names = ["reference", "shuffled", "similar_positives"]
    if not results_all.get("inductive", {}).get("skipped"):
        ablation_names.append("inductive")

    display_names = {
        "reference": f"Co-lethal\n({best_neg_type})",
        "shuffled": "Shuffled",
        "similar_positives": "Similar\nPositives",
        "inductive": "Inductive\n70/30",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Overall AUC
    aucs = [results_all[name]["auc_pam"] for name in ablation_names]
    cos_aucs = [results_all[name]["auc_cos"] for name in ablation_names]
    x = range(len(ablation_names))
    width = 0.35

    axes[0].bar([xi - width / 2 for xi in x], aucs, width,
                label="PAM", color="steelblue")
    axes[0].bar([xi + width / 2 for xi in x], cos_aucs, width,
                label="Cosine", color="orange")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels([display_names[n] for n in ablation_names], fontsize=9)
    axes[0].set_ylabel("AUC")
    axes[0].set_title("Overall AUC: Ablation Comparison")
    axes[0].legend()
    axes[0].axhline(0.5, color="gray", ls="--", alpha=0.3)

    # Cross-boundary AUC
    cb_aucs = [results_all[name].get("auc_pam_cb") for name in ablation_names]
    cb_cos = [results_all[name].get("auc_cos_cb") for name in ablation_names]
    valid_idx = [i for i, v in enumerate(cb_aucs) if v is not None]

    if valid_idx:
        valid_names = [ablation_names[i] for i in valid_idx]
        valid_pam = [cb_aucs[i] for i in valid_idx]
        valid_cos = [cb_cos[i] for i in valid_idx]
        x2 = range(len(valid_names))
        axes[1].bar([xi - width / 2 for xi in x2], valid_pam, width,
                    label="PAM", color="steelblue")
        axes[1].bar([xi + width / 2 for xi in x2], valid_cos, width,
                    label="Cosine", color="orange")
        axes[1].set_xticks(list(x2))
        axes[1].set_xticklabels([display_names[n] for n in valid_names], fontsize=9)
        axes[1].set_ylabel("AUC")
        axes[1].set_title("Cross-boundary AUC (|cos| < 0.2)")
        axes[1].legend()
        axes[1].axhline(0.5, color="gray", ls="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES / "04_ablations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: figures/04_ablations.png")

    # ── Save results ───────────────────────────────────────────────────────

    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(RESULTS / "04_ablations.json", "w") as f:
        json.dump(clean_for_json(results_all), f, indent=2)
    print(f"  Saved: results/04_ablations.json")

    # ── Summary ────────────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("PHASE 4 SUMMARY")
    print("=" * 60)

    for name in ablation_names:
        r = results_all[name]
        if isinstance(r, dict) and "auc_pam" in r:
            cb_str = ""
            if r.get("auc_pam_cb") is not None:
                cb_str = f", CB={r['auc_pam_cb']:.4f}"
            print(f"  {display_names[name].replace(chr(10), ' ')}: "
                  f"PAM={r['auc_pam']:.4f}, Cos={r['auc_cos']:.4f}{cb_str}")

    # Kill criteria
    ref = results_all["reference"]
    shuf = results_all["shuffled"]
    sim = results_all["similar_positives"]

    if shuf["auc_pam"] >= ref["auc_pam"] * 0.95:
        print("\n  ⚠ KILL: Shuffled ablation doesn't destroy signal.")
        print("  Model may have learned L1000 embedding geometry, not association.")

    if sim["auc_pam"] >= ref["auc_pam"] * 0.95:
        print("\n  ⚠ KILL: Similar-positives matches co-lethality performance.")
        print("  Transcriptional similarity already captures co-lethality.")

    print("\nPhase 4 complete.")
