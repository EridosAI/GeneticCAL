#!/usr/bin/env python3
"""
Experiment 3: Experimental-Channel-Only STRING Filtering
=========================================================
Filter STRING pairs by the 'experimental' channel score instead of combined_score.
Train and evaluate at experimental >= 700 and experimental >= 400.
"""

import json, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

BASE = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE / "Gene AAR" / "data"
RESULTS_DIR = BASE / "Gene AAR" / "results"
OUT_DIR = BASE / "Paper" / "strengthening_results"

BULK_FILE = DATA_DIR / "K562_essential_normalized_bulk_01.h5ad"
STRING_DETAILED = DATA_DIR / "9606.protein.links.detailed.v12.0.txt.gz"
STRING_INFO = DATA_DIR / "9606.protein.info.v12.0.txt.gz"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED = 42
PCA_DIM = 50
HIDDEN_DIM = 1024
N_LAYERS = 4
BATCH_SIZE = 512
LR = 3e-4
EPOCHS = 100
TEMPERATURE = 0.05


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def contrastive_loss(emb_a, emb_b, temperature=TEMPERATURE):
    logits = emb_a @ emb_b.T / temperature
    labels = torch.arange(len(emb_a), device=emb_a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def train_model(X_norm_tensor, pairs, epochs=EPOCHS):
    set_seed(SEED)
    model = AssociationMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_pairs = len(pairs)

    for epoch in range(epochs):
        model.train()
        perm = np.random.permutation(n_pairs)
        for start in range(0, n_pairs, BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            if len(batch_idx) < 8:
                continue
            emb_a = X_norm_tensor[pairs[batch_idx, 0]]
            emb_b = X_norm_tensor[pairs[batch_idx, 1]]
            pred_a = model(emb_a)
            loss = contrastive_loss(pred_a, emb_b)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, n_pairs, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n_pairs)
            emb_a = X_norm_tensor[pairs[start:end, 0]]
            emb_b = X_norm_tensor[pairs[start:end, 1]]
            pred_a = model(emb_a)
            sims = pred_a @ emb_b.T
            correct += (sims.argmax(dim=1) == torch.arange(end - start, device=DEVICE)).sum().item()
            total += end - start

    return model, correct / total


@torch.no_grad()
def compute_scores(model, X_norm_tensor, idx_a, idx_b):
    model.eval()
    emb_a = X_norm_tensor[idx_a]
    emb_b = X_norm_tensor[idx_b]
    pred_a = model(emb_a)
    pred_b = model(emb_b)
    assoc = 0.5 * ((pred_a * emb_b).sum(dim=-1) + (pred_b * emb_a).sum(dim=-1))
    cosine = (emb_a * emb_b).sum(dim=-1)
    return assoc.cpu().numpy(), cosine.cpu().numpy()


def evaluate(model, X_norm_tensor, pos_pairs, n_pert):
    pos_assoc, pos_cosine = compute_scores(model, X_norm_tensor,
                                           pos_pairs[:, 0], pos_pairs[:, 1])
    n_neg = min(len(pos_pairs) * 5, 50000)
    pos_set = set()
    for i in range(len(pos_pairs)):
        pos_set.add((int(pos_pairs[i, 0]), int(pos_pairs[i, 1])))
        pos_set.add((int(pos_pairs[i, 1]), int(pos_pairs[i, 0])))

    rng = np.random.RandomState(42)
    neg_list = []
    while len(neg_list) < n_neg:
        i, j = rng.randint(0, n_pert), rng.randint(0, n_pert)
        if i != j and (i, j) not in pos_set:
            neg_list.append([i, j])
    neg_pairs = np.array(neg_list, dtype=np.int32)

    neg_assoc, neg_cosine = compute_scores(model, X_norm_tensor,
                                           neg_pairs[:, 0], neg_pairs[:, 1])

    labels = np.concatenate([np.ones(len(pos_pairs)), np.zeros(len(neg_pairs))])
    all_cos = np.concatenate([pos_cosine, neg_cosine])
    all_assoc = np.concatenate([pos_assoc, neg_assoc])

    res = {
        'cosine_auc': float(roc_auc_score(labels, all_cos)),
        'assoc_only_auc': float(roc_auc_score(labels, all_assoc)),
    }
    for lam in [0.9]:
        blended = (1 - lam) * all_cos + lam * all_assoc
        res[f'auc_lam_{lam}'] = float(roc_auc_score(labels, blended))

    cb_pos = np.abs(pos_cosine) < 0.2
    cb_neg = np.abs(neg_cosine) < 0.2
    if cb_pos.sum() > 10 and cb_neg.sum() > 10:
        cb_labels = np.concatenate([np.ones(cb_pos.sum()), np.zeros(cb_neg.sum())])
        cb_cos = np.concatenate([pos_cosine[cb_pos], neg_cosine[cb_neg]])
        cb_assoc = np.concatenate([pos_assoc[cb_pos], neg_assoc[cb_neg]])
        res['cb_cosine_auc'] = float(roc_auc_score(cb_labels, cb_cos))
        res['cb_assoc_auc'] = float(roc_auc_score(cb_labels, cb_assoc))
        for lam in [0.9]:
            blended = (1 - lam) * cb_cos + lam * cb_assoc
            res[f'cb_auc_lam_{lam}'] = float(roc_auc_score(cb_labels, blended))
        res['cb_n_pos'] = int(cb_pos.sum())

    return res


def main():
    t0 = time.time()
    print("Experiment 3: Experimental-Channel-Only STRING Filtering")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    # Load expression data
    import anndata as ad
    adata = ad.read_h5ad(BULK_FILE)
    X_raw = adata.X.toarray() if hasattr(adata.X, 'toarray') else np.array(adata.X)
    n_pert = X_raw.shape[0]

    set_seed(SEED)
    pca = PCA(n_components=PCA_DIM)
    X_pca = pca.fit_transform(X_raw)
    norms = np.linalg.norm(X_pca, axis=1, keepdims=True)
    X_norm = X_pca / (norms + 1e-8)
    X_norm_tensor = torch.tensor(X_norm, dtype=torch.float32, device=DEVICE)

    # Gene names and name-to-index mapping
    gene_names = []
    for idx_str in adata.obs.index:
        parts = str(idx_str).split('_')
        gene_names.append(parts[1].upper() if len(parts) >= 2 else str(idx_str))
    gene_to_idx = {g: i for i, g in enumerate(gene_names)}
    print(f"  {n_pert} perturbations, PCA-{PCA_DIM}")

    # Load ENSP -> gene name mapping
    print("\nLoading STRING protein info...")
    info_df = pd.read_csv(STRING_INFO, sep='\t')
    ensp_to_gene = {}
    for _, row in info_df.iterrows():
        ensp_to_gene[row['#string_protein_id']] = str(row['preferred_name']).upper()

    # Load detailed STRING and filter by experimental channel
    print("Loading detailed STRING links and filtering by experimental channel...")
    exp_pairs_400 = []
    exp_pairs_700 = []
    channel_stats = {'textmining_gt0': 0, 'coexpression_gt0': 0, 'total_exp700': 0}

    # Also load combined_score >= 900 pairs for overlap analysis
    combined_900_set = set()
    string_high_df = pd.read_csv(DATA_DIR / "string_pairs_high.csv")
    for _, row in string_high_df.iterrows():
        combined_900_set.add((row['gene1'], row['gene2']))
        combined_900_set.add((row['gene2'], row['gene1']))

    chunk_iter = pd.read_csv(STRING_DETAILED, sep=' ', chunksize=500000)
    for chunk in chunk_iter:
        for _, row in chunk.iterrows():
            exp_score = row.get('experimental', 0)
            if exp_score < 400:
                continue
            g1 = ensp_to_gene.get(row['protein1'], '')
            g2 = ensp_to_gene.get(row['protein2'], '')
            if g1 not in gene_to_idx or g2 not in gene_to_idx:
                continue
            if g1 == g2:
                continue
            idx1, idx2 = gene_to_idx[g1], gene_to_idx[g2]
            # Canonicalize
            if idx1 > idx2:
                idx1, idx2 = idx2, idx1
                g1, g2 = g2, g1

            pair_info = {
                'g1': g1, 'g2': g2, 'idx1': idx1, 'idx2': idx2,
                'experimental': int(exp_score),
                'textmining': int(row.get('textmining', 0)),
                'coexpression': int(row.get('coexpression', 0)),
                'combined_score': int(row.get('combined_score', 0)),
            }
            exp_pairs_400.append(pair_info)
            if exp_score >= 700:
                exp_pairs_700.append(pair_info)

    # Deduplicate
    seen_400 = set()
    deduped_400 = []
    for p in exp_pairs_400:
        key = (p['idx1'], p['idx2'])
        if key not in seen_400:
            seen_400.add(key)
            deduped_400.append(p)
    exp_pairs_400 = deduped_400

    seen_700 = set()
    deduped_700 = []
    for p in exp_pairs_700:
        key = (p['idx1'], p['idx2'])
        if key not in seen_700:
            seen_700.add(key)
            deduped_700.append(p)
    exp_pairs_700 = deduped_700

    print(f"  Experimental >= 400: {len(exp_pairs_400)} pairs")
    print(f"  Experimental >= 700: {len(exp_pairs_700)} pairs")

    # Channel purity analysis for exp >= 700
    n_tm = sum(1 for p in exp_pairs_700 if p['textmining'] > 0)
    n_coex = sum(1 for p in exp_pairs_700 if p['coexpression'] > 0)
    n_in_combined900 = sum(1 for p in exp_pairs_700 if (p['g1'], p['g2']) in combined_900_set)
    print(f"  Of experimental >= 700:")
    print(f"    With textmining > 0: {n_tm} ({100*n_tm/len(exp_pairs_700):.1f}%)")
    print(f"    With coexpression > 0: {n_coex} ({100*n_coex/len(exp_pairs_700):.1f}%)")
    print(f"    Also in combined >= 900: {n_in_combined900} ({100*n_in_combined900/len(exp_pairs_700):.1f}%)")

    all_results = {
        'pair_counts': {
            'experimental_400': len(exp_pairs_400),
            'experimental_700': len(exp_pairs_700),
        },
        'channel_purity_exp700': {
            'pct_textmining_gt0': float(n_tm / len(exp_pairs_700)) if exp_pairs_700 else 0,
            'pct_coexpression_gt0': float(n_coex / len(exp_pairs_700)) if exp_pairs_700 else 0,
            'pct_in_combined900': float(n_in_combined900 / len(exp_pairs_700)) if exp_pairs_700 else 0,
        }
    }

    # Train and evaluate for experimental >= 700
    for label, pair_list in [('experimental_700', exp_pairs_700), ('experimental_400', exp_pairs_400)]:
        if len(pair_list) < 100:
            print(f"\n  Skipping {label}: only {len(pair_list)} pairs")
            continue

        pairs_np = np.array([[p['idx1'], p['idx2']] for p in pair_list], dtype=np.int32)
        print(f"\n{'='*70}")
        print(f"Training: {label} ({len(pairs_np)} pairs)")
        print(f"{'='*70}")

        model, train_acc = train_model(X_norm_tensor, pairs_np)
        alpha = torch.sigmoid(model.alpha_logit).item()
        print(f"  Train acc: {train_acc*100:.1f}%, alpha: {alpha:.3f}")

        eval_res = evaluate(model, X_norm_tensor, pairs_np, n_pert)
        eval_res['n_pairs'] = len(pairs_np)
        eval_res['train_acc'] = float(train_acc)
        eval_res['alpha'] = float(alpha)

        all_results[label] = eval_res

        print(f"  Cosine AUC: {eval_res['cosine_auc']:.4f}")
        print(f"  Overall AUC (lam=0.9): {eval_res['auc_lam_0.9']:.4f}")
        print(f"  Assoc-only AUC: {eval_res['assoc_only_auc']:.4f}")
        if 'cb_cosine_auc' in eval_res:
            print(f"  CB cosine AUC: {eval_res['cb_cosine_auc']:.4f}")
            print(f"  CB assoc AUC: {eval_res['cb_assoc_auc']:.4f}")

    elapsed = time.time() - t0
    all_results['elapsed_seconds'] = elapsed

    # Save JSON
    json_path = OUT_DIR / "03_experimental_string.json"
    with open(json_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Write markdown
    md_path = OUT_DIR / "03_experimental_string.md"
    with open(md_path, 'w') as f:
        f.write("# Experiment 3: Experimental-Channel-Only STRING Filtering\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"**Runtime:** {elapsed:.0f}s\n\n")

        f.write("## Pair Counts\n\n")
        f.write(f"- Experimental >= 400: {all_results['pair_counts']['experimental_400']} pairs\n")
        f.write(f"- Experimental >= 700: {all_results['pair_counts']['experimental_700']} pairs\n\n")

        cp = all_results['channel_purity_exp700']
        f.write("## Channel Purity (experimental >= 700)\n\n")
        f.write(f"- With textmining > 0: {cp['pct_textmining_gt0']*100:.1f}%\n")
        f.write(f"- With coexpression > 0: {cp['pct_coexpression_gt0']*100:.1f}%\n")
        f.write(f"- Also in combined >= 900: {cp['pct_in_combined900']*100:.1f}%\n\n")

        f.write("## Results Comparison\n\n")
        f.write("| Filter | Pairs | Overall AUC (lam=0.9) | Assoc-only AUC | CB AUC (assoc) |\n")
        f.write("|--------|-------|-----------------------|----------------|----------------|\n")

        for label in ['experimental_400', 'experimental_700']:
            if label in all_results:
                r = all_results[label]
                cb = r.get('cb_assoc_auc', 'N/A')
                if isinstance(cb, float):
                    cb = f"{cb:.4f}"
                f.write(f"| {label} | {r['n_pairs']} | {r['auc_lam_0.9']:.4f} | "
                        f"{r['assoc_only_auc']:.4f} | {cb} |\n")

        f.write(f"| combined >= 700 (ref) | 41,434 | 0.8397 | 0.8771 | 0.8562 |\n")
        f.write(f"| combined >= 900 (ref) | 23,268 | 0.8834 | 0.9104 | 0.9018 |\n")
        f.write("\n")

        f.write("## Regression Canary\n\n")
        f.write("Reference combined >= 700: AUC 0.8397, CB AUC 0.8562\n")
        f.write("Reference combined >= 900: AUC 0.8834, CB AUC 0.9018\n")

    print(f"\nResults saved to {md_path} and {json_path}")
    print(f"Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
