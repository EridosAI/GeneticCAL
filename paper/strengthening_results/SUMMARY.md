# Strengthening Experiments — Summary

**Date:** 2026-03-18
**All regression canaries passed.** Seed 42 results reproduced exactly in every experiment.

---

## Experiment 1: Node-Split Inductive Test

**Result: CAL generalises to completely unseen genes.** With 30% of genes held out (node split), the model achieves test AUC 0.826 (assoc-only) on high-confidence pairs — a +0.127 improvement over cosine. This is a genuine cold-start result: genes never seen during training are correctly positioned in association space. The drop from the transductive AUC (0.907) to inductive (0.826) is modest and expected. This directly addresses the reviewer concern that edge-split inductive results could reflect gene-specific memorisation.

## Experiment 2: Seed Variance (3 seeds)

**Result: Results are highly stable across seeds.** Overall AUC (λ=0.9): 0.8836 ± 0.0005. Cross-boundary AUC (λ=0.9): 0.9015 ± 0.0005. The standard deviation is less than 0.1%, confirming that the contrastive learning signal is robust and not an artefact of initialisation. All three seeds converge to nearly identical alpha values (~0.490) and training losses (~4.110).

## Experiment 3: Experimental-Channel-Only STRING Filtering

**Result: Experimental-only pairs produce equivalent performance.** Filtering to STRING experimental channel ≥ 700 yields 23,438 pairs and AUC 0.884 (λ=0.9), CB AUC 0.902 — effectively identical to the combined-score ≥ 900 baseline (0.883, 0.902). This validates that the training signal comes from experimentally supported interactions, not text-mining or coexpression artefacts. Note: 87.6% of experimental ≥ 700 pairs also have textmining > 0, and 96.9% have coexpression > 0, reflecting genuine biological correlation across evidence types.

## Experiment 4: Cross-Boundary AUC at Multiple Thresholds

**Result: CAL advantage is robust across all cross-boundary cutoffs.** CAL AUC remains stable at ~0.91 across |cos| < 0.30, 0.20, 0.15, 0.10, and 0.05, while the cosine baseline drops from 0.54 to 0.49 (approaching chance). The delta actually *increases* at stricter thresholds (+0.37 at |cos| < 0.30 to +0.42 at |cos| < 0.05). The |cos| < 0.2 threshold used in the paper is not cherry-picked — any reasonable cutoff yields the same conclusion.

## Experiment 5: Expression-Space Cluster GO Enrichment

**Result: Expression clusters also show high enrichment (35/36 coherent, 97%).** This means the 22/22 (100%) PAM-specific cluster enrichment, while perfect, should be contextualised against a high baseline. However, the random baseline (matched cluster sizes) shows only 13.8 ± 1.5 / 34 coherent (~41%), confirming both expression and PAM clustering capture real biological structure far above chance. The key claim for PAM-specific clusters remains valid: these clusters group genes that are *invisible* to expression similarity but are functionally related.

**Paper implication:** The paper should report that expression clusters also show ~97% GO enrichment, and frame the PAM-specific cluster result as showing that association space discovers *additional* coherent groupings beyond what expression clustering already captures — not that only PAM clusters are biologically meaningful.

## Experiment 6: Degree-Matched Negatives

**Result: Degree-matching does not reduce AUC — it slightly increases it.** With degree-matched negatives, overall AUC (λ=0.9) = 0.893 vs random 0.883 (+0.010), and CB AUC = 0.935 vs 0.908 (+0.027). This means the random negative baseline was, if anything, *conservative*. The AUC is not inflated by degree bias in the negative set.

---

## Changes to Paper Claims

1. **Node-split inductive (Exp 1):** Paper can now claim genuine cold-start generalisation, not just compositional generalisation from edge splits.
2. **Seed variance (Exp 2):** Paper should report mean ± SD. The numbers are tight enough that single-seed reporting is defensible, but reporting variance is good practice.
3. **STRING filtering (Exp 3):** Paper can now state "results hold when training exclusively on experimentally-validated interactions (STRING experimental channel ≥ 700)."
4. **Cross-boundary threshold (Exp 4):** Paper should include the sensitivity table showing robustness across cutoffs.
5. **Cluster enrichment (Exp 5):** Paper must acknowledge the high expression-space baseline (97%) and frame PAM-specific clusters as *additional* discoveries rather than *unique* biological signal.
6. **Degree-matched negatives (Exp 6):** Paper can state that degree-matched negatives yield equivalent or higher AUC, ruling out degree-bias confound.
