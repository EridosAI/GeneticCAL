# Experiment 6: Degree-Matched Negatives (high confidence >= 900)

**Date:** 2026-03-18 05:26
**Runtime:** 4s

## Results

| Negative type | N neg | Overall AUC (lam=0.9) | Assoc-only AUC | CB AUC (assoc) | Delta vs random |
|---------------|-------|-----------------------|----------------|----------------|------------------|
| Random (existing) | 50000 | 0.8834 | 0.9104 | 0.9082 | — |
| Degree-matched (±20%) | 116340 | 0.8933 | 0.9149 | 0.9349 | +0.0099 |

## Degree Statistics

- Positive pair mean degree: 105.9
- Degree-matched neg mean degree: 102.1
- Pairs needing ±50% widening: 9

## Regression Canary

- Random neg AUC (lam=0.9): 0.883415 (ref: 0.883415)
- Random neg CB AUC: 0.908218280881078 (ref: 0.9082)
