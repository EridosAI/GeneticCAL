# Experiment 3: Experimental-Channel-Only STRING Filtering

**Date:** 2026-03-18 05:09
**Runtime:** 179s

## Pair Counts

- Experimental >= 400: 35266 pairs
- Experimental >= 700: 23438 pairs

## Channel Purity (experimental >= 700)

- With textmining > 0: 87.6%
- With coexpression > 0: 96.9%
- Also in combined >= 900: 79.1%

## Results Comparison

| Filter | Pairs | Overall AUC (lam=0.9) | Assoc-only AUC | CB AUC (assoc) |
|--------|-------|-----------------------|----------------|----------------|
| experimental_400 | 35266 | 0.8165 | 0.8537 | 0.8476 |
| experimental_700 | 23438 | 0.8838 | 0.9069 | 0.9023 |
| combined >= 700 (ref) | 41,434 | 0.8397 | 0.8771 | 0.8562 |
| combined >= 900 (ref) | 23,268 | 0.8834 | 0.9104 | 0.9018 |

## Regression Canary

Reference combined >= 700: AUC 0.8397, CB AUC 0.8562
Reference combined >= 900: AUC 0.8834, CB AUC 0.9018
