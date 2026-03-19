# Experiment 1: Node-Split Inductive Test

**Date:** 2026-03-18 04:51
**Runtime:** 16s

## Node-Split Results (medium confidence >= 700)

- Train genes: 1291, Test genes: 554
- Train pairs: 20263, Test pairs: 21171
- Test pairs (both genes unseen): 3767, (one gene unseen): 17404
- Train accuracy: 4.3%
- Learned alpha: 0.490

| Metric | Cosine | CAL (lam=0.9) | CAL (lam=1.0) | Delta (lam=1.0) |
|--------|--------|---------------|---------------|------------------|
| Train AUC (transductive) | 0.6379 | 0.8498 | 0.8887 | +0.2508 |
| Test AUC (inductive) | 0.6503 | 0.7585 | 0.7763 | +0.1261 |
| Test CB AUC (|cos|<0.2) | 0.5381 | 0.6895289209431918 | 0.6885 | +0.1504 |
| Train CB AUC (|cos|<0.2) | 0.5304 | 0.8583056737398976 | 0.8630 | +0.3326 |

## Node-Split Results (high confidence >= 900)

- Train genes: 1181, Test genes: 507
- Train pairs: 11833, Test pairs: 11435
- Test pairs (both genes unseen): 1854, (one gene unseen): 9581
- Train accuracy: 6.4%
- Learned alpha: 0.497

| Metric | Cosine | CAL (lam=0.9) | CAL (lam=1.0) | Delta (lam=1.0) |
|--------|--------|---------------|---------------|------------------|
| Train AUC (transductive) | 0.6860 | 0.8809 | 0.9072 | +0.2212 |
| Test AUC (inductive) | 0.6991 | 0.8131 | 0.8261 | +0.1270 |
| Test CB AUC (|cos|<0.2) | 0.5191 | 0.7484465592386741 | 0.7518 | +0.2327 |
| Train CB AUC (|cos|<0.2) | 0.5174 | 0.8961358587289965 | 0.9022 | +0.3848 |

## Comparison with Edge-Split Inductive

| Split Type | Test AUC (lam=0.9) | Test AUC (lam=1.0) |
|------------|--------------------|--------------------|  
| Edge-split (medium, from 03_train_results) | 0.7960 | 0.8266 |
| Node-split (medium) | 0.7585 | 0.7763 |
| Node-split (high) | 0.8131 | 0.8261 |

## Regression Canary

- medium train AUC (transductive): 0.6379 cosine, 0.8498 lam=0.9
- high train AUC (transductive): 0.6860 cosine, 0.8809 lam=0.9
