# Experiment 5: Expression-Space Cluster GO Enrichment

**Date:** 2026-03-18 05:22
**Runtime:** 622s

## Cluster GO Enrichment Comparison

| Cluster source | Total clusters | Coherent | Partial | No signal | % coherent |
|----------------|---------------|----------|---------|-----------|------------|
| PAM-specific (association space) | 22 | 22 | 0 | 0 | 100% |
| Expression-space HDBSCAN | 36 | 35 | 1 | 0 | 97% |
| Random (matched sizes, 10 repeats) | 34 | 13.8 +/- 1.4 | | | 41% |

## Top 5 Expression-Space Enrichments

1. **Cluster 27** (108 genes): GO:BP: ribosome biogenesis (p=5.68e-132, 81% coverage)
2. **Cluster 31** (28 genes): GO:CC: cytosolic large ribosomal subunit (p=1.17e-70, 96% coverage)
3. **Cluster 35** (29 genes): GO:CC: organellar large ribosomal subunit (p=1.68e-69, 93% coverage)
4. **Cluster 6** (36 genes): GO:CC: proteasome complex (p=2.52e-68, 83% coverage)
5. **Cluster 17** (46 genes): GO:BP: DNA replication (p=2.79e-56, 78% coverage)

## Regression Canary

PAM-specific cluster results: 22/22 coherent (from 06_validation_results.json)
