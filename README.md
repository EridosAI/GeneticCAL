# Contrastive Association Learning Generalises to Biology

Code and results for "Contrastive Association Learning Generalises to Biology: Cross-Domain Validation of Association ≠ Similarity" (Dury, 2026).

## Summary

Contrastive Association Learning (CAL) trains a lightweight MLP to learn associative relationships between items that co-occur in shared contexts, distinct from embedding similarity. This paper tests whether the principle — established in text retrieval ([AAR](https://github.com/EridosAI/AAR)) and corpus-scale concept discovery — transfers to molecular biology.

**Four experiments, two domains:**

| Experiment | Embedding | Association | Outcome |
|---|---|---|---|
| Replogle K562 CRISPRi | Gene perturbation profiles (PCA-50) | STRING protein interactions | **CB AUC 0.908** (cosine: 0.518) |
| DepMap CRISPR screens | Gene expression (PCA-100) | Co-essentiality | **CB AUC 0.947** (cosine: 0.570) |
| Drug fingerprints | Morgan FP (2048-bit) | PRISM co-lethality | Negative (no latent signal) |
| Drug L1000 | Transcriptional signatures | PRISM co-lethality | Negative (degree confounded) |

**Key findings:**
- Association ≠ similarity is a cross-domain phenomenon, not a text artefact
- Inductive transfer succeeds in biology (node-split Δ +0.127) where it fails in text (±0.10)
- Improvement concentrates on low-degree (understudied) genes
- Association quality outperforms quantity, reversing the text pattern

## Repository Structure

```
experiments/
├── replogle/          # Experiment 1: Gene perturbation × STRING PPI
├── depmap/            # Experiment 2: Gene expression × co-essentiality
├── drug_structure/    # Experiment 3: Morgan fingerprints × co-lethality
└── drug_l1000/        # Experiment 4: L1000 signatures × co-lethality
paper/                 # LaTeX source, figures, supplementary results
```

## Requirements

```
pip install -r requirements.txt
```

Key dependencies: PyTorch, NumPy, scikit-learn, FAISS, scanpy (for .h5ad loading), gprofiler-official.

## Data

Embeddings and association pairs are derived from public datasets:

- **Replogle K562:** [Replogle et al. (2022)](https://doi.org/10.1038/s41588-022-01106-y) — available via [CellXGene](https://cellxgene.cziscience.com/)
- **STRING:** [v12.0](https://string-db.org/) — `9606.protein.links.detailed.v12.0.txt.gz`
- **DepMap:** [DepMap Public 23Q4](https://depmap.org/portal/)
- **PRISM:** [Corsello et al. (2020)](https://doi.org/10.1038/s43018-019-0018-6)
- **L1000:** [LINCS L1000](https://lincsproject.org/)

Raw data files are not included in this repository due to size. Download scripts are provided in each experiment's `scripts/` directory.

## Related Work

This paper is part of the PAM research programme:

1. **PAM** — [Predictive Associative Memory: Retrieval Beyond Similarity](https://github.com/EridosAI/PAM-Benchmark) (Dury, 2026)
2. **AAR** — [Association-Augmented Retrieval for Multi-Hop QA](https://github.com/EridosAI/AAR) (Dury, 2026)
3. **Concept Discovery** — Corpus-scale concept discovery via temporal co-occurrence (Dury, 2026)
4. **This paper** — Cross-domain biological validation of CAL

## Citation

```bibtex
@article{dury2026cal,
  title={Contrastive Association Learning Generalises to Biology: Cross-Domain Validation of Association $\neq$ Similarity},
  author={Dury, Jason},
  year={2026},
  journal={arXiv preprint}
}
```

## License

MIT
