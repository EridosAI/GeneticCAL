# Repository Survey — GeneticCAL

Survey conducted 2026-03-19.

## Gene AAR/ → experiments/replogle/

### Code (.py) — 10 scripts
- scripts/01_explore.py, 02_pairs.py, 03_train.py, 04_ablations.py
- scripts/04_cluster_analysis.py, 05_threshold_sweep.py, 05_validate.py
- scripts/06_analysis.py, 06_cluster_validation.py
- scripts/degree_analysis.py, find_examples.py (also copied to analysis/)

### Results (.json, .csv, .png) — INCLUDED
- 01_exploration_stats.json, 02_pairs_results.json, 03_train_results.json
- 04_cluster_results.json, 05_threshold_results.json, 06_validation_results.json
- 06_validation_summary.csv, degree_vs_improvement_genes.csv (123KB)
- degree_vs_improvement_pairs.csv (1.7MB — gitignored)
- 01-05 plots .png

### Large data — EXCLUDED
- K562_essential_normalized_bulk_01.h5ad (76MB)
- CRISPRGeneEffect.csv (412MB), OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv (518MB)
- 9606.protein.links.detailed.v12.0.txt.gz (133MB)
- string_pairs_{high,medium,low}.npy
- model_{high,medium,low}.pt

### Small data — INCLUDED
- string_pairs_high.csv (1.4MB), string_pairs_medium.csv (2.6MB), string_pairs_low.csv (5.8MB)

---

## Gene AAR_v2/ → experiments/depmap/

### Code (.py) — 7 scripts
- scripts/01_explore.py through 06_analysis.py
- scripts/03b_train_random_neg.py

### Results (.json, .png) — INCLUDED
- 01-06 stats/results .json, 03b_randneg_results.json
- 01-06 plots .png
- cell_line_list.json, gene_list.json

### Large data — EXCLUDED
- CRISPRGeneEffect.csv (412MB), OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv (518MB)
- 9606.protein.links.detailed.v12.0.txt.gz (133MB), Model.csv
- crispr_matrix.npy (150MB), expr_matrix.npy (150MB)
- gene_embeddings_pca{50,100}.npy, pairs_*.npy, pairs_*_coess_scores.npy
- model_*.pt (4 models + 1 randneg)

---

## Gene AAR Drug Structure/ → experiments/drug_structure/

### Code (.py) — 7 scripts
- scripts/01_explore.py, 01b_rdkit_descriptors.py, 02_pairs.py
- scripts/03_train.py, 04_ablations.py, 05_validate.py, 06_analysis.py

### Results (.json, .png) — INCLUDED
- 02_pairs_stats.json, 03_train.json, 04_ablations.json, 05_validation.json, 06_analysis.json
- figures/01-06 .png (12 figure files)

### Large data — EXCLUDED
- co_lethality.npy (84MB), sensitivity_matrix.npy, embeddings_*.npy, pairs_*.npy
- models/pam_*.pt (4 models)
- primary-screen-replicate-collapsed-logfold-change.csv (39MB)
- drug_info.csv (1.1MB — borderline, excluded with other data files)

---

## Gene AAR Drug L1000/ → experiments/drug_l1000/

### Code (.py) — 7 scripts
- scripts/01_explore.py through 06_analysis.py
- scripts/04b_shuffle_analysis.py

### Results (.json, .png) — INCLUDED
- 02_pairs_stats.json, 03_train.json, 04_ablations.json, 04b_shuffle_analysis.json
- figures/01-04b .png (9 figure files)

### Large data — EXCLUDED
- GSE92742_Broad_LINCS_Level5 (22GB!), Level2 (185MB), .gz files
- co_lethality.npy, sensitivity_matrix.npy, embeddings_*.npy, pairs_*.npy
- models/pam_*.pt (12 models)
- primary-screen-replicate-collapsed-logfold-change.csv (39MB)

---

## Paper/ → paper/

### Included
- cal_biology_paper_v3.tex (final version)
- cal_references.bib
- generate_figures.py
- figures/ (6 .png files including fig4_degree_scatter.png)
- strengthening_results/ (6 experiments × {.json, .md, run_*.py} + SUMMARY.md + lambda_1.0_numbers.md)

### Excluded
- cal_biology_paper.tex (v1 — superseded)
- cal_biology_paper_v2.md (v2 — superseded)
- experiment_details_for_paper.md (working notes)
- __pycache__/

---

## Gene AAR Epigenomics/ — EXCLUDED ENTIRELY
Parked exploration, not part of the paper.

---

## analysis/
Cross-experiment scripts copied from Gene AAR:
- degree_analysis.py
- find_examples.py
