# Simulated disease-wise detection results

Illustrative synthetic-score simulation. These values are generated from specified probability distributions; they are not predictions from ThorPruneViT or clinical evaluation.

Three seeds (42, 123, 456), 2,000 synthetic samples per seed, 14 disease labels, and threshold 0.5. Each label has assumed probability 0.15. Positive and negative logits follow normal distributions with means +1 and -1 and standard deviation 1. The same assumptions apply to every disease. Differences between diseases reflect random sampling.

Values below are mean ± sample SD across simulation seeds. AUROC is on the 0–1 scale; other values are percentages.

| Disease | AUROC | Accuracy (%) | Precision (%) | Sensitivity (%) | Specificity (%) | F1 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.913 ± 0.008 | 82.83 ± 0.29 | 46.41 ± 1.38 | 82.11 ± 2.50 | 82.95 ± 0.72 | 59.30 ± 1.68 |
| Cardiomegaly | 0.921 ± 0.008 | 84.10 ± 0.58 | 47.62 ± 2.21 | 84.93 ± 1.75 | 83.97 ± 0.91 | 60.99 ± 1.35 |
| Effusion | 0.923 ± 0.012 | 84.57 ± 1.30 | 49.48 ± 1.97 | 85.04 ± 1.58 | 84.48 ± 1.28 | 62.55 ± 1.99 |
| Infiltration | 0.923 ± 0.012 | 83.52 ± 0.45 | 47.07 ± 1.85 | 84.67 ± 1.24 | 83.31 ± 0.37 | 60.50 ± 1.82 |
| Mass | 0.927 ± 0.002 | 84.82 ± 0.40 | 50.56 ± 1.88 | 85.02 ± 0.78 | 84.78 ± 0.35 | 63.41 ± 1.69 |
| Nodule | 0.917 ± 0.006 | 83.77 ± 1.01 | 47.90 ± 2.91 | 81.81 ± 2.46 | 84.12 ± 1.47 | 60.37 ± 2.23 |
| Pneumonia | 0.928 ± 0.004 | 85.05 ± 0.43 | 50.20 ± 1.76 | 85.26 ± 1.20 | 85.01 ± 0.40 | 63.18 ± 1.70 |
| Pneumothorax | 0.923 ± 0.005 | 84.12 ± 0.93 | 49.52 ± 0.92 | 85.48 ± 0.76 | 83.86 ± 1.11 | 62.71 ± 0.90 |
| Consolidation | 0.921 ± 0.004 | 84.73 ± 1.44 | 50.39 ± 2.46 | 84.34 ± 2.03 | 84.81 ± 1.74 | 63.06 ± 1.95 |
| Edema | 0.917 ± 0.010 | 83.78 ± 0.73 | 48.03 ± 3.25 | 85.14 ± 1.77 | 83.54 ± 0.72 | 61.38 ± 2.90 |
| Emphysema | 0.925 ± 0.009 | 84.47 ± 0.88 | 48.99 ± 3.22 | 84.92 ± 2.87 | 84.37 ± 0.66 | 62.13 ± 3.35 |
| Fibrosis | 0.920 ± 0.007 | 83.52 ± 0.03 | 46.67 ± 1.63 | 84.52 ± 1.51 | 83.35 ± 0.28 | 60.11 ± 1.20 |
| Pleural Thickening | 0.918 ± 0.010 | 83.67 ± 0.83 | 46.58 ± 1.64 | 84.40 ± 2.63 | 83.54 ± 0.54 | 60.03 ± 2.02 |
| Hernia | 0.926 ± 0.014 | 84.87 ± 1.06 | 49.89 ± 1.34 | 84.33 ± 1.94 | 84.97 ± 0.90 | 62.69 ± 1.54 |

## Aggregate simulation results

| Metric | Mean ± SD |
|---|---:|
| macro_auroc | 0.9215 ± 0.0007 |
| macro_f1 | 0.6160 ± 0.0036 |
| macro_precision | 0.4852 ± 0.0041 |
| macro_sensitivity | 0.8443 ± 0.0026 |
| macro_specificity | 0.8407 ± 0.0022 |
| hamming_accuracy | 0.8413 ± 0.0021 |
| micro_f1 | 0.6161 ± 0.0036 |
| micro_auroc | 0.9215 ± 0.0007 |
| exact_match_accuracy | 0.0882 ± 0.0099 |

## Results text for a simulation section

A reproducible illustrative simulation generated synthetic labels and detection scores for 14 thoracic disease categories across three random seeds. Disease-wise AUROC, accuracy, precision, sensitivity, specificity, and F1 were computed from the generated labels and scores. The tables summarize variability across simulation runs under identical assumptions for all disease categories. This simulation demonstrates the reporting workflow; model performance is evaluated separately through inference on labelled images.

Reproduce with `python scripts/generate_simulated_detection_results.py`. Raw labels and scores, confusion counts, per-seed metrics, aggregate metrics, and distribution assumptions accompany this report.
