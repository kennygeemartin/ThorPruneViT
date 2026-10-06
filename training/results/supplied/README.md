# Table 8: Disease-wise performance on the NIH ChestX-ray14 held-out test set (mean across three seeds)

| Disease | AUROC (ViT-B/16) | AUROC (ThorPruneViT) | F1 (ViT-B/16) | F1 (ThorPruneViT) | AUPRC (ViT-B/16) | AUPRC (ThorPruneViT) |
|---|---:|---:|---:|---:|---:|---:|
| Atelectasis | 0.842 | 0.838 | 0.360 | 0.357 | 0.418 | 0.414 |
| Cardiomegaly | 0.913 | 0.909 | 0.475 | 0.471 | 0.589 | 0.584 |
| Consolidation | 0.801 | 0.796 | 0.289 | 0.286 | 0.315 | 0.311 |
| Edema | 0.901 | 0.897 | 0.461 | 0.457 | 0.562 | 0.557 |
| Effusion | 0.861 | 0.857 | 0.382 | 0.379 | 0.448 | 0.444 |
| Emphysema | 0.874 | 0.870 | 0.408 | 0.404 | 0.483 | 0.478 |
| Fibrosis | 0.812 | 0.808 | 0.315 | 0.312 | 0.336 | 0.332 |
| Hernia | 0.742 | 0.736 | 0.176 | 0.168 | 0.148 | 0.142 |
| Infiltration | 0.821 | 0.817 | 0.327 | 0.324 | 0.372 | 0.368 |
| Mass | 0.846 | 0.842 | 0.349 | 0.346 | 0.401 | 0.397 |
| Nodule | 0.827 | 0.822 | 0.322 | 0.319 | 0.359 | 0.355 |
| Pleural Thickening | 0.836 | 0.831 | 0.329 | 0.326 | 0.381 | 0.377 |
| Pneumonia | 0.789 | 0.781 | 0.281 | 0.262 | 0.291 | 0.283 |
| Pneumothorax | 0.910 | 0.906 | 0.482 | 0.477 | 0.428 | 0.424 |

Disease-wise held-out test performance is presented in Table 8. Cardiomegaly, Edema, and Pneumothorax achieved the highest AUROC values for both models, whereas Hernia had the lowest AUROC. ThorPruneViT retained similar disease-specific performance across most labels, with AUROC reductions of 0.004-0.008; 13 of 14 labels had reductions of 0.004-0.006. The largest F1-score reduction occurred for Pneumonia (0.281 to 0.262, a decrease of 0.019). Across all 14 labels, macro AUROC changed from 0.8411 to 0.8364, macro F1 from 0.3540 to 0.3491, and macro AUPRC from 0.3951 to 0.3904. These supplied values indicate a small reduction in disease-level discrimination at the reported pruning operating point.

## Source and reporting

Source: user-supplied Table 8, corrected on 2026-10-06. The title and mean-across-three-seeds designation follow the supplied table. This repository stores these values as supplied summaries, separately from generated simulation results. Per-seed predictions, split identifiers, checkpoints, variability estimates, and the AUPRC integration convention were not included with the table. Macro summaries are unweighted arithmetic means of the 14 rounded disease values.

The interpretation above uses the supplied numeric values. Disease prevalence and resource savings are reported from their respective data and profiling sources.
