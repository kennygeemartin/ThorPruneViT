# ThorPruneViT executable reproducibility package

This package implements the reviewer-requested computational workflow without inventing results. It is designed for **NIH ChestX-ray14** and contains a separate CheXpert loader because CheXpert has a different 14-observation taxonomy and therefore must be reported as a separate benchmark rather than treated as label-identical to NIH.

## What is implemented

- patient-disjoint 70/10/20 NIH splitting and export of exact split CSVs;
- ViT-B/16 multilabel baseline (14 disease labels);
- class-balanced BCE (`pos_weight` computed from the training split only);
- explicit Taylor importance: `mean(abs(activation ⊙ gradient))`;
- structural magnitude importance;
- weighted Taylor+magnitude score (default 0.5/0.5, globally normalized within component type);
- **physical structured pruning** of attention heads and FFN neurons by model surgery;
- 53% cumulative **total-parameter** target reached in five 10.6-percentage-point stages;
- recovery fine-tuning at `1e-5` after each stage;
- DenseNet-121 CNN baseline;
- magnitude-only pruning baseline;
- heads-only, FFN-only, Taylor-only, magnitude-only, and combined ablations;
- sensitivity targets 30%, 40%, 50%, 53%, and 60%;
- mean per-class/Hamming accuracy, exact-match accuracy, macro/micro precision, recall, F1, AUROC;
- 14-disease NIH breakdown with AUROC, precision, sensitivity, specificity, F1;
- latency protocol with explicit warm-up, timed iterations, batch size, and device;
- multi-seed aggregation with mean, SD, and 95% CI.

## Important scientific limitation

The current ChatGPT execution environment does **not** contain the NIH ChestX-ray14 or CheXpert image datasets and has **CPU-only PyTorch**. Therefore the package has been smoke-tested on synthetic tensors for code correctness, but the reviewer-facing empirical tables must be generated on the real datasets (preferably on the authors' GPU) before submission. The synthetic smoke-test numbers are **not manuscript results**.

## Setup

Edit `configs/config.yaml` with the real dataset paths. Then run from the repository root:

```bash
export PYTHONPATH=$PWD/src
python scripts/train_baseline.py --seed 42
python scripts/train_baseline.py --seed 123
python scripts/train_baseline.py --seed 456

python scripts/run_cnn_baseline.py --seed 42
python scripts/run_cnn_baseline.py --seed 123
python scripts/run_cnn_baseline.py --seed 456

python scripts/run_ablations.py
python scripts/run_sparsity_sensitivity.py
python scripts/summarize_results.py
```

The scripts write exact split files, checkpoints, aggregate metrics, disease-wise metrics, pruning histories, and latency measurements into `data_splits/` and `results/`.

## Smoke test

```bash
export PYTHONPATH=$PWD/src
python tests/smoke_test.py
```

The smoke test verifies that the structured-pruning implementation physically reduces parameters/FLOPs and that the five-stage schedule targets 10.6%, 21.2%, 31.8%, 42.4%, and 53.0% cumulative parameter reduction.
