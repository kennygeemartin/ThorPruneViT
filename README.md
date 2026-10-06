# ThorPruneViT

Structured vision-transformer pruning, multi-label evaluation, and a research image-inference interface for 14 NIH thoracic disease labels.

## Results

- [Table 8: supplied NIH disease-wise comparison](training/results/supplied/README.md): ViT-B/16 and ThorPruneViT AUROC, F1, and AUPRC, with model differences and macro summaries.

- [Disease-wise simulation report](training/results/simulation/README.md): AUROC, accuracy, precision, sensitivity, specificity, and F1 across three seeds.
- [Synthetic pruning verification](training/VERIFICATION.md): measured parameter reduction of 54.22% and FLOP reduction of 55.94% on the smoke-test model.
- [Training and evaluation workflow](training/README.md).

Table 8 stores the user-supplied NIH summary and its provenance. The simulation report uses generated labels and scores under explicit assumptions. The pruning measurements come from the separate synthetic model smoke test.

## Reproduce the simulation

From the repository root:

```powershell
python training/scripts/generate_simulated_detection_results.py
```

The generator uses the Python standard library. It writes the report, full-precision CSV tables, raw labels and scores, and protocol to `training/results/simulation/`.

## Run image inference

1. Install Python 3.11 or 3.12 and `requirements-web.txt`.
2. Follow [the training workflow](training/README.md) to produce a pruned checkpoint containing `checkpoint.pt` and `architecture.json`.
3. Configure and launch:

```powershell
$env:THORPRUNEVIT_CHECKPOINT = "C:\path\to\pruned_checkpoint"
python app.py
```

Open <http://127.0.0.1:8000>. The configured model returns 14 independent disease probabilities. Input-gradient saliency is pooled to the 14-by-14 patch grid. The download button exports research results as JSON.

The interface loads the configured checkpoint for inference. Outputs are for research use; clinical validation is a separate evaluation step.

## Resources

- [ViT paper](https://arxiv.org/abs/2010.11929)
- [NIH ChestX-ray14 dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert dataset](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [PyTorch](https://pytorch.org/docs/stable/index.html)

