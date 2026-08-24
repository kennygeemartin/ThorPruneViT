# ThorPruneViT clinical inference interface

This repository couples the supplied HTML interface to the physically structured-pruned ViT-B/16 implementation in `Model Training/reproducibility_fixed`. It does **not** generate simulated clinical predictions. The API remains unavailable until a real NIH ChestX-ray14-trained pruned checkpoint is configured.

## Run locally

1. Install Python 3.11 or 3.12 and the packages in `requirements-web.txt`.
2. Train and prune the model using `Model Training/reproducibility_fixed/README.md`, producing a checkpoint directory containing `checkpoint.pt` and `architecture.json`.
3. Configure and launch:

```powershell
$env:THORPRUNEVIT_CHECKPOINT = "C:\path\to\pruned_checkpoint"
python app.py
```

Open <http://127.0.0.1:8000>. Predictions are 14 independent NIH disease probabilities. The visualization is input-gradient saliency for the highest-probability output, pooled to the ViT 14×14 patch grid.

## Important limitations

- No trained weights or NIH/CheXpert images are included in this workspace. A checkpoint must be produced from authorized data before inference is available.
- Outputs are research-only and are not a diagnosis or a validated medical device result.
- The download button exports JSON research results, not a standards-compliant DICOM Structured Report.

## Resource materials

- ViT paper: https://arxiv.org/abs/2010.11929
- NIH ChestX-ray14 paper and dataset description: https://nihcc.app.box.com/v/ChestXray-NIHCC
- CheXpert dataset: https://stanfordmlgroup.github.io/competitions/chexpert/
- PyTorch documentation: https://pytorch.org/docs/stable/index.html
- Torchvision ViT-B/16 reference: https://pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html

Review each dataset's terms and access requirements before downloading or redistributing it.

