"""NIH-to-CheXpert external transfer evaluation. Never trains or tunes the model."""
import argparse, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np, pandas as pd, torch, yaml
from torch.utils.data import DataLoader
from common import ROOT, load_config, device_from
from thorprunevit.data import CheXpertDataset, CHEXPERT_LABELS
from thorprunevit.metrics import masked_multilabel_metrics
from thorprunevit.model import load_pruned_model, make_vit_b16

DESCRIPTION='NIH-to-CheXpert external transfer evaluation'
def load_mapping(path=ROOT/'configs/label_mappings.yaml'):
    entries=yaml.safe_load(Path(path).read_text(encoding='utf-8'))['entries']; return [x for x in entries if x.get('compatible')]
def validate_mapping(entries, output_dim=14, available_columns=None):
    good=[]
    for x in entries:
        i=int(x['nih_output_index'])
        if not 0<=i<output_dim: raise ValueError(f'NIH output index {i} outside checkpoint dimension {output_dim}')
        if available_columns is not None and x['chexpert_column'] not in available_columns: raise ValueError(f"CheXpert column missing: {x['chexpert_column']}")
        good.append(x)
    if not good: raise ValueError('No scientifically compatible shared labels configured')
    return good
def evaluate_transfer_arrays(logits, chexpert_targets, entries, out_dir, seed, checkpoint='synthetic', policy='ignore'):
    entries=validate_mapping(entries,logits.shape[1]); oi=[x['nih_output_index'] for x in entries]; ci=[CHEXPERT_LABELS.index(x['chexpert_column']) for x in entries]
    prob=torch.sigmoid(torch.as_tensor(logits)).cpu().numpy()[:,oi]; truth=np.asarray(chexpert_targets)[:,ci]; names=[x['nih_label'] for x in entries]
    if policy=='u-zeros': truth=np.where((truth==-1)|~np.isfinite(truth),0,truth)
    elif policy=='u-ones': truth=np.where((truth==-1)|~np.isfinite(truth),1,truth)
    agg,per=masked_multilabel_metrics(truth,prob,.5,names); d=Path(out_dir); d.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(d/'raw_predictions.npz',ground_truth=truth,predicted_probabilities=prob,binary_predictions=(prob>=.5).astype(np.uint8),valid_mask=np.isfinite(truth)&(truth!=-1),nih_output_indices=np.array(oi),labels=np.array(names))
    pd.DataFrame([agg]).to_csv(d/'metrics.csv',index=False); pd.DataFrame(per).to_csv(d/'per_class_metrics.csv',index=False); pd.DataFrame(entries).to_csv(d/'mapping_used.csv',index=False)
    (d/'run_manifest.json').write_text(json.dumps({'experiment':DESCRIPTION,'seed':seed,'checkpoint':str(checkpoint),'uncertainty_policy':policy,'threshold':.5,'labels':names,'timestamp':datetime.now(timezone.utc).isoformat(),'training_or_finetuning':False},indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser(description=DESCRIPTION); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); ap.add_argument('--seed',type=int,required=True); ap.add_argument('--checkpoint',required=True); a=ap.parse_args(); cfg=load_config(a.config); cp=Path(a.checkpoint)
    if not cp.exists(): print(f'TRANSFER BLOCKED - checkpoint directory missing: {cp}'); return 2
    valid=Path(cfg['chexpert']['valid_csv']); root=Path(cfg['chexpert']['root'])
    if 'PATH_TO_' in str(valid) or not valid.is_file() or 'PATH_TO_' in str(root) or not root.is_dir(): print(f'TRANSFER BLOCKED - configure CheXpert root and valid CSV: {root}; {valid}'); return 2
    raw=pd.read_csv(valid); entries=validate_mapping(load_mapping(),14,raw.columns)
    if (cp/'checkpoint.pt').exists(): model=load_pruned_model(cp)
    elif (cp/'model.pt').exists(): model=make_vit_b16(14,False,cfg['training']['dropout']); model.load_state_dict(torch.load(cp/'model.pt',map_location='cpu'))
    else: print(f'TRANSFER BLOCKED - checkpoint.pt or model.pt missing in {cp}'); return 2
    device=device_from(cfg); model.to(device).eval(); ds=CheXpertDataset(str(valid),str(root),False,'ignore',cfg['experiment']['image_size']); dl=DataLoader(ds,batch_size=cfg['training']['batch_size'],shuffle=False,num_workers=cfg['training']['num_workers']); logits=[]; targets=[]
    with torch.no_grad():
        for x,y in dl: logits.append(model(x.to(device)).cpu()); targets.append(y)
    evaluate_transfer_arrays(torch.cat(logits),torch.cat(targets).numpy(),entries,ROOT/'results/chexpert_transfer'/f'seed_{a.seed}',a.seed,cp,cfg['chexpert']['primary_uncertainty_policy']); print(DESCRIPTION); return 0
if __name__=='__main__': sys.exit(main())
