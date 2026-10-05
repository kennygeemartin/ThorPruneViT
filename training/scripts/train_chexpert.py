import argparse, json
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
from common import load_config, device_from, ROOT
from thorprunevit.data import prepare_chexpert_dataframe, chexpert_patient_split, CheXpertDataset, CHEXPERT_LABELS, MaskedBCEWithLogitsLoss
from thorprunevit.model import PrunableViT, count_parameters, approximate_flops
from thorprunevit.train import set_seed, train_model, evaluate, evaluate_masked

ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); cfg=load_config(args.config); device=device_from(cfg); set_seed(args.seed)
unc=cfg['chexpert']['uncertainty']; full=prepare_chexpert_dataframe(cfg['chexpert']['train_csv'],unc); tr,va=chexpert_patient_split(full,0.9,args.seed); te=prepare_chexpert_dataframe(cfg['chexpert']['valid_csv'],unc)
(ROOT/'data_splits').mkdir(exist_ok=True); tr.to_csv(ROOT/'data_splits'/f'chexpert_train_seed_{args.seed}.csv',index=False); va.to_csv(ROOT/'data_splits'/f'chexpert_validation_seed_{args.seed}.csv',index=False); te.to_csv(ROOT/'data_splits'/'chexpert_official_validation.csv',index=False)
root=cfg['chexpert']['root']; bs=cfg['training']['batch_size']; nw=cfg['training']['num_workers']
# Save temporary mapped CSVs because the dataset class reads CSV paths.
trp=ROOT/'data_splits'/f'chexpert_train_seed_{args.seed}.csv'; vap=ROOT/'data_splits'/f'chexpert_validation_seed_{args.seed}.csv'; tep=ROOT/'data_splits'/'chexpert_official_validation.csv'
trdl=DataLoader(CheXpertDataset(str(trp),root,True,unc),batch_size=bs,shuffle=True,num_workers=nw); vadl=DataLoader(CheXpertDataset(str(vap),root,False,unc),batch_size=bs,shuffle=False,num_workers=nw); tedl=DataLoader(CheXpertDataset(str(tep),root,False,unc),batch_size=bs,shuffle=False,num_workers=nw)
y=tr[CHEXPERT_LABELS].to_numpy(float)
if unc=='ignore':
    known=(y!=-1) & ~pd.isna(y); pos=((y==1)&known).sum(0); neg=((y==0)&known).sum(0)
else: pos=y.sum(0); neg=len(y)-pos
pw=torch.tensor(neg/(pos.clip(min=1)),dtype=torch.float32)
# A separate randomly initialized task head is used; ImageNet weight loading can be added if cached locally.
from thorprunevit.model import make_vit_b16
criterion=MaskedBCEWithLogitsLoss(unc,pw) if unc=='ignore' else None
m=make_vit_b16(14,cfg['training']['pretrained_imagenet'],cfg['training']['dropout']); m,h=train_model(m,trdl,vadl,device,cfg['training']['epochs'],cfg['training']['learning_rate'],cfg['training']['weight_decay'],pw,True,cfg['training']['early_stopping_patience'],criterion)
agg,per,_,_=(evaluate_masked if unc=='ignore' else evaluate)(m,tedl,device,CHEXPERT_LABELS)
out=ROOT/'results'/f'chexpert_baseline_seed_{args.seed}'; out.mkdir(exist_ok=True); torch.save(m.state_dict(),out/'model.pt'); pd.DataFrame(h).to_csv(out/'history.csv',index=False); pd.DataFrame(per).to_csv(out/'disease_wise.csv',index=False); pd.DataFrame([agg|{'seed':args.seed,'parameters':count_parameters(m),'approx_flops':approximate_flops(m)}]).to_csv(out/'aggregate.csv',index=False); print(json.dumps(agg,indent=2))
