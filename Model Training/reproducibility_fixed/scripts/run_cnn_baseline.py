import argparse, json
import pandas as pd, torch
from torch.utils.data import DataLoader
from torchvision.models import densenet121, DenseNet121_Weights
from common import load_config, device_from, ROOT, require_split
from thorprunevit.data import NIHChestXray14, NIH_LABELS
from thorprunevit.train import set_seed, train_model, evaluate, measure_latency

ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args(); cfg=load_config(args.config); device=device_from(cfg); set_seed(args.seed)
sp=require_split(args.seed); tr=pd.read_csv(sp['train']); va=pd.read_csv(sp['val']); te=pd.read_csv(sp['test']); root=cfg['nih']['image_root']; bs=cfg['training']['batch_size']; nw=cfg['training']['num_workers']
trdl=DataLoader(NIHChestXray14(tr,root,True),batch_size=bs,shuffle=True,num_workers=nw); vadl=DataLoader(NIHChestXray14(va,root,False),batch_size=bs,shuffle=False,num_workers=nw); tedl=DataLoader(NIHChestXray14(te,root,False),batch_size=bs,shuffle=False,num_workers=nw)
pw=torch.tensor(pd.read_csv(sp['train'].parent/'positive_weights.csv')['pos_weight'].values,dtype=torch.float32); w=DenseNet121_Weights.DEFAULT if cfg['training']['pretrained_imagenet'] else None; m=densenet121(weights=w); m.classifier=torch.nn.Linear(m.classifier.in_features,14)
m,h=train_model(m,trdl,vadl,device,cfg['training']['epochs'],cfg['training']['learning_rate'],cfg['training']['weight_decay'],pw,True,cfg['training']['early_stopping_patience']); agg,per,_,_=evaluate(m,tedl,device,NIH_LABELS,raw_output=ROOT/'results/raw_predictions'/f'densenet121_seed_{args.seed}.npz',metadata={'split':'test','seed':args.seed,'model':'DenseNet-121'}); lat=measure_latency(m,device,cfg['experiment']['image_size'],cfg['latency']['batch_size'],cfg['latency']['warmup_iterations'],cfg['latency']['timed_iterations']); row=agg|lat|{'seed':args.seed,'parameters':sum(p.numel() for p in m.parameters())}
out=ROOT/'results'/f'densenet121_seed_{args.seed}'; out.mkdir(exist_ok=True); torch.save(m.state_dict(),out/'model.pt'); pd.DataFrame(h).to_csv(out/'history.csv',index=False); pd.DataFrame(per).to_csv(out/'disease_wise.csv',index=False); pd.DataFrame([row]).to_csv(out/'aggregate.csv',index=False); print(json.dumps(row,indent=2))
