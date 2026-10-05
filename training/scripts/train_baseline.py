import argparse, json
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
from common import load_config, device_from, ROOT, require_split
from thorprunevit.data import NIHChestXray14, NIH_LABELS
from thorprunevit.model import make_vit_b16, count_parameters, approximate_flops
from thorprunevit.train import set_seed, train_model, evaluate

ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
cfg=load_config(args.config); device=device_from(cfg); set_seed(args.seed)
sp=require_split(args.seed); tr=pd.read_csv(sp['train']); va=pd.read_csv(sp['val']); te=pd.read_csv(sp['test']); bs=cfg['training']['batch_size']; nw=cfg['training']['num_workers']
train_ds=NIHChestXray14(tr,cfg['nih']['image_root'],True,cfg['experiment']['image_size']); val_ds=NIHChestXray14(va,cfg['nih']['image_root'],False,cfg['experiment']['image_size']); test_ds=NIHChestXray14(te,cfg['nih']['image_root'],False,cfg['experiment']['image_size'])
train_dl=DataLoader(train_ds,batch_size=bs,shuffle=True,num_workers=nw,pin_memory=device.type=='cuda'); val_dl=DataLoader(val_ds,batch_size=bs,shuffle=False,num_workers=nw); test_dl=DataLoader(test_ds,batch_size=bs,shuffle=False,num_workers=nw)
pw=torch.tensor(pd.read_csv(sp['train'].parent/'positive_weights.csv')['pos_weight'].values,dtype=torch.float32)
model=make_vit_b16(14,cfg['training']['pretrained_imagenet'],cfg['training']['dropout'])
model,hist=train_model(model,train_dl,val_dl,device,epochs=cfg['training']['epochs'],lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay'],pos_weight=pw,patience=cfg['training']['early_stopping_patience'])
out=ROOT/'results'/f'baseline_seed_{args.seed}'; out.mkdir(exist_ok=True)
agg,per,yt,yp=evaluate(model,test_dl,device,NIH_LABELS,raw_output=ROOT/'results/raw_predictions'/f'baseline_vit_seed_{args.seed}.npz',metadata={'split':'test','seed':args.seed,'model':'ViT-B/16','target':0})
torch.save(model.state_dict(),out/'model.pt'); pd.DataFrame(hist).to_csv(out/'history.csv',index=False); pd.DataFrame(per).to_csv(out/'disease_wise.csv',index=False); pd.DataFrame([agg|{'seed':args.seed,'parameters':count_parameters(model),'approx_flops':approximate_flops(model)}]).to_csv(out/'aggregate.csv',index=False)
print(json.dumps(agg,indent=2))
