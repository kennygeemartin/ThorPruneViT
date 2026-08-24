import argparse, json
from pathlib import Path
import torch
from common import ROOT
from thorprunevit.model import load_pruned_model
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',action='append',default=[],help='NAME=CHECKPOINT_DIR'); ap.add_argument('--image-size',type=int,default=224); a=ap.parse_args(); rows=[]
    try: from thop import profile
    except ImportError: raise SystemExit('thop is required: install requirements.txt')
    for spec in a.model:
        name,path=spec.split('=',1); m=load_pruned_model(path).eval(); x=torch.zeros(1,3,a.image_size,a.image_size); macs,params=profile(m,inputs=(x,),verbose=False)
        cp=Path(path)/'checkpoint.pt'; rows.append({'model':name,'parameters':int(params),'trainable_parameters':sum(p.numel() for p in m.parameters() if p.requires_grad),
          'serialized_model_bytes':cp.stat().st_size,'MACs':int(macs),'FLOPs':int(2*macs),'flop_convention':'2 FLOPs per MAC (thop)'})
    import pandas as pd; out=ROOT/'results/architecture_efficiency.csv'; out.parent.mkdir(exist_ok=True); pd.DataFrame(rows).to_csv(out,index=False)
if __name__=='__main__': main()
