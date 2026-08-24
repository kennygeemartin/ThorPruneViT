import os, sys, json, math, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
import torch
from torch.utils.data import TensorDataset, DataLoader
from thorprunevit.model import PrunableViT, count_parameters, approximate_flops
from thorprunevit.prune import iterative_prune
from thorprunevit.train import set_seed, evaluate, measure_latency

set_seed(42)
device=torch.device('cpu')
X=torch.randn(24,3,32,32)
y=(torch.rand(24,3)>0.65).float()
tr=DataLoader(TensorDataset(X[:16],y[:16]),batch_size=4,shuffle=False)
va=DataLoader(TensorDataset(X[16:20],y[16:20]),batch_size=4,shuffle=False)
te=DataLoader(TensorDataset(X[20:],y[20:]),batch_size=4,shuffle=False)
m=PrunableViT(image_size=32,patch_size=8,embed_dim=64,depth=2,heads_per_layer=[4,4],head_dim=16,mlp_dims=[128,128],num_classes=3,dropout=0.0)
base=count_parameters(m); basefl=approximate_flops(m,32)
crit=torch.nn.BCEWithLogitsLoss()
p,h=iterative_prune(m,tr,tr,va,device,crit,target_sparsity=0.53,steps=5,taylor_weight=0.5,magnitude_weight=0.5,finetune_epochs=1,finetune_lr=1e-4,max_score_batches=1)
final=count_parameters(p); finalfl=approximate_flops(p,32)
agg,per,_,_=evaluate(p,te,device,['a','b','c'])
report={
 'baseline_params':base,'final_params':final,'parameter_reduction':1-final/base,
 'baseline_flops':basefl,'final_flops':finalfl,'flops_reduction':1-finalfl/basefl,
 'stages':h,'metrics_keys':sorted(agg),'structure':p.active_structure()
}
assert final < base
assert len(h)==5
assert abs((1-final/base)-0.53) < 0.08, report
assert all(x>=1 for x in p.active_structure()['heads_per_layer'])
assert all(x>=1 for x in p.active_structure()['ffn_dims'])
(ROOT/'results').mkdir(exist_ok=True)
(ROOT/'results'/'smoke_test_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
