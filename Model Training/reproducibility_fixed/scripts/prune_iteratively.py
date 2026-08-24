import argparse, json, copy
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader
from common import load_config, device_from, ROOT, require_split
from thorprunevit.data import NIHChestXray14, NIH_LABELS, pos_weight_from_dataframe
from thorprunevit.model import make_vit_b16, count_parameters, approximate_flops, save_pruned_model
from thorprunevit.train import set_seed, evaluate, measure_latency
from thorprunevit.prune import iterative_prune

ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--target',type=float,default=None); ap.add_argument('--taylor-weight',type=float,default=None); ap.add_argument('--magnitude-weight',type=float,default=None); ap.add_argument('--heads-only',action='store_true'); ap.add_argument('--ffn-only',action='store_true'); args=ap.parse_args()
cfg=load_config(args.config); device=device_from(cfg); set_seed(args.seed)
sp=require_split(args.seed); tr=pd.read_csv(sp['train']); va=pd.read_csv(sp['val']); te=pd.read_csv(sp['test'])
bs=cfg['training']['batch_size']; nw=cfg['training']['num_workers']; root=cfg['nih']['image_root']
trdl=DataLoader(NIHChestXray14(tr,root,True),batch_size=bs,shuffle=True,num_workers=nw); vadl=DataLoader(NIHChestXray14(va,root,False),batch_size=bs,shuffle=False,num_workers=nw); tedl=DataLoader(NIHChestXray14(te,root,False),batch_size=bs,shuffle=False,num_workers=nw)
pw=torch.tensor(pd.read_csv(sp['train'].parent/'positive_weights.csv')['pos_weight'].values,dtype=torch.float32); crit=torch.nn.BCEWithLogitsLoss(pos_weight=pw.to(device))
model=make_vit_b16(14,False,cfg['training']['dropout']); state=torch.load(ROOT/'results'/f'baseline_seed_{args.seed}'/'model.pt',map_location='cpu'); model.load_state_dict(state); model.to(device)
tw=args.taylor_weight if args.taylor_weight is not None else 0.5; mw=args.magnitude_weight if args.magnitude_weight is not None else 1-tw; target=cfg['pruning']['target_sparsity'] if args.target is None else args.target
alpha_file=ROOT/'results/alpha_selection'/f'seed_{args.seed}.csv'
if args.taylor_weight is None:
    candidates=[]
    for alpha in cfg['pruning']['alpha_grid']:
        candidate,_=iterative_prune(copy.deepcopy(model),trdl,trdl,vadl,device,crit,target,cfg['pruning']['steps'],alpha,1-alpha,not args.ffn_only,not args.heads_only,cfg['pruning']['finetune_epochs_per_step'],cfg['pruning']['finetune_learning_rate'],pw,cfg['pruning']['score_batches'])
        vm,_,_,_=evaluate(candidate,vadl,device,NIH_LABELS); candidates.append({'alpha':alpha,'validation_metric':vm[cfg['pruning']['selection_metric']]})
    best=max(candidates,key=lambda x:(x['validation_metric'],x['alpha'])); tw=best['alpha']; mw=1-tw
    alpha_file.parent.mkdir(parents=True,exist_ok=True)
    for x in candidates: x['selected']=x['alpha']==tw
    pd.DataFrame(candidates).to_csv(alpha_file,index=False)
imp_dir=ROOT/'results/importance'/f'seed_{args.seed}'
pruned,hist=iterative_prune(model,trdl,trdl,vadl,device,crit,target,cfg['pruning']['steps'],tw,mw,not args.ffn_only,not args.heads_only,cfg['pruning']['finetune_epochs_per_step'],cfg['pruning']['finetune_learning_rate'],pw,cfg['pruning']['score_batches'],imp_dir)
tag=f"target_{target:.2f}_tw_{tw:.2f}_mw_{mw:.2f}" + ('_heads' if args.heads_only else '_ffn' if args.ffn_only else '_joint')
out=ROOT/'results'/f'pruned_seed_{args.seed}_{tag}'; out.mkdir(exist_ok=True)
agg,per,_,_=evaluate(pruned,tedl,device,NIH_LABELS,raw_output=ROOT/'results/raw_predictions'/f'{tag}_seed_{args.seed}.npz',metadata={'split':'test','seed':args.seed,'model':'pruned_vit','target':target}); lat=measure_latency(pruned,device,cfg['experiment']['image_size'],**{'batch_size':cfg['latency']['batch_size'],'warmup':cfg['latency']['warmup_iterations'],'iters':cfg['latency']['timed_iterations']})
mask_dir=ROOT/'masks'/tag/f'seed_{args.seed}'; mask_dir.mkdir(parents=True,exist_ok=True)
pd.DataFrame([{'layer':li,'original_head':i,'active':int(i in keep)} for li,keep in enumerate(pruned.original_head_indices) for i in range(12)]).to_csv(mask_dir/'head_masks.csv',index=False)
pd.DataFrame([{'layer':li,'original_neuron':i,'active':int(i in keep)} for li,keep in enumerate(pruned.original_ffn_indices) for i in range(3072)]).to_csv(mask_dir/'ffn_masks.csv',index=False)
pd.DataFrame([{'active_heads':sum(map(len,pruned.original_head_indices)),'total_heads':144,'active_ffn':sum(map(len,pruned.original_ffn_indices)),'total_ffn':36864}]).to_csv(mask_dir/'mask_summary.csv',index=False)
save_pruned_model(pruned,out,{'target':target,'taylor_weight':tw,'magnitude_weight':mw,'mask_directory':str(mask_dir)},cfg)
(mask_dir/'architecture.json').write_text((out/'architecture.json').read_text(encoding='utf-8'),encoding='utf-8')
sched=ROOT/'results/pruning_schedule'; sched.mkdir(parents=True,exist_ok=True); pd.DataFrame(hist).to_csv(sched/f'seed_{args.seed}.csv',index=False); pd.DataFrame(hist).to_csv(out/'pruning_history.csv',index=False); pd.DataFrame(per).to_csv(out/'disease_wise.csv',index=False); row=agg|lat|{'seed':args.seed,'parameters':count_parameters(pruned),'approx_flops':approximate_flops(pruned),'target_sparsity':target,'taylor_weight':tw,'magnitude_weight':mw}; pd.DataFrame([row]).to_csv(out/'aggregate.csv',index=False); print(json.dumps(row,indent=2))
