import argparse, subprocess, sys
from common import load_config, ROOT
ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); args=ap.parse_args(); cfg=load_config(args.config)
for seed in cfg['experiment']['seeds']:
    for s in cfg['sensitivity']['sparsities']:
        if s==0: continue
        subprocess.run([sys.executable,str(ROOT/'scripts/prune_iteratively.py'),'--config',args.config,'--seed',str(seed),'--target',str(s)],check=True)
    import pandas as pd
    rows=[]
    base=ROOT/'results'/f'baseline_seed_{seed}'/'aggregate.csv'
    if base.exists(): r=pd.read_csv(base).iloc[0].to_dict(); r['target_sparsity']=0.0; rows.append(r)
    for d in (ROOT/'results').glob(f'pruned_seed_{seed}_target_*_joint'):
        f=d/'aggregate.csv'
        if f.exists(): rows.append(pd.read_csv(f).iloc[0].to_dict())
    out=ROOT/'results/sensitivity'; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/f'sensitivity_seed_{seed}.csv',index=False)
