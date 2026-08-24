import argparse, subprocess, sys
from pathlib import Path
from common import load_config, ROOT
ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); args=ap.parse_args(); cfg=load_config(args.config)
for seed in cfg['experiment']['seeds']:
    variants=[[],['--heads-only'],['--ffn-only'],['--taylor-weight','1','--magnitude-weight','0'],['--taylor-weight','0','--magnitude-weight','1']]
    for v in variants:
        subprocess.run([sys.executable,str(ROOT/'scripts/prune_iteratively.py'),'--config',args.config,'--seed',str(seed),*v],check=True)
    import pandas as pd
    rows=[]
    for d in (ROOT/'results').glob(f'*seed_{seed}*'):
        f=d/'aggregate.csv'
        if f.exists(): r=pd.read_csv(f).iloc[0].to_dict(); r['experiment']=d.name; rows.append(r)
    out=ROOT/'results/ablations'; out.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(out/f'ablation_seed_{seed}.csv',index=False)
