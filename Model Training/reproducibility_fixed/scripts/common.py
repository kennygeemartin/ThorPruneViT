import os, sys, yaml
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

def load_config(path):
    with open(path,'r',encoding='utf-8') as f: cfg=yaml.safe_load(f)
    nr=os.getenv('THORPRUNEVIT_NIH_ROOT')
    cr=os.getenv('THORPRUNEVIT_CHEXPERT_ROOT')
    if nr:
        cfg['nih']['image_root']=str(Path(nr)/'images')
        cfg['nih']['metadata_csv']=str(Path(nr)/'Data_Entry_2017.csv')
    if cr:
        cfg['chexpert']['root']=cr
        cfg['chexpert']['train_csv']=str(Path(cr)/'train.csv')
        cfg['chexpert']['valid_csv']=str(Path(cr)/'valid.csv')
    return cfg

def split_dir(seed): return ROOT/'splits'/f'seed_{seed}'

def require_split(seed):
    d=split_dir(seed); files={k:d/f'nih_{k}.csv' for k in ('train','val','test')}
    missing=[str(p) for p in files.values() if not p.exists()]
    if missing: raise SystemExit('Seed-specific split files missing: '+', '.join(missing)+'. Run validate_datasets.py first.')
    return files

def device_from(cfg):
    import torch
    d=cfg['experiment'].get('device','auto')
    if d=='auto': return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.device(d)
