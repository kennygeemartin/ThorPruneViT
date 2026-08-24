import importlib.util, json
from pathlib import Path
from common import ROOT, load_config
def status_module(name): return 'READY' if importlib.util.find_spec(name) else 'BLOCKED'
cfg=load_config(ROOT/'configs/config.yaml'); rows={x:status_module(x) for x in ['torch','torchvision','thop','scipy','docx']}
try:
 import torch; rows['CUDA']='READY' if torch.cuda.is_available() else 'BLOCKED'
except Exception: rows['CUDA']='BLOCKED'
rows['NIH paths']='READY' if Path(cfg['nih']['metadata_csv']).is_file() and Path(cfg['nih']['image_root']).is_dir() else 'BLOCKED'
rows['CheXpert paths']='READY' if Path(cfg['chexpert']['train_csv']).is_file() and Path(cfg['chexpert']['valid_csv']).is_file() and Path(cfg['chexpert']['root']).is_dir() else 'BLOCKED'
for k,v in rows.items(): print(f'{k}: {v}')
raise SystemExit(0 if all(v=='READY' for v in rows.values()) else 2)
