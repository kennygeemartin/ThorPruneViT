import argparse, csv, sys
from pathlib import Path
import torch
from common import ROOT, load_config
from thorprunevit.model import load_pruned_model
from thorprunevit.train import measure_latency

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(ROOT/'configs/config.yaml')); ap.add_argument('--device',default='cuda'); ap.add_argument('--model',action='append',default=[]); a=ap.parse_args()
    if a.device!='cuda' or not torch.cuda.is_available():
        print('GPU LATENCY BLOCKED - CUDA DEVICE REQUIRED'); return 2
    cfg=load_config(a.config); rows=[]
    for spec in a.model:
        name,path=spec.split('=',1); model=load_pruned_model(path)
        torch.cuda.reset_peak_memory_stats(); r=measure_latency(model,torch.device('cuda'),cfg['experiment']['image_size'],cfg['latency']['batch_size'],cfg['latency']['warmup_iterations'],cfg['latency']['timed_iterations'])
        r.update(model=name,device_name=torch.cuda.get_device_name(0),vram_bytes=torch.cuda.get_device_properties(0).total_memory,
          cuda_version=torch.version.cuda,pytorch_version=torch.__version__,input_size=cfg['experiment']['image_size'],precision='float32',peak_allocated_bytes=torch.cuda.max_memory_allocated())
        rows.append(r)
    out=ROOT/'results/gpu_latency.csv'; out.parent.mkdir(exist_ok=True)
    if rows:
        with out.open('w',newline='',encoding='utf-8') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    return 0
if __name__=='__main__': sys.exit(main())
