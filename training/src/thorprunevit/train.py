from __future__ import annotations
import json, random, time
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import multilabel_metrics, masked_multilabel_metrics


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, val_loader, device, epochs=50, lr=1e-4, weight_decay=0.05,
                pos_weight=None, cosine=True, patience=10, criterion=None):
    model.to(device)
    criterion = criterion or nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device) if pos_weight is not None else None)
    criterion=criterion.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9,0.999))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if cosine else None
    best = None; best_loss=float('inf'); stale=0; history=[]
    for ep in range(1, epochs+1):
        model.train(); tr_loss=0.0; n=0
        for x,y in train_loader:
            x=x.to(device); y=y.to(device).float(); opt.zero_grad(set_to_none=True)
            logits=model(x); loss=criterion(logits,y); loss.backward(); opt.step()
            tr_loss += loss.item()*x.shape[0]; n += x.shape[0]
        model.eval(); va=0.0; vn=0
        with torch.no_grad():
            for x,y in val_loader:
                x=x.to(device); y=y.to(device).float(); loss=criterion(model(x),y)
                va += loss.item()*x.shape[0]; vn += x.shape[0]
        if sched: sched.step()
        tr_loss/=max(n,1); va/=max(vn,1)
        history.append({'epoch':ep,'train_loss':tr_loss,'val_loss':va,'lr':opt.param_groups[0]['lr']})
        if va < best_loss - 1e-6:
            best_loss=va; stale=0; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience: break
    if best is not None: model.load_state_dict(best)
    return model, history


def evaluate(model, loader, device, label_names=None, threshold=0.5, raw_output=None, metadata=None):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader:
            x=x.to(device); logits=model(x); ps.append(torch.sigmoid(logits).cpu().numpy()); ys.append(y.numpy())
    y_true=np.concatenate(ys); y_prob=np.concatenate(ps)
    agg, per=multilabel_metrics(y_true,y_prob,threshold,label_names)
    if raw_output is not None:
        from pathlib import Path
        Path(raw_output).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(raw_output,y_true=y_true,y_prob=y_prob,y_pred=(y_prob>=threshold).astype(np.uint8),
                            threshold=threshold,metadata=json.dumps(metadata or {}))
    return agg, per, y_true, y_prob

def evaluate_masked(model, loader, device, label_names=None, threshold=0.5, raw_output=None, metadata=None):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x,y in loader: ps.append(torch.sigmoid(model(x.to(device))).cpu().numpy()); ys.append(y.numpy())
    y_true=np.concatenate(ys); y_prob=np.concatenate(ps); agg,per=masked_multilabel_metrics(y_true,y_prob,threshold,label_names)
    if raw_output is not None:
        Path(raw_output).parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(raw_output,y_true=y_true,y_prob=y_prob,y_pred=(y_prob>=threshold).astype(np.uint8),valid_mask=np.isfinite(y_true)&(y_true!=-1),metadata=json.dumps(metadata or {}))
    return agg,per,y_true,y_prob


def measure_latency(model, device, image_size=224, batch_size=1, warmup=20, iters=100):
    model.eval().to(device)
    x=torch.randn(batch_size,3,image_size,image_size,device=device)
    with torch.no_grad():
        for _ in range(warmup): _=model(x)
        if device.type=='cuda': torch.cuda.synchronize()
        times=[]
        for _ in range(iters):
            if device.type=='cuda': torch.cuda.synchronize()
            t0=time.perf_counter(); _=model(x)
            if device.type=='cuda': torch.cuda.synchronize()
            times.append((time.perf_counter()-t0)*1000)
    a=np.asarray(times)
    return {'latency_ms_mean':float(a.mean()),'latency_ms_sd':float(a.std(ddof=1)) if len(a)>1 else 0.0,
            'latency_ms_median':float(np.median(a)),'latency_ms_p95':float(np.percentile(a,95)),
            'batch_size':batch_size,'warmup':warmup,'iterations':iters,'device':str(device)}
