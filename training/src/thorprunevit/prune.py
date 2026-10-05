from __future__ import annotations
from typing import List
import torch

from .importance import collect_importance, combined_scores
from .model import PrunableViT, clone_pruned, count_parameters, approximate_flops
from .train import train_model, evaluate


def _unit_costs(model: PrunableViT):
    """Exact parameter counts removed by deleting one structured unit.

    Attention-head cost excludes the shared output bias; FFN-neuron cost excludes
    the shared fc2 bias. Those shared terms remain in the compact model.
    """
    E=model.embed_dim; hd=model.head_dim
    head_cost=3*hd*E + 3*hd + E*hd
    ffn_cost=E + 1 + E
    return head_cost, ffn_cost


def _select_to_param_budget(model: PrunableViT, head_scores, ffn_scores,
                            params_to_remove: int, prune_heads=True, prune_ffn=True,
                            min_heads_per_layer=1, min_ffn_per_layer=1):
    head_cost, ffn_cost=_unit_costs(model)
    keep_h=[set(range(s.numel())) for s in head_scores]
    keep_f=[set(range(s.numel())) for s in ffn_scores]
    candidates=[]
    if prune_heads:
        for li,s in enumerate(head_scores):
            for ui,val in enumerate(s.detach().cpu().tolist()):
                candidates.append((float(val), 'h', li, ui, head_cost))
    if prune_ffn:
        for li,s in enumerate(ffn_scores):
            for ui,val in enumerate(s.detach().cpu().tolist()):
                candidates.append((float(val), 'f', li, ui, ffn_cost))
    candidates.sort(key=lambda x:x[0])
    removed=0
    for _,kind,li,ui,cost in candidates:
        if removed >= params_to_remove: break
        if kind=='h':
            if len(keep_h[li]) <= min_heads_per_layer: continue
            if ui in keep_h[li]: keep_h[li].remove(ui); removed += cost
        else:
            if len(keep_f[li]) <= min_ffn_per_layer: continue
            if ui in keep_f[li]: keep_f[li].remove(ui); removed += cost
    return [sorted(x) for x in keep_h],[sorted(x) for x in keep_f],removed


def iterative_prune(model: PrunableViT, score_loader, finetune_loader, val_loader, device,
                    criterion, target_sparsity: float=0.53, steps: int=5,
                    taylor_weight: float=0.5, magnitude_weight: float=0.5,
                    prune_heads: bool=True, prune_ffn: bool=True,
                    finetune_epochs: int=5, finetune_lr: float=1e-5,
                    pos_weight=None, max_score_batches: int=8, artifacts_dir=None):
    """Iterative compact structured pruning to a total-parameter sparsity target.

    For a 53% target and five stages, the cumulative parameter-reduction targets
    are 10.6%, 21.2%, 31.8%, 42.4%, and 53.0% of the *original total parameter
    count*. At each stage the lowest-scoring structured units are physically
    removed until the cumulative parameter budget is reached. This makes the
    reviewer's 53%/five-step requirement mathematically explicit and avoids the
    incorrect interpretation that repeatedly removing 10.6% of the remaining
    units equals 53%.
    """
    current=model
    original_params=count_parameters(model)
    original_heads=sum(b.self_attention.num_heads for b in model.layers)
    original_ffn=sum(b.mlp.hidden_dim for b in model.layers)
    history=[]
    for stage in range(1,steps+1):
        imp=collect_importance(current, score_loader, criterion, device, max_batches=max_score_batches)
        if artifacts_dir:
            from pathlib import Path
            import pandas as pd
            d=Path(artifacts_dir); d.mkdir(parents=True,exist_ok=True)
            hr=[]; fr=[]
            for li,x in enumerate(imp):
                hr += [{'stage':stage,'layer':li,'head':i,'taylor':float(t),'magnitude':float(m)} for i,(t,m) in enumerate(zip(x.head_taylor.cpu(),x.head_magnitude.cpu()))]
                fr += [{'stage':stage,'layer':li,'neuron':i,'taylor':float(t),'magnitude':float(m)} for i,(t,m) in enumerate(zip(x.ffn_taylor.cpu(),x.ffn_magnitude.cpu()))]
            pd.DataFrame(hr).to_csv(d/f'head_importance_stage_{stage}.csv',index=False); pd.DataFrame(fr).to_csv(d/f'ffn_importance_stage_{stage}.csv',index=False)
        hs,fs=combined_scores(imp,taylor_weight,magnitude_weight)
        desired_cum=target_sparsity*stage/steps
        desired_params=int(round(original_params*(1-desired_cum)))
        current_params=count_parameters(current)
        need=max(0,current_params-desired_params)
        keep_h,keep_f,removed=_select_to_param_budget(current,hs,fs,need,prune_heads,prune_ffn)
        current=clone_pruned(current,keep_h,keep_f)
        current,_=train_model(current,finetune_loader,val_loader,device,epochs=finetune_epochs,lr=finetune_lr,
                              weight_decay=0.0,pos_weight=pos_weight,cosine=True,patience=finetune_epochs)
        cur_h=sum(b.self_attention.num_heads for b in current.layers); cur_f=sum(b.mlp.hidden_dim for b in current.layers)
        cur_params=count_parameters(current)
        va_metrics,_,_,_=evaluate(current,val_loader,device)
        history.append({
            'stage':stage,
            'target_cumulative_parameter_sparsity':desired_cum,
            'actual_parameter_sparsity':1-cur_params/original_params,
            'actual_head_sparsity':1-cur_h/original_heads,
            'actual_ffn_sparsity':1-cur_f/original_ffn,
            'heads_remaining':cur_h,
            'ffn_neurons_remaining':cur_f,
            'parameters_remaining':cur_params,
            'FLOPs':approximate_flops(current),
            'validation_macro_AUROC':va_metrics['macro_auroc'],
            'validation_mean_per_class_accuracy':va_metrics['mean_per_class_accuracy'],
        })
    return current, history
