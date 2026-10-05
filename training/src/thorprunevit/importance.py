from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .model import PrunableViT


@dataclass
class LayerImportance:
    head_taylor: torch.Tensor
    head_magnitude: torch.Tensor
    ffn_taylor: torch.Tensor
    ffn_magnitude: torch.Tensor


def _minmax(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    if x.numel() <= 1:
        return torch.ones_like(x)
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + eps)


def collect_importance(model: PrunableViT, loader, criterion,
                       device: torch.device, max_batches: int = 8) -> List[LayerImportance]:
    model.train(False)
    sums = []
    counts = []
    for b in model.layers:
        sums.append({
            'head_t': torch.zeros(b.self_attention.num_heads, device=device),
            'ffn_t': torch.zeros(b.mlp.hidden_dim, device=device),
        })
        counts.append(0)

    for bi, batch in enumerate(loader):
        if bi >= max_batches:
            break
        images, targets = batch[:2]
        images = images.to(device)
        targets = targets.to(device).float()
        model.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        for li, b in enumerate(model.layers):
            h = b.self_attention.last_head_output
            f = b.mlp.last_hidden
            if h is None or h.grad is None or f is None or f.grad is None:
                raise RuntimeError('Activation gradients were not retained; importance cannot be computed.')
            # Explicit element-wise Taylor sensitivity |activation ⊙ gradient|.
            hs = (h.detach().abs() * h.grad.detach().abs()).mean(dim=(0, 2, 3))
            fs = (f.detach().abs() * f.grad.detach().abs()).mean(dim=(0, 1))
            sums[li]['head_t'] += hs
            sums[li]['ffn_t'] += fs
            counts[li] += 1

    out: List[LayerImportance] = []
    for li, b in enumerate(model.layers):
        c = max(counts[li], 1)
        ht = sums[li]['head_t'] / c
        ft = sums[li]['ffn_t'] / c
        # Structural magnitude scores from the parameters tied to each unit.
        a = b.self_attention
        H, hd, E = a.num_heads, a.head_dim, a.embed_dim
        qkv = a.qkv.weight.detach().view(3, H, hd, E).abs().mean(dim=(0, 2, 3))
        outw = a.out_proj.weight.detach().view(E, H, hd).abs().mean(dim=(0, 2))
        hm = 0.5 * (qkv + outw)
        fm = 0.5 * (
            b.mlp.fc1.weight.detach().abs().mean(dim=1) +
            b.mlp.fc2.weight.detach().abs().mean(dim=0)
        )
        out.append(LayerImportance(ht, hm, ft, fm))
    return out


def combined_scores(importance: List[LayerImportance], taylor_weight: float = 0.5,
                    magnitude_weight: float = 0.5,
                    component: str = 'both') -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    if abs(taylor_weight + magnitude_weight - 1.0) > 1e-6:
        raise ValueError('Weights must sum to 1.')
    head_scores, ffn_scores = [], []
    # Normalize globally within each component type to make the weighting operational.
    all_ht = torch.cat([x.head_taylor for x in importance])
    all_hm = torch.cat([x.head_magnitude for x in importance])
    all_ft = torch.cat([x.ffn_taylor for x in importance])
    all_fm = torch.cat([x.ffn_magnitude for x in importance])
    ht_n, hm_n, ft_n, fm_n = _minmax(all_ht), _minmax(all_hm), _minmax(all_ft), _minmax(all_fm)
    h_off = f_off = 0
    for x in importance:
        nh, nf = x.head_taylor.numel(), x.ffn_taylor.numel()
        hs = taylor_weight * ht_n[h_off:h_off+nh] + magnitude_weight * hm_n[h_off:h_off+nh]
        fs = taylor_weight * ft_n[f_off:f_off+nf] + magnitude_weight * fm_n[f_off:f_off+nf]
        head_scores.append(hs)
        ffn_scores.append(fs)
        h_off += nh; f_off += nf
    return head_scores, ffn_scores


def select_keep_indices(scores: List[torch.Tensor], prune_fraction_of_current: float,
                        min_keep_per_layer: int = 1) -> List[List[int]]:
    """Globally prune the lowest scoring units while preserving a minimum per layer."""
    total = sum(s.numel() for s in scores)
    target_prune = int(round(total * prune_fraction_of_current))
    candidates = []
    for li, s in enumerate(scores):
        for ui, val in enumerate(s.detach().cpu().tolist()):
            candidates.append((val, li, ui))
    candidates.sort(key=lambda x: x[0])
    keep = [set(range(s.numel())) for s in scores]
    pruned = 0
    for _, li, ui in candidates:
        if pruned >= target_prune:
            break
        if len(keep[li]) <= min_keep_per_layer:
            continue
        if ui in keep[li]:
            keep[li].remove(ui)
            pruned += 1
    return [sorted(x) for x in keep]
