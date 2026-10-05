from __future__ import annotations
import copy
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableSelfAttention(nn.Module):
    """Multi-head self-attention with an explicit, shrinkable inner dimension.

    The input/output embedding dimension stays fixed, while the number of retained
    heads can be reduced. This allows *physical* structured pruning rather than
    merely zeroing masks.
    """
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.inner_dim = num_heads * head_dim
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(embed_dim, 3 * self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.last_head_output: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.view(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        h = attn @ v  # [B,H,N,D]
        # Retain the per-head activation for Taylor importance.
        if h.requires_grad:
            h.retain_grad()
        self.last_head_output = h
        h = h.transpose(1, 2).contiguous().view(b, n, self.inner_dim)
        return self.out_proj(h)


class PrunableMLP(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop2 = nn.Dropout(dropout)
        self.last_hidden: Optional[torch.Tensor] = None

    @property
    def hidden_dim(self) -> int:
        return self.fc1.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc1(x)
        h = self.act(h)
        if h.requires_grad:
            h.retain_grad()
        self.last_hidden = h
        h = self.drop1(h)
        h = self.fc2(h)
        return self.drop2(h)


class PrunableEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, mlp_dim: int,
                 dropout: float = 0.0, attention_dropout: float = 0.0, eps: float = 1e-6):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim, eps=eps)
        self.self_attention = PrunableSelfAttention(embed_dim, num_heads, head_dim, attention_dropout)
        self.dropout = nn.Dropout(dropout)
        self.ln_2 = nn.LayerNorm(embed_dim, eps=eps)
        self.mlp = PrunableMLP(embed_dim, mlp_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.self_attention(self.ln_1(x)))
        x = x + self.mlp(self.ln_2(x))
        return x


class PrunableViT(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 16, embed_dim: int = 768,
                 depth: int = 12, heads_per_layer: Optional[Sequence[int]] = None,
                 head_dim: int = 64, mlp_dims: Optional[Sequence[int]] = None,
                 num_classes: int = 14, dropout: float = 0.1, attention_dropout: float = 0.0):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.head_dim = head_dim
        if heads_per_layer is None:
            heads_per_layer = [embed_dim // head_dim] * depth
        if mlp_dims is None:
            mlp_dims = [embed_dim * 4] * depth
        self.heads_per_layer = list(heads_per_layer)
        self.mlp_dims = list(mlp_dims)
        self.original_head_indices = [list(range(n)) for n in heads_per_layer]
        self.original_ffn_indices = [list(range(n)) for n in mlp_dims]
        self.conv_proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        seq_length = (image_size // patch_size) ** 2 + 1
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, embed_dim).normal_(std=0.02))
        self.encoder_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            PrunableEncoderBlock(embed_dim, heads_per_layer[i], head_dim, mlp_dims[i], dropout, attention_dropout)
            for i in range(depth)
        ])
        self.encoder_ln = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.zeros_(self.head.bias)

    def _process_input(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._process_input(x)
        cls = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embedding[:, :x.shape[1]]
        x = self.encoder_dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.encoder_ln(x)
        return self.head(x[:, 0])

    def active_structure(self) -> dict:
        return {
            "heads_per_layer": [b.self_attention.num_heads for b in self.layers],
            "ffn_dims": [b.mlp.hidden_dim for b in self.layers],
        }


def make_vit_b16(num_classes: int = 14, pretrained: bool = True, dropout: float = 0.1) -> PrunableViT:
    target = PrunableViT(num_classes=num_classes, dropout=dropout)
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
    source = vit_b_16(weights=weights)
    load_from_torchvision_vit(target, source, num_classes=num_classes)
    return target

def save_pruned_model(model, checkpoint_dir, metadata=None, config=None):
    import json, yaml
    from pathlib import Path
    d=Path(checkpoint_dir); d.mkdir(parents=True,exist_ok=True)
    arch={'base_architecture':'ViT-B/16','image_size':model.image_size,'patch_size':model.patch_size,
          'embedding_dimension':model.embed_dim,'depth':model.depth,'head_dimension':model.head_dim,
          'heads_per_layer':model.active_structure()['heads_per_layer'],'ffn_dimensions':model.active_structure()['ffn_dims'],
          'surviving_head_indices':model.original_head_indices,'surviving_ffn_indices':model.original_ffn_indices,
          'number_of_labels':model.head.out_features}
    torch.save({'state_dict':model.state_dict(),'architecture':arch},d/'checkpoint.pt')
    (d/'architecture.json').write_text(json.dumps(arch,indent=2),encoding='utf-8')
    (d/'mask_metadata.json').write_text(json.dumps(metadata or {},indent=2),encoding='utf-8')
    (d/'config_snapshot.yaml').write_text(yaml.safe_dump(config or {},sort_keys=False),encoding='utf-8')

def load_pruned_model(checkpoint_dir, map_location='cpu'):
    from pathlib import Path
    ck=torch.load(Path(checkpoint_dir)/'checkpoint.pt',map_location=map_location,weights_only=False); a=ck['architecture']
    m=PrunableViT(image_size=a['image_size'],patch_size=a['patch_size'],embed_dim=a['embedding_dimension'],depth=a['depth'],
        heads_per_layer=a['heads_per_layer'],head_dim=a['head_dimension'],mlp_dims=a['ffn_dimensions'],num_classes=a['number_of_labels'])
    m.original_head_indices=a.get('surviving_head_indices',m.original_head_indices); m.original_ffn_indices=a.get('surviving_ffn_indices',m.original_ffn_indices)
    m.load_state_dict(ck['state_dict']); return m


def load_from_torchvision_vit(target: PrunableViT, source: nn.Module, num_classes: int = 14) -> None:
    """Load a torchvision ViT-B/16 into the explicit-head implementation."""
    with torch.no_grad():
        target.conv_proj.weight.copy_(source.conv_proj.weight)
        if source.conv_proj.bias is not None:
            target.conv_proj.bias.copy_(source.conv_proj.bias)
        target.class_token.copy_(source.class_token)
        target.pos_embedding.copy_(source.encoder.pos_embedding)
        target.encoder_ln.load_state_dict(source.encoder.ln.state_dict())
        for i, (tb, sb) in enumerate(zip(target.layers, source.encoder.layers)):
            tb.ln_1.load_state_dict(sb.ln_1.state_dict())
            tb.ln_2.load_state_dict(sb.ln_2.state_dict())
            mha = sb.self_attention
            E = mha.embed_dim
            H = mha.num_heads
            hd = E // H
            assert tb.self_attention.num_heads == H and tb.self_attention.head_dim == hd
            qkv_w = mha.in_proj_weight.reshape(3, H, hd, E).reshape(3 * E, E)
            qkv_b = mha.in_proj_bias.reshape(3, H, hd).reshape(3 * E)
            tb.self_attention.qkv.weight.copy_(qkv_w)
            tb.self_attention.qkv.bias.copy_(qkv_b)
            tb.self_attention.out_proj.weight.copy_(mha.out_proj.weight)
            tb.self_attention.out_proj.bias.copy_(mha.out_proj.bias)
            tb.mlp.fc1.weight.copy_(sb.mlp[0].weight)
            tb.mlp.fc1.bias.copy_(sb.mlp[0].bias)
            tb.mlp.fc2.weight.copy_(sb.mlp[3].weight)
            tb.mlp.fc2.bias.copy_(sb.mlp[3].bias)
        # Classification head intentionally reinitialized for the 14-label task.
        nn.init.trunc_normal_(target.head.weight, std=0.02)
        nn.init.zeros_(target.head.bias)


def clone_pruned(model: PrunableViT,
                 keep_heads: Sequence[Sequence[int]],
                 keep_ffn: Sequence[Sequence[int]]) -> PrunableViT:
    """Physically compact a PrunableViT by slicing selected heads and FFN neurons.

    keep_heads/keep_ffn indices are relative to the *current* model structure.
    """
    assert len(keep_heads) == model.depth == len(keep_ffn)
    new = PrunableViT(
        image_size=model.image_size,
        patch_size=model.patch_size,
        embed_dim=model.embed_dim,
        depth=model.depth,
        heads_per_layer=[len(k) for k in keep_heads],
        head_dim=model.head_dim,
        mlp_dims=[len(k) for k in keep_ffn],
        num_classes=model.head.out_features,
        dropout=model.encoder_dropout.p,
        attention_dropout=model.layers[0].self_attention.dropout.p if model.layers else 0.0,
    )
    device = next(model.parameters()).device
    new.to(device)
    new.original_head_indices=[[model.original_head_indices[li][i] for i in keep_heads[li]] for li in range(model.depth)]
    new.original_ffn_indices=[[model.original_ffn_indices[li][i] for i in keep_ffn[li]] for li in range(model.depth)]
    with torch.no_grad():
        new.conv_proj.load_state_dict(model.conv_proj.state_dict())
        new.class_token.copy_(model.class_token)
        new.pos_embedding.copy_(model.pos_embedding)
        new.encoder_ln.load_state_dict(model.encoder_ln.state_dict())
        new.head.load_state_dict(model.head.state_dict())
        for old_b, new_b, h_idx, f_idx in zip(model.layers, new.layers, keep_heads, keep_ffn):
            new_b.ln_1.load_state_dict(old_b.ln_1.state_dict())
            new_b.ln_2.load_state_dict(old_b.ln_2.state_dict())
            h_idx = torch.tensor(list(h_idx), dtype=torch.long, device=device)
            f_idx = torch.tensor(list(f_idx), dtype=torch.long, device=device)
            old_attn = old_b.self_attention
            H_old, hd, E = old_attn.num_heads, old_attn.head_dim, old_attn.embed_dim
            qkv_w = old_attn.qkv.weight.view(3, H_old, hd, E)[:, h_idx]
            qkv_b = old_attn.qkv.bias.view(3, H_old, hd)[:, h_idx]
            new_b.self_attention.qkv.weight.copy_(qkv_w.reshape(-1, E))
            new_b.self_attention.qkv.bias.copy_(qkv_b.reshape(-1))
            col_idx = (h_idx[:, None] * hd + torch.arange(hd, device=device)[None, :]).reshape(-1)
            new_b.self_attention.out_proj.weight.copy_(old_attn.out_proj.weight[:, col_idx])
            new_b.self_attention.out_proj.bias.copy_(old_attn.out_proj.bias)
            new_b.mlp.fc1.weight.copy_(old_b.mlp.fc1.weight[f_idx])
            new_b.mlp.fc1.bias.copy_(old_b.mlp.fc1.bias[f_idx])
            new_b.mlp.fc2.weight.copy_(old_b.mlp.fc2.weight[:, f_idx])
            new_b.mlp.fc2.bias.copy_(old_b.mlp.fc2.bias)
    return new


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def approximate_flops(model: PrunableViT, image_size: Optional[int] = None) -> int:
    """Approximate multiply-add FLOPs for a single forward pass.

    This is a transparent architecture-level estimator intended for consistent
    baseline/pruned comparisons, not a hardware-specific profiler.
    """
    S = (image_size or model.image_size) // model.patch_size
    N = S * S + 1
    E = model.embed_dim
    flops = 0
    # Patch projection conv.
    flops += S * S * E * 3 * model.patch_size * model.patch_size
    for b in model.layers:
        H = b.self_attention.num_heads
        I = H * b.self_attention.head_dim
        M = b.mlp.hidden_dim
        # qkv + attention score/value + output projection
        flops += N * E * (3 * I)
        flops += H * N * N * b.self_attention.head_dim * 2
        flops += N * I * E
        # FFN two linears
        flops += N * E * M + N * M * E
    flops += E * model.head.out_features
    return int(flops)
