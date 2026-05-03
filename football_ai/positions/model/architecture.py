from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SetAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ff_expansion: int = 2) -> None:
        super().__init__()
        ff_dim = int(embed_dim) * int(ff_expansion)
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), ff_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, int(embed_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class PoolingMultiheadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float, ff_expansion: int = 2) -> None:
        super().__init__()
        ff_dim = int(embed_dim) * int(ff_expansion)
        self.seed = nn.Parameter(torch.randn(1, 1, int(embed_dim)))
        self.attn = nn.MultiheadAttention(
            embed_dim=int(embed_dim),
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(int(embed_dim))
        self.norm2 = nn.LayerNorm(int(embed_dim))
        self.ff = nn.Sequential(
            nn.Linear(int(embed_dim), ff_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ff_dim, int(embed_dim)),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        seeds = self.seed.expand(x.shape[0], -1, -1)
        attn_out, _ = self.attn(
            seeds,
            x,
            x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        pooled = self.norm1(seeds + attn_out)
        pooled = self.norm2(pooled + self.ff(pooled))
        return pooled


class RoleSetTransformer(nn.Module):
    def __init__(
        self,
        objective_dim: int,
        teammate_dim: int,
        num_classes: int,
        config,
    ) -> None:
        super().__init__()
        if int(config.objective_num_layers) < 1:
            raise ValueError("objective_num_layers debe ser >= 1.")
        objective_hidden_dims = tuple(
            int(config.objective_hidden_dim)
            for _ in range(int(config.objective_num_layers))
        )
        self.objective_encoder = MLP(
            input_dim=int(objective_dim),
            hidden_dims=objective_hidden_dims,
            output_dim=int(config.set_hidden_dim),
            dropout=float(config.dropout),
        )
        self.teammate_encoder = MLP(
            input_dim=int(teammate_dim),
            hidden_dims=(int(config.teammate_embed_dim),),
            output_dim=int(config.set_hidden_dim),
            dropout=float(config.dropout),
        )
        self.set_blocks = nn.ModuleList(
            [
                SetAttentionBlock(
                    embed_dim=int(config.set_hidden_dim),
                    num_heads=int(config.num_heads),
                    dropout=float(config.dropout),
                    ff_expansion=int(config.ff_expansion),
                )
                for _ in range(int(config.num_set_blocks))
            ]
        )
        self.pool = PoolingMultiheadAttention(
            embed_dim=int(config.set_hidden_dim),
            num_heads=int(config.num_heads),
            dropout=float(config.dropout),
            ff_expansion=int(config.ff_expansion),
        )
        self.classifier = nn.Sequential(
            nn.Linear(int(config.set_hidden_dim) * 2, int(config.fusion_hidden_dim)),
            nn.GELU(),
            nn.Dropout(float(config.dropout)),
            nn.Linear(int(config.fusion_hidden_dim), int(num_classes)),
        )

    def forward(
        self,
        objective: torch.Tensor,
        teammates: torch.Tensor,
        teammate_mask: torch.Tensor,
    ) -> torch.Tensor:
        h_obj = self.objective_encoder(objective)
        z = self.teammate_encoder(teammates)

        safe_mask = teammate_mask.clone()
        empty_rows = ~safe_mask.any(dim=1)
        if empty_rows.any():
            safe_mask[empty_rows, 0] = True
            z = z.clone()
            z[empty_rows, 0, :] = 0.0

        key_padding_mask = ~safe_mask
        for block in self.set_blocks:
            z = block(z, key_padding_mask=key_padding_mask)

        h_set = self.pool(z, key_padding_mask=key_padding_mask).squeeze(1)
        logits = self.classifier(torch.cat([h_obj, h_set], dim=-1))
        return logits


__all__ = [
    "MLP",
    "PoolingMultiheadAttention",
    "RoleSetTransformer",
    "SetAttentionBlock",
]
