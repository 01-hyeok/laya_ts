from __future__ import annotations

import math
import random
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LayaModelConfig
from .losses import SIGReg


def normalize_channel_mixer_type(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "mixer": "mixer",
        "query": "mixer",
        "query_mixer": "mixer",
        "independent": "independent",
        "ci": "independent",
        "ci_adapter": "independent",
        "channel_independent": "independent",
        "none": "independent",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported channel_mixer_type: {value}")
    return aliases[normalized]


def normalize_channel_metadata_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "coordinates": "coordinates",
        "position": "coordinates",
        "positions": "coordinates",
        "onehot": "onehot",
        "one_hot": "onehot",
        "channel_id": "onehot",
        "channel_ids": "onehot",
        "text": "text",
        "description": "text",
        "descriptions": "text",
        "stats": "stats",
        "statistics": "stats",
        "statistical": "stats",
        "summary_stats": "stats",
        "text_stats_avg": "text_stats_avg",
        "text_stats": "text_stats_avg",
        "description_stats": "text_stats_avg",
        "text_stats_joint": "text_stats_joint",
        "description_stats_joint": "text_stats_joint",
        "stats_description_joint": "text_stats_joint",
        "none": "none",
        "no_channel_info": "none",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported channel_metadata_mode: {value}")
    return aliases[normalized]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    even = x[..., ::2]
    odd = x[..., 1::2]
    return torch.stack((-odd, even), dim=-1).flatten(start_dim=-2)


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError(f"RoPE head_dim must be even, got {head_dim}")

    if position_ids is None:
        position = torch.arange(seq_len, device=device, dtype=torch.float32)
    else:
        if position_ids.dim() not in {1, 2} or position_ids.shape[-1] != seq_len:
            raise ValueError(
                f"Expected position_ids [N] or [B, N] with N={seq_len}, got {tuple(position_ids.shape)}"
            )
        position = position_ids.to(device=device, dtype=torch.float32)

    inv_freq = 1.0 / (
        10000
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                device=device,
                dtype=torch.float32,
            )
            / head_dim
        )
    )

    freqs = position.unsqueeze(-1) * inv_freq
    emb = torch.repeat_interleave(freqs, 2, dim=-1)

    return emb.cos().to(dtype=dtype), emb.sin().to(dtype=dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    else:
        raise ValueError(f"Expected RoPE cache [N, D] or [B, N, D], got {tuple(cos.shape)}")
    return (x * cos) + (rotate_half(x) * sin)


class FourierChannelEncoding(nn.Module):
    """Fixed Fourier electrode-coordinate encoding.

    Laya/LUNA-style flow:
        channel_positions [B, C, 3]
        -> fixed Fourier features
        -> added to per-channel patch tokens before channel cross-attention

    This module has no learnable parameters.
    """

    def __init__(
        self,
        out_dim: int,
        num_bands: int = 5,
        include_raw_xyz: bool = True,
    ) -> None:
        super().__init__()

        self.out_dim = out_dim
        self.num_bands = num_bands
        self.include_raw_xyz = include_raw_xyz

        feature_width = (3 if include_raw_xyz else 0) + (3 * 2 * num_bands)
        if feature_width > out_dim:
            raise ValueError(
                "FourierChannelEncoding width exceeds out_dim at initialization: "
                f"feature_width={feature_width}, out_dim={out_dim}. "
                "Reduce fourier_num_bands or increase channel_mixer_dim."
            )

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        if positions.dim() != 3 or positions.shape[-1] != 3:
            raise ValueError(
                f"Expected [B, C, 3] channel positions, got {tuple(positions.shape)}"
            )

        device = positions.device
        dtype = positions.dtype

        freqs = (
            2.0
            ** torch.arange(
                self.num_bands,
                device=device,
                dtype=dtype,
            )
            * math.pi
        )

        scaled = positions.unsqueeze(-1) * freqs
        fourier = torch.cat(
            [scaled.sin(), scaled.cos()],
            dim=-1,
        ).flatten(start_dim=-2)

        pieces = [fourier]

        if self.include_raw_xyz:
            pieces.insert(0, positions)

        encoded = torch.cat(pieces, dim=-1)

        if encoded.shape[-1] < self.out_dim:
            encoded = F.pad(encoded, (0, self.out_dim - encoded.shape[-1]))
        elif encoded.shape[-1] > self.out_dim:
            raise ValueError(
                "FourierChannelEncoding width exceeds out_dim: "
                f"got {encoded.shape[-1]} features for out_dim={self.out_dim}. "
                "Reduce fourier_num_bands or increase channel_mixer_dim instead of silently truncating coordinates."
            )

        return encoded


class TemporalPatchEmbedding(nn.Module):
    """Shared single-channel temporal patch embedder."""

    def __init__(self, patch_size: int, out_dim: int) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.proj = nn.Conv1d(
            1,
            out_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")

        batch, channels, time = x.shape

        pad = (-time) % self.patch_size

        if pad > 0:
            x = F.pad(x, (0, pad))

        flat = x.reshape(batch * channels, 1, x.shape[-1])
        tokens = self.proj(flat).transpose(1, 2)

        return tokens.reshape(
            batch,
            channels,
            tokens.shape[1],
            tokens.shape[2],
        )


class QueryChannelMixer(nn.Module):
    """LUNA/Laya-style learned-query channel mixer.

    Input:
        [B, C, N, channel_mixer_dim]

    Output:
        [B, N, encoder_dim]

    At each temporal patch, learned queries attend over channels.
    """

    def __init__(
        self,
        mixer_dim: int,
        encoder_dim: int,
        num_queries: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__()

        if mixer_dim % num_heads != 0:
            raise ValueError("mixer_dim must be divisible by num_heads")

        self.mixer_dim = mixer_dim
        self.encoder_dim = encoder_dim
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = mixer_dim // num_heads

        self.query_bank = nn.Parameter(
            torch.randn(num_heads, num_queries, self.head_dim) * 0.02
        )

        self.key_proj = nn.Linear(mixer_dim, mixer_dim)
        self.value_proj = nn.Linear(mixer_dim, mixer_dim)
        self.query_ffn_norm = nn.LayerNorm(mixer_dim)
        self.query_ffn = nn.Sequential(
            nn.Linear(mixer_dim, mixer_dim * 4),
            nn.GELU(),
            nn.Dropout(0.0),
            nn.Linear(mixer_dim * 4, mixer_dim),
        )
        self.query_self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=mixer_dim,
                nhead=num_heads,
                dim_feedforward=mixer_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),
            num_layers=3,
        )

        self.out_proj = nn.Linear(
            num_queries * mixer_dim,
            encoder_dim,
        )

    def _refine_query_tokens(
        self,
        mixed: torch.Tensor,
        *,
        batch: int,
        patches: int,
    ) -> torch.Tensor:
        query_tokens = mixed.permute(0, 1, 3, 2, 4).reshape(
            batch * patches,
            self.num_queries,
            self.mixer_dim,
        )
        # LUNA-style channel unification refines query latents with an FFN
        # residual block before a shallow query-side transformer stack.
        query_tokens = query_tokens + self.query_ffn(self.query_ffn_norm(query_tokens))
        query_tokens = self.query_self_attn(query_tokens)
        return query_tokens.reshape(
            batch,
            patches,
            self.num_queries,
            self.mixer_dim,
        )

    def _specialization_loss(self, attn: torch.Tensor) -> torch.Tensor:
        # attn: [B, N, H, Q, C]
        affinity = attn.mean(dim=2)  # [B, N, Q, C]

        affinity = F.normalize(
            affinity.reshape(
                -1,
                affinity.shape[-2],
                affinity.shape[-1],
            ),
            dim=-1,
        )

        gram = affinity @ affinity.transpose(1, 2)

        eye = torch.eye(
            gram.shape[-1],
            device=gram.device,
            dtype=torch.bool,
        ).unsqueeze(0)

        off_diag = gram.masked_fill(eye, 0.0)

        denom = max(
            1,
            gram.shape[-1] * max(1, gram.shape[-1] - 1),
        )

        return off_diag.square().sum() / max(
            1,
            off_diag.shape[0] * denom,
        )

    def forward(
        self,
        tokens: torch.Tensor,
        channel_metadata: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if tokens.dim() != 4:
            raise ValueError(
                f"Expected [B, C, N, D], got {tuple(tokens.shape)}"
            )

        batch, channels, patches, dim = tokens.shape

        if dim != self.mixer_dim:
            raise ValueError(
                f"Expected mixer dim {self.mixer_dim}, got {dim}"
            )

        if channel_mask is not None:
            if channel_mask.shape != (batch, channels):
                raise ValueError(
                    f"Expected channel_mask [B, C] = {(batch, channels)}, "
                    f"got {tuple(channel_mask.shape)}"
                )

            channel_mask = channel_mask.to(
                device=tokens.device,
                dtype=torch.bool,
            )

        patch_tokens = tokens.permute(0, 2, 1, 3)  # [B, N, C, D]

        keys = self.key_proj(patch_tokens).reshape(
            batch,
            patches,
            channels,
            self.num_heads,
            self.head_dim,
        )

        values = self.value_proj(patch_tokens).reshape(
            batch,
            patches,
            channels,
            self.num_heads,
            self.head_dim,
        )

        keys = keys.permute(0, 1, 3, 2, 4)    # [B, N, H, C, Hd]
        values = values.permute(0, 1, 3, 2, 4)

        queries = (
            self.query_bank.to(dtype=tokens.dtype, device=tokens.device)
            .unsqueeze(0)
            .unsqueeze(0)
        )  # [1, 1, H, Q, Hd]

        scores = torch.einsum(
            "bnhqd,bnhcd->bnhqc",
            queries.expand(batch, patches, -1, -1, -1),
            keys,
        )

        scores = scores / math.sqrt(self.head_dim)

        if channel_mask is not None:
            scores = scores.masked_fill(
                ~channel_mask[:, None, None, None, :],
                torch.finfo(scores.dtype).min,
            )

        attn = torch.softmax(scores, dim=-1)

        mixed = torch.einsum(
            "bnhqc,bnhcd->bnhqd",
            attn,
            values,
        )  # [B, N, H, Q, Hd]

        query_tokens = self._refine_query_tokens(
            mixed,
            batch=batch,
            patches=patches,
        )
        latent_tokens = query_tokens.permute(0, 2, 1, 3)

        mixed = query_tokens.reshape(
            batch,
            patches,
            self.num_queries * self.mixer_dim,
        )

        return self.out_proj(mixed), self._specialization_loss(attn), attn, {
            "latent_tokens": latent_tokens,
        }


class ChannelRelationBlock(nn.Module):
    """Patch-wise description-aware channel relation refinement.

    Input/Output:
        tokens: [B, C, N, D]

    The block keeps the patch axis intact and performs channel attention at
    every patch location. Description/metadata embeddings do not merely add to
    tokens; they produce a pairwise channel gate that directly modulates the
    channel-attention scores before QueryChannelMixer compression.
    """

    def __init__(self, token_dim: int, num_heads: int = 1) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads

        self.input_norm = nn.LayerNorm(token_dim)
        self.q_proj = nn.Linear(token_dim, token_dim)
        self.k_proj = nn.Linear(token_dim, token_dim)
        self.v_proj = nn.Linear(token_dim, token_dim)
        self.out_proj = nn.Linear(token_dim, token_dim)
        self.meta_q_proj = nn.Linear(token_dim, token_dim)
        self.meta_k_proj = nn.Linear(token_dim, token_dim)
        self.metadata_gate_scale = nn.Parameter(torch.tensor(0.1))
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        tokens: torch.Tensor,
        metadata: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens [B, C, N, D], got {tuple(tokens.shape)}")
        if metadata.shape != (tokens.shape[0], tokens.shape[1], tokens.shape[3]):
            raise ValueError(
                f"Expected metadata [B, C, D] = {(tokens.shape[0], tokens.shape[1], tokens.shape[3])}, got {tuple(metadata.shape)}"
            )

        batch, channels, patches, dim = tokens.shape
        tokens_patch_first = tokens.transpose(1, 2)  # [B, N, C, D]
        metadata_patch = metadata.unsqueeze(1).expand(batch, patches, channels, dim)
        relation_input = self.input_norm(tokens_patch_first + metadata_patch)

        q = self.q_proj(relation_input).reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(relation_input).reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(tokens_patch_first).reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        meta_q = self.meta_q_proj(metadata).reshape(batch, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        meta_k = self.meta_k_proj(metadata).reshape(batch, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        metadata_scores = torch.matmul(meta_q, meta_k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = token_scores + self.metadata_gate_scale * metadata_scores.unsqueeze(1)

        if channel_mask is not None:
            if channel_mask.shape != (batch, channels):
                raise ValueError(f"Expected channel_mask [B, C] = {(batch, channels)}, got {tuple(channel_mask.shape)}")
            valid = channel_mask.to(device=tokens.device, dtype=torch.bool)
            scores = scores.masked_fill(~valid[:, None, None, None, :], torch.finfo(scores.dtype).min)
            scores = scores.masked_fill(~valid[:, None, None, :, None], torch.finfo(scores.dtype).min)

        relation_attn = torch.softmax(scores, dim=-1)
        relation_update = torch.matmul(relation_attn, v).permute(0, 1, 3, 2, 4).reshape(batch, patches, channels, dim)
        relation_update = self.out_proj(relation_update).transpose(1, 2)
        refined = tokens + self.residual_scale * relation_update
        return refined, relation_attn


class RopeSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if self.head_dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        qkv = (
            self.qkv(x)
            .reshape(
                batch,
                seq_len,
                3,
                self.num_heads,
                self.head_dim,
            )
            .permute(2, 0, 3, 1, 4)
        )

        q, k, v = qkv[0], qkv[1], qkv[2]

        cos, sin = build_rope_cache(
            seq_len,
            self.head_dim,
            x.device,
            x.dtype,
            position_ids=position_ids,
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(attn, dim=-1))

        out = (attn @ v).transpose(1, 2).reshape(
            batch,
            seq_len,
            self.embed_dim,
        )

        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attn = RopeSelfAttention(
            embed_dim,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position_ids=position_ids)
        x = x + self.mlp(self.norm2(x))
        return x


class Projector(nn.Module):
    """3-layer projector with BatchNorm."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()

        hidden = in_dim * 4

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        flat = x.reshape(-1, shape[-1])

        for module in self.net:
            if (
                isinstance(module, nn.BatchNorm1d)
                and self.training
                and flat.shape[0] == 1
            ):
                flat = F.batch_norm(
                    flat,
                    module.running_mean,
                    module.running_var,
                    module.weight,
                    module.bias,
                    training=False,
                    momentum=module.momentum,
                    eps=module.eps,
                )
            else:
                flat = module(flat)

        return flat.reshape(*shape[:-1], -1)


class Predictor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        depth: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attn_dropout: float,
    ) -> None:
        super().__init__()

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.blocks = nn.ModuleList(
            TransformerBlock(
                embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attn_dropout=attn_dropout,
            )
            for _ in range(max(1, depth))
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        patch_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if patch_mask is not None:
            if patch_mask.shape != x.shape[:2]:
                raise ValueError(
                    f"Expected patch_mask shape {tuple(x.shape[:2])}, "
                    f"got {tuple(patch_mask.shape)}"
                )

            patch_mask = patch_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

            x = torch.where(
                patch_mask.unsqueeze(-1),
                self.mask_token.to(dtype=x.dtype, device=x.device).expand_as(x),
                x,
            )

        for block in self.blocks:
            x = block(x, position_ids=position_ids)

        return self.norm(x)


class LayaEncoder(nn.Module):
    def __init__(self, config: Optional[LayaModelConfig] = None) -> None:
        super().__init__()

        self.config = config or LayaModelConfig()
        self.channel_mixer_type = normalize_channel_mixer_type(self.config.channel_mixer_type)
        self.channel_metadata_mode = normalize_channel_metadata_mode(self.config.channel_metadata_mode)
        self.channel_token_dim = (
            self.config.channel_mixer_dim
            if self.channel_mixer_type == "mixer"
            else self.config.embed_dim
        )

        self.patch_embed = TemporalPatchEmbedding(
            self.config.patch_size,
            self.channel_token_dim,
        )

        self.channel_encoding: FourierChannelEncoding | None = None
        if self.channel_metadata_mode == "coordinates":
            self.channel_encoding = FourierChannelEncoding(
                self.channel_token_dim,
                num_bands=self.config.fourier_num_bands,
            )

        self.channel_id_projector: nn.Module | None = None
        if self.channel_metadata_mode == "onehot":
            if self.config.onehot_channel_vocab_size <= 0:
                raise ValueError("onehot_channel_vocab_size must be positive when channel_metadata_mode='onehot'.")
            self.channel_id_projector = nn.Sequential(
                nn.Linear(self.config.onehot_channel_vocab_size, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )

        self.channel_text_projector: nn.Module | None = None
        if self.channel_metadata_mode in {"text", "text_stats_avg", "text_stats_joint"}:
            self.channel_text_projector = nn.Sequential(
                nn.Linear(self.config.text_metadata_dim, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )
        self.channel_stats_projector: nn.Module | None = None
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            self.channel_stats_projector = nn.Sequential(
                nn.Linear(self.config.stats_metadata_dim, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )

        self.channel_mixer: QueryChannelMixer | None = None
        if self.channel_mixer_type == "mixer":
            self.channel_mixer = QueryChannelMixer(
                mixer_dim=self.config.channel_mixer_dim,
                encoder_dim=self.config.embed_dim,
                num_queries=self.config.num_queries,
                num_heads=self.config.channel_mixer_heads,
            )

        self.channel_relation_block: ChannelRelationBlock | None = None
        if self.config.use_channel_relation_block:
            self.channel_relation_block = ChannelRelationBlock(
                token_dim=self.channel_token_dim,
                num_heads=self.config.channel_relation_heads,
            )

        self.blocks = nn.ModuleList(
            TransformerBlock(
                self.config.embed_dim,
                num_heads=self.config.num_heads,
                mlp_ratio=self.config.mlp_ratio,
                dropout=self.config.dropout,
                attn_dropout=self.config.attn_dropout,
            )
            for _ in range(self.config.depth)
        )

        self.norm = nn.LayerNorm(self.config.embed_dim)

    def get_config(self) -> Dict[str, object]:
        return asdict(self.config)

    def _resolve_channel_metadata(
        self,
        *,
        batch: int,
        channels: int,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor],
        channel_text_embeddings: Optional[torch.Tensor],
        channel_stats_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if self.channel_metadata_mode == "coordinates":
            if channel_positions is None:
                raise ValueError(
                    "channel_positions are required when channel_metadata_mode='coordinates'."
                )
            if channel_positions.dim() != 3 or channel_positions.shape[-1] != 3:
                raise ValueError(
                    f"Expected channel_positions [B, C, 3], got {tuple(channel_positions.shape)}"
                )
            if channel_positions.shape[:2] != (batch, channels):
                raise ValueError(
                    f"channel_positions shape {tuple(channel_positions.shape)} does not match input {(batch, channels)}"
                )
            if self.channel_encoding is None:
                raise RuntimeError("channel_encoding is not initialized.")
            return self.channel_encoding(channel_positions.to(device=x.device, dtype=x.dtype))

        if self.channel_metadata_mode == "onehot":
            if self.channel_id_projector is None:
                raise RuntimeError("channel_id_projector is not initialized.")
            if channels > self.config.onehot_channel_vocab_size:
                raise ValueError(
                    f"Input channels {channels} exceed onehot_channel_vocab_size={self.config.onehot_channel_vocab_size}."
                )
            channel_ids = torch.arange(channels, device=x.device)
            one_hot = F.one_hot(channel_ids, num_classes=self.config.onehot_channel_vocab_size).to(dtype=x.dtype)
            projected = self.channel_id_projector(one_hot)
            return projected.unsqueeze(0).expand(batch, -1, -1)

        if self.channel_metadata_mode in {"text", "stats", "text_stats_avg", "text_stats_joint"}:
            def _prepare_metadata_tensor(
                value: Optional[torch.Tensor],
                *,
                name: str,
            ) -> torch.Tensor:
                if value is None:
                    raise ValueError(
                        f"{name} are required when channel_metadata_mode='{self.channel_metadata_mode}'."
                    )
                metadata = value.to(device=x.device, dtype=x.dtype)
                if metadata.dim() == 2:
                    if metadata.shape[0] != channels:
                        raise ValueError(f"Expected {name} [C, E] with C={channels}, got {tuple(metadata.shape)}")
                    metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
                elif metadata.dim() == 3:
                    if metadata.shape[:2] != (batch, channels):
                        raise ValueError(
                            f"Expected {name} [B, C, E] = {(batch, channels, 'E')}, got {tuple(metadata.shape)}"
                        )
                else:
                    raise ValueError(f"Unsupported {name} shape: {tuple(metadata.shape)}")
                return metadata

            if self.channel_metadata_mode in {"text", "text_stats_joint"}:
                metadata = _prepare_metadata_tensor(
                    channel_text_embeddings,
                    name="channel_text_embeddings",
                )
                if self.channel_text_projector is None:
                    raise RuntimeError("channel_text_projector is not initialized.")
                projected = self.channel_text_projector(metadata.reshape(-1, metadata.shape[-1]))
                return projected.reshape(batch, channels, self.channel_token_dim)

            if self.channel_metadata_mode == "stats":
                metadata = _prepare_metadata_tensor(
                    channel_stats_embeddings,
                    name="channel_stats_embeddings",
                )
                if self.channel_stats_projector is None:
                    raise RuntimeError("channel_stats_projector is not initialized.")
                projected = self.channel_stats_projector(metadata.reshape(-1, metadata.shape[-1]))
                return projected.reshape(batch, channels, self.channel_token_dim)

            text_metadata = _prepare_metadata_tensor(
                channel_text_embeddings,
                name="channel_text_embeddings",
            )
            stats_metadata = _prepare_metadata_tensor(
                channel_stats_embeddings,
                name="channel_stats_embeddings",
            )
            if self.channel_text_projector is None:
                raise RuntimeError("channel_text_projector is not initialized.")
            if self.channel_stats_projector is None:
                raise RuntimeError("channel_stats_projector is not initialized.")
            projected_text = self.channel_text_projector(
                text_metadata.reshape(-1, text_metadata.shape[-1])
            ).reshape(batch, channels, self.channel_token_dim)
            projected_stats = self.channel_stats_projector(
                stats_metadata.reshape(-1, stats_metadata.shape[-1])
            ).reshape(batch, channels, self.channel_token_dim)
            return 0.5 * (projected_text + projected_stats)

        if self.channel_metadata_mode == "none":
            return x.new_zeros(batch, channels, self.channel_token_dim)

        raise AssertionError("Unsupported channel metadata mode")

    def forward_features(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        channel_text_embeddings: Optional[torch.Tensor] = None,
        channel_stats_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")

        batch, channels, time = x.shape
        num_patches = math.ceil(time / self.config.patch_size)

        metadata_encoding = self._resolve_channel_metadata(
            batch=batch,
            channels=channels,
            x=x,
            channel_positions=channel_positions,
            channel_text_embeddings=channel_text_embeddings,
            channel_stats_embeddings=channel_stats_embeddings,
        )

        if channel_mask is None:
            inferred_mask = x.abs().sum(dim=-1) > 0
            if self.channel_metadata_mode == "coordinates" and channel_positions is not None:
                inferred_mask = inferred_mask | (channel_positions.to(device=x.device, dtype=x.dtype).abs().sum(dim=-1) > 0)
            if inferred_mask.any() and not inferred_mask.all():
                channel_mask = inferred_mask.to(device=x.device, dtype=torch.bool)
            else:
                channel_mask = torch.ones(
                    batch,
                    channels,
                    device=x.device,
                    dtype=torch.bool,
                )
        else:
            if channel_mask.shape != (batch, channels):
                raise ValueError(
                    f"Expected channel_mask [B, C] = {(batch, channels)}, "
                    f"got {tuple(channel_mask.shape)}"
                )

            channel_mask = channel_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

        if patch_mask is not None:
            if patch_mask.shape != (batch, num_patches):
                raise ValueError(
                    f"Expected patch_mask [B, N] = {(batch, num_patches)}, "
                    f"got {tuple(patch_mask.shape)}"
                )

            patch_mask = patch_mask.to(
                device=x.device,
                dtype=torch.bool,
            )

        channel_tokens = self.patch_embed(x)

        if channel_tokens.shape[2] != num_patches:
            raise RuntimeError(
                f"Patch count mismatch: patch_embed produced {channel_tokens.shape[2]}, "
                f"expected {num_patches}"
            )

        channel_tokens = channel_tokens + metadata_encoding.unsqueeze(2)
        relation_affinity = None

        if self.channel_relation_block is not None:
            channel_tokens, relation_affinity = self.channel_relation_block(
                channel_tokens,
                metadata_encoding,
                channel_mask=channel_mask,
            )

        if self.channel_mixer_type == "mixer":
            if self.channel_mixer is None:
                raise RuntimeError("channel_mixer is not initialized.")
            mixed_tokens, query_loss, affinity = self.channel_mixer(
                channel_tokens,
                channel_mask=channel_mask,
            )
            if patch_mask is not None:
                mixed_tokens = mixed_tokens.masked_fill(
                    patch_mask.unsqueeze(-1),
                    0.0,
                )
            encoded = mixed_tokens
            for block in self.blocks:
                encoded = block(encoded)
            encoded = self.norm(encoded)
            return {
                "channel_tokens": channel_tokens,
                "mixed_tokens_pre_encoder": mixed_tokens,
                "mixed_tokens": encoded,
                "mixed_repr": encoded.mean(dim=1),
                "channel_repr": channel_tokens.mean(dim=2),
                "query_loss": query_loss,
                "channel_affinity": affinity,
                "relation_affinity": relation_affinity,
            }

        independent_tokens = channel_tokens
        if patch_mask is not None:
            independent_tokens = independent_tokens.masked_fill(
                patch_mask[:, None, :, None],
                0.0,
            )
        flat_tokens = independent_tokens.reshape(batch * channels, num_patches, self.config.embed_dim)
        encoded = flat_tokens
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.norm(encoded)
        encoded = encoded.reshape(batch, channels, num_patches, self.config.embed_dim)
        channel_repr = encoded.mean(dim=2)
        return {
            "channel_tokens": channel_tokens,
            "independent_tokens": encoded,
            "mixed_tokens_pre_encoder": flat_tokens,
            "mixed_tokens": encoded.reshape(batch * channels, num_patches, self.config.embed_dim),
            "mixed_repr": channel_repr.mean(dim=1),
            "channel_repr": channel_repr,
            "query_loss": encoded.new_zeros(()),
            "channel_affinity": None,
            "relation_affinity": relation_affinity,
        }

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        channel_text_embeddings: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_features(
            x,
            channel_positions=channel_positions,
            channel_mask=channel_mask,
            channel_text_embeddings=channel_text_embeddings,
        )


class LayaPretrainer(nn.Module):
    def __init__(self, config: Optional[LayaModelConfig] = None) -> None:
        super().__init__()

        self.config = config or LayaModelConfig()

        self.encoder = LayaEncoder(self.config)

        self.projector = Projector(
            self.config.embed_dim,
            self.config.proj_dim,
        )

        self.predictor = Predictor(
            self.config.proj_dim,
            depth=self.config.predictor_depth,
            num_heads=self.config.predictor_heads,
            mlp_ratio=self.config.predictor_mlp_ratio,
            dropout=self.config.dropout,
            attn_dropout=self.config.attn_dropout,
        )

        self.sigreg = SIGReg(
            num_slices=self.config.sigreg_num_slices,
            quadrature_points=self.config.sigreg_quadrature_points,
            cf_t_max=self.config.sigreg_cf_t_max,
            cf_bandwidth=self.config.sigreg_cf_bandwidth,
        )

    def _sample_patch_mask(
        self,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        rng: Optional[random.Random] = None,
    ) -> torch.Tensor:
        rng = rng or random
        mask = torch.zeros(
            batch_size,
            num_patches,
            dtype=torch.bool,
            device=device,
        )

        min_block, max_block = self.config.mask_patch_span

        min_block = max(1, min(min_block, num_patches))
        max_block = max(min_block, min(max_block, num_patches))

        target = max(
            1,
            min(
                num_patches,
                int(round(self.config.mask_ratio * num_patches)),
            ),
        )

        for batch_idx in range(batch_size):
            runs = []
            remaining = target

            while remaining > 0:
                candidates = [
                    length
                    for length in range(min_block, max_block + 1)
                    if (
                        length <= remaining
                        and (
                            remaining - length == 0
                            or remaining - length >= min_block
                        )
                    )
                ]

                if not candidates:
                    if runs and runs[-1] + remaining <= max_block:
                        runs[-1] += remaining
                    else:
                        runs.append(remaining)
                    break

                block_len = rng.choice(candidates)
                runs.append(block_len)
                remaining -= block_len

            required = sum(runs) + max(0, len(runs) - 1)

            if required > num_patches:
                cursor = 0
                for run_len in runs:
                    end = min(num_patches, cursor + run_len)
                    mask[batch_idx, cursor:end] = True
                    cursor = end
                continue

            extra_space = num_patches - required
            gap_slots = [0] + [1] * max(0, len(runs) - 1) + [0]

            for _ in range(extra_space):
                gap_slots[rng.randint(0, len(gap_slots) - 1)] += 1

            cursor = gap_slots[0]

            for run_index, run_len in enumerate(runs):
                mask[batch_idx, cursor : cursor + run_len] = True
                cursor += run_len

                if run_index < len(runs) - 1:
                    cursor += gap_slots[run_index + 1]

        return mask

    def _resolve_patch_mask(
        self,
        *,
        batch_size: int,
        num_patches: int,
        device: torch.device,
        patch_mask: Optional[torch.Tensor] = None,
        patch_mask_seed: Optional[int] = None,
    ) -> torch.Tensor:
        if patch_mask is not None:
            expected_shape = (batch_size, num_patches)
            if patch_mask.shape != expected_shape:
                raise ValueError(
                    f"Expected patch_mask [B, N] = {expected_shape}, got {tuple(patch_mask.shape)}"
                )
            return patch_mask.to(device=device, dtype=torch.bool)
        rng = None if patch_mask_seed is None else random.Random(int(patch_mask_seed))
        return self._sample_patch_mask(batch_size, num_patches, device, rng=rng)

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor],
        channel_mask: Optional[torch.Tensor] = None,
        channel_text_embeddings: Optional[torch.Tensor] = None,
        channel_stats_embeddings: Optional[torch.Tensor] = None,
        return_aux: bool = False,
        patch_mask: Optional[torch.Tensor] = None,
        patch_mask_seed: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")

        if self.encoder.channel_metadata_mode == "coordinates" and channel_positions is None:
            raise ValueError(
                "channel_positions must be provided. "
                "They should be generated in dataset.py from channel_names."
            )

        if channel_positions is not None and (channel_positions.dim() != 3 or channel_positions.shape[-1] != 3):
            raise ValueError(
                f"Expected channel_positions [B, C, 3], "
                f"got {tuple(channel_positions.shape)}"
            )

        if channel_positions is not None and channel_positions.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"channel_positions shape {tuple(channel_positions.shape)} "
                f"does not match input shape {tuple(x.shape)}"
            )

        num_patches = math.ceil(x.shape[-1] / self.config.patch_size)

        patch_mask = self._resolve_patch_mask(
            batch_size=x.shape[0],
            num_patches=num_patches,
            device=x.device,
            patch_mask=patch_mask,
            patch_mask_seed=patch_mask_seed,
        )

        full = self.encoder.forward_features(
            x,
            channel_positions=channel_positions,
            channel_mask=channel_mask,
            channel_text_embeddings=channel_text_embeddings,
            channel_stats_embeddings=channel_stats_embeddings,
        )

        context = self.encoder.forward_features(
            x,
            channel_positions=channel_positions,
            channel_mask=channel_mask,
            patch_mask=patch_mask,
            channel_text_embeddings=channel_text_embeddings,
            channel_stats_embeddings=channel_stats_embeddings,
        )

        # StopGrad target branch.
        target_tokens = self.projector(full["mixed_tokens"]).detach()

        context_tokens = self.projector(context["mixed_tokens"])

        if self.encoder.channel_mixer_type == "independent":
            patch_mask = patch_mask.unsqueeze(1).expand(x.shape[0], x.shape[1], num_patches).reshape(x.shape[0] * x.shape[1], num_patches)

        pred_tokens = self.predictor(
            context_tokens,
            patch_mask=patch_mask,
        )

        pred_loss = F.mse_loss(
            pred_tokens[patch_mask],
            target_tokens[patch_mask],
        )

        # SIGReg regularizes the global context representation.
        if self.encoder.channel_mixer_type == "mixer":
            context_global = self.projector(context["mixed_repr"])
        else:
            context_global = self.projector(context["channel_repr"].reshape(-1, self.config.embed_dim))
        sigreg_loss = self.sigreg(context_global.unsqueeze(0))

        query_loss = 0.5 * (
            full["query_loss"] + context["query_loss"]
        )

        loss = (
            pred_loss
            + (self.config.sigreg_weight * sigreg_loss)
            + (self.config.query_loss_weight * query_loss)
        )

        outputs = {
            "loss": loss,
            "pred_loss": pred_loss,
            "sigreg_loss": sigreg_loss,
            "query_loss": query_loss,
            "patch_mask": patch_mask,
            "target_tokens": target_tokens,
            "pred_tokens": pred_tokens,
            "mixed_tokens": full["mixed_tokens"],
            "mixed_repr": full["mixed_repr"],
        }

        if return_aux:
            outputs.update(
                {
                    "full_features": full,
                    "context_features": context,
                }
            )

        return outputs
