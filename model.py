from __future__ import annotations

from dataclasses import asdict, fields, is_dataclass
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import FourierChannelEncoding, LayaEncoder, LayaPretrainer, QueryChannelMixer, TemporalPatchEmbedding
from .config import LayaModelConfig


def normalize_patchifier_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "single": "single",
        "baseline": "single",
        "fixed": "single",
        "multiscale": "multiscale",
        "multi_scale": "multiscale",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported patchifier_mode: {value}")
    return aliases[normalized]


def normalize_temporal_patchifier_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "fixed",
        "baseline": "fixed",
        "multiscale": "multiscale",
        "multi_scale": "multiscale",
        "charm_like": "charm_like",
        "charmlike": "charm_like",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported temporal_patchifier_mode: {value}")
    return aliases[normalized]


def normalize_charm_scale_gate_source(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "learned": "learned",
        "global": "learned",
        "text": "text",
        "text_aware": "text",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported charm_scale_gate_source: {value}")
    return aliases[normalized]


def normalize_charm_patchifier_fusion(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "replace": "replace",
        "residual": "residual",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported charm_patchifier_fusion: {value}")
    return aliases[normalized]


def normalize_encoder_variant(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "default": "default",
        "base": "default",
        "baseline": "default",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported encoder_variant: {value}")
    return aliases[normalized]


def normalize_channel_mixer_relation_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "laya_relation": "laya_relation",
        "relation": "laya_relation",
        "description_bias": "laya_relation",
        "metadata_query_gate": "metadata_query_gate",
        "metadata_query_bias": "metadata_query_gate",
        "metadata_aware_query": "metadata_query_gate",
        "metadata_aware_query_mixer": "metadata_query_gate",
        "query_bias": "metadata_query_gate",
        "routing_bias": "metadata_query_gate",
        "query_gate": "metadata_query_gate",
        "routing_gate": "metadata_query_gate",
        "description_relation": "description_relation",
        "relation_aware": "description_relation",
        "description_relation_aware": "description_relation",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported channel_mixer_relation_mode: {value}")
    return aliases[normalized]


def normalize_description_relation_metric(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "projected_dot": "projected_dot",
        "dot": "projected_dot",
        "projected": "projected_dot",
        "cosine": "cosine",
        "cos": "cosine",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported description_relation_metric: {value}")
    return aliases[normalized]


def normalize_metadata_fusion_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "none": "none",
        "off": "none",
        "add": "add",
        "sum": "add",
        "concat_kv": "concat_kv",
        "concat_proj": "concat_kv",
        "concat_projection": "concat_kv",
        "kv_concat": "concat_kv",
        "concat_key_value": "concat_kv",
        "attention_gate": "attention_gate",
        "gate": "attention_gate",
        "attention": "attention_gate",
        "attention_suppress_gate": "attention_suppress_gate",
        "suppress_gate": "attention_suppress_gate",
        "suppression_gate": "attention_suppress_gate",
        "attention_suppression": "attention_suppress_gate",
        "charm_suppress_gate": "attention_suppress_gate",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported metadata_fusion_mode: {value}")
    return aliases[normalized]


def normalize_relation_adapter_position(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "post_encoder": "post_encoder",
        "after_encoder": "post_encoder",
        "encoder_output": "post_encoder",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported relation_adapter_position: {value}")
    return aliases[normalized]


def summarize_metadata_usage(features: dict[str, torch.Tensor | None]) -> dict[str, float]:
    """Extract compact metadata-conditioning diagnostics from encoder features."""
    key_map = {
        "relation_scale_mean": "channel_mixer_relation_scale",
        "signal_score_mean_abs": "channel_mixer_signal_score_mean_abs",
        "score_delta_mean_abs": "channel_mixer_score_delta_mean_abs",
        "relation_scores_mean_abs": "channel_mixer_relation_scores_mean_abs",
        "relation_threshold_mean": "channel_mixer_relation_threshold_mean",
        "relation_gate_mean": "channel_mixer_relation_gate_mean",
        "relation_gate_sparsity": "channel_mixer_relation_gate_sparsity",
        "metadata_norm_mean": "channel_mixer_metadata_norm_mean",
        "adapter_scale_mean": "relation_adapter_scale",
        "adapter_metadata_scale_mean": "relation_adapter_metadata_scale",
        "adapter_gate_mean": "relation_adapter_gate_mean",
        "adapter_bias_mean_abs": "relation_adapter_metadata_bias_mean_abs",
        "adapter_metadata_present": "relation_adapter_metadata_present",
        "adapter_metadata_nonzero_fraction": "relation_adapter_metadata_nonzero_fraction",
        "adapter_input_norm_mean": "relation_adapter_input_norm_mean",
        "adapter_output_norm_mean": "relation_adapter_output_norm_mean",
        "adapter_update_norm_mean": "relation_adapter_update_norm_mean",
        "adapter_delta_ratio": "relation_adapter_delta_ratio",
        "adapter_attention_entropy": "relation_adapter_attention_entropy",
    }
    summary: dict[str, float] = {}
    for out_key, feature_key in key_map.items():
        value = features.get(feature_key)
        if value is None:
            continue
        if torch.is_tensor(value):
            if value.numel() == 0:
                continue
            summary[out_key] = float(value.detach().float().mean().item())
        else:
            summary[out_key] = float(value)
    signal_mag = summary.get("signal_score_mean_abs")
    delta_mag = summary.get("score_delta_mean_abs")
    if signal_mag is not None and delta_mag is not None and signal_mag > 0.0:
        summary["score_delta_ratio"] = float(delta_mag / signal_mag)
    return summary


class MultiScaleFusionGate(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_scales: int,
        *,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if num_scales <= 0:
            raise ValueError(f"num_scales must be positive, got {num_scales}")
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        input_dim = latent_dim * num_scales
        hidden_dim = max(latent_dim, num_scales * 4)
        self.temperature = float(temperature)
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, scale_latents: torch.Tensor) -> torch.Tensor:
        if scale_latents.dim() != 5:
            raise ValueError(
                f"Expected scale_latents [B, C, N, S, D], got {tuple(scale_latents.shape)}"
            )
        batch, channels, patches, scales, dim = scale_latents.shape
        fused_input = scale_latents.reshape(batch, channels, patches, scales * dim)
        logits = self.mlp(self.norm(fused_input))
        return torch.softmax(logits / self.temperature, dim=-1)


def apply_metadata_dropout(
    metadata: Optional[torch.Tensor],
    p: float,
    training: bool,
) -> Optional[torch.Tensor]:
    if metadata is None:
        return None
    if not training or p <= 0.0:
        return metadata
    if metadata.dim() == 2:
        keep = (torch.rand(metadata.shape[0], device=metadata.device) > p).to(
            dtype=metadata.dtype
        )
        return metadata * keep.unsqueeze(-1)
    if metadata.dim() == 3:
        keep = (
            torch.rand(metadata.shape[0], metadata.shape[1], device=metadata.device) > p
        ).to(dtype=metadata.dtype)
        return metadata * keep.unsqueeze(-1)
    return metadata


def normalize_description_gate_mode(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "bias": "bias",
        "additive_bias": "bias",
        "attention_gate": "bias",
        "suppress": "suppress",
        "suppression": "suppress",
        "suppress_gate": "suppress",
        "attention_suppress_gate": "suppress",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported description_gate_mode: {value}")
    return aliases[normalized]


def infer_temporal_patchifier_num_patches(config: LayaModelConfig, time: int) -> int:
    if time <= 0:
        raise ValueError(f"time must be positive, got {time}")
    mode = normalize_temporal_patchifier_mode(config.temporal_patchifier_mode)
    fixed_patches = math.ceil(time / config.patch_size)
    if mode == "fixed":
        return fixed_patches
    charm_patches = math.ceil(time / config.charm_stride)
    fusion = normalize_charm_patchifier_fusion(config.charm_patchifier_fusion)
    if fusion == "residual":
        return min(fixed_patches, charm_patches)
    return charm_patches


class CHARMLikeMultiScaleTemporalPatchifier(nn.Module):
    """CHARM-inspired multi-scale temporal patchifier compatible with LayaTS.

    This mode does not fully reproduce CHARM. In particular, it does not
    implement CHARM's full channel-time attention or contextual kernel
    generation. Instead, it extends the fixed patch embedding used by LayaTS
    into a channel-adaptive multi-scale temporal patchifier that preserves the
    exact structure expected by QueryChannelMixer.

    The key compatibility constraints are:
    - the final output shape must stay [B, C, N, D]
    - every scale branch must end with the same N
    - the same patch index n must correspond to the same time anchor

    We enforce the anchor rule by using a shared stride across all scale
    branches and right-padding each branch so output index n always starts at
    the same temporal anchor n * stride. Smaller kernels see a shorter context
    around that anchor, while larger kernels see a longer receptive field.
    """

    def __init__(
        self,
        out_dim: int,
        kernel_sizes: tuple[int, ...],
        stride: int,
        dropout: float = 0.0,
        scale_gate_source: str = "learned",
        scale_gate_temperature: float = 1.0,
        text_metadata_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError(f"out_dim must be positive, got {out_dim}")
        if stride <= 0:
            raise ValueError(f"stride must be positive, got {stride}")
        if not kernel_sizes:
            raise ValueError("kernel_sizes must contain at least one kernel size")
        if any(kernel_size <= 0 for kernel_size in kernel_sizes):
            raise ValueError(f"kernel_sizes must be positive, got {kernel_sizes}")
        if scale_gate_temperature <= 0:
            raise ValueError(
                f"scale_gate_temperature must be positive, got {scale_gate_temperature}"
            )

        self.out_dim = out_dim
        self.kernel_sizes = tuple(int(kernel_size) for kernel_size in kernel_sizes)
        self.num_scales = len(self.kernel_sizes)
        self.stride = int(stride)
        self.scale_gate_source = normalize_charm_scale_gate_source(scale_gate_source)
        self.scale_gate_temperature = float(scale_gate_temperature)
        self.branches = nn.ModuleList(
            nn.Conv1d(
                in_channels=1,
                out_channels=out_dim,
                kernel_size=kernel_size,
                stride=self.stride,
                padding=0,
            )
            for kernel_size in self.kernel_sizes
        )
        self.output_dropout = nn.Dropout(dropout)
        self.learned_scale_logits = nn.Parameter(torch.zeros(self.num_scales))
        self.scale_text_projector: nn.Module | None = None
        if text_metadata_dim is not None and text_metadata_dim > 0:
            hidden_dim = max(self.num_scales * 2, min(out_dim, 128))
            self.scale_text_projector = nn.Sequential(
                nn.Linear(text_metadata_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, self.num_scales),
            )

    def num_patches_for_length(self, time: int) -> int:
        if time <= 0:
            raise ValueError(f"time must be positive, got {time}")
        return math.ceil(time / self.stride)

    def _prepare_text_metadata(
        self,
        channel_text_embeddings: Optional[torch.Tensor],
        *,
        batch: int,
        channels: int,
        x: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if channel_text_embeddings is None:
            return None
        metadata = channel_text_embeddings.to(device=x.device, dtype=x.dtype)
        if metadata.dim() == 2:
            if metadata.shape[0] != channels:
                raise ValueError(
                    f"Expected channel_text_embeddings [C, E] with C={channels}, got {tuple(metadata.shape)}"
                )
            metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
        elif metadata.dim() == 3:
            if metadata.shape[:2] != (batch, channels):
                raise ValueError(
                    f"Expected channel_text_embeddings [B, C, E] = {(batch, channels, 'E')}, got {tuple(metadata.shape)}"
                )
        else:
            raise ValueError(f"Unsupported channel_text_embeddings shape: {tuple(metadata.shape)}")
        return metadata

    def _resolve_scale_logits(
        self,
        *,
        batch: int,
        channels: int,
        x: torch.Tensor,
        channel_text_embeddings: Optional[torch.Tensor],
    ) -> torch.Tensor:
        learned_logits = self.learned_scale_logits.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        learned_logits = learned_logits.expand(batch, channels, -1)
        if self.scale_gate_source != "text":
            return learned_logits
        metadata = self._prepare_text_metadata(
            channel_text_embeddings,
            batch=batch,
            channels=channels,
            x=x,
        )
        if metadata is None or self.scale_text_projector is None:
            return learned_logits
        projected = self.scale_text_projector(metadata.reshape(-1, metadata.shape[-1]))
        return projected.reshape(batch, channels, self.num_scales).to(dtype=x.dtype)

    def forward(
        self,
        x: torch.Tensor,
        channel_text_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, object]]:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")

        batch, channels, time = x.shape
        target_patches = self.num_patches_for_length(time)
        x_flat = x.reshape(batch * channels, 1, time)
        branch_outputs: list[torch.Tensor] = []
        branch_patch_counts: list[int] = []

        for kernel_size, branch in zip(self.kernel_sizes, self.branches):
            required_time = ((target_patches - 1) * self.stride) + kernel_size
            pad_right = max(0, required_time - time)
            branch_input = F.pad(x_flat, (0, pad_right)) if pad_right > 0 else x_flat
            z_k = branch(branch_input)
            branch_patch_counts.append(z_k.shape[-1])
            z_k = z_k.transpose(1, 2).reshape(batch, channels, z_k.shape[-1], self.out_dim)
            branch_outputs.append(z_k)

        aligned_patches = min(branch_output.shape[2] for branch_output in branch_outputs)
        if any(branch_output.shape[2] != aligned_patches for branch_output in branch_outputs):
            branch_outputs = [
                branch_output[:, :, :aligned_patches, :]
                for branch_output in branch_outputs
            ]

        scale_logits = self._resolve_scale_logits(
            batch=batch,
            channels=channels,
            x=x,
            channel_text_embeddings=channel_text_embeddings,
        )
        scale_gate = torch.softmax(scale_logits / self.scale_gate_temperature, dim=-1)
        z_stack = torch.stack(branch_outputs, dim=2)  # [B, C, K, N, D]
        tokens = (z_stack * scale_gate[:, :, :, None, None]).sum(dim=2)
        tokens = self.output_dropout(tokens)
        aux = {
            "scale_gate": scale_gate,
            "scale_logits": scale_logits,
            "scale_gate_mean": scale_gate.mean(dim=(0, 1)),
            "patchifier_N": aligned_patches,
            "branch_patch_counts": tuple(branch_patch_counts),
        }
        return tokens, aux


class SamePadConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.causal = bool(causal)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        effective_kernel = (self.kernel_size - 1) * self.dilation + 1
        total_padding = max(0, effective_kernel - 1)
        if self.causal:
            left_padding = total_padding
            right_padding = 0
        else:
            left_padding = total_padding // 2
            right_padding = total_padding - left_padding
        if left_padding > 0 or right_padding > 0:
            x = F.pad(x, (left_padding, right_padding))
        return self.conv(x)


class MetadataAwareQueryChannelMixer(QueryChannelMixer):
    """Metadata-aware query channel mixer.

    Learned queries aggregate channels per temporal patch as in the original
    Laya mixer, but metadata contributes only as a signed query-to-channel
    routing modulation. No explicit C x C inter-channel relation matrix is formed.
    """

    def __init__(
        self,
        mixer_dim: int,
        encoder_dim: int,
        num_queries: int,
        *,
        metadata_dim: int,
        num_heads: int = 1,
        relation_scale_init: float = 0.01,
    ) -> None:
        super().__init__(
            mixer_dim=mixer_dim,
            encoder_dim=encoder_dim,
            num_queries=num_queries,
            num_heads=num_heads,
        )
        if metadata_dim <= 0:
            raise ValueError(f"metadata_dim must be positive, got {metadata_dim}")
        self.metadata_key_proj = nn.Linear(metadata_dim, mixer_dim)
        self.relation_scale = nn.Parameter(torch.tensor(float(relation_scale_init)))

    def forward(
        self,
        tokens: torch.Tensor,
        channel_metadata: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B, C, N, D], got {tuple(tokens.shape)}")

        batch, channels, patches, dim = tokens.shape
        if dim != self.mixer_dim:
            raise ValueError(f"Expected mixer dim {self.mixer_dim}, got {dim}")
        if channel_metadata is not None and channel_metadata.shape != (batch, channels, dim):
            raise ValueError(
                f"Expected channel_metadata [B, C, D] = {(batch, channels, dim)}, got {tuple(channel_metadata.shape)}"
            )

        if channel_mask is not None:
            if channel_mask.shape != (batch, channels):
                raise ValueError(
                    f"Expected channel_mask [B, C] = {(batch, channels)}, got {tuple(channel_mask.shape)}"
                )
            channel_mask = channel_mask.to(device=tokens.device, dtype=torch.bool)

        patch_tokens = tokens.permute(0, 2, 1, 3)
        keys = self.key_proj(patch_tokens).reshape(batch, patches, channels, self.num_heads, self.head_dim)
        values = self.value_proj(patch_tokens).reshape(batch, patches, channels, self.num_heads, self.head_dim)
        keys = keys.permute(0, 1, 3, 2, 4)
        values = values.permute(0, 1, 3, 2, 4)

        queries = self.query_bank.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0).unsqueeze(0)
        token_scores = torch.einsum(
            "bnhqd,bnhcd->bnhqc",
            queries.expand(batch, patches, -1, -1, -1),
            keys,
        )
        token_scores = token_scores / math.sqrt(self.head_dim)

        relation_scores = None
        relation_gate = None
        effective_scale = torch.tanh(self.relation_scale).to(device=tokens.device, dtype=tokens.dtype)
        score_delta = None
        metadata_norm_mean = None
        metadata_norm_std = None
        metadata_norm_min = None
        metadata_norm_max = None
        if channel_metadata is not None:
            metadata_norms = channel_metadata.detach().norm(dim=-1)
            metadata_norm_mean = metadata_norms.mean()
            metadata_norm_std = metadata_norms.std(unbiased=False)
            metadata_norm_min = metadata_norms.min()
            metadata_norm_max = metadata_norms.max()
            relation_keys = self.metadata_key_proj(channel_metadata).reshape(
                batch,
                channels,
                self.num_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)
            relation_scores = torch.einsum(
                "bnhqd,bhcd->bnhqc",
                queries.expand(batch, patches, -1, -1, -1),
                relation_keys,
            )
            relation_scores = relation_scores / math.sqrt(self.head_dim)
            relation_gate = torch.tanh(relation_scores)
            modulation = 1.0 + effective_scale * relation_gate
            scores = token_scores * modulation
            score_delta = scores - token_scores
        else:
            scores = token_scores

        if channel_mask is not None:
            scores = scores.masked_fill(
                ~channel_mask[:, None, None, None, :],
                torch.finfo(scores.dtype).min,
            )

        attn = torch.softmax(scores, dim=-1)
        mixed = torch.einsum("bnhqc,bnhcd->bnhqd", attn, values)
        query_tokens = self._refine_query_tokens(
            mixed,
            batch=batch,
            patches=patches,
        )
        latent_tokens = query_tokens.permute(0, 2, 1, 3)
        mixed = query_tokens.reshape(batch, patches, self.num_queries * self.mixer_dim)
        aux = {
            "token_scores": token_scores,
            "relation_scores": relation_scores,
            "relation_scale": effective_scale.detach(),
            "relation_gate": relation_gate,
            "latent_tokens": latent_tokens,
            "signal_score_mean_abs": token_scores.detach().abs().mean(),
            "score_delta_mean_abs": None if score_delta is None else score_delta.detach().abs().mean(),
            "relation_scores_mean_abs": None if relation_scores is None else relation_scores.detach().abs().mean(),
            "relation_gate_mean": None if relation_gate is None else relation_gate.detach().mean(),
            "relation_gate_sparsity": None if relation_gate is None else (relation_gate.detach().abs() <= 1e-6).float().mean(),
            "metadata_norm_mean": metadata_norm_mean,
            "metadata_norm_std": metadata_norm_std,
            "metadata_norm_min": metadata_norm_min,
            "metadata_norm_max": metadata_norm_max,
        }
        return self.out_proj(mixed), self._specialization_loss(attn), attn, aux


class ConcatProjectedMetadataQueryChannelMixer(QueryChannelMixer):
    """Query mixer that injects metadata through concatenated K/V inputs.

    Instead of adding metadata onto patch tokens before mixing, this module
    concatenates per-channel metadata with each patch token and learns new key
    and value projections from the concatenated representation.
    """

    def __init__(
        self,
        mixer_dim: int,
        encoder_dim: int,
        num_queries: int,
        *,
        metadata_dim: int,
        num_heads: int = 1,
    ) -> None:
        super().__init__(
            mixer_dim=mixer_dim,
            encoder_dim=encoder_dim,
            num_queries=num_queries,
            num_heads=num_heads,
        )
        if metadata_dim <= 0:
            raise ValueError(f"metadata_dim must be positive, got {metadata_dim}")
        self.metadata_dim = metadata_dim
        self.kv_input_norm = nn.LayerNorm(mixer_dim + metadata_dim)
        self.concat_key_proj = nn.Linear(mixer_dim + metadata_dim, mixer_dim)
        self.concat_value_proj = nn.Linear(mixer_dim + metadata_dim, mixer_dim)

    def forward(
        self,
        tokens: torch.Tensor,
        channel_metadata: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B, C, N, D], got {tuple(tokens.shape)}")

        batch, channels, patches, dim = tokens.shape
        if dim != self.mixer_dim:
            raise ValueError(f"Expected mixer dim {self.mixer_dim}, got {dim}")

        if channel_mask is not None:
            if channel_mask.shape != (batch, channels):
                raise ValueError(
                    f"Expected channel_mask [B, C] = {(batch, channels)}, got {tuple(channel_mask.shape)}"
                )
            channel_mask = channel_mask.to(device=tokens.device, dtype=torch.bool)

        patch_tokens = tokens.permute(0, 2, 1, 3)  # [B, N, C, D]
        metadata_norm_mean = None
        metadata_norm_std = None
        metadata_norm_min = None
        metadata_norm_max = None
        if channel_metadata is not None:
            if channel_metadata.shape != (batch, channels, self.metadata_dim):
                raise ValueError(
                    f"Expected channel_metadata [B, C, Dm] = {(batch, channels, self.metadata_dim)}, "
                    f"got {tuple(channel_metadata.shape)}"
                )
            metadata_norms = channel_metadata.detach().norm(dim=-1)
            metadata_norm_mean = metadata_norms.mean()
            metadata_norm_std = metadata_norms.std(unbiased=False)
            metadata_norm_min = metadata_norms.min()
            metadata_norm_max = metadata_norms.max()
            metadata_patch = channel_metadata.unsqueeze(1).expand(batch, patches, channels, self.metadata_dim)
            kv_input = self.kv_input_norm(torch.cat([patch_tokens, metadata_patch], dim=-1))
            keys_raw = self.concat_key_proj(kv_input)
            values_raw = self.concat_value_proj(kv_input)
        else:
            keys_raw = self.key_proj(patch_tokens)
            values_raw = self.value_proj(patch_tokens)

        keys = keys_raw.reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        values = values_raw.reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        queries = self.query_bank.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0).unsqueeze(0)
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
        mixed = torch.einsum("bnhqc,bnhcd->bnhqd", attn, values)
        query_tokens = self._refine_query_tokens(
            mixed,
            batch=batch,
            patches=patches,
        )
        latent_tokens = query_tokens.permute(0, 2, 1, 3)
        mixed = query_tokens.reshape(batch, patches, self.num_queries * self.mixer_dim)
        return self.out_proj(mixed), self._specialization_loss(attn), attn, {
            "latent_tokens": latent_tokens,
            "metadata_norm_mean": metadata_norm_mean,
            "metadata_norm_std": metadata_norm_std,
            "metadata_norm_min": metadata_norm_min,
            "metadata_norm_max": metadata_norm_max,
        }


class MetadataGuidedInterChannelAdapter(nn.Module):
    """Optional post-encoder inter-channel adapter for CI Laya."""

    def __init__(
        self,
        token_dim: int,
        *,
        num_heads: int = 4,
        dropout: float = 0.1,
        relation_scale_init: float = 1e-3,
        use_metadata_bias: bool = True,
        use_metadata_gate: bool = True,
        metadata_scale_init: float = 1e-3,
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if token_dim % num_heads != 0:
            raise ValueError(
                f"token_dim={token_dim} must be divisible by num_heads={num_heads}"
            )
        self.token_dim = int(token_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.token_dim // self.num_heads
        self.use_metadata_bias = bool(use_metadata_bias)
        self.use_metadata_gate = bool(use_metadata_gate)

        self.q_proj = nn.Linear(self.token_dim, self.token_dim)
        self.k_proj = nn.Linear(self.token_dim, self.token_dim)
        self.v_proj = nn.Linear(self.token_dim, self.token_dim)
        self.out_proj = nn.Linear(self.token_dim, self.token_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        self.meta_proj = nn.Linear(self.token_dim, self.token_dim)
        self.meta_pair_mlp = nn.Sequential(
            nn.Linear(self.token_dim * 4, self.token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.token_dim, self.num_heads),
        )
        self.meta_gate_mlp = nn.Sequential(
            nn.Linear(self.token_dim, self.token_dim),
            nn.GELU(),
            nn.Linear(self.token_dim, 1),
        )

        self.relation_scale = nn.Parameter(torch.tensor(float(relation_scale_init)))
        self.metadata_scale = nn.Parameter(torch.tensor(float(metadata_scale_init)))

    def _prepare_metadata(
        self,
        metadata: Optional[torch.Tensor],
        *,
        batch: int,
        channels: int,
        z: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if metadata is None:
            return None
        metadata = metadata.to(device=z.device, dtype=z.dtype)
        if metadata.dim() == 2:
            if metadata.shape[0] != channels:
                raise ValueError(
                    f"Expected metadata [C, M] with C={channels}, got {tuple(metadata.shape)}"
                )
            metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
        elif metadata.dim() == 3:
            if metadata.shape[:2] != (batch, channels):
                raise ValueError(
                    f"Expected metadata [B, C, M] = {(batch, channels, 'M')}, got {tuple(metadata.shape)}"
                )
        else:
            raise ValueError(
                f"Unsupported metadata shape for relation adapter: {tuple(metadata.shape)}"
            )
        if metadata.shape[-1] != self.token_dim:
            raise ValueError(
                "Relation adapter expects projected metadata width to match token_dim. "
                f"Expected {self.token_dim}, got {metadata.shape[-1]}."
            )
        return metadata

    def forward(
        self,
        z: torch.Tensor,
        *,
        metadata: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        if z.dim() != 4:
            raise ValueError(f"Expected relation adapter input [B, C, L, D], got {tuple(z.shape)}")

        batch, channels, patches, dim = z.shape
        if dim != self.token_dim:
            raise ValueError(
                f"Expected relation adapter token_dim={self.token_dim}, got {dim}"
            )
        if channels <= 1:
            return z, {
                "relation_adapter_scale": self.relation_scale.detach(),
                "relation_adapter_metadata_scale": self.metadata_scale.detach(),
                "relation_adapter_gate_mean": None,
                "relation_adapter_metadata_bias_mean_abs": None,
                "relation_adapter_metadata_present": z.new_tensor(0.0),
                "relation_adapter_metadata_nonzero_fraction": None,
                "relation_adapter_input_norm_mean": z.detach().norm(dim=-1).mean(),
                "relation_adapter_output_norm_mean": z.new_tensor(0.0),
                "relation_adapter_update_norm_mean": z.new_tensor(0.0),
                "relation_adapter_delta_ratio": z.new_tensor(0.0),
                "relation_adapter_attention_entropy": None,
                "relation_adapter_metadata_shape": None,
                "relation_adapter_attention": None,
            }

        if channel_mask is not None:
            if channel_mask.shape != (batch, channels):
                raise ValueError(
                    f"Expected channel_mask [B, C] = {(batch, channels)}, got {tuple(channel_mask.shape)}"
                )
            channel_mask = channel_mask.to(device=z.device, dtype=torch.bool)
            empty_rows = ~channel_mask.any(dim=1)
            if empty_rows.any():
                channel_mask = channel_mask.clone()
                channel_mask[empty_rows] = True

        z_bl = z.permute(0, 2, 1, 3).reshape(batch * patches, channels, dim)
        q = self.q_proj(z_bl).reshape(batch * patches, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(z_bl).reshape(batch * patches, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(z_bl).reshape(batch * patches, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)

        prepared_metadata = self._prepare_metadata(
            metadata,
            batch=batch,
            channels=channels,
            z=z,
        )
        projected_metadata = None
        metadata_bias = None
        metadata_nonzero_fraction = None
        if prepared_metadata is not None:
            metadata_nonzero_fraction = (
                (prepared_metadata.detach().abs() > 0).float().mean()
            )
            projected_metadata = self.meta_proj(
                prepared_metadata.reshape(-1, prepared_metadata.shape[-1])
            ).reshape(batch, channels, dim)
            if self.use_metadata_bias:
                meta_i = projected_metadata.unsqueeze(2).expand(-1, -1, channels, -1)
                meta_j = projected_metadata.unsqueeze(1).expand(-1, channels, -1, -1)
                pairwise_features = torch.cat(
                    [meta_i, meta_j, (meta_i - meta_j).abs(), meta_i * meta_j],
                    dim=-1,
                )
                metadata_bias = self.meta_pair_mlp(pairwise_features).permute(0, 3, 1, 2)
                metadata_bias = (
                    metadata_bias.unsqueeze(1)
                    .expand(batch, patches, self.num_heads, channels, channels)
                    .reshape(batch * patches, self.num_heads, channels, channels)
                )
                scores = scores + self.metadata_scale.to(
                    device=z.device,
                    dtype=z.dtype,
                ) * metadata_bias

        if channel_mask is not None:
            scores = scores.masked_fill(
                ~channel_mask[:, None, None, :]
                .expand(batch, patches, channels, channels)
                .reshape(batch * patches, 1, channels, channels),
                torch.finfo(scores.dtype).min,
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)
        attn_entropy = -(attn.clamp_min(1e-12) * attn.clamp_min(1e-12).log()).sum(dim=-1).mean()
        out = torch.matmul(attn, v).permute(0, 2, 1, 3).reshape(batch * patches, channels, dim)
        out = self.out_proj(out)
        out = out.reshape(batch, patches, channels, dim).permute(0, 2, 1, 3)

        gate = None
        if self.use_metadata_gate and projected_metadata is not None:
            gate = torch.sigmoid(
                self.meta_gate_mlp(projected_metadata.reshape(-1, dim))
            ).reshape(batch, channels, 1)
            out = out * gate.unsqueeze(2)

        update = self.relation_scale.to(device=z.device, dtype=z.dtype) * self.output_dropout(out)
        z_out = z + update
        if z_out.shape != z.shape:
            raise RuntimeError(
                f"Relation adapter must preserve shape {tuple(z.shape)}, got {tuple(z_out.shape)}"
            )
        input_norm_mean = z.detach().norm(dim=-1).mean()
        output_norm_mean = out.detach().norm(dim=-1).mean()
        update_norm_mean = update.detach().norm(dim=-1).mean()
        delta_ratio = update_norm_mean / input_norm_mean.clamp_min(1e-12)
        return z_out, {
            "relation_adapter_scale": self.relation_scale.detach(),
            "relation_adapter_metadata_scale": self.metadata_scale.detach(),
            "relation_adapter_gate_mean": None if gate is None else gate.detach().mean(),
            "relation_adapter_metadata_bias_mean_abs": None
            if metadata_bias is None
            else metadata_bias.detach().abs().mean(),
            "relation_adapter_metadata_present": z.new_tensor(
                0.0 if prepared_metadata is None else 1.0
            ),
            "relation_adapter_metadata_nonzero_fraction": metadata_nonzero_fraction,
            "relation_adapter_input_norm_mean": input_norm_mean,
            "relation_adapter_output_norm_mean": output_norm_mean,
            "relation_adapter_update_norm_mean": update_norm_mean,
            "relation_adapter_delta_ratio": delta_ratio,
            "relation_adapter_attention_entropy": attn_entropy.detach(),
            "relation_adapter_metadata_shape": None
            if prepared_metadata is None
            else tuple(prepared_metadata.shape),
            "relation_adapter_attention": attn.reshape(
                batch,
                patches,
                self.num_heads,
                channels,
                channels,
            ),
        }


class DescriptionAwareQueryChannelMixer(MetadataAwareQueryChannelMixer):
    """Backward-compatible alias for the legacy laya_relation path."""


class DescriptionAwareInterChannelGate(nn.Module):
    """CHARM-style description-aware inter-channel attention gate."""

    def __init__(
        self,
        *,
        token_dim: int,
        metadata_dim: int,
        num_heads: int = 1,
        relation_metric: str = "projected_dot",
        relation_scale_init: float = 0.0,
        residual_scale_init: float = 0.0,
        gate_mode: str = "bias",
    ) -> None:
        super().__init__()
        if token_dim <= 0:
            raise ValueError(f"token_dim must be positive, got {token_dim}")
        if metadata_dim <= 0:
            raise ValueError(f"metadata_dim must be positive, got {metadata_dim}")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        self.token_dim = token_dim
        self.metadata_dim = metadata_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.relation_metric = normalize_description_relation_metric(relation_metric)
        self.gate_mode = normalize_description_gate_mode(gate_mode)

        self.input_norm = nn.LayerNorm(token_dim)
        self.qkv_proj = nn.Linear(token_dim, token_dim * 3)
        self.out_proj = nn.Linear(token_dim, token_dim)

        self.desc_q_proj = nn.Linear(metadata_dim, token_dim)
        self.desc_k_proj = nn.Linear(metadata_dim, token_dim)
        self.meta_pair_norm = nn.LayerNorm(metadata_dim)
        self.meta_bilinear = nn.Parameter(torch.empty(num_heads, metadata_dim, metadata_dim))

        self.relation_scale = nn.Parameter(torch.full((num_heads,), float(relation_scale_init)))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_scale_init)))
        # CHARM-style suppression should behave like a real attention penalty
        # rather than a near-zero additive tweak at initialization.
        self.suppression_penalty_base = 5.0
        nn.init.xavier_uniform_(self.meta_bilinear)

    def _prepare_metadata(
        self,
        channel_metadata_embeddings: Optional[torch.Tensor],
        *,
        batch: int,
        channels: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if channel_metadata_embeddings is None:
            raise ValueError(
                "channel metadata embeddings are required for attention-gated metadata fusion."
            )
        metadata = channel_metadata_embeddings.to(device=x.device, dtype=x.dtype)
        if metadata.dim() == 2:
            if metadata.shape[0] != channels:
                raise ValueError(
                    f"Expected channel metadata embeddings [C, M] with C={channels}, got {tuple(metadata.shape)}"
                )
            metadata = metadata.unsqueeze(0).expand(batch, -1, -1)
        elif metadata.dim() == 3:
            if metadata.shape[:2] != (batch, channels):
                raise ValueError(
                    f"Expected channel metadata embeddings [B, C, M] = {(batch, channels, 'M')}, got {tuple(metadata.shape)}"
                )
        else:
            raise ValueError(f"Unsupported channel metadata embeddings shape: {tuple(metadata.shape)}")
        return metadata

    def _description_relation(self, metadata: torch.Tensor) -> torch.Tensor:
        batch, channels, _ = metadata.shape
        desc_q = self.desc_q_proj(metadata).reshape(batch, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        desc_k = self.desc_k_proj(metadata).reshape(batch, channels, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        if self.relation_metric == "cosine":
            desc_q = F.normalize(desc_q, dim=-1)
            desc_k = F.normalize(desc_k, dim=-1)
            desc_relation = torch.matmul(desc_q, desc_k.transpose(-2, -1))
            return torch.tanh(desc_relation)
        relation = torch.matmul(desc_q, desc_k.transpose(-2, -1))
        return torch.tanh(relation / math.sqrt(self.head_dim))

    def _description_suppression_gate(
        self,
        metadata: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        metadata = self.meta_pair_norm(metadata)
        if self.relation_metric == "cosine":
            similarity_base = F.normalize(metadata, dim=-1)
            similarity = torch.matmul(similarity_base, similarity_base.transpose(-2, -1))
        else:
            similarity = torch.matmul(metadata, metadata.transpose(-2, -1))
            similarity = torch.tanh(similarity / math.sqrt(metadata.shape[-1]))
        similarity = similarity.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        threshold = torch.einsum("bid,hde,bje->bhij", metadata, self.meta_bilinear, metadata)
        threshold = torch.sigmoid(threshold)
        gate = torch.relu(threshold - similarity)
        diagonal = torch.eye(
            metadata.shape[1],
            device=metadata.device,
            dtype=metadata.dtype,
        ).view(1, 1, metadata.shape[1], metadata.shape[1])
        gate = gate * (1.0 - diagonal)
        return similarity, threshold, gate

    def _effective_relation_scale(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw = self.relation_scale.to(device=device, dtype=dtype)
        if self.gate_mode == "suppress":
            # Map raw scale to a strictly-positive, mask-like penalty so
            # suppression is active from the start and can still grow.
            return self.suppression_penalty_base * F.softplus(raw)
        return raw

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        channel_metadata_embeddings: Optional[torch.Tensor],
        channel_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor | None]]:
        if tokens.dim() != 4:
            raise ValueError(f"Expected [B, C, N, D], got {tuple(tokens.shape)}")

        batch, channels, patches, dim = tokens.shape
        if dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {dim}")

        metadata = self._prepare_metadata(
            channel_metadata_embeddings,
            batch=batch,
            channels=channels,
            x=tokens,
        )
        if channel_mask is not None:
            if channel_mask.shape != (batch, channels):
                raise ValueError(
                    f"Expected channel_mask [B, C] = {(batch, channels)}, got {tuple(channel_mask.shape)}"
                )
            channel_mask = channel_mask.to(device=tokens.device, dtype=torch.bool)

        patch_tokens = tokens.permute(0, 2, 1, 3)
        normalized = self.input_norm(patch_tokens)
        qkv = self.qkv_proj(normalized).reshape(
            batch,
            patches,
            channels,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=3)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        v = v.permute(0, 1, 3, 2, 4)

        signal_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        relation_scale = self._effective_relation_scale(device=tokens.device, dtype=tokens.dtype)
        lambda_rel = relation_scale.view(1, 1, self.num_heads, 1, 1)
        relation_scores: torch.Tensor
        relation_threshold: torch.Tensor | None = None
        relation_gate: torch.Tensor | None = None
        if self.gate_mode == "bias":
            relation_scores = self._description_relation(metadata)  # [B, H, C, C]
            attn_score = signal_score + lambda_rel * relation_scores[:, None, :, :, :]
        else:
            relation_scores, relation_threshold, relation_gate = self._description_suppression_gate(metadata)
            attn_score = signal_score - lambda_rel * relation_gate[:, None, :, :, :]

        if channel_mask is not None:
            key_mask = channel_mask[:, None, None, None, :]
            attn_score = attn_score.masked_fill(~key_mask, torch.finfo(attn_score.dtype).min)

        attn = torch.softmax(attn_score, dim=-1)
        out = torch.matmul(attn, v)
        if channel_mask is not None:
            query_mask = channel_mask[:, None, None, :, None]
            out = out.masked_fill(~query_mask, 0.0)
        out = out.permute(0, 1, 3, 2, 4).reshape(batch, patches, channels, dim)
        out = self.out_proj(out).permute(0, 2, 1, 3)
        residual_scale = self.residual_scale.to(device=tokens.device, dtype=tokens.dtype)
        refined_tokens = tokens + (residual_scale * out)
        aux = {
            "relation_scores": relation_scores,
            "relation_scale": relation_scale.detach(),
            "relation_threshold": relation_threshold,
            "relation_gate": relation_gate,
            "signal_score_mean_abs": signal_score.detach().abs().mean(),
            "score_delta_mean_abs": (attn_score.detach() - signal_score.detach()).abs().mean(),
            "relation_scores_mean_abs": relation_scores.detach().abs().mean(),
            "relation_threshold_mean": None if relation_threshold is None else relation_threshold.detach().mean(),
            "relation_gate_mean": None if relation_gate is None else relation_gate.detach().mean(),
            "relation_gate_sparsity": None if relation_gate is None else (relation_gate.detach().abs() <= 1e-6).float().mean(),
            "metadata_norm_mean": metadata.detach().norm(dim=-1).mean(),
            "metadata_norm_std": metadata.detach().norm(dim=-1).std(unbiased=False),
            "metadata_norm_min": metadata.detach().norm(dim=-1).min(),
            "metadata_norm_max": metadata.detach().norm(dim=-1).max(),
        }
        return refined_tokens, attn, aux


DescriptionRelationChannelRefiner = DescriptionAwareInterChannelGate


class LayaTSChannelRelationBlock(nn.Module):
    """CHARM-style patch-wise metadata-gated channel relation block for laya_ts.

    This stays local to laya_ts so TS-specific relation experiments do not have
    to rely on the shared standalone Laya backbone implementation.
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
        self.meta_pair_norm = nn.LayerNorm(token_dim)
        self.meta_bilinear = nn.Parameter(torch.empty(num_heads, token_dim, token_dim))
        self.metadata_gate_scale = nn.Parameter(torch.tensor(0.01))
        self.residual_scale = nn.Parameter(torch.tensor(0.05))
        nn.init.xavier_uniform_(self.meta_bilinear)

    def _metadata_gate_bias(self, metadata: torch.Tensor) -> torch.Tensor:
        metadata = self.meta_pair_norm(metadata)
        similarity = torch.matmul(metadata, metadata.transpose(-2, -1)) / math.sqrt(metadata.shape[-1])
        threshold = torch.einsum("bid,hde,bje->bhij", metadata, self.meta_bilinear, metadata)
        threshold = torch.sigmoid(threshold)
        gate = torch.relu(threshold - similarity.unsqueeze(1))
        return gate

    def forward(
        self,
        tokens: torch.Tensor,
        metadata: torch.Tensor,
        channel_mask: Optional[torch.Tensor] = None,
    ):
        if tokens.dim() != 4:
            raise ValueError(f"Expected tokens [B, C, N, D], got {tuple(tokens.shape)}")
        if metadata.shape != (tokens.shape[0], tokens.shape[1], tokens.shape[3]):
            raise ValueError(
                f"Expected metadata [B, C, D] = {(tokens.shape[0], tokens.shape[1], tokens.shape[3])}, got {tuple(metadata.shape)}"
            )

        batch, channels, patches, dim = tokens.shape
        tokens_patch_first = tokens.transpose(1, 2)
        metadata_patch = metadata.unsqueeze(1).expand(batch, patches, channels, dim)
        relation_input = self.input_norm(tokens_patch_first + metadata_patch)

        q = self.q_proj(relation_input).reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k_proj(relation_input).reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v = self.v_proj(tokens_patch_first).reshape(batch, patches, channels, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        token_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        metadata_scores = self._metadata_gate_bias(metadata)
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
        return refined, relation_attn, metadata_scores


class LayaTSEncoder(LayaEncoder):
    def __init__(self, config: Optional[LayaModelConfig] = None) -> None:
        super().__init__(config)
        raw_channel_mixer_type = str(self.config.channel_mixer_type).strip().lower().replace("-", "_")
        self.patchifier_mode = normalize_patchifier_mode(self.config.patchifier_mode)
        if str(self.config.model_id).strip().lower() == "laya_ci_multiscale":
            self.patchifier_mode = "multiscale"
        self.encoder_variant = normalize_encoder_variant(self.config.encoder_variant)
        self.temporal_patchifier_mode = normalize_temporal_patchifier_mode(self.config.temporal_patchifier_mode)
        self.charm_patchifier_fusion = normalize_charm_patchifier_fusion(self.config.charm_patchifier_fusion)
        self.metadata_fusion_mode = normalize_metadata_fusion_mode(self.config.metadata_fusion_mode)
        self.use_relation_adapter = bool(
            self.config.use_relation_adapter or raw_channel_mixer_type == "ci_adapter"
        )
        self.relation_adapter_position = normalize_relation_adapter_position(
            self.config.relation_adapter_position
        )
        self.channel_mixer_relation_mode = normalize_channel_mixer_relation_mode(
            self.config.channel_mixer_relation_mode
        )
        if self.channel_mixer_relation_mode == "description_relation":
            self.metadata_fusion_mode = "attention_gate"
            self.channel_mixer_relation_mode = "none"
            self.config.metadata_fusion_mode = "attention_gate"
            self.config.channel_mixer_relation_mode = "none"
        self.temporal_patchifier: CHARMLikeMultiScaleTemporalPatchifier | None = None
        self.charm_patchifier_residual_beta: nn.Parameter | None = None
        self.scale_text_projector: nn.Module | None = None
        self.multiscale_patch_embedders: nn.ModuleDict = nn.ModuleDict()
        self.description_relation_refiner: DescriptionRelationChannelRefiner | None = None
        self.relation_adapter: MetadataGuidedInterChannelAdapter | None = None
        if self.channel_mixer_relation_mode != "none" and self.channel_mixer_type != "mixer":
            raise ValueError(
                "channel_mixer_relation_mode requires channel_mixer_type='mixer'."
            )
        if self.use_relation_adapter and self.channel_mixer_type != "independent":
            raise ValueError(
                "use_relation_adapter requires channel_mixer_type='independent' (CI path)."
            )
        if self.use_relation_adapter and self.relation_adapter_position != "post_encoder":
            raise ValueError(
                "Only relation_adapter_position='post_encoder' is currently supported."
            )
        if self.metadata_fusion_mode in {"attention_gate", "attention_suppress_gate"}:
            if self.channel_metadata_mode not in {"text", "stats", "text_stats_avg", "text_stats_joint"}:
                raise ValueError(
                    "attention-gated metadata fusion currently requires "
                    "channel_metadata_mode to be one of: text, stats, text_stats_avg, text_stats_joint."
                )
            if self.config.use_channel_relation_block:
                raise ValueError(
                    "attention-gated metadata fusion should be used without "
                    "use_channel_relation_block so description-aware gating is applied only once."
                )
            if self.temporal_patchifier_mode != "fixed":
                raise ValueError(
                    "attention-gated metadata fusion requires temporal_patchifier_mode='fixed'."
                )
            if self.channel_mixer_relation_mode != "none":
                raise ValueError(
                    "attention-gated metadata fusion should be used with "
                    "channel_mixer_relation_mode='none' so QueryChannelMixer remains unchanged."
                )
        if self.metadata_fusion_mode == "concat_kv":
            if self.channel_mixer_type != "mixer":
                raise ValueError("concat_kv metadata fusion requires channel_mixer_type='mixer'.")
            if self.channel_mixer_relation_mode != "none":
                raise ValueError(
                    "concat_kv metadata fusion should be used with "
                    "channel_mixer_relation_mode='none' so K/V conditioning is applied only once."
                )
            if self.channel_metadata_mode == "none":
                raise ValueError(
                    "concat_kv metadata fusion requires channel_metadata_mode to provide per-channel metadata."
                )
        if self.temporal_patchifier_mode != "fixed":
            scale_gate_source = normalize_charm_scale_gate_source(self.config.charm_scale_gate_source)
            if self.temporal_patchifier_mode == "charm_like":
                scale_gate_source = "text"
            self.temporal_patchifier = CHARMLikeMultiScaleTemporalPatchifier(
                out_dim=self.channel_token_dim,
                kernel_sizes=self.config.charm_kernel_sizes,
                stride=self.config.charm_stride,
                dropout=self.config.charm_patchifier_dropout,
                scale_gate_source=scale_gate_source,
                scale_gate_temperature=self.config.charm_scale_gate_temperature,
                text_metadata_dim=self.config.text_metadata_dim,
            )
            self.scale_text_projector = self.temporal_patchifier.scale_text_projector
            if self.charm_patchifier_fusion == "residual":
                self.charm_patchifier_residual_beta = nn.Parameter(
                    torch.tensor(float(self.config.charm_patchifier_residual_init))
                )
        if self.patchifier_mode == "multiscale":
            for patch_size in self.config.multiscale_patch_sizes:
                if patch_size == self.config.patch_size:
                    continue
                self.multiscale_patch_embedders[str(patch_size)] = TemporalPatchEmbedding(
                    patch_size,
                    self.channel_token_dim,
                )

        self.relation_channel_encoding: Optional[FourierChannelEncoding] = None
        if self.channel_metadata_mode == "coordinates":
            self.relation_channel_encoding = FourierChannelEncoding(
                self.channel_token_dim,
                num_bands=self.config.fourier_num_bands,
            )

        self.relation_channel_id_projector: nn.Module | None = None
        if self.channel_metadata_mode == "onehot":
            self.relation_channel_id_projector = nn.Sequential(
                nn.Linear(self.config.onehot_channel_vocab_size, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )

        self.relation_channel_text_projector: nn.Module | None = None
        if self.channel_metadata_mode in {"text", "text_stats_avg", "text_stats_joint"}:
            self.relation_channel_text_projector = nn.Sequential(
                nn.Linear(self.config.text_metadata_dim, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )
        self.relation_channel_stats_projector: nn.Module | None = None
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            self.relation_channel_stats_projector = nn.Sequential(
                nn.Linear(self.config.stats_metadata_dim, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )

        self.mixer_relation_channel_encoding: Optional[FourierChannelEncoding] = None
        if self.channel_metadata_mode == "coordinates":
            self.mixer_relation_channel_encoding = FourierChannelEncoding(
                self.channel_token_dim,
                num_bands=self.config.fourier_num_bands,
            )

        self.mixer_relation_channel_id_projector: nn.Module | None = None
        if self.channel_metadata_mode == "onehot":
            self.mixer_relation_channel_id_projector = nn.Sequential(
                nn.Linear(self.config.onehot_channel_vocab_size, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )

        self.mixer_relation_channel_text_projector: nn.Module | None = None
        if self.channel_metadata_mode in {"text", "text_stats_avg", "text_stats_joint"}:
            self.mixer_relation_channel_text_projector = nn.Sequential(
                nn.Linear(self.config.text_metadata_dim, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )
        self.mixer_relation_channel_stats_projector: nn.Module | None = None
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            self.mixer_relation_channel_stats_projector = nn.Sequential(
                nn.Linear(self.config.stats_metadata_dim, self.channel_token_dim),
                nn.LayerNorm(self.channel_token_dim),
            )

        if self.channel_mixer_type == "mixer" and self.channel_mixer_relation_mode == "laya_relation":
            self.channel_mixer = DescriptionAwareQueryChannelMixer(
                mixer_dim=self.config.channel_mixer_dim,
                encoder_dim=self.config.embed_dim,
                num_queries=self.config.num_queries,
                num_heads=self.config.channel_mixer_heads,
                metadata_dim=self.channel_token_dim,
                relation_scale_init=self.config.channel_mixer_relation_scale_init,
            )
        elif self.channel_mixer_type == "mixer" and self.channel_mixer_relation_mode == "metadata_query_gate":
            self.channel_mixer = MetadataAwareQueryChannelMixer(
                mixer_dim=self.config.channel_mixer_dim,
                encoder_dim=self.config.embed_dim,
                num_queries=self.config.num_queries,
                num_heads=self.config.channel_mixer_heads,
                metadata_dim=self.channel_token_dim,
                relation_scale_init=self.config.channel_mixer_relation_scale_init,
            )
        elif self.channel_mixer_type == "mixer" and self.metadata_fusion_mode == "concat_kv":
            self.channel_mixer = ConcatProjectedMetadataQueryChannelMixer(
                mixer_dim=self.config.channel_mixer_dim,
                encoder_dim=self.config.embed_dim,
                num_queries=self.config.num_queries,
                num_heads=self.config.channel_mixer_heads,
                metadata_dim=self.channel_token_dim,
            )
        elif self.channel_mixer_type == "mixer" and self.metadata_fusion_mode in {"attention_gate", "attention_suppress_gate"}:
            self.description_relation_refiner = DescriptionAwareInterChannelGate(
                token_dim=self.config.channel_mixer_dim,
                metadata_dim=self.channel_token_dim,
                num_heads=self.config.channel_mixer_heads,
                relation_metric=self.config.description_relation_metric,
                relation_scale_init=self.config.description_relation_lambda_init,
                residual_scale_init=self.config.description_relation_gamma_init,
                gate_mode="suppress" if self.metadata_fusion_mode == "attention_suppress_gate" else "bias",
            )

        if self.config.use_channel_relation_block:
            self.channel_relation_block = LayaTSChannelRelationBlock(
                token_dim=self.channel_token_dim,
                num_heads=self.config.channel_relation_heads,
            )
            self.channel_relation_block.metadata_gate_scale.data.fill_(self.config.channel_relation_gate_scale_init)
            self.channel_relation_block.residual_scale.data.fill_(self.config.channel_relation_residual_scale_init)
        else:
            self.channel_relation_block = None

        if self.use_relation_adapter:
            self.relation_adapter = MetadataGuidedInterChannelAdapter(
                self.config.embed_dim,
                num_heads=self.config.relation_num_heads,
                dropout=self.config.relation_dropout,
                relation_scale_init=self.config.relation_scale_init,
                use_metadata_bias=self.config.use_metadata_bias,
                use_metadata_gate=self.config.use_metadata_gate,
                metadata_scale_init=self.config.metadata_scale_init,
            )

    def infer_num_patches(self, time: int) -> int:
        return infer_temporal_patchifier_num_patches(self.config, time)

    def _should_add_channel_metadata_to_tokens(self) -> bool:
        return self.metadata_fusion_mode == "add"

    def _prepare_metadata_tensor(
        self,
        value: Optional[torch.Tensor],
        *,
        name: str,
        batch: int,
        channels: int,
        x: torch.Tensor,
    ) -> torch.Tensor:
        if value is None:
            raise ValueError(f"{name} are required when channel_metadata_mode='{self.channel_metadata_mode}'.")
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

    def _resolve_projected_metadata(
        self,
        *,
        batch: int,
        channels: int,
        x: torch.Tensor,
        channel_text_embeddings: Optional[torch.Tensor],
        channel_stats_embeddings: Optional[torch.Tensor],
        text_projector: nn.Module | None,
        stats_projector: nn.Module | None,
    ) -> torch.Tensor:
        if self.channel_metadata_mode in {"text", "text_stats_joint"}:
            metadata = self._prepare_metadata_tensor(
                channel_text_embeddings,
                name="channel_text_embeddings",
                batch=batch,
                channels=channels,
                x=x,
            )
            if text_projector is None:
                raise RuntimeError("text projector is not initialized.")
            projected = text_projector(metadata.reshape(-1, metadata.shape[-1]))
            return projected.reshape(batch, channels, self.channel_token_dim)

        if self.channel_metadata_mode == "stats":
            metadata = self._prepare_metadata_tensor(
                channel_stats_embeddings,
                name="channel_stats_embeddings",
                batch=batch,
                channels=channels,
                x=x,
            )
            if stats_projector is None:
                raise RuntimeError("stats projector is not initialized.")
            projected = stats_projector(metadata.reshape(-1, metadata.shape[-1]))
            return projected.reshape(batch, channels, self.channel_token_dim)

        if self.channel_metadata_mode == "text_stats_avg":
            text_metadata = self._prepare_metadata_tensor(
                channel_text_embeddings,
                name="channel_text_embeddings",
                batch=batch,
                channels=channels,
                x=x,
            )
            stats_metadata = self._prepare_metadata_tensor(
                channel_stats_embeddings,
                name="channel_stats_embeddings",
                batch=batch,
                channels=channels,
                x=x,
            )
            if text_projector is None:
                raise RuntimeError("text projector is not initialized.")
            if stats_projector is None:
                raise RuntimeError("stats projector is not initialized.")
            projected_text = text_projector(
                text_metadata.reshape(-1, text_metadata.shape[-1])
            ).reshape(batch, channels, self.channel_token_dim)
            projected_stats = stats_projector(
                stats_metadata.reshape(-1, stats_metadata.shape[-1])
            ).reshape(batch, channels, self.channel_token_dim)
            return 0.5 * (projected_text + projected_stats)

        raise AssertionError("Projected metadata is only defined for text/stats/text_stats_avg/text_stats_joint modes")

    def _resolve_relation_metadata(
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
                raise ValueError("channel_positions are required when channel_metadata_mode='coordinates'.")
            if channel_positions.dim() != 3 or channel_positions.shape[-1] != 3:
                raise ValueError(f"Expected channel_positions [B, C, 3], got {tuple(channel_positions.shape)}")
            if channel_positions.shape[:2] != (batch, channels):
                raise ValueError(f"channel_positions shape {tuple(channel_positions.shape)} does not match input {(batch, channels)}")
            if self.relation_channel_encoding is None:
                raise RuntimeError("relation_channel_encoding is not initialized.")
            return self.relation_channel_encoding(channel_positions.to(device=x.device, dtype=x.dtype))

        if self.channel_metadata_mode == "onehot":
            if self.relation_channel_id_projector is None:
                raise RuntimeError("relation_channel_id_projector is not initialized.")
            if channels > self.config.onehot_channel_vocab_size:
                raise ValueError(
                    f"Input channels {channels} exceed onehot_channel_vocab_size={self.config.onehot_channel_vocab_size}."
                )
            channel_ids = torch.arange(channels, device=x.device)
            one_hot = F.one_hot(channel_ids, num_classes=self.config.onehot_channel_vocab_size).to(dtype=x.dtype)
            projected = self.relation_channel_id_projector(one_hot)
            return projected.unsqueeze(0).expand(batch, -1, -1)

        if self.channel_metadata_mode in {"text", "stats", "text_stats_avg", "text_stats_joint"}:
            return self._resolve_projected_metadata(
                batch=batch,
                channels=channels,
                x=x,
                channel_text_embeddings=channel_text_embeddings,
                channel_stats_embeddings=channel_stats_embeddings,
                text_projector=self.relation_channel_text_projector,
                stats_projector=self.relation_channel_stats_projector,
            )

        if self.channel_metadata_mode == "none":
            return x.new_zeros(batch, channels, self.channel_token_dim)

        raise AssertionError("Unsupported channel metadata mode")

    def _resolve_optional_relation_metadata(
        self,
        *,
        batch: int,
        channels: int,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor],
        channel_text_embeddings: Optional[torch.Tensor],
        channel_stats_embeddings: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if self.channel_metadata_mode == "none":
            return None
        if self.channel_metadata_mode == "coordinates":
            if channel_positions is None:
                return None
        elif self.channel_metadata_mode == "text":
            if channel_text_embeddings is None:
                return None
        elif self.channel_metadata_mode == "stats":
            if channel_stats_embeddings is None:
                return None
        elif self.channel_metadata_mode == "text_stats_avg":
            if channel_text_embeddings is None or channel_stats_embeddings is None:
                return None
        elif self.channel_metadata_mode == "text_stats_joint":
            if channel_text_embeddings is None:
                return None
        return self._resolve_relation_metadata(
            batch=batch,
            channels=channels,
            x=x,
            channel_positions=channel_positions,
            channel_text_embeddings=channel_text_embeddings,
            channel_stats_embeddings=channel_stats_embeddings,
        )

    def _resolve_mixer_relation_metadata(
        self,
        *,
        batch: int,
        channels: int,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor],
        channel_text_embeddings: Optional[torch.Tensor],
        channel_stats_embeddings: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if (
            self.channel_mixer_relation_mode not in {"laya_relation", "metadata_query_gate"}
            and self.metadata_fusion_mode != "concat_kv"
        ):
            return None

        if self.channel_metadata_mode == "coordinates":
            if channel_positions is None:
                raise ValueError("channel_positions are required when channel_metadata_mode='coordinates'.")
            if channel_positions.dim() != 3 or channel_positions.shape[-1] != 3:
                raise ValueError(f"Expected channel_positions [B, C, 3], got {tuple(channel_positions.shape)}")
            if channel_positions.shape[:2] != (batch, channels):
                raise ValueError(f"channel_positions shape {tuple(channel_positions.shape)} does not match input {(batch, channels)}")
            if self.mixer_relation_channel_encoding is None:
                raise RuntimeError("mixer_relation_channel_encoding is not initialized.")
            return self.mixer_relation_channel_encoding(channel_positions.to(device=x.device, dtype=x.dtype))

        if self.channel_metadata_mode == "onehot":
            if self.mixer_relation_channel_id_projector is None:
                raise RuntimeError("mixer_relation_channel_id_projector is not initialized.")
            if channels > self.config.onehot_channel_vocab_size:
                raise ValueError(
                    f"Input channels {channels} exceed onehot_channel_vocab_size={self.config.onehot_channel_vocab_size}."
                )
            channel_ids = torch.arange(channels, device=x.device)
            one_hot = F.one_hot(channel_ids, num_classes=self.config.onehot_channel_vocab_size).to(dtype=x.dtype)
            projected = self.mixer_relation_channel_id_projector(one_hot)
            return projected.unsqueeze(0).expand(batch, -1, -1)

        if self.channel_metadata_mode in {"text", "stats", "text_stats_avg", "text_stats_joint"}:
            return self._resolve_projected_metadata(
                batch=batch,
                channels=channels,
                x=x,
                channel_text_embeddings=channel_text_embeddings,
                channel_stats_embeddings=channel_stats_embeddings,
                text_projector=self.mixer_relation_channel_text_projector,
                stats_projector=self.mixer_relation_channel_stats_projector,
            )

        if self.channel_metadata_mode == "none":
            return None

        raise AssertionError("Unsupported channel metadata mode")

    def _resolve_channel_tokens(
        self,
        x: torch.Tensor,
        channel_text_embeddings: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, object]]:
        fixed_tokens = self.patch_embed(x)
        if self.temporal_patchifier_mode == "fixed" or self.temporal_patchifier is None:
            return fixed_tokens, {
                "scale_gate": None,
                "scale_logits": None,
                "scale_gate_mean": None,
                "patchifier_N": fixed_tokens.shape[2],
                "branch_patch_counts": (fixed_tokens.shape[2],),
            }

        charm_tokens, aux = self.temporal_patchifier(
            x,
            channel_text_embeddings=channel_text_embeddings,
        )
        aux = dict(aux)
        if self.charm_patchifier_fusion == "replace":
            return charm_tokens, aux

        if self.charm_patchifier_residual_beta is None:
            raise RuntimeError("charm_patchifier_residual_beta is not initialized for residual fusion.")

        aligned_patches = min(fixed_tokens.shape[2], charm_tokens.shape[2])
        fixed_tokens = fixed_tokens[:, :, :aligned_patches, :]
        charm_tokens = charm_tokens[:, :, :aligned_patches, :]
        aux["patchifier_N"] = aligned_patches
        channel_tokens = fixed_tokens + self.charm_patchifier_residual_beta.to(
            device=fixed_tokens.device,
            dtype=fixed_tokens.dtype,
        ) * charm_tokens
        return channel_tokens, aux

    def embed_with_patch_size(self, x: torch.Tensor, patch_size: int) -> torch.Tensor:
        if patch_size == self.config.patch_size:
            return self.patch_embed(x)
        branch_key = str(int(patch_size))
        branch = self.multiscale_patch_embedders[branch_key] if branch_key in self.multiscale_patch_embedders else None
        if branch is None:
            branch = TemporalPatchEmbedding(int(patch_size), self.channel_token_dim).to(
                device=x.device,
                dtype=x.dtype,
            )
            self.multiscale_patch_embedders[branch_key] = branch
        return branch(x)

    def _resolve_channel_mask(
        self,
        *,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor],
        channel_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch, channels, _ = x.shape
        if channel_mask is None:
            inferred_mask = x.abs().sum(dim=-1) > 0
            if self.channel_metadata_mode == "coordinates" and channel_positions is not None:
                inferred_mask = inferred_mask | (
                    channel_positions.to(device=x.device, dtype=x.dtype).abs().sum(dim=-1) > 0
                )
            if inferred_mask.any() and not inferred_mask.all():
                return inferred_mask.to(device=x.device, dtype=torch.bool)
            return torch.ones(batch, channels, device=x.device, dtype=torch.bool)
        if channel_mask.shape != (batch, channels):
            raise ValueError(f"Expected channel_mask [B, C] = {(batch, channels)}, got {tuple(channel_mask.shape)}")
        return channel_mask.to(device=x.device, dtype=torch.bool)

    def _run_channel_mixer(
        self,
        channel_tokens: torch.Tensor,
        *,
        channel_mask: torch.Tensor,
        mixer_relation_metadata: Optional[torch.Tensor],
        refiner_metadata: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict[str, torch.Tensor | None]]:
        if self.channel_mixer is None:
            raise RuntimeError("channel_mixer is not initialized.")
        mixer_input = channel_tokens
        refiner_aux_outputs: dict[str, torch.Tensor | None] = {
            "token_scores": None,
            "relation_scores": None,
            "relation_scale": None,
            "relation_threshold": None,
            "relation_gate": None,
            "latent_tokens": None,
            "refined_tokens": None,
            "refiner_attention": None,
        }
        if self.description_relation_refiner is not None:
            mixer_input, refiner_attention, refiner_aux = self.description_relation_refiner(
                channel_tokens,
                channel_metadata_embeddings=refiner_metadata,
                channel_mask=channel_mask,
            )
            refiner_aux_outputs["relation_scores"] = refiner_aux.get("relation_scores")
            refiner_aux_outputs["relation_scale"] = refiner_aux.get("relation_scale")
            refiner_aux_outputs["relation_threshold"] = refiner_aux.get("relation_threshold")
            refiner_aux_outputs["relation_gate"] = refiner_aux.get("relation_gate")
            refiner_aux_outputs["refined_tokens"] = mixer_input
            refiner_aux_outputs["refiner_attention"] = refiner_attention
        outputs = self.channel_mixer(
            mixer_input,
            channel_metadata=mixer_relation_metadata,
            channel_mask=channel_mask,
        )
        if isinstance(outputs, tuple) and len(outputs) == 4:
            mixed_tokens, query_loss, affinity, mixer_aux = outputs
            merged_aux = {**refiner_aux_outputs, **mixer_aux}
            if merged_aux.get("refined_tokens") is None:
                merged_aux["refined_tokens"] = mixer_input if self.description_relation_refiner is not None else None
            return mixed_tokens, query_loss, affinity, merged_aux
        mixed_tokens, query_loss, affinity = outputs
        return mixed_tokens, query_loss, affinity, refiner_aux_outputs

    def _forward_features_default(
        self,
        x: torch.Tensor,
        *,
        channel_positions: Optional[torch.Tensor],
        channel_mask: Optional[torch.Tensor],
        patch_mask: Optional[torch.Tensor],
        channel_text_embeddings: Optional[torch.Tensor],
        channel_stats_embeddings: Optional[torch.Tensor],
        channel_tokens_override: Optional[torch.Tensor] = None,
        patchifier_aux_override: Optional[dict[str, object]] = None,
    ) -> dict[str, torch.Tensor]:
        batch, channels, _ = x.shape
        metadata_encoding: Optional[torch.Tensor] = None
        if self._should_add_channel_metadata_to_tokens():
            metadata_encoding = self._resolve_channel_metadata(
                batch=batch,
                channels=channels,
                x=x,
                channel_positions=channel_positions,
                channel_text_embeddings=channel_text_embeddings,
                channel_stats_embeddings=channel_stats_embeddings,
            )
        relation_metadata: Optional[torch.Tensor] = None
        if self.channel_relation_block is not None or self.description_relation_refiner is not None:
            relation_metadata = self._resolve_relation_metadata(
                batch=batch,
                channels=channels,
                x=x,
                channel_positions=channel_positions,
                channel_text_embeddings=channel_text_embeddings,
                channel_stats_embeddings=channel_stats_embeddings,
            )
        adapter_metadata: Optional[torch.Tensor] = None
        if self.use_relation_adapter:
            adapter_metadata = (
                relation_metadata
                if relation_metadata is not None
                else self._resolve_optional_relation_metadata(
                    batch=batch,
                    channels=channels,
                    x=x,
                    channel_positions=channel_positions,
                    channel_text_embeddings=channel_text_embeddings,
                    channel_stats_embeddings=channel_stats_embeddings,
                )
            )
        mixer_relation_metadata = self._resolve_mixer_relation_metadata(
            batch=batch,
            channels=channels,
            x=x,
            channel_positions=channel_positions,
            channel_text_embeddings=channel_text_embeddings,
            channel_stats_embeddings=channel_stats_embeddings,
        )

        resolved_channel_mask = self._resolve_channel_mask(
            x=x,
            channel_positions=channel_positions,
            channel_mask=channel_mask,
        )
        if channel_tokens_override is None:
            channel_tokens, patchifier_aux = self._resolve_channel_tokens(
                x,
                channel_text_embeddings=channel_text_embeddings,
            )
        else:
            channel_tokens = channel_tokens_override.to(device=x.device, dtype=x.dtype)
            patchifier_aux = dict(patchifier_aux_override or {})
            patchifier_aux.setdefault("scale_gate", None)
            patchifier_aux.setdefault("scale_logits", None)
            patchifier_aux.setdefault("scale_gate_mean", None)
            patchifier_aux.setdefault("patchifier_N", channel_tokens.shape[2])
            patchifier_aux.setdefault("branch_patch_counts", (channel_tokens.shape[2],))
        num_patches = channel_tokens.shape[2]

        if patch_mask is not None:
            if patch_mask.shape != (batch, num_patches):
                raise ValueError(f"Expected patch_mask [B, N] = {(batch, num_patches)}, got {tuple(patch_mask.shape)}")
            patch_mask = patch_mask.to(device=x.device, dtype=torch.bool)

        # This path preserves the original patch-first LayaTS flow. The
        # CHARM-like patchifier, when enabled, still returns [B, C, N, D] so
        # QueryChannelMixer continues to receive the exact same interface.
        if metadata_encoding is not None:
            channel_tokens = channel_tokens + metadata_encoding.unsqueeze(2)
        relation_affinity = None
        relation_gate = None

        if self.channel_relation_block is not None:
            if relation_metadata is None:
                raise RuntimeError("relation_metadata must be available when channel_relation_block is enabled.")
            channel_tokens, relation_affinity, relation_gate = self.channel_relation_block(
                channel_tokens,
                relation_metadata,
                channel_mask=resolved_channel_mask,
            )

        shared_outputs = {
            "encoder_variant": self.encoder_variant,
            "channel_tokens": channel_tokens,
            "relation_affinity": relation_affinity,
            "relation_gate": relation_gate,
            "scale_gate": patchifier_aux["scale_gate"],
            "scale_logits": patchifier_aux["scale_logits"],
            "scale_gate_mean": patchifier_aux["scale_gate_mean"],
            "patchifier_N": patchifier_aux["patchifier_N"],
            "branch_patch_counts": patchifier_aux.get("branch_patch_counts"),
            "patchifier_mode": self.temporal_patchifier_mode,
            "channel_mixer_relation_scores": None,
            "channel_mixer_relation_scale": None,
            "channel_mixer_relation_threshold": None,
            "channel_mixer_relation_gate": None,
            "channel_mixer_signal_score_mean_abs": None,
            "channel_mixer_score_delta_mean_abs": None,
            "channel_mixer_relation_scores_mean_abs": None,
            "channel_mixer_relation_threshold_mean": None,
            "channel_mixer_relation_gate_mean": None,
            "channel_mixer_relation_gate_sparsity": None,
            "channel_mixer_metadata_norm_mean": None,
            "channel_mixer_metadata_norm_std": None,
            "channel_mixer_metadata_norm_min": None,
            "channel_mixer_metadata_norm_max": None,
            "channel_mixer_latent_tokens": None,
            "channel_mixer_refined_tokens": None,
            "channel_mixer_refiner_attention": None,
            "relation_adapter_scale": None,
            "relation_adapter_metadata_scale": None,
            "relation_adapter_gate_mean": None,
            "relation_adapter_metadata_bias_mean_abs": None,
            "relation_adapter_metadata_present": None,
            "relation_adapter_metadata_nonzero_fraction": None,
            "relation_adapter_input_norm_mean": None,
            "relation_adapter_output_norm_mean": None,
            "relation_adapter_update_norm_mean": None,
            "relation_adapter_delta_ratio": None,
            "relation_adapter_attention_entropy": None,
            "relation_adapter_metadata_shape": None,
            "relation_adapter_attention": None,
        }

        if self.channel_mixer_type == "mixer":
            mixed_tokens, query_loss, affinity, mixer_aux = self._run_channel_mixer(
                channel_tokens,
                channel_mask=resolved_channel_mask,
                mixer_relation_metadata=mixer_relation_metadata,
                refiner_metadata=relation_metadata,
            )
            if patch_mask is not None:
                mixed_tokens = mixed_tokens.masked_fill(patch_mask.unsqueeze(-1), 0.0)
            encoded = mixed_tokens
            for block in self.blocks:
                encoded = block(encoded)
            encoded = self.norm(encoded)
            return {
                **shared_outputs,
                "mixed_tokens_pre_encoder": mixed_tokens,
                "mixed_tokens": encoded,
                "mixed_repr": encoded.mean(dim=1),
                "channel_repr": channel_tokens.mean(dim=2),
                "query_loss": query_loss,
                "channel_affinity": affinity,
                "channel_mixer_relation_scores": mixer_aux.get("relation_scores"),
                "channel_mixer_relation_scale": mixer_aux.get("relation_scale"),
                "channel_mixer_relation_threshold": mixer_aux.get("relation_threshold"),
                "channel_mixer_relation_gate": mixer_aux.get("relation_gate"),
                "channel_mixer_signal_score_mean_abs": mixer_aux.get("signal_score_mean_abs"),
                "channel_mixer_score_delta_mean_abs": mixer_aux.get("score_delta_mean_abs"),
                "channel_mixer_relation_scores_mean_abs": mixer_aux.get("relation_scores_mean_abs"),
                "channel_mixer_relation_threshold_mean": mixer_aux.get("relation_threshold_mean"),
                "channel_mixer_relation_gate_mean": mixer_aux.get("relation_gate_mean"),
                "channel_mixer_relation_gate_sparsity": mixer_aux.get("relation_gate_sparsity"),
                "channel_mixer_metadata_norm_mean": mixer_aux.get("metadata_norm_mean"),
                "channel_mixer_metadata_norm_std": mixer_aux.get("metadata_norm_std"),
                "channel_mixer_metadata_norm_min": mixer_aux.get("metadata_norm_min"),
                "channel_mixer_metadata_norm_max": mixer_aux.get("metadata_norm_max"),
                "channel_mixer_latent_tokens": mixer_aux.get("latent_tokens"),
                "channel_mixer_refined_tokens": mixer_aux.get("refined_tokens"),
                "channel_mixer_refiner_attention": mixer_aux.get("refiner_attention"),
            }

        independent_tokens = channel_tokens
        if patch_mask is not None:
            independent_tokens = independent_tokens.masked_fill(patch_mask[:, None, :, None], 0.0)
        flat_tokens = independent_tokens.reshape(batch * channels, num_patches, self.config.embed_dim)
        encoded = flat_tokens
        for block in self.blocks:
            encoded = block(encoded)
        encoded = self.norm(encoded)
        encoded = encoded.reshape(batch, channels, num_patches, self.config.embed_dim)
        adapter_outputs: dict[str, torch.Tensor | None] = {
            "relation_adapter_scale": None,
            "relation_adapter_metadata_scale": None,
            "relation_adapter_gate_mean": None,
            "relation_adapter_metadata_bias_mean_abs": None,
            "relation_adapter_metadata_present": None,
            "relation_adapter_metadata_nonzero_fraction": None,
            "relation_adapter_input_norm_mean": None,
            "relation_adapter_output_norm_mean": None,
            "relation_adapter_update_norm_mean": None,
            "relation_adapter_delta_ratio": None,
            "relation_adapter_attention_entropy": None,
            "relation_adapter_metadata_shape": None,
            "relation_adapter_attention": None,
        }
        if (
            self.use_relation_adapter
            and self.relation_adapter is not None
            and channels > 1
        ):
            relation_metadata_for_adapter = apply_metadata_dropout(
                adapter_metadata,
                self.config.metadata_dropout,
                self.training,
            )
            encoded, adapter_outputs = self.relation_adapter(
                encoded,
                metadata=relation_metadata_for_adapter,
                channel_mask=resolved_channel_mask,
            )
        channel_repr = encoded.mean(dim=2)
        return {
            **shared_outputs,
            "independent_tokens": encoded,
            "mixed_tokens_pre_encoder": flat_tokens,
            "mixed_tokens": encoded.reshape(batch * channels, num_patches, self.config.embed_dim),
            "mixed_repr": channel_repr.mean(dim=1),
            "channel_repr": channel_repr,
            "query_loss": encoded.new_zeros(()),
            "channel_affinity": None,
            **adapter_outputs,
        }

    def forward_features(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
        channel_text_embeddings: Optional[torch.Tensor] = None,
        channel_stats_embeddings: Optional[torch.Tensor] = None,
        channel_tokens_override: Optional[torch.Tensor] = None,
        patchifier_aux_override: Optional[dict[str, object]] = None,
    ):
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")
        return self._forward_features_default(
            x,
            channel_positions=channel_positions,
            channel_mask=channel_mask,
            patch_mask=patch_mask,
            channel_text_embeddings=channel_text_embeddings,
            channel_stats_embeddings=channel_stats_embeddings,
            channel_tokens_override=channel_tokens_override,
            patchifier_aux_override=patchifier_aux_override,
        )


class LayaTSPretrainer(LayaPretrainer):
    """Exact standalone Laya pretrainer reused for time-series data."""

    def __init__(self, config: Optional[LayaModelConfig] = None) -> None:
        super().__init__(config)
        self.encoder = LayaTSEncoder(self.config)
        self.patchifier_mode = normalize_patchifier_mode(self.config.patchifier_mode)
        if str(self.config.model_id).strip().lower() == "laya_ci_multiscale":
            self.patchifier_mode = "multiscale"
        self.multiscale_patch_sizes = tuple(int(value) for value in self.config.multiscale_patch_sizes)
        self.multiscale_base_patch = int(self.config.multiscale_base_patch)
        self.multiscale_fusion_gate: MultiScaleFusionGate | None = None
        if self.patchifier_mode == "multiscale":
            self.multiscale_fusion_gate = MultiScaleFusionGate(
                self.config.proj_dim,
                len(self.multiscale_patch_sizes),
                temperature=self.config.multiscale_gate_temperature,
            )

    def _use_multiscale_jepa(self) -> bool:
        return self.patchifier_mode == "multiscale"

    def _reshape_ci_tokens(
        self,
        tokens: torch.Tensor,
        *,
        batch: int,
        channels: int,
    ) -> torch.Tensor:
        if tokens.dim() != 3:
            raise ValueError(f"Expected CI tokens [B*C, N, D], got {tuple(tokens.shape)}")
        if tokens.shape[0] != batch * channels:
            raise ValueError(
                f"Expected leading dim {batch * channels} for CI tokens, got {tokens.shape[0]}"
            )
        return tokens.reshape(batch, channels, tokens.shape[1], tokens.shape[2])

    def _align_latents_to_base_grid(
        self,
        latents: torch.Tensor,
        *,
        patch_size: int,
        base_patch: int,
        base_num_patches: int,
    ) -> torch.Tensor:
        if latents.dim() != 4:
            raise ValueError(f"Expected latents [B, C, N, D], got {tuple(latents.shape)}")
        if patch_size == base_patch:
            return latents[:, :, :base_num_patches, :]
        if patch_size < base_patch:
            ratio = base_patch // patch_size
            if base_patch % patch_size != 0:
                raise ValueError(
                    f"Base patch {base_patch} must be divisible by patch size {patch_size}"
                )
            target_tokens = base_num_patches * ratio
            if latents.shape[2] < target_tokens:
                pad = target_tokens - latents.shape[2]
                latents = F.pad(latents, (0, 0, 0, pad))
            latents = latents[:, :, :target_tokens, :]
            batch, channels, _, dim = latents.shape
            return latents.reshape(batch, channels, base_num_patches, ratio, dim).mean(dim=3)
        ratio = patch_size // base_patch
        if patch_size % base_patch != 0:
            raise ValueError(
                f"Patch size {patch_size} must be divisible by base patch {base_patch}"
            )
        expanded = latents.repeat_interleave(ratio, dim=2)
        if expanded.shape[2] < base_num_patches:
            pad = base_num_patches - expanded.shape[2]
            expanded = F.pad(expanded, (0, 0, 0, pad))
        return expanded[:, :, :base_num_patches, :]

    def _base_mask_to_scale_mask(
        self,
        base_mask: torch.Tensor,
        *,
        patch_size: int,
        base_patch: int,
        scale_num_patches: int,
    ) -> torch.Tensor:
        if base_mask.dim() != 2:
            raise ValueError(f"Expected base_mask [B, N], got {tuple(base_mask.shape)}")
        if patch_size == base_patch:
            return base_mask[:, :scale_num_patches]
        if patch_size < base_patch:
            ratio = base_patch // patch_size
            repeated = base_mask.repeat_interleave(ratio, dim=1)
            if repeated.shape[1] < scale_num_patches:
                pad = scale_num_patches - repeated.shape[1]
                repeated = F.pad(repeated, (0, pad), value=False)
            return repeated[:, :scale_num_patches]
        ratio = patch_size // base_patch
        target_base_tokens = scale_num_patches * ratio
        if base_mask.shape[1] < target_base_tokens:
            pad = target_base_tokens - base_mask.shape[1]
            base_mask = F.pad(base_mask, (0, pad), value=False)
        reduced = base_mask[:, :target_base_tokens].reshape(base_mask.shape[0], scale_num_patches, ratio)
        return reduced.any(dim=-1)

    def _masked_mse(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(
                f"pred and target must share shape, got {tuple(pred.shape)} vs {tuple(target.shape)}"
            )
        if mask.shape != pred.shape[:-1]:
            raise ValueError(
                f"mask must match pred/target without latent dim, got {tuple(mask.shape)} vs {tuple(pred.shape[:-1])}"
            )
        masked_pred = pred[mask]
        masked_target = target[mask]
        if masked_pred.numel() == 0 or masked_target.numel() == 0:
            return pred.new_zeros(())
        return F.mse_loss(masked_pred, masked_target)

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
    ) -> dict[str, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"Expected [B, C, T], got {tuple(x.shape)}")

        if self.encoder.channel_metadata_mode == "coordinates" and channel_positions is None:
            raise ValueError(
                "channel_positions must be provided. "
                "They should be generated in dataset.py from channel_names."
            )

        if channel_positions is not None and (channel_positions.dim() != 3 or channel_positions.shape[-1] != 3):
            raise ValueError(
                f"Expected channel_positions [B, C, 3], got {tuple(channel_positions.shape)}"
            )

        if channel_positions is not None and channel_positions.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"channel_positions shape {tuple(channel_positions.shape)} does not match input shape {tuple(x.shape)}"
            )

        if self._use_multiscale_jepa():
            if self.encoder.channel_mixer_type != "independent":
                raise ValueError(
                    "laya_ci_multiscale currently supports channel_mixer_type='independent' only."
                )
            if self.multiscale_fusion_gate is None:
                raise RuntimeError("multiscale_fusion_gate is not initialized.")

            batch, channels, time = x.shape
            base_num_patches = math.ceil(time / self.multiscale_base_patch)
            base_patch_mask = self._resolve_patch_mask(
                batch_size=batch,
                num_patches=base_num_patches,
                device=x.device,
                patch_mask=patch_mask,
                patch_mask_seed=patch_mask_seed,
            )

            gate_input_scales: list[torch.Tensor] = []
            target_fused: torch.Tensor | None = None
            base_full: dict[str, torch.Tensor] | None = None
            base_context: dict[str, torch.Tensor] | None = None

            for patch_size in self.multiscale_patch_sizes:
                scale_channel_tokens = self.encoder.embed_with_patch_size(x, patch_size)
                scale_num_patches = scale_channel_tokens.shape[2]
                scale_patch_mask = self._base_mask_to_scale_mask(
                    base_patch_mask,
                    patch_size=patch_size,
                    base_patch=self.multiscale_base_patch,
                    scale_num_patches=scale_num_patches,
                )
                patchifier_aux = {
                    "scale_gate": None,
                    "scale_logits": None,
                    "scale_gate_mean": None,
                    "patchifier_N": scale_num_patches,
                    "branch_patch_counts": (scale_num_patches,),
                    "patchifier_mode": f"multiscale_p{patch_size}",
                }
                with torch.no_grad():
                    full_scale = self.encoder.forward_features(
                        x,
                        channel_positions=channel_positions,
                        channel_mask=channel_mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                        channel_tokens_override=scale_channel_tokens,
                        patchifier_aux_override=patchifier_aux,
                    )
                with torch.no_grad():
                    context_scale = self.encoder.forward_features(
                        x,
                        channel_positions=channel_positions,
                        channel_mask=channel_mask,
                        patch_mask=scale_patch_mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                        channel_tokens_override=scale_channel_tokens,
                        patchifier_aux_override=patchifier_aux,
                    )
                if patch_size == self.multiscale_base_patch:
                    base_full = full_scale

                target_scale = self.projector(
                    self._reshape_ci_tokens(
                        full_scale["mixed_tokens"],
                        batch=batch,
                        channels=channels,
                    )
                ).detach()
                context_scale_tokens = self.projector(
                    self._reshape_ci_tokens(
                        context_scale["mixed_tokens"],
                        batch=batch,
                        channels=channels,
                    )
                )
                aligned_context_scale = self._align_latents_to_base_grid(
                    context_scale_tokens.detach(),
                    patch_size=patch_size,
                    base_patch=self.multiscale_base_patch,
                    base_num_patches=base_num_patches,
                )
                gate_input_scales.append(aligned_context_scale)
                aligned_target_scale = self._align_latents_to_base_grid(
                    target_scale,
                    patch_size=patch_size,
                    base_patch=self.multiscale_base_patch,
                    base_num_patches=base_num_patches,
                )
            gate_inputs = torch.stack(gate_input_scales, dim=3)
            alpha = self.multiscale_fusion_gate(gate_inputs)
            target_fused = gate_inputs.new_zeros(batch, channels, base_num_patches, self.config.proj_dim)
            pred_fused: torch.Tensor | None = None

            for scale_index, patch_size in enumerate(self.multiscale_patch_sizes):
                scale_channel_tokens = self.encoder.embed_with_patch_size(x, patch_size)
                scale_num_patches = scale_channel_tokens.shape[2]
                scale_patch_mask = self._base_mask_to_scale_mask(
                    base_patch_mask,
                    patch_size=patch_size,
                    base_patch=self.multiscale_base_patch,
                    scale_num_patches=scale_num_patches,
                )
                patchifier_aux = {
                    "scale_gate": None,
                    "scale_logits": None,
                    "scale_gate_mean": None,
                    "patchifier_N": scale_num_patches,
                    "branch_patch_counts": (scale_num_patches,),
                    "patchifier_mode": f"multiscale_p{patch_size}",
                }
                with torch.no_grad():
                    full_scale = self.encoder.forward_features(
                        x,
                        channel_positions=channel_positions,
                        channel_mask=channel_mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                        channel_tokens_override=scale_channel_tokens,
                        patchifier_aux_override=patchifier_aux,
                    )
                context_scale = self.encoder.forward_features(
                    x,
                    channel_positions=channel_positions,
                    channel_mask=channel_mask,
                    patch_mask=scale_patch_mask,
                    channel_text_embeddings=channel_text_embeddings,
                    channel_stats_embeddings=channel_stats_embeddings,
                    channel_tokens_override=scale_channel_tokens,
                    patchifier_aux_override=patchifier_aux,
                )
                if patch_size == self.multiscale_base_patch:
                    base_context = context_scale
                target_scale = self.projector(
                    self._reshape_ci_tokens(
                        full_scale["mixed_tokens"],
                        batch=batch,
                        channels=channels,
                    )
                ).detach()
                aligned_target_scale = self._align_latents_to_base_grid(
                    target_scale,
                    patch_size=patch_size,
                    base_patch=self.multiscale_base_patch,
                    base_num_patches=base_num_patches,
                )
                target_fused = target_fused + (
                    alpha[:, :, :, scale_index].unsqueeze(-1) * aligned_target_scale
                )
                context_scale_tokens = self.projector(
                    self._reshape_ci_tokens(
                        context_scale["mixed_tokens"],
                        batch=batch,
                        channels=channels,
                    )
                )
                ci_scale_mask = scale_patch_mask.unsqueeze(1).expand(batch, channels, scale_num_patches)
                pred_scale = self.predictor(
                    context_scale_tokens.reshape(batch * channels, scale_num_patches, self.config.proj_dim),
                    patch_mask=ci_scale_mask.reshape(batch * channels, scale_num_patches),
                ).reshape(batch, channels, scale_num_patches, self.config.proj_dim)
                aligned_pred_scale = self._align_latents_to_base_grid(
                    pred_scale,
                    patch_size=patch_size,
                    base_patch=self.multiscale_base_patch,
                    base_num_patches=base_num_patches,
                )
                weighted_pred_scale = alpha[:, :, :, scale_index].unsqueeze(-1) * aligned_pred_scale
                if pred_fused is None:
                    pred_fused = weighted_pred_scale
                else:
                    pred_fused = pred_fused + weighted_pred_scale

            if pred_fused is None:
                raise RuntimeError("pred_fused was not initialized during multiscale forward.")
            target_mask_base = base_patch_mask.unsqueeze(1).expand(batch, channels, base_num_patches)
            pred_loss = self._masked_mse(pred_fused, target_fused.detach(), target_mask_base)

            if base_full is None or base_context is None:
                raise RuntimeError(
                    f"Base patch {self.multiscale_base_patch} features were not collected during multiscale forward."
                )
            context_global = self.projector(base_context["channel_repr"].reshape(-1, self.config.embed_dim))
            sigreg_loss = self.sigreg(context_global.unsqueeze(0))
            query_loss = 0.5 * (base_full["query_loss"] + base_context["query_loss"])
            loss = pred_loss + (self.config.sigreg_weight * sigreg_loss) + (self.config.query_loss_weight * query_loss)

            metadata_usage = summarize_metadata_usage(base_full)
            scale_means = alpha.detach().float().mean(dim=(0, 1, 2))
            for idx, patch_size in enumerate(self.multiscale_patch_sizes):
                metadata_usage[f"scale_weight_p{patch_size}"] = float(scale_means[idx].item())

            pred_tokens = pred_fused.reshape(batch * channels, base_num_patches, self.config.proj_dim)
            target_tokens = target_fused.detach().reshape(batch * channels, base_num_patches, self.config.proj_dim)
            flat_patch_mask = target_mask_base.reshape(batch * channels, base_num_patches)
            outputs = {
                "loss": loss,
                "pred_loss": pred_loss,
                "sigreg_loss": sigreg_loss,
                "query_loss": query_loss,
                "patch_mask": flat_patch_mask,
                "target_tokens": target_tokens,
                "pred_tokens": pred_tokens,
                "mixed_tokens": base_full["mixed_tokens"],
                "mixed_repr": base_full["mixed_repr"],
                "metadata_usage": metadata_usage,
            }

            if return_aux:
                outputs.update(
                    {
                        "full_features": base_full,
                        "context_features": base_context,
                        "multiscale_alpha": alpha,
                    }
                )
            return outputs

        num_patches = self.encoder.infer_num_patches(x.shape[-1])
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

        target_tokens = self.projector(full["mixed_tokens"]).detach()
        context_tokens = self.projector(context["mixed_tokens"])

        if self.encoder.channel_mixer_type == "independent":
            patch_mask = patch_mask.unsqueeze(1).expand(x.shape[0], x.shape[1], num_patches).reshape(x.shape[0] * x.shape[1], num_patches)

        pred_tokens = self.predictor(context_tokens, patch_mask=patch_mask)
        pred_loss = F.mse_loss(pred_tokens[patch_mask], target_tokens[patch_mask])

        if self.encoder.channel_mixer_type == "mixer":
            context_global = self.projector(context["mixed_repr"])
        else:
            context_global = self.projector(context["channel_repr"].reshape(-1, self.config.embed_dim))
        sigreg_loss = self.sigreg(context_global.unsqueeze(0))

        query_loss = 0.5 * (full["query_loss"] + context["query_loss"])
        loss = pred_loss + (self.config.sigreg_weight * sigreg_loss) + (self.config.query_loss_weight * query_loss)

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
            "metadata_usage": summarize_metadata_usage(full),
        }

        if return_aux:
            outputs.update(
                {
                    "full_features": full,
                    "context_features": context,
                }
            )

        return outputs


class LayaTSClassifier(nn.Module):
    def __init__(self, config: Optional[LayaModelConfig] = None, num_classes: int = 2) -> None:
        super().__init__()
        self.encoder = LayaTSEncoder(config)
        self.norm = nn.LayerNorm(self.encoder.config.embed_dim)
        self.head = nn.Linear(self.encoder.config.embed_dim, num_classes)

    def forward(self, x: torch.Tensor, channel_positions: Optional[torch.Tensor], channel_mask: Optional[torch.Tensor] = None, channel_text_embeddings: Optional[torch.Tensor] = None, channel_stats_embeddings: Optional[torch.Tensor] = None, return_features: bool = False):
        features = self.encoder.forward_features(x, channel_positions=channel_positions, channel_mask=channel_mask, channel_text_embeddings=channel_text_embeddings, channel_stats_embeddings=channel_stats_embeddings)
        pooled = self.norm(features["mixed_repr"])
        logits = self.head(pooled)
        if return_features:
            return logits, features
        return logits


class RevIN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        *,
        eps: float = 1e-5,
        affine: bool = True,
        subtract_last: bool = False,
    ) -> None:
        super().__init__()
        self.num_channels = int(num_channels)
        self.eps = float(eps)
        self.affine = bool(affine)
        self.subtract_last = bool(subtract_last)
        if self.affine:
            self.weight = nn.Parameter(torch.ones(1, self.num_channels, 1))
            self.bias = nn.Parameter(torch.zeros(1, self.num_channels, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def _compute_stats(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 3:
            raise ValueError(f"RevIN expects [B, C, T] input, got shape {tuple(x.shape)}")
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"RevIN was initialized for {self.num_channels} channels, but got {x.shape[1]} channels."
            )
        if self.subtract_last:
            center = x[:, :, -1:].detach()
        else:
            center = x.mean(dim=-1, keepdim=True).detach()
        scale = torch.sqrt(x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).detach()
        return center, scale

    def normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        center, scale = self._compute_stats(x)
        normalized = (x - center) / scale
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized, (center, scale)

    def denormalize(self, x: torch.Tensor, stats: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        center, scale = stats
        if x.dim() != 3:
            raise ValueError(f"RevIN expects [B, C, T] input for denormalize, got shape {tuple(x.shape)}")
        if x.shape[1] != center.shape[1]:
            raise ValueError(
                f"RevIN denormalize channel mismatch: prediction has {x.shape[1]} channels, stats have {center.shape[1]}."
            )
        restored = x
        if self.affine:
            restored = (restored - self.bias) / self.weight.clamp_min(self.eps)
        return restored * scale + center


class LayaTSForecaster(nn.Module):
    def __init__(
        self,
        config: Optional[LayaModelConfig] = None,
        pred_len: int = 96,
        out_channels: int = 1,
        num_patches: Optional[int] = None,
        *,
        use_revin: bool = False,
        revin_affine: bool = False,
        revin_subtract_last: bool = False,
        revin_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.encoder = LayaTSEncoder(config)
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.use_revin = bool(use_revin)
        self.revin_affine = bool(revin_affine)
        self.revin_subtract_last = bool(revin_subtract_last)
        self.revin_eps = float(revin_eps)
        default_time = int(round(self.encoder.config.input_seconds * self.encoder.config.sample_rate))
        self.num_patches = num_patches if num_patches is not None else self.encoder.infer_num_patches(default_time)
        # Keep a channel-wise forecasting probe for both CI and mixer paths, but
        # respect the actual token width being probed in each branch.
        if self.encoder.channel_mixer_type == "independent":
            self.probe_token_dim = self.encoder.config.embed_dim
        else:
            self.probe_token_dim = self.encoder.config.channel_mixer_dim
        self.head = nn.Linear(self.num_patches * self.probe_token_dim, pred_len)
        self.revin = (
            RevIN(
                out_channels,
                eps=self.revin_eps,
                affine=self.revin_affine,
                subtract_last=self.revin_subtract_last,
            )
            if self.use_revin
            else None
        )

    def _resolve_channelwise_probe_tokens(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.encoder.channel_mixer_type == "independent":
            return features["independent_tokens"]

        # Prefer refined tokens when an inter-channel refiner was applied before
        # the mixer. Otherwise fall back to the metadata-conditioned channel
        # tokens that feed the mixer.
        channel_tokens = features.get("channel_mixer_refined_tokens")
        if channel_tokens is None:
            channel_tokens = features["channel_tokens"]

        affinity = features.get("channel_affinity")
        if affinity is not None:
            # affinity: [B, N, H, Q, C]
            # Collapse heads/queries to obtain a per-patch channel importance and
            # rescale it so values are centered around 1.0 instead of 1/C.
            channel_importance = affinity.mean(dim=(2, 3))  # [B, N, C]
            channel_importance = channel_importance * channel_tokens.shape[1]
            channel_tokens = channel_tokens * channel_importance.permute(0, 2, 1).unsqueeze(-1)
        return channel_tokens

    def forward(self, x: torch.Tensor, channel_positions: Optional[torch.Tensor], channel_mask: Optional[torch.Tensor] = None, channel_text_embeddings: Optional[torch.Tensor] = None, channel_stats_embeddings: Optional[torch.Tensor] = None, return_features: bool = False):
        revin_stats = None
        encoder_input = x
        if self.revin is not None:
            encoder_input, revin_stats = self.revin.normalize(x)
        features = self.encoder.forward_features(encoder_input, channel_positions=channel_positions, channel_mask=channel_mask, channel_text_embeddings=channel_text_embeddings, channel_stats_embeddings=channel_stats_embeddings)
        tokens = self._resolve_channelwise_probe_tokens(features)
        batch, channels, patches, dim = tokens.shape
        if patches != self.num_patches:
            raise ValueError(
                f"Forecast head expected {self.num_patches} patches from configuration, "
                f"but encoder produced {patches}. Check seq_len and patch_size alignment."
            )
        out = self.head(tokens.reshape(batch * channels, patches * dim)).reshape(batch, channels, self.pred_len)
        if self.revin is not None:
            if revin_stats is None:
                raise RuntimeError("RevIN stats were not populated before denormalization.")
            out = self.revin.denormalize(out, revin_stats)
        if return_features:
            return out, features
        return out


def load_model_config_from_checkpoint(checkpoint_path: str) -> LayaModelConfig:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    raw_config = ckpt.get("model_config")
    if raw_config is None:
        return LayaModelConfig()
    valid_fields = {field.name for field in fields(LayaModelConfig)}

    def sanitize_config_dict(config_dict: dict[str, object]) -> dict[str, object]:
        sanitized = {key: value for key, value in config_dict.items() if key in valid_fields}
        if sanitized.get("encoder_variant") not in {None, "default"}:
            sanitized["encoder_variant"] = "default"
        return sanitized

    if is_dataclass(raw_config):
        return LayaModelConfig(**sanitize_config_dict(asdict(raw_config)))
    if isinstance(raw_config, dict):
        return LayaModelConfig(**sanitize_config_dict(raw_config))
    return LayaModelConfig()


def load_encoder_from_checkpoint_report(model: nn.Module, checkpoint_path: str) -> dict[str, object]:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    encoder_state = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
    skipped_keys: list[str] = []
    missing_keys: list[str] = []
    unexpected_keys: list[str] = []
    matched_keys = 0
    matched_key_names: list[str] = []
    if encoder_state:
        current_state = model.encoder.state_dict()
        compatible_state = {}
        for key, value in encoder_state.items():
            if key not in current_state or current_state[key].shape != value.shape:
                skipped_keys.append(key)
                continue
            compatible_state[key] = value
        matched_keys = len(compatible_state)
        matched_key_names = list(compatible_state.keys())
        incompatible = model.encoder.load_state_dict(compatible_state, strict=False)
        missing_keys = list(incompatible.missing_keys)
        unexpected_keys = list(incompatible.unexpected_keys)
    return {
        "matched_keys": matched_keys,
        "matched_key_names": matched_key_names,
        "total_encoder_keys": len(encoder_state),
        "skipped_keys": skipped_keys,
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def load_encoder_from_checkpoint(model: nn.Module, checkpoint_path: str) -> list[str]:
    return list(load_encoder_from_checkpoint_report(model, checkpoint_path)["skipped_keys"])
