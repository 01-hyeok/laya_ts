from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def normalize_variant_name(variant: str) -> str:
    value = variant.lower()
    if value in {"s", "small", "laya-s"}:
        return "s"
    if value in {"b", "base", "laya-b"}:
        return "b"
    raise ValueError(f"Unknown Laya variant: {variant!r}")


@dataclass(frozen=True)
class EEGPretrainDatasetConfig:
    root: Optional[Path] = None
    manifest_path: Optional[Path] = None
    crop_seconds: float = 16.0
    source_segment_seconds: float = 120.0
    sampling_rate: int = 250
    preprocess_raw: bool = False
    bandpass_low_hz: float = 0.5
    bandpass_high_hz: float = 100.0
    notch_freqs_hz: Tuple[float, ...] = (50.0, 60.0)
    robust_scale: bool = True
    reject_bad_windows: bool = False
    clipping_quantile: float = 0.999
    low_variance_eps: float = 1e-8
    leading_trim_window_seconds: float = 2.0
    require_channel_positions: bool = True
    supported_extensions: Sequence[str] = field(
        default_factory=lambda: (".npy", ".npz", ".pt", ".pth", ".h5")
    )

    @property
    def crop_samples(self) -> int:
        return int(round(self.crop_seconds * self.sampling_rate))

    @property
    def source_segment_samples(self) -> int:
        return int(round(self.source_segment_seconds * self.sampling_rate))


@dataclass(frozen=True)
class LayaModelConfig:
    variant: str = "s"
    sample_rate: int = 250
    input_seconds: float = 16.0
    patch_size: int = 25
    num_queries: int = 16
    channel_mixer_dim: int = 32
    channel_mixer_type: str = "mixer"
    channel_metadata_mode: str = "coordinates"
    metadata_fusion_mode: str = "add"
    onehot_channel_vocab_size: int = 0
    text_metadata_dim: int = 384
    stats_metadata_dim: int = 384
    channel_mixer_relation_mode: str = "none"
    channel_mixer_relation_scale_init: float = 0.01
    use_relation_adapter: bool = False
    relation_num_heads: int = 4
    relation_dropout: float = 0.1
    relation_scale_init: float = 1e-3
    use_metadata_bias: bool = True
    use_metadata_gate: bool = True
    metadata_scale_init: float = 1e-3
    metadata_dropout: float = 0.0
    relation_adapter_position: str = "post_encoder"
    description_relation_num_latents: int = 1
    description_relation_metric: str = "cosine"
    description_relation_lambda_init: float = 0.0
    description_relation_gamma_init: float = 1.0
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    proj_dim: int = 128
    predictor_depth: int = 4
    predictor_heads: int = 4
    mask_ratio: float = 0.6
    mask_min_patches: int = 5
    mask_max_patches: int = 10
    query_loss_weight: float = 1.0
    sigreg_weight_s: float = 0.05
    sigreg_weight_b: float = 0.02
    sigreg_num_slices: int = 256
    sigreg_quadrature_points: int = 17
    sigreg_cf_t_max: float = 3.0
    sigreg_cf_bandwidth: float = 1.0
    channel_mixer_heads: int = 2
    use_channel_relation_block: bool = False
    channel_relation_heads: int = 1
    channel_relation_gate_scale_init: float = 0.01
    channel_relation_residual_scale_init: float = 0.05
    encoder_variant: str = "default"
    temporal_patchifier_mode: str = "fixed"
    charm_kernel_sizes: Tuple[int, ...] = (16, 32, 64)
    charm_stride: int = 0
    charm_patchifier_dropout: float = 0.0
    charm_scale_gate_source: str = "learned"
    charm_scale_gate_temperature: float = 1.0
    charm_patchifier_fusion: str = "residual"
    charm_patchifier_residual_init: float = 0.0
    fourier_num_bands: int = 4
    mlp_ratio: float = 4.0
    predictor_mlp_ratio: float = 4.0
    dropout: float = 0.0
    attn_dropout: float = 0.0

    def __post_init__(self) -> None:
        normalized_channel_mixer_type = str(self.channel_mixer_type).strip().lower().replace("-", "_")
        if normalized_channel_mixer_type == "ci_adapter" and not self.use_relation_adapter:
            object.__setattr__(self, "use_relation_adapter", True)
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.relation_num_heads <= 0:
            raise ValueError(
                f"relation_num_heads must be positive, got {self.relation_num_heads}"
            )
        if not 0.0 <= self.relation_dropout <= 1.0:
            raise ValueError(
                f"relation_dropout must be in [0, 1], got {self.relation_dropout}"
            )
        if not 0.0 <= self.metadata_dropout <= 1.0:
            raise ValueError(
                f"metadata_dropout must be in [0, 1], got {self.metadata_dropout}"
            )
        if self.relation_adapter_position != "post_encoder":
            raise ValueError(
                "relation_adapter_position must currently be 'post_encoder', "
                f"got {self.relation_adapter_position!r}"
            )
        if self.description_relation_num_latents <= 0:
            raise ValueError(
                "description_relation_num_latents must be positive, "
                f"got {self.description_relation_num_latents}"
            )
        if self.description_relation_metric not in {"projected_dot", "cosine"}:
            raise ValueError(
                "description_relation_metric must be one of: projected_dot, cosine. "
                f"Got {self.description_relation_metric!r}."
            )
        if self.metadata_fusion_mode not in {
            "none",
            "add",
            "concat_kv",
            "attention_gate",
            "attention_suppress_gate",
        }:
            raise ValueError(
                "metadata_fusion_mode must be one of: none, add, concat_kv, attention_gate, attention_suppress_gate. "
                f"Got {self.metadata_fusion_mode!r}."
            )
        if self.stats_metadata_dim <= 0:
            raise ValueError(
                f"stats_metadata_dim must be positive, got {self.stats_metadata_dim}"
            )
        if not self.charm_kernel_sizes:
            raise ValueError(
                "charm_kernel_sizes must contain at least one kernel size"
            )
        normalized_kernel_sizes = tuple(
            int(kernel_size) for kernel_size in self.charm_kernel_sizes
        )
        if any(kernel_size <= 0 for kernel_size in normalized_kernel_sizes):
            raise ValueError(
                f"All charm_kernel_sizes must be positive, got {normalized_kernel_sizes}"
            )
        object.__setattr__(self, "charm_kernel_sizes", normalized_kernel_sizes)
        if self.charm_stride <= 0:
            object.__setattr__(self, "charm_stride", self.patch_size)
        if self.charm_stride <= 0:
            raise ValueError(
                f"charm_stride must be positive after resolution, got {self.charm_stride}"
            )
        if self.charm_scale_gate_temperature <= 0:
            raise ValueError(
                f"charm_scale_gate_temperature must be positive, got {self.charm_scale_gate_temperature}"
            )

    @property
    def sigreg_weight(self) -> float:
        variant = normalize_variant_name(self.variant)
        if variant == "s":
            return self.sigreg_weight_s
        if variant == "b":
            return self.sigreg_weight_b
        raise AssertionError("normalize_variant_name should have handled the variant")

    @property
    def patch_duration_seconds(self) -> float:
        return self.patch_size / float(self.sample_rate)

    @property
    def mask_patch_span(self) -> Tuple[int, int]:
        return self.mask_min_patches, self.mask_max_patches

    @property
    def num_input_patches(self) -> int:
        return int(round(self.input_seconds * self.sample_rate / self.patch_size))

    def paper_variant_summary(self) -> Dict[str, Any]:
        return {
            "variant": normalize_variant_name(self.variant),
            "training_data_fraction": 0.1
            if normalize_variant_name(self.variant) == "s"
            else 1.0,
            "max_input_duration_seconds": self.input_seconds,
            "num_input_patches": self.num_input_patches,
            "encoder_dim": self.embed_dim,
            "encoder_depth": self.depth,
            "encoder_heads": self.num_heads,
            "predictor_depth": self.predictor_depth,
            "predictor_heads": self.predictor_heads,
            "projection_dim": self.proj_dim,
            "predictor_dim": self.proj_dim,
            "patch_size": self.patch_size,
            "channel_queries": self.num_queries,
            "channel_mixer_dim": self.channel_mixer_dim,
            "mask_ratio": self.mask_ratio,
            "mask_block_sizes": self.mask_patch_span,
            "global_crops": 1,
            "sigreg_weight": self.sigreg_weight,
            "query_loss_weight": self.query_loss_weight,
        }


@dataclass(frozen=True)
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 5e-2
    warmup_steps: int = 1_000
    min_lr: float = 1e-6
    data_fraction_s: float = 0.1
    data_fraction_b: float = 1.0
    max_steps_s: int = 10_000
    max_steps_b: int = 20_000
    batch_size: int = 256
    num_workers: int = 4
    grad_clip_norm: Optional[float] = None
    device: str = "cuda"
    precision: str = "bf16-mixed"

    def data_fraction_for_variant(self, variant: str) -> float:
        value = normalize_variant_name(variant)
        if value == "s":
            return self.data_fraction_s
        if value == "b":
            return self.data_fraction_b
        raise AssertionError("normalize_variant_name should have handled the variant")

    def max_steps_for_variant(self, variant: str) -> int:
        value = normalize_variant_name(variant)
        if value == "s":
            return self.max_steps_s
        if value == "b":
            return self.max_steps_b
        raise AssertionError("normalize_variant_name should have handled the variant")

    def paper_variant_summary(self, variant: str) -> Dict[str, Any]:
        return {
            "variant": normalize_variant_name(variant),
            "batch_size": self.batch_size,
            "learning_rate": self.lr,
            "weight_decay": self.weight_decay,
            "lr_schedule": "warmup+cosine",
            "warmup_steps": self.warmup_steps,
            "min_lr": self.min_lr,
            "max_steps": self.max_steps_for_variant(variant),
            "precision": self.precision,
            "data_fraction": self.data_fraction_for_variant(variant),
        }


@dataclass(frozen=True)
class PretrainConfig:
    dataset_type: str = "tslib"  # tsld | tslib | electricity
    data_path: str = ""
    seq_len: int = 512
    patch_size: int = 16
    stride: int = 512
    tsld_mode: str = "univariate"  # univariate | multivariate
    tslib_mode: str = "univariate"  # univariate | multivariate
    max_files: Optional[int] = None
    onehot_channel_vocab_size: int = 256
    batch_size: int = 256
    num_workers: int = 4
    variant: str = "s"
    epochs: int = 100
    warmup_epochs: int = 10
    save_dir: str = "checkpoints/laya_ts"
    log_dir: str = "laya_ts/runs"
    log_every: int = 10


@dataclass(frozen=True)
class ForecastingConfig:
    dataset_type: str = "Electricity"
    data_path: str = ""
    seq_len: int = 512
    pred_len: int = 96
    batch_size: int = 64
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 20
    checkpoint_path: Optional[str] = None


@dataclass(frozen=True)
class ClassificationConfig:
    data_root: str = ""
    seq_len: int = 0
    batch_size: int = 64
    num_workers: int = 4
    val_ratio: float = 0.1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    epochs: int = 20
    checkpoint_path: Optional[str] = None


__all__ = [
    "ClassificationConfig",
    "EEGPretrainDatasetConfig",
    "ForecastingConfig",
    "LayaModelConfig",
    "PretrainConfig",
    "TrainingConfig",
    "normalize_variant_name",
]
