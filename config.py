from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from laya.config import LayaModelConfig, TrainingConfig


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
    "ForecastingConfig",
    "LayaModelConfig",
    "PretrainConfig",
    "TrainingConfig",
]
