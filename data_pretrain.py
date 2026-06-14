from __future__ import annotations

import math
import os
import random
import hashlib
from collections import defaultdict
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset, IterableDataset, Sampler

from .utils import (
    build_joint_text_stats_channel_metadata,
    build_statistical_channel_metadata,
    build_text_channel_metadata,
    collate_with_positions,
    infer_csv_context,
    infer_tsld_context,
    infer_tslib_context,
    list_tsld_csv_files,
    list_tslib_files,
    load_tsld_frame,
    load_tslib_frame,
    load_csv_frame,
    synthetic_channel_positions,
    tsld_split_bounds,
    tslib_split_bounds,
    validate_tsld_mode,
    validate_tslib_mode,
)


_BEIJING_AIR_QUALITY_FEATURES = [
    "PM2.5",
    "PM10",
    "SO2",
    "NO2",
    "CO",
    "O3",
    "TEMP",
    "PRES",
    "DEWP",
    "RAIN",
    "WSPM",
]

_BEIJING_AIR_QUALITY_FEATURES_WITH_WD = _BEIJING_AIR_QUALITY_FEATURES[:-1] + ["wd", "WSPM"]

_AUSTRALIAN_ELECTRICITY_STATES = [
    "Victoria",
    "New South Wales",
    "Queensland",
    "Tasmania",
    "South Australia",
]

_LOTSA_DEBUG_ENABLED = os.environ.get("LAYA_TS_DEBUG_LOTSA", "0").strip().lower() in {"1", "true", "yes", "on"}
_LOTSA_SUBSET_PROBABILITY_CAP = 0.001


class CSVTimeSeriesPretrainDataset(Dataset):
    def __init__(self, csv_path: str, seq_len: int = 512, stride: int = 128, mode: str = "train", dataset_name: str = "csv", channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False) -> None:
        self.seq_len = seq_len
        self.stride = stride
        self.channel_metadata_mode = str(channel_metadata_mode).strip().lower()
        df = load_csv_frame(csv_path)
        total_len = len(df)
        val_len = int(total_len * 0.1)
        test_len = int(total_len * 0.2)
        train_len = total_len - val_len - test_len
        scaler = StandardScaler()
        scaler.fit(df.values[:train_len])
        df_norm = scaler.transform(df.values)
        train_values_norm = df_norm[:train_len].astype(np.float32, copy=False)
        if mode == "train":
            split = df_norm[:train_len]
        elif mode == "val":
            split = df_norm[train_len:train_len + val_len]
        else:
            split = df_norm[train_len + val_len:]
        self.data = torch.from_numpy(split.astype(np.float32))
        self.feature_names = list(df.columns)
        self.positions = synthetic_channel_positions(len(self.feature_names))
        self.channel_text_embeddings = None
        self.channel_stats_embeddings = None
        domain_name, resolved_dataset_name = infer_csv_context(dataset_name, csv_path)
        if self.channel_metadata_mode in {"text", "text_stats_avg"}:
            self.channel_text_embeddings = build_text_channel_metadata(
                channel_names=self.feature_names,
                domain=domain_name,
                dataset_name=resolved_dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        elif self.channel_metadata_mode == "text_stats_joint":
            self.channel_text_embeddings = build_joint_text_stats_channel_metadata(
                train_values_norm,
                channel_names=self.feature_names,
                domain=domain_name,
                dataset_name=resolved_dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            self.channel_stats_embeddings = build_statistical_channel_metadata(
                train_values_norm,
                channel_names=self.feature_names,
                domain=domain_name,
                dataset_name=resolved_dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        self.indices = [i for i in range(0, len(self.data) - self.seq_len + 1, self.stride)]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        window = self.data[start:start + self.seq_len].T
        return {
            "series": window,
            "channel_positions": self.positions,
            "channel_mask": torch.ones(window.shape[0], dtype=torch.bool),
            "channel_names": self.feature_names,
            "channel_text_embeddings": self.channel_text_embeddings,
            "channel_stats_embeddings": self.channel_stats_embeddings,
        }


class LOTSABatchStreamingPretrainDataset(IterableDataset):
    def __init__(
        self,
        *,
        dataset_name: str,
        subset_names: list[str] | None,
        batch_size: int,
        seq_len: int = 512,
        stride: int = 128,
        patch_size: int = 16,
        mode: str = "train",
        channel_metadata_mode: str = "onehot",
        text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
        text_metadata_cache_dir: str = "./metadata_cache",
        text_encoder_local_files_only: bool = False,
        lotsa_split_mode: str = "temporal_70_10_20",
        lotsa_sampling_mode: str = "sliding_window",
        lotsa_preprocessing_mode: str = "standardize",
        lotsa_sample_time_series: str = "proportional",
        lotsa_subset_sampling: str = "uniform",
        lotsa_min_patches: int = 2,
        lotsa_max_channel: int | None = None,
        lotsa_windows_per_series: int = 32,
        max_samples: int | None = None,
        skip_samples: int = 0,
        shuffle_buffer_size: int = 128,
        random_seed: int = 42,
    ) -> None:
        self.dataset_name = dataset_name
        self.subset_names = subset_names or []
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.stride = stride
        self.patch_size = patch_size
        self.mode = mode
        self.channel_metadata_mode = str(channel_metadata_mode).strip().lower()
        self.lotsa_split_mode = str(lotsa_split_mode).strip().lower()
        self.lotsa_sampling_mode = str(lotsa_sampling_mode).strip().lower()
        self.lotsa_preprocessing_mode = str(lotsa_preprocessing_mode).strip().lower()
        self.lotsa_sample_time_series = str(lotsa_sample_time_series).strip().lower()
        self.lotsa_subset_sampling = str(lotsa_subset_sampling).strip().lower()
        self.lotsa_min_patches = int(lotsa_min_patches)
        self.lotsa_max_channel = None if lotsa_max_channel is None else int(lotsa_max_channel)
        self.lotsa_windows_per_series = int(lotsa_windows_per_series)
        self.text_encoder_name_or_path = text_encoder_name_or_path
        self.text_metadata_cache_dir = text_metadata_cache_dir
        self.text_encoder_local_files_only = text_encoder_local_files_only
        self.max_samples = max_samples
        self.skip_samples = skip_samples
        self.shuffle_buffer_size = shuffle_buffer_size
        self.random_seed = random_seed
        self._text_metadata_cache: dict[tuple[str, tuple[str, ...]], torch.Tensor | None] = {}
        self._count_cache: tuple[int, int] | None = None
        self._count_cache_token: tuple[object, ...] | None = None
        self._channel_count_cache: int | None = None
        self._subset_warning_cache: set[tuple[str, str]] = set()
        self._channel_name_warning_cache: set[str] = set()
        self._subset_iteration_index = 0
        self._available_split_cache: dict[str, tuple[str, ...]] = {}
        self._subset_dataset_cache: dict[str, object] = {}
        self._subset_index_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._subset_weight_cache: dict[tuple[str, ...], dict[str, float]] = {}

        if self.lotsa_split_mode not in {"official", "temporal_70_10_20"}:
            raise ValueError(
                f"Unsupported LOTSA split mode: {lotsa_split_mode}. "
                "Expected one of: official, temporal_70_10_20."
            )
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {patch_size}")
        if self.patch_size > self.seq_len:
            raise ValueError(
                f"patch_size must be <= seq_len for LOTSA patch crop, got patch_size={patch_size}, seq_len={seq_len}"
            )
        if self.lotsa_sampling_mode not in {"official", "sliding_window", "hierarchical"}:
            raise ValueError(
                f"Unsupported LOTSA sampling mode: {lotsa_sampling_mode}. "
                "Expected one of: official, sliding_window, hierarchical."
            )
        if self.lotsa_preprocessing_mode not in {"official", "standardize"}:
            raise ValueError(
                f"Unsupported LOTSA preprocessing mode: {lotsa_preprocessing_mode}. "
                "Expected one of: official, standardize."
            )
        if self.lotsa_sample_time_series not in {"none", "uniform", "proportional"}:
            raise ValueError(
                f"Unsupported LOTSA sample_time_series mode: {lotsa_sample_time_series}. "
                "Expected one of: none, uniform, proportional."
            )
        if self.lotsa_min_patches <= 0:
            raise ValueError(f"lotsa_min_patches must be positive, got {lotsa_min_patches}")
        if self.lotsa_max_channel is not None and self.lotsa_max_channel <= 0:
            raise ValueError(f"lotsa_max_channel must be positive when provided, got {lotsa_max_channel}")
        if self.lotsa_subset_sampling not in {"exhaustive", "uniform", "official"}:
            raise ValueError(
                f"Unsupported LOTSA subset sampling mode: {lotsa_subset_sampling}. "
                "Expected one of: exhaustive, uniform, official."
            )
        if self.lotsa_windows_per_series <= 0:
            raise ValueError(f"lotsa_windows_per_series must be positive, got {lotsa_windows_per_series}")

    def _debug(self, message: str) -> None:
        if _LOTSA_DEBUG_ENABLED:
            print(f"[LOTSA-DEBUG] {message}")

    def __len__(self):
        _, total_batches = self._count_windows_and_batches()
        return total_batches

    @property
    def num_samples(self) -> int:
        total_windows, _ = self._count_windows_and_batches()
        return total_windows

    @property
    def num_channels(self) -> int:
        if self._channel_count_cache is not None:
            return self._channel_count_cache

        if self.lotsa_sampling_mode == "official":
            for subset_name in self._resolve_subset_names():
                try:
                    valid_indices, _ = self._subset_valid_indices_and_lengths(subset_name)
                    if len(valid_indices) == 0:
                        continue
                    dataset = self._load_subset_dataset(subset_name)
                    record = dataset[int(valid_indices[0])]
                    series_ct = self._infer_series_array(record)
                    self._channel_count_cache = self._resolved_max_channel(int(series_ct.shape[0]))
                    return self._channel_count_cache
                except Exception as exc:
                    self._warn_skipped_subset(subset_name, "channel-count", exc)
                    continue
            self._channel_count_cache = 0
            return self._channel_count_cache

        for subset_name in self._resolve_subset_names():
            try:
                for record in self._iter_subset(subset_name):
                    try:
                        series_ct = self._infer_series_array(record)
                        num_channels, total_len = series_ct.shape
                        window_count = self._split_window_count(total_len)
                        if window_count <= 0:
                            continue
                        self._channel_count_cache = int(num_channels)
                        return self._channel_count_cache
                    except Exception:
                        continue
            except Exception as exc:
                self._warn_skipped_subset(subset_name, "channel-count", exc)
                continue

        self._channel_count_cache = 0
        return self._channel_count_cache

    def _resolve_subset_names(self) -> list[str]:
        if self.subset_names:
            return self.subset_names
        local_repo = self._local_lotsa_repo_path()
        if local_repo is not None:
            subset_names = self._list_local_subset_names(local_repo)
            if not subset_names:
                raise ValueError(f"No LOTSA subset directories with arrow files found under local path {local_repo}")
            self.subset_names = subset_names
            return self.subset_names
        try:
            from datasets import get_dataset_config_names  # type: ignore
        except Exception as exc:
            raise ImportError(
                "datasets is required to enumerate LOTSA subset names. Install huggingface-datasets."
            ) from exc
        subset_names = get_dataset_config_names(self.dataset_name)
        if not subset_names:
            raise ValueError(f"No LOTSA subset/config names found for dataset {self.dataset_name}")
        self.subset_names = list(subset_names)
        return self.subset_names

    def _local_lotsa_repo_path(self) -> Path | None:
        dataset_path = Path(str(self.dataset_name)).expanduser()
        if dataset_path.is_dir():
            return dataset_path
        return None

    @staticmethod
    def _is_local_subset_dir(path: Path) -> bool:
        if not path.is_dir() or path.name.startswith("."):
            return False
        return any(path.glob("*.arrow"))

    def _list_local_subset_names(self, repo_path: Path) -> list[str]:
        return [child.name for child in sorted(repo_path.iterdir()) if self._is_local_subset_dir(child)]

    def _local_subset_dir(self, subset_name: str) -> Path:
        repo_path = self._local_lotsa_repo_path()
        if repo_path is None:
            raise ValueError("LOTSA local subset directory requested, but dataset_name is not a local directory.")
        subset_dir = repo_path / subset_name
        if not self._is_local_subset_dir(subset_dir):
            raise FileNotFoundError(
                f"LOTSA local subset '{subset_name}' was not found under {repo_path} or does not contain arrow files."
            )
        return subset_dir

    @staticmethod
    def _infer_series_array(record: dict) -> np.ndarray:
        for key in ("target", "series", "values"):
            if key in record:
                raw = np.asarray(record[key], dtype=np.float32)
                break
        else:
            raise ValueError(f"LOTSA record does not contain a supported series key: {list(record.keys())[:10]}")

        if raw.ndim == 1:
            return raw.reshape(1, -1)
        if raw.ndim != 2:
            raise ValueError(f"Expected LOTSA series with ndim 1 or 2, got shape {raw.shape}")

        # Normalize to [C, T]. LOTSA examples are usually [T, C], but some subsets
        # may already be channel-first. Prefer the orientation where the channel axis
        # is the smaller dimension and the time axis is the larger dimension.
        if raw.shape[0] > raw.shape[1]:
            raw = raw.T
        return raw

    @staticmethod
    def _default_channel_names(num_channels: int) -> list[str]:
        return [f"ch_{idx}" for idx in range(num_channels)]

    @staticmethod
    def _generic_fallback_channel_names(subset_name: str, num_channels: int) -> list[str]:
        if num_channels == 1:
            return ["target"]
        return [f"feature_{idx}" for idx in range(num_channels)]

    @staticmethod
    def _record_scalar_string(value, *keys: str, depth: int = 0) -> str | None:
        if depth > 3:
            return None
        if isinstance(value, dict):
            for key in keys:
                if key in value:
                    nested = LOTSABatchStreamingPretrainDataset._record_scalar_string(value[key], depth=depth + 1)
                    if nested is not None:
                        return nested
            for nested_value in value.values():
                nested = LOTSABatchStreamingPretrainDataset._record_scalar_string(nested_value, *keys, depth=depth + 1)
                if nested is not None:
                    return nested
            return None
        if isinstance(value, np.ndarray):
            if value.ndim != 0:
                return None
            value = value.item()
        if isinstance(value, (list, tuple, set)):
            return None
        if value is None:
            return None
        text_value = str(value).strip()
        return text_value or None

    @staticmethod
    def _record_static_index(record: dict) -> int | None:
        raw = record.get("feat_static_cat")
        if isinstance(raw, np.ndarray):
            raw = raw.tolist()
        if isinstance(raw, (list, tuple)) and len(raw) == 1:
            try:
                return int(raw[0])
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _coerce_channel_names(value, num_channels: int) -> list[str] | None:
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                return None
            value = value.tolist()

        if isinstance(value, (list, tuple)):
            if len(value) != num_channels or len(value) == 0:
                return None
            names: list[str] = []
            for item in value:
                if isinstance(item, np.ndarray):
                    if item.ndim != 0:
                        return None
                    item = item.item()
                if isinstance(item, dict):
                    nested_name = None
                    for key in ("name", "feature_name", "channel_name", "sensor_name", "station_name", "id"):
                        if key in item:
                            nested_name = str(item[key]).strip()
                            break
                    if not nested_name:
                        return None
                    names.append(nested_name)
                    continue
                if isinstance(item, (list, tuple, set)):
                    return None
                name = str(item).strip()
                if not name:
                    return None
                names.append(name)
            return names

        if isinstance(value, dict):
            if "names" in value:
                return LOTSABatchStreamingPretrainDataset._coerce_channel_names(value["names"], num_channels)
            if "columns" in value:
                return LOTSABatchStreamingPretrainDataset._coerce_channel_names(value["columns"], num_channels)

        return None

    @classmethod
    def _search_channel_names(cls, value, num_channels: int, *, depth: int = 0) -> list[str] | None:
        if depth > 3:
            return None

        direct = cls._coerce_channel_names(value, num_channels)
        if direct is not None:
            return direct

        priority_keys = (
            "channel_names",
            "feature_names",
            "target_names",
            "column_names",
            "columns",
            "variate_names",
            "component_names",
            "sensor_names",
            "station_names",
            "features",
        )

        if isinstance(value, dict):
            for key in priority_keys:
                if key in value:
                    found = cls._search_channel_names(value[key], num_channels, depth=depth + 1)
                    if found is not None:
                        return found
            for nested in value.values():
                if isinstance(nested, (dict, list, tuple)):
                    found = cls._search_channel_names(nested, num_channels, depth=depth + 1)
                    if found is not None:
                        return found
            return None

        if isinstance(value, (list, tuple)) and len(value) <= max(32, num_channels * 4):
            for nested in value:
                if isinstance(nested, (dict, list, tuple)):
                    found = cls._search_channel_names(nested, num_channels, depth=depth + 1)
                    if found is not None:
                        return found

        return None

    def _warn_fallback_channel_names(self, subset_name: str, num_channels: int) -> None:
        if subset_name in self._channel_name_warning_cache:
            return
        self._channel_name_warning_cache.add(subset_name)
        print(
            f"[LOTSA] Falling back to synthetic channel names for subset '{subset_name}' "
            f"(channels={num_channels}); no explicit feature names were found in the record metadata."
        )

    @staticmethod
    def _normalize_split_with_observed_values(
        split: np.ndarray,
        train_reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        split = np.asarray(split, dtype=np.float32)
        train_reference = np.asarray(train_reference, dtype=np.float32)

        observed_mask = np.isfinite(split)
        reference_mask = np.isfinite(train_reference)
        if not reference_mask.any():
            raise ValueError("LOTSA split has no finite training observations for normalization.")

        reference_filled = np.where(reference_mask, train_reference, 0.0)
        reference_count = reference_mask.sum(axis=1, keepdims=True)
        safe_count = np.clip(reference_count, 1, None)
        mean = reference_filled.sum(axis=1, keepdims=True) / safe_count

        centered_reference = np.where(reference_mask, train_reference - mean, 0.0)
        var = (centered_reference * centered_reference).sum(axis=1, keepdims=True) / safe_count
        std = np.sqrt(var).astype(np.float32)
        std = np.where(std < 1e-8, 1.0, std)

        normalized = (np.where(observed_mask, split, mean) - mean) / std
        normalized = np.where(observed_mask, normalized, 0.0).astype(np.float32)
        return normalized, observed_mask

    @staticmethod
    def _zero_impute_with_observed_mask(split: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        split = np.asarray(split, dtype=np.float32)
        observed_mask = np.isfinite(split)
        imputed = np.where(observed_mask, split, 0.0).astype(np.float32)
        return imputed, observed_mask

    @staticmethod
    def _infer_lotsa_domain(subset_name: str) -> str:
        name = str(subset_name).strip().lower()
        rules = [
            ("weather", "weather"),
            ("traffic", "traffic"),
            ("taxi", "mobility"),
            ("uber", "mobility"),
            ("rideshare", "mobility"),
            ("vehicle", "mobility"),
            ("electric", "electricity"),
            ("power", "electricity"),
            ("solar", "energy"),
            ("wind", "energy"),
            ("load", "energy"),
            ("meter", "energy"),
            ("air_quality", "environment"),
            ("air", "environment"),
            ("precip", "climate"),
            ("seasonal", "climate"),
            ("climate", "climate"),
            ("temperature", "climate"),
            ("rain", "climate"),
            ("wiki", "web_traffic"),
            ("web", "web_traffic"),
            ("tourism", "tourism"),
            ("sales", "commerce"),
            ("transactions", "commerce"),
            ("cluster", "cloud_systems"),
            ("azure", "cloud_systems"),
            ("vm", "cloud_systems"),
            ("borg", "cloud_systems"),
            ("flu", "public_health"),
            ("covid", "public_health"),
            ("birth", "demography"),
            ("bitcoin", "finance"),
            ("spain", "energy"),
            ("residential", "energy"),
        ]
        for token, domain in rules:
            if token in name:
                return domain
        return name

    def _state_name_for_australian_electricity(self, record: dict) -> str | None:
        for key in ("state", "state_name", "item_id"):
            value = self._record_scalar_string(record, key)
            if value and not value.isdigit():
                return value
        static_index = self._record_static_index(record)
        if static_index is not None and 0 <= static_index < len(_AUSTRALIAN_ELECTRICITY_STATES):
            return _AUSTRALIAN_ELECTRICITY_STATES[static_index]
        return None

    def _subset_specific_channel_names(self, subset_name: str, record: dict, num_channels: int) -> list[str] | None:
        if subset_name == "beijing_air_quality":
            if num_channels == len(_BEIJING_AIR_QUALITY_FEATURES):
                return list(_BEIJING_AIR_QUALITY_FEATURES)
            if num_channels == len(_BEIJING_AIR_QUALITY_FEATURES_WITH_WD):
                return list(_BEIJING_AIR_QUALITY_FEATURES_WITH_WD)

        if num_channels != 1:
            return None

        if subset_name == "weather":
            series_type = self._record_scalar_string(record, "series_type", "type", "data_column")
            if series_type:
                return [series_type]

        if subset_name == "australian_electricity_demand":
            state_name = self._state_name_for_australian_electricity(record)
            if state_name:
                return [f"{state_name} electricity demand"]

        if subset_name == "PEMS_BAY":
            sensor_id = self._record_scalar_string(record, "sensor_id", "station_id", "item_id")
            if sensor_id:
                return [f"sensor {sensor_id} traffic speed"]

        if subset_name == "Q-TRAFFIC":
            segment_id = self._record_scalar_string(record, "road_segment_id", "link_id", "item_id")
            if segment_id:
                return [f"road segment {segment_id} traffic speed"]

        if subset_name == "SZ_TAXI":
            segment_id = self._record_scalar_string(record, "road_segment_id", "link_id", "item_id")
            if segment_id:
                return [f"road segment {segment_id} taxi speed"]

        return None

    def _metadata_dataset_name(self, subset_name: str, record: dict) -> str:
        if subset_name == "beijing_air_quality":
            station = self._record_scalar_string(record, "station", "station_name", "item_id")
            if station:
                return f"{subset_name} station {station}"

        if subset_name == "weather":
            station = self._record_scalar_string(record, "station_id", "item_id")
            if station:
                return f"{subset_name} station {station}"

        if subset_name == "australian_electricity_demand":
            state_name = self._state_name_for_australian_electricity(record)
            if state_name:
                return f"{subset_name} state {state_name}"

        if subset_name == "PEMS_BAY":
            sensor_id = self._record_scalar_string(record, "sensor_id", "station_id", "item_id")
            if sensor_id:
                return f"{subset_name} sensor {sensor_id}"

        if subset_name == "Q-TRAFFIC":
            segment_id = self._record_scalar_string(record, "road_segment_id", "link_id", "item_id")
            if segment_id:
                return f"{subset_name} road segment {segment_id}"

        if subset_name == "SZ_TAXI":
            segment_id = self._record_scalar_string(record, "road_segment_id", "link_id", "item_id")
            if segment_id:
                return f"{subset_name} road segment {segment_id}"

        return subset_name

    def _channel_names(self, subset_name: str, record: dict, num_channels: int) -> list[str]:
        names = self._search_channel_names(record, num_channels)
        if names is not None:
            return names

        names = self._subset_specific_channel_names(subset_name, record, num_channels)
        if names is not None:
            return names

        self._warn_fallback_channel_names(subset_name, num_channels)
        return self._generic_fallback_channel_names(subset_name, num_channels)

    def _channel_metadata(
        self,
        subset_name: str,
        record: dict,
        channel_names: list[str],
        train_reference_ct: np.ndarray,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        dataset_name = self._metadata_dataset_name(subset_name, record)
        domain_name = self._infer_lotsa_domain(subset_name)
        metadata_reference_ct, _ = self._preprocess_split(train_reference_ct, train_reference_ct)
        train_reference_tc = np.asarray(metadata_reference_ct, dtype=np.float32).T

        text_metadata = None
        stats_metadata = None

        if self.channel_metadata_mode in {"text", "text_stats_avg"}:
            cache_key = (dataset_name, tuple(channel_names))
            if cache_key not in self._text_metadata_cache:
                self._text_metadata_cache[cache_key] = build_text_channel_metadata(
                    channel_names=channel_names,
                    domain=domain_name,
                    dataset_name=dataset_name,
                    text_encoder_name_or_path=self.text_encoder_name_or_path,
                    text_metadata_cache_dir=self.text_metadata_cache_dir,
                    text_encoder_local_files_only=self.text_encoder_local_files_only,
                )
            text_metadata = self._text_metadata_cache[cache_key]
        elif self.channel_metadata_mode == "text_stats_joint":
            text_metadata = build_joint_text_stats_channel_metadata(
                train_reference_tc,
                channel_names=channel_names,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=self.text_encoder_name_or_path,
                text_metadata_cache_dir=self.text_metadata_cache_dir,
                text_encoder_local_files_only=self.text_encoder_local_files_only,
            )

        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            stats_metadata = build_statistical_channel_metadata(
                train_reference_tc,
                channel_names=channel_names,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=self.text_encoder_name_or_path,
                text_metadata_cache_dir=self.text_metadata_cache_dir,
                text_encoder_local_files_only=self.text_encoder_local_files_only,
            )

        return text_metadata, stats_metadata

    def _available_hf_split_names(self, subset_name: str) -> tuple[str, ...]:
        cached = self._available_split_cache.get(subset_name)
        if cached is not None:
            return cached

        local_repo = self._local_lotsa_repo_path()
        if local_repo is not None:
            subset_dir = self._local_subset_dir(subset_name)
            info_path = subset_dir / "dataset_info.json"
            if info_path.exists():
                try:
                    payload = json.loads(info_path.read_text())
                except Exception:
                    payload = {}
                splits = payload.get("splits")
                if isinstance(splits, dict) and splits:
                    resolved = tuple(str(name).strip() for name in splits.keys() if str(name).strip())
                    self._available_split_cache[subset_name] = resolved
                    return resolved
            resolved = ("train",)
            self._available_split_cache[subset_name] = resolved
            return resolved

        local_names: list[str] = []
        dataset_cache_root = Path.home() / ".cache" / "huggingface" / "datasets"
        dataset_cache_name = self.dataset_name.replace("/", "___")
        info_pattern = f"{dataset_cache_name}/{subset_name}/*/*/dataset_info.json"
        info_paths = sorted(dataset_cache_root.glob(info_pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        for info_path in info_paths:
            try:
                payload = json.loads(info_path.read_text())
            except Exception:
                continue
            splits = payload.get("splits")
            if isinstance(splits, dict) and splits:
                local_names = [str(name).strip() for name in splits.keys() if str(name).strip()]
                if local_names:
                    break

        if local_names:
            resolved = tuple(local_names)
            self._available_split_cache[subset_name] = resolved
            return resolved

        try:
            from datasets import get_dataset_split_names  # type: ignore
        except Exception:
            resolved = tuple()
            self._available_split_cache[subset_name] = resolved
            return resolved

        try:
            names = get_dataset_split_names(self.dataset_name, subset_name)
        except Exception:
            names = []
        resolved = tuple(str(name).strip() for name in names if str(name).strip())
        self._available_split_cache[subset_name] = resolved
        return resolved

    def _resolve_hf_split_name(self, subset_name: str) -> str:
        if self.lotsa_split_mode != "official":
            return "train"

        available = self._available_hf_split_names(subset_name)
        normalized = {name.lower(): name for name in available}
        if self.mode == "train":
            preferred = ("train",)
        elif self.mode == "val":
            preferred = ("validation", "valid", "val", "dev", "test")
        else:
            preferred = ("test",)

        for alias in preferred:
            if alias in normalized:
                return normalized[alias]

        raise ValueError(
            f"LOTSA official split mode requested for subset '{subset_name}', "
            f"but no held-out split is available for mode='{self.mode}'. "
            f"Available splits: {list(available) or ['<none>']}. "
            "Use --lotsa_split_mode temporal_70_10_20 to opt back into the legacy within-series split."
        )

    def _select_series_split(self, series_ct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        total_len = series_ct.shape[1]
        if self.lotsa_split_mode == "official":
            return series_ct, series_ct

        train_len = int(total_len * 0.7)
        val_len = int(total_len * 0.1)
        if self.mode == "train":
            return series_ct[:, :train_len], series_ct[:, :train_len]
        if self.mode == "val":
            return series_ct[:, train_len:train_len + val_len], series_ct[:, :train_len]
        return series_ct[:, train_len + val_len:], series_ct[:, :train_len]

    def _warn_skipped_subset(self, subset_name: str, phase: str, exc: Exception) -> None:
        cache_key = (phase, subset_name)
        if cache_key in self._subset_warning_cache:
            return
        self._subset_warning_cache.add(cache_key)
        print(f"[LOTSA] Skipping subset '{subset_name}' during {phase}: {exc}")

    def _iter_subset_names_for_epoch(self) -> list[str]:
        subset_names = list(self._resolve_subset_names())
        if self.mode == "train" and len(subset_names) > 1:
            rng = random.Random(self.random_seed + self._subset_iteration_index)
            rng.shuffle(subset_names)
            self._subset_iteration_index += 1
        return subset_names

    def _planned_official_iteration_state(self) -> tuple[list[str], int, int]:
        subset_names = list(self._resolve_subset_names())
        epoch_iteration_index = self._subset_iteration_index
        sampling_iteration_index = self._subset_iteration_index
        if self.mode == "train" and len(subset_names) > 1:
            rng = random.Random(self.random_seed + epoch_iteration_index)
            rng.shuffle(subset_names)
            sampling_iteration_index = epoch_iteration_index + 1
        return subset_names, epoch_iteration_index, sampling_iteration_index

    def _load_local_subset_dataset(self, subset_name: str):
        cached = self._subset_dataset_cache.get(subset_name)
        if cached is not None:
            return cached
        try:
            from datasets import Dataset, concatenate_datasets  # type: ignore
        except Exception as exc:
            raise ImportError(
                "datasets is required for LOTSA local arrow loading. Install huggingface-datasets."
            ) from exc

        subset_dir = self._local_subset_dir(subset_name)
        arrow_files = sorted(subset_dir.glob("*.arrow"))
        if not arrow_files:
            raise FileNotFoundError(f"No arrow files found for local LOTSA subset '{subset_name}' in {subset_dir}")

        datasets_list = [Dataset.from_file(str(arrow_path)) for arrow_path in arrow_files]
        dataset = datasets_list[0] if len(datasets_list) == 1 else concatenate_datasets(datasets_list)
        self._subset_dataset_cache[subset_name] = dataset
        return dataset

    def _iter_local_subset(self, subset_name: str):
        dataset = self._load_local_subset_dataset(subset_name)
        indices = list(range(len(dataset)))
        if self.mode == "train" and self.shuffle_buffer_size > 1 and len(indices) > 1:
            rng = random.Random(self.random_seed + self._subset_iteration_index)
            rng.shuffle(indices)
        if self.skip_samples > 0:
            indices = indices[self.skip_samples:]
        for index in indices:
            yield dataset[int(index)]

    def _iter_subset(self, subset_name: str):
        if self._local_lotsa_repo_path() is not None:
            yield from self._iter_local_subset(subset_name)
            return
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:
            raise ImportError("datasets is required for LOTSA streaming support. Install huggingface-datasets.") from exc

        split_name = self._resolve_hf_split_name(subset_name)
        dataset = load_dataset(self.dataset_name, subset_name, split=split_name, streaming=True)
        if self.mode == "train" and self.shuffle_buffer_size > 1:
            dataset = dataset.shuffle(buffer_size=self.shuffle_buffer_size, seed=self.random_seed)
        if self.skip_samples > 0:
            dataset = dataset.skip(self.skip_samples)
        yield from dataset

    def _load_subset_dataset(self, subset_name: str):
        if self._local_lotsa_repo_path() is not None:
            return self._load_local_subset_dataset(subset_name)
        cached = self._subset_dataset_cache.get(subset_name)
        if cached is not None:
            return cached
        try:
            from datasets import load_dataset  # type: ignore
        except Exception as exc:
            raise ImportError("datasets is required for LOTSA dataset loading. Install huggingface-datasets.") from exc

        split_name = self._resolve_hf_split_name(subset_name)
        dataset = load_dataset(self.dataset_name, subset_name, split=split_name, streaming=False)
        self._subset_dataset_cache[subset_name] = dataset
        return dataset

    def _official_max_time_patches(self) -> int:
        return max(1, self.seq_len // self.patch_size)

    def _valid_patch_crop_capacity(self, total_len: int) -> int:
        if total_len < self.seq_len:
            return 0
        return self._official_max_time_patches()

    def _subset_valid_indices_and_lengths(self, subset_name: str) -> tuple[np.ndarray, np.ndarray]:
        cached = self._subset_index_cache.get(subset_name)
        if cached is not None:
            return cached

        dataset = self._load_subset_dataset(subset_name)
        valid_indices: list[int] = []
        lengths: list[int] = []
        for index in range(len(dataset)):
            try:
                record = dataset[index]
                series_ct = self._infer_series_array(record)
                _, total_len = series_ct.shape
                if self._valid_patch_crop_capacity(total_len) < self.lotsa_min_patches:
                    continue
                valid_indices.append(index)
                lengths.append(total_len)
            except Exception:
                continue

        resolved = (
            np.asarray(valid_indices, dtype=np.int64),
            np.asarray(lengths, dtype=np.int64),
        )
        self._subset_index_cache[subset_name] = resolved
        return resolved

    def _resolve_subset_budget(self, available_count: int, remaining: int | None) -> int:
        if remaining is None:
            return available_count
        return max(0, min(available_count, remaining))

    def _resolved_max_channel(self, num_channels: int) -> int:
        if self.lotsa_max_channel is None:
            return int(num_channels)
        return int(min(num_channels, self.lotsa_max_channel))

    def _record_identity(self, subset_name: str, record: dict, fallback: str) -> str:
        for key in (
            "item_id",
            "sensor_id",
            "station_id",
            "station_name",
            "road_segment_id",
            "link_id",
            "state",
            "state_name",
        ):
            value = self._record_scalar_string(record, key)
            if value is not None:
                return f"{subset_name}:{key}:{value}"
        static_index = self._record_static_index(record)
        if static_index is not None:
            return f"{subset_name}:static_index:{static_index}"
        return f"{subset_name}:fallback:{fallback}"

    def _fixed_channel_subset_seed(self, subset_name: str, record: dict, record_key: str, num_channels: int) -> int:
        identity = self._record_identity(subset_name, record, record_key)
        target_channels = self._resolved_max_channel(num_channels)
        digest = hashlib.sha256(f"{identity}|C={num_channels}|K={target_channels}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def _subset_sampling_weights(self, subset_names: list[str] | None = None) -> dict[str, float]:
        resolved_subset_names = tuple(subset_names or self._resolve_subset_names())
        cached = self._subset_weight_cache.get(resolved_subset_names)
        if cached is not None:
            return cached

        if not resolved_subset_names:
            return {}

        observation_counts: dict[str, float] = {}
        total_observations = 0.0
        for subset_name in resolved_subset_names:
            try:
                _, lengths = self._subset_valid_indices_and_lengths(subset_name)
                observation_count = float(lengths.sum())
            except Exception as exc:
                self._warn_skipped_subset(subset_name, "subset-weight", exc)
                observation_count = 0.0
            observation_counts[subset_name] = observation_count
            total_observations += observation_count

        if total_observations <= 0.0:
            uniform_weight = 1.0 / float(len(resolved_subset_names))
            weights = {subset_name: uniform_weight for subset_name in resolved_subset_names}
            self._subset_weight_cache[resolved_subset_names] = weights
            return weights

        capped_weights: dict[str, float] = {}
        capped_total = 0.0
        for subset_name in resolved_subset_names:
            base_probability = observation_counts[subset_name] / total_observations
            capped_weight = min(base_probability, _LOTSA_SUBSET_PROBABILITY_CAP)
            capped_weights[subset_name] = capped_weight
            capped_total += capped_weight

        if capped_total <= 0.0:
            uniform_weight = 1.0 / float(len(resolved_subset_names))
            weights = {subset_name: uniform_weight for subset_name in resolved_subset_names}
            self._subset_weight_cache[resolved_subset_names] = weights
            return weights

        weights = {
            subset_name: capped_weights[subset_name] / capped_total
            for subset_name in resolved_subset_names
        }
        self._subset_weight_cache[resolved_subset_names] = weights
        return weights

    def _series_sampling_probabilities(self, lengths: np.ndarray) -> np.ndarray | None:
        if self.lotsa_sample_time_series in {"none", "uniform"}:
            return None
        weights = lengths.astype(np.float64, copy=False)
        total = float(weights.sum())
        if total <= 0.0:
            return None
        return (weights / total).astype(np.float64, copy=False)

    def _sample_record_indices_for_subset(
        self,
        subset_name: str,
        *,
        remaining: int | None,
        sampling_iteration_index: int | None = None,
    ) -> list[int]:
        valid_indices, lengths = self._subset_valid_indices_and_lengths(subset_name)
        sample_count = self._resolve_subset_budget(len(valid_indices), remaining)
        if sample_count <= 0:
            return []

        if sampling_iteration_index is None:
            sampling_iteration_index = self._subset_iteration_index

        if self.mode != "train":
            return valid_indices[:sample_count].tolist()

        if self.lotsa_sample_time_series == "none":
            indices = valid_indices.copy()
            rng = np.random.default_rng(self.random_seed + sampling_iteration_index)
            rng.shuffle(indices)
            return indices[:sample_count].tolist()

        rng = np.random.default_rng(self.random_seed + sampling_iteration_index)
        probabilities = self._series_sampling_probabilities(lengths)
        chosen = rng.choice(len(valid_indices), size=sample_count, replace=True, p=probabilities)
        return valid_indices[chosen].tolist()

    def _official_crop_bounds(self, total_len: int, *, rng: np.random.Generator) -> tuple[int, int] | None:
        max_time_patches = self._official_max_time_patches()
        if max_time_patches < self.lotsa_min_patches:
            return None
        if total_len < self.seq_len:
            return None
        max_start = total_len - self.seq_len
        start = 0 if max_start <= 0 else int(rng.integers(max_start + 1))
        stop = start + self.seq_len
        return start, stop

    def _preprocess_split(
        self,
        split: np.ndarray,
        train_reference: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.lotsa_preprocessing_mode == "official":
            return self._zero_impute_with_observed_mask(split)
        return self._normalize_split_with_observed_values(split, train_reference)

    def _sample_dimension_indices(
        self,
        num_channels: int,
        *,
        subset_name: str,
        record: dict,
        record_key: str,
    ) -> np.ndarray:
        target_channels = self._resolved_max_channel(num_channels)
        if num_channels <= target_channels:
            return np.arange(num_channels, dtype=np.int64)
        rng = np.random.default_rng(
            self._fixed_channel_subset_seed(subset_name, record, record_key, num_channels)
        )
        selected = rng.choice(num_channels, size=target_channels, replace=False)
        selected.sort()
        return selected.astype(np.int64, copy=False)

    def _count_valid_sliding_windows_for_series(self, series_ct: np.ndarray) -> int:
        _, total_len = series_ct.shape
        if total_len < self.seq_len:
            return 0

        split, train_reference = self._select_series_split(series_ct)
        if split.shape[1] < self.seq_len:
            return 0

        try:
            split, observed_mask = self._preprocess_split(split, train_reference)
        except Exception:
            return 0

        if not np.isfinite(split).all() or not observed_mask.any():
            return 0

        valid_starts = self._valid_window_starts(observed_mask, split.shape[1])
        return self._effective_window_count_from_starts(valid_starts)

    def _count_valid_sliding_windows_for_record(
        self,
        subset_name: str,
        record: dict,
    ) -> int:
        window_count, _ = self._diagnose_sliding_windows_for_record(subset_name, record)
        return window_count

    def _diagnose_sliding_windows_for_record(
        self,
        subset_name: str,
        record: dict,
    ) -> tuple[int, str]:
        series_ct = self._infer_series_array(record)
        num_channels, total_len = series_ct.shape
        if total_len < self.seq_len:
            return 0, "series_too_short"

        split, train_reference = self._select_series_split(series_ct)
        if split.shape[1] < self.seq_len:
            return 0, "split_too_short"

        try:
            split, observed_mask = self._preprocess_split(split, train_reference)
        except Exception as exc:
            return 0, f"preprocess_error:{type(exc).__name__}"
        if not np.isfinite(split).all() or not observed_mask.any():
            if not np.isfinite(split).all():
                return 0, "non_finite_split"
            return 0, "empty_observed_mask"

        channel_names = self._channel_names(subset_name, record, num_channels)
        channel_text_embeddings, channel_stats_embeddings = self._channel_metadata(
            subset_name,
            record,
            channel_names,
            train_reference,
        )
        if channel_text_embeddings is not None and not torch.isfinite(channel_text_embeddings).all():
            return 0, "non_finite_text_metadata"
        if channel_stats_embeddings is not None and not torch.isfinite(channel_stats_embeddings).all():
            return 0, "non_finite_stats_metadata"

        valid_windows = 0
        valid_starts = self._valid_window_starts(observed_mask, split.shape[1])
        valid_windows = self._effective_window_count_from_starts(valid_starts)
        if valid_windows <= 0:
            return 0, "no_valid_windows"
        return valid_windows, "ok"

    def _valid_window_starts(self, observed_mask: np.ndarray, split_len: int) -> list[int]:
        valid_starts: list[int] = []
        for start in range(0, split_len - self.seq_len + 1, self.stride):
            if observed_mask[:, start:start + self.seq_len].any():
                valid_starts.append(start)
        return valid_starts

    def _effective_window_count_from_starts(self, valid_starts: list[int]) -> int:
        if self.lotsa_sampling_mode == "hierarchical":
            return min(len(valid_starts), self.lotsa_windows_per_series)
        return len(valid_starts)

    def _select_window_starts_for_record(
        self,
        valid_starts: list[int],
        *,
        rng: np.random.Generator | None,
    ) -> list[int]:
        if self.lotsa_sampling_mode != "hierarchical":
            return valid_starts
        target = min(len(valid_starts), self.lotsa_windows_per_series)
        if target <= 0:
            return []
        if self.mode != "train":
            return valid_starts[:target]
        assert rng is not None
        selected = rng.choice(len(valid_starts), size=target, replace=False)
        selected.sort()
        return [valid_starts[int(index)] for index in selected.tolist()]

    @staticmethod
    def _collate_homogeneous_batch(items: list[dict]):
        batch_size = len(items)
        series = torch.stack([item["series"] for item in items], dim=0)
        channel_positions = torch.stack([item["channel_positions"] for item in items], dim=0)
        channel_mask = torch.stack([item["channel_mask"] for item in items], dim=0)
        channel_names = [item["channel_names"] for item in items]
        embeddings = items[0]["channel_text_embeddings"]
        if embeddings is None:
            channel_text_embeddings = None
        else:
            channel_text_embeddings = torch.stack([item["channel_text_embeddings"] for item in items], dim=0)
        stats_embeddings = items[0].get("channel_stats_embeddings")
        if stats_embeddings is None:
            channel_stats_embeddings = None
        else:
            channel_stats_embeddings = torch.stack([item["channel_stats_embeddings"] for item in items], dim=0)
        return {
            "series": series,
            "channel_positions": channel_positions,
            "channel_mask": channel_mask,
            "channel_names": channel_names,
            "channel_text_embeddings": channel_text_embeddings,
            "channel_stats_embeddings": channel_stats_embeddings,
        }

    def _split_window_count(self, total_len: int) -> int:
        if self.lotsa_sampling_mode == "official":
            return int(self._valid_patch_crop_capacity(total_len) >= self.lotsa_min_patches)
        if total_len < self.seq_len:
            return 0
        if self.lotsa_split_mode == "official":
            split_len = total_len
        else:
            train_len = int(total_len * 0.7)
            val_len = int(total_len * 0.1)
            split_len = train_len if self.mode == "train" else val_len
        if split_len < self.seq_len:
            return 0
        return 1 + (split_len - self.seq_len) // self.stride

    def _count_windows_and_batches(self) -> tuple[int, int]:
        cache_token = (
            self.lotsa_sampling_mode,
            self.mode,
            self._subset_iteration_index,
        )
        if self._count_cache is not None and self._count_cache_token == cache_token:
            return self._count_cache

        total_windows = 0
        total_batches = 0
        remaining = self.max_samples if self.max_samples is not None and self.max_samples > 0 else None

        if self.lotsa_sampling_mode == "official":
            subset_names, epoch_iteration_index, sampling_iteration_index = self._planned_official_iteration_state()
            for subset_offset, subset_name in enumerate(subset_names):
                subset_exception_count = 0
                first_exception: str | None = None
                try:
                    dataset = self._load_subset_dataset(subset_name)
                    selected_indices = self._sample_record_indices_for_subset(
                        subset_name,
                        remaining=remaining,
                        sampling_iteration_index=sampling_iteration_index,
                    )
                except Exception as exc:
                    self._warn_skipped_subset(subset_name, "counting", exc)
                    continue
                if not selected_indices:
                    continue
                subset_batch_counts: dict[tuple[int, int], int] = defaultdict(int)
                subset_windows = 0
                for sample_offset, record_index in enumerate(selected_indices):
                    try:
                        record = dataset[int(record_index)]
                        full_series_ct = self._infer_series_array(record)
                        full_num_channels, _ = full_series_ct.shape
                        rng = np.random.default_rng(
                            self.random_seed + epoch_iteration_index + subset_offset * 100_003 + sample_offset
                        )
                        dim_indices = self._sample_dimension_indices(
                            full_num_channels,
                            subset_name=subset_name,
                            record=record,
                            record_key=f"index:{int(record_index)}",
                        )
                        series_ct = full_series_ct[dim_indices]
                        split, train_reference = self._select_series_split(series_ct)
                        if split.shape[1] <= 0:
                            continue
                        crop_bounds = self._official_crop_bounds(split.shape[1], rng=rng)
                        if crop_bounds is None:
                            continue
                        start, stop = crop_bounds
                        crop = split[:, start:stop]
                        _, observed_mask = self._preprocess_split(crop, train_reference)
                        if not observed_mask.any():
                            continue
                        subset_batch_counts[(series_ct.shape[0], crop.shape[1])] += 1
                        subset_windows += 1
                    except Exception as exc:
                        subset_exception_count += 1
                        if first_exception is None:
                            first_exception = f"{type(exc).__name__}: {exc}"
                        continue

                subset_batches = sum(
                    math.ceil(count / self.batch_size) for count in subset_batch_counts.values()
                ) if subset_windows > 0 else 0
                self._debug(
                    "count subset="
                    f"{subset_name} mode={self.mode} sampling=official selected_records={len(selected_indices)} "
                    f"windows={subset_windows} batches={subset_batches} exceptions={subset_exception_count}"
                    + (f" first_exception={first_exception}" if first_exception else "")
                )
                if subset_windows <= 0:
                    continue
                total_windows += subset_windows
                total_batches += subset_batches
                if remaining is not None:
                    remaining -= subset_windows
                    if remaining <= 0:
                        break

            self._count_cache = (total_windows, total_batches)
            self._count_cache_token = cache_token
            return self._count_cache

        for subset_name in self._resolve_subset_names():
            subset_windows = 0
            subset_reason_counts: dict[str, int] = defaultdict(int)
            subset_exception_count = 0
            first_exception: str | None = None
            try:
                for record in self._iter_subset(subset_name):
                    try:
                        window_count, reason = self._diagnose_sliding_windows_for_record(subset_name, record)
                        subset_reason_counts[reason] += 1
                        if window_count <= 0:
                            continue
                        if remaining is not None:
                            if remaining <= 0:
                                break
                            window_count = min(window_count, remaining)
                        subset_windows += window_count
                        total_windows += window_count
                        if remaining is not None:
                            remaining -= window_count
                            if remaining <= 0:
                                break
                    except Exception as exc:
                        subset_exception_count += 1
                        if first_exception is None:
                            first_exception = f"{type(exc).__name__}: {exc}"
                        continue
            except Exception as exc:
                self._warn_skipped_subset(subset_name, "counting", exc)
                continue
            subset_batches = math.ceil(subset_windows / self.batch_size) if subset_windows > 0 else 0
            reason_summary = ", ".join(
                f"{reason}={count}" for reason, count in sorted(subset_reason_counts.items())
            ) or "none"
            self._debug(
                "count subset="
                f"{subset_name} mode={self.mode} sampling={self.lotsa_sampling_mode} windows={subset_windows} "
                f"batches={subset_batches} reasons=[{reason_summary}] exceptions={subset_exception_count}"
                + (f" first_exception={first_exception}" if first_exception else "")
            )
            if subset_windows > 0:
                total_batches += subset_batches
            if remaining is not None and remaining <= 0:
                break

        self._count_cache = (total_windows, total_batches)
        self._count_cache_token = cache_token
        return self._count_cache

    def __iter__(self):
        remaining = self.max_samples if self.max_samples is not None and self.max_samples > 0 else None
        if self.lotsa_sampling_mode == "official":
            epoch_seed = self.random_seed + self._subset_iteration_index
            for subset_offset, subset_name in enumerate(self._iter_subset_names_for_epoch()):
                batch_buckets: dict[tuple[int, int], list[dict]] = {}
                try:
                    dataset = self._load_subset_dataset(subset_name)
                    selected_indices = self._sample_record_indices_for_subset(
                        subset_name,
                        remaining=remaining,
                    )
                    for sample_offset, record_index in enumerate(selected_indices):
                        if remaining is not None and remaining <= 0:
                            break
                        record = dataset[int(record_index)]
                        full_series_ct = self._infer_series_array(record)
                        full_num_channels, _ = full_series_ct.shape
                        rng = np.random.default_rng(epoch_seed + subset_offset * 100_003 + sample_offset)
                        dim_indices = self._sample_dimension_indices(
                            full_num_channels,
                            subset_name=subset_name,
                            record=record,
                            record_key=f"index:{int(record_index)}",
                        )
                        series_ct = full_series_ct[dim_indices]
                        split, train_reference = self._select_series_split(series_ct)
                        num_channels = series_ct.shape[0]
                        if split.shape[1] <= 0:
                            continue
                        crop_bounds = self._official_crop_bounds(split.shape[1], rng=rng)
                        if crop_bounds is None:
                            continue
                        start, stop = crop_bounds
                        crop = split[:, start:stop]
                        crop, observed_mask = self._preprocess_split(crop, train_reference)
                        if not observed_mask.any():
                            continue
                        full_channel_names = self._channel_names(subset_name, record, full_num_channels)
                        channel_names = [full_channel_names[int(index)] for index in dim_indices.tolist()]
                        positions = synthetic_channel_positions(num_channels)
                        channel_text_embeddings, channel_stats_embeddings = self._channel_metadata(
                            subset_name,
                            record,
                            channel_names,
                            train_reference,
                        )
                        batch_key = (num_channels, crop.shape[1])
                        batch_bucket = batch_buckets.setdefault(batch_key, [])
                        batch_bucket.append(
                            {
                                "series": torch.from_numpy(crop),
                                "channel_positions": positions,
                                "channel_mask": torch.from_numpy(observed_mask.any(axis=1)),
                                "channel_names": channel_names,
                                "channel_text_embeddings": channel_text_embeddings,
                                "channel_stats_embeddings": channel_stats_embeddings,
                            }
                        )
                        if remaining is not None:
                            remaining -= 1
                        if len(batch_bucket) == self.batch_size:
                            yield self._collate_homogeneous_batch(batch_bucket)
                            batch_buckets[batch_key] = []
                except Exception as exc:
                    self._warn_skipped_subset(subset_name, "iteration", exc)
                for batch_bucket in batch_buckets.values():
                    if batch_bucket:
                        yield self._collate_homogeneous_batch(batch_bucket)
                if remaining is not None and remaining <= 0:
                    return
            return

        for subset_offset, subset_name in enumerate(self._iter_subset_names_for_epoch()):
            batch_bucket = []
            try:
                for record_offset, record in enumerate(self._iter_subset(subset_name)):
                    try:
                        series_ct = self._infer_series_array(record)  # [C, T]
                        num_channels, total_len = series_ct.shape
                        if total_len < self.seq_len:
                            continue
                        split, train_reference = self._select_series_split(series_ct)
                        if split.shape[1] < self.seq_len:
                            continue
                        split, observed_mask = self._preprocess_split(split, train_reference)
                        if not np.isfinite(split).all() or not observed_mask.any():
                            continue
                        channel_names = self._channel_names(subset_name, record, num_channels)
                        positions = synthetic_channel_positions(num_channels)
                        channel_text_embeddings, channel_stats_embeddings = self._channel_metadata(
                            subset_name,
                            record,
                            channel_names,
                            train_reference,
                        )
                        if channel_text_embeddings is not None and not torch.isfinite(channel_text_embeddings).all():
                            continue
                        if channel_stats_embeddings is not None and not torch.isfinite(channel_stats_embeddings).all():
                            continue
                        valid_starts = self._valid_window_starts(observed_mask, split.shape[1])
                        if not valid_starts:
                            continue
                        rng = None
                        if self.lotsa_sampling_mode == "hierarchical":
                            rng = np.random.default_rng(
                                self.random_seed
                                + self._subset_iteration_index
                                + subset_offset * 100_003
                                + record_offset
                            )
                        selected_starts = self._select_window_starts_for_record(valid_starts, rng=rng)
                        for start in selected_starts:
                            if remaining is not None and remaining <= 0:
                                break
                            window_observed = observed_mask[:, start:start + self.seq_len]
                            batch_bucket.append(
                                {
                                    "series": torch.from_numpy(split[:, start:start + self.seq_len]),
                                    "channel_positions": positions,
                                    "channel_mask": torch.from_numpy(window_observed.any(axis=1)),
                                    "channel_names": channel_names,
                                    "channel_text_embeddings": channel_text_embeddings,
                                    "channel_stats_embeddings": channel_stats_embeddings,
                                }
                            )
                            if remaining is not None:
                                remaining -= 1
                            if len(batch_bucket) == self.batch_size:
                                yield self._collate_homogeneous_batch(batch_bucket)
                                batch_bucket = []
                        if remaining is not None and remaining <= 0:
                            break
                    except Exception:
                        continue
            except Exception as exc:
                self._warn_skipped_subset(subset_name, "iteration", exc)
            if batch_bucket:
                yield self._collate_homogeneous_batch(batch_bucket)
            if remaining is not None and remaining <= 0:
                return


class TSLDTimeSeriesPretrainDataset(Dataset):
    def __init__(self, root_path: str, seq_len: int = 512, stride: int = 512, mode: str = "train", max_files: int | None = None, tsld_mode: str = "univariate", channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False, selected_channel_count: int | None = None, csv_files: list[str] | None = None, dataset_name_override: str | None = None, domain_name_override: str | None = None) -> None:
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode
        self.tsld_mode = validate_tsld_mode(tsld_mode)
        self.channel_metadata_mode = str(channel_metadata_mode).strip().lower()
        self.selected_channel_count = selected_channel_count
        self.dataset_name_override = dataset_name_override
        self.domain_name_override = domain_name_override
        self.sample_indices = []
        self.time_series_list = []
        self.channel_names_list = []
        self.positions_list = []
        self.channel_text_embeddings_list = []
        self.channel_stats_embeddings_list = []
        self.series_summaries = []
        discovered_csv_files = list_tsld_csv_files(root_path, max_files=max_files) if csv_files is None else sorted(csv_files)
        if max_files and csv_files is not None:
            discovered_csv_files = discovered_csv_files[:max_files]
        multivariate_candidates = []
        channel_count_histogram: dict[int, int] = {}
        for file_path in discovered_csv_files:
            try:
                df = load_tsld_frame(file_path)
                cols = [str(col) for col in df.columns]
                if self.tsld_mode == "multivariate":
                    series_data = df.values.astype(np.float32)
                    total_len = len(series_data)
                    if total_len < self.seq_len:
                        continue
                    train_len, offset, start_limit = tsld_split_bounds(total_len, mode)
                    if start_limit - offset < self.seq_len or train_len <= 0:
                        continue
                    channel_count = series_data.shape[1]
                    channel_count_histogram[channel_count] = channel_count_histogram.get(channel_count, 0) + 1
                    if self.selected_channel_count is not None and channel_count != self.selected_channel_count:
                        continue
                    mean = series_data[:train_len].mean(axis=0, keepdims=True)
                    std = series_data[:train_len].std(axis=0, keepdims=True) + 1e-8
                    split_data = ((series_data[offset:start_limit] - mean) / std).astype(np.float32)
                    multivariate_candidates.append((split_data, list(cols), channel_count, file_path))
                else:
                    for col in cols:
                        col_data = pd.Series(df[col].values.astype(np.float32)).ffill().fillna(0).values
                        total_len = len(col_data)
                        if total_len < self.seq_len:
                            continue
                        train_len, offset, start_limit = tsld_split_bounds(total_len, mode)
                        if start_limit - offset < self.seq_len or train_len <= 0:
                            continue
                        mean = col_data[:train_len].mean()
                        std = col_data[:train_len].std() + 1e-8
                        split_data = ((col_data[offset:start_limit] - mean) / std).reshape(-1, 1).astype(np.float32)
                        self._append_series(
                            split_data,
                            [col],
                            file_path=file_path,
                            root_path=root_path,
                            text_encoder_name_or_path=text_encoder_name_or_path,
                            text_metadata_cache_dir=text_metadata_cache_dir,
                            text_encoder_local_files_only=text_encoder_local_files_only,
                        )
            except Exception:
                continue
        self.observed_channel_counts = sorted(channel_count_histogram)
        if self.tsld_mode == "multivariate" and multivariate_candidates:
            for split_data, cols, _, file_path in multivariate_candidates:
                self._append_series(
                    split_data,
                    cols,
                    file_path=file_path,
                    root_path=root_path,
                    text_encoder_name_or_path=text_encoder_name_or_path,
                    text_metadata_cache_dir=text_metadata_cache_dir,
                    text_encoder_local_files_only=text_encoder_local_files_only,
                )
        if self.selected_channel_count is not None:
            self.num_channels = self.selected_channel_count
        elif self.time_series_list:
            self.num_channels = self.time_series_list[0].shape[1]
        else:
            self.num_channels = 0

    def _append_series(self, split_data: np.ndarray, cols, *, file_path: str, root_path: str, text_encoder_name_or_path: str, text_metadata_cache_dir: str, text_encoder_local_files_only: bool):
        series_idx = len(self.time_series_list)
        self.time_series_list.append(split_data)
        self.channel_names_list.append(list(cols))
        self.positions_list.append(synthetic_channel_positions(split_data.shape[1]))
        metadata_embeddings = None
        stats_embeddings = None
        domain_name, dataset_name = infer_tsld_context(root_path, file_path)
        if self.domain_name_override:
            domain_name = self.domain_name_override
        if self.dataset_name_override:
            dataset_name = self.dataset_name_override
        if self.channel_metadata_mode in {"text", "text_stats_avg"}:
            metadata_embeddings = build_text_channel_metadata(
                channel_names=cols,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        elif self.channel_metadata_mode == "text_stats_joint":
            metadata_embeddings = build_joint_text_stats_channel_metadata(
                split_data,
                channel_names=cols,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            stats_embeddings = build_statistical_channel_metadata(
                split_data,
                channel_names=cols,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        self.channel_text_embeddings_list.append(metadata_embeddings)
        self.channel_stats_embeddings_list.append(stats_embeddings)
        current_len = split_data.shape[0]
        num_windows = 0
        for start in range(0, current_len - self.seq_len + 1, self.stride):
            self.sample_indices.append((series_idx, start))
            num_windows += 1
        self.series_summaries.append(
            {
                "series_idx": series_idx,
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "dataset_name": dataset_name,
                "domain_name": domain_name,
                "split_shape": tuple(split_data.shape),
                "num_channels": int(split_data.shape[1]),
                "num_timesteps": int(split_data.shape[0]),
                "num_windows": int(num_windows),
            }
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        series_idx, start = self.sample_indices[idx]
        series_data = self.time_series_list[series_idx]
        window = torch.from_numpy(series_data[start:start + self.seq_len].T)
        return {
            "series": window,
            "channel_positions": self.positions_list[series_idx],
            "channel_mask": torch.ones(window.shape[0], dtype=torch.bool),
            "channel_names": self.channel_names_list[series_idx],
            "channel_text_embeddings": self.channel_text_embeddings_list[series_idx],
            "channel_stats_embeddings": self.channel_stats_embeddings_list[series_idx],
        }


class TSLibTimeSeriesPretrainDataset(Dataset):
    def __init__(self, root_path: str, seq_len: int = 512, stride: int = 512, mode: str = "train", max_files: int | None = None, tslib_mode: str = "univariate", channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False, selected_channel_count: int | None = None, csv_files: list[str] | None = None, dataset_name_override: str | None = None, domain_name_override: str | None = None) -> None:
        self.seq_len = seq_len
        self.stride = stride
        self.mode = mode
        self.tslib_mode = validate_tslib_mode(tslib_mode)
        self.channel_metadata_mode = str(channel_metadata_mode).strip().lower()
        self.selected_channel_count = selected_channel_count
        self.dataset_name_override = dataset_name_override
        self.domain_name_override = domain_name_override
        self.sample_indices = []
        self.time_series_list = []
        self.channel_names_list = []
        self.positions_list = []
        self.channel_text_embeddings_list = []
        self.channel_stats_embeddings_list = []
        self.series_summaries = []
        csv_files = list_tslib_files(root_path, max_files=max_files) if csv_files is None else sorted(csv_files)
        if max_files and csv_files is not None:
            csv_files = csv_files[:max_files]
        multivariate_candidates = []
        channel_count_histogram: dict[int, int] = {}
        for file_path in csv_files:
            try:
                df = load_tslib_frame(file_path)
                cols = [str(col) for col in df.columns]
                if self.tslib_mode == "multivariate":
                    series_data = df.values.astype(np.float32)
                    total_len = len(series_data)
                    if total_len < self.seq_len:
                        continue
                    train_len, offset, start_limit = tslib_split_bounds(total_len, mode)
                    if start_limit - offset < self.seq_len or train_len <= 0:
                        continue
                    channel_count = series_data.shape[1]
                    channel_count_histogram[channel_count] = channel_count_histogram.get(channel_count, 0) + 1
                    if self.selected_channel_count is not None and channel_count != self.selected_channel_count:
                        continue
                    mean = series_data[:train_len].mean(axis=0, keepdims=True)
                    std = series_data[:train_len].std(axis=0, keepdims=True) + 1e-8
                    split_data = ((series_data[offset:start_limit] - mean) / std).astype(np.float32)
                    multivariate_candidates.append((split_data, list(cols), channel_count, file_path))
                else:
                    for col in cols:
                        col_data = pd.Series(df[col].values.astype(np.float32)).ffill().fillna(0).values
                        total_len = len(col_data)
                        if total_len < self.seq_len:
                            continue
                        train_len, offset, start_limit = tslib_split_bounds(total_len, mode)
                        if start_limit - offset < self.seq_len or train_len <= 0:
                            continue
                        mean = col_data[:train_len].mean()
                        std = col_data[:train_len].std() + 1e-8
                        split_data = ((col_data[offset:start_limit] - mean) / std).reshape(-1, 1).astype(np.float32)
                        self._append_series(
                            split_data,
                            [col],
                            file_path=file_path,
                            root_path=root_path,
                            text_encoder_name_or_path=text_encoder_name_or_path,
                            text_metadata_cache_dir=text_metadata_cache_dir,
                            text_encoder_local_files_only=text_encoder_local_files_only,
                        )
            except Exception:
                continue
        self.observed_channel_counts = sorted(channel_count_histogram)
        if self.tslib_mode == "multivariate" and multivariate_candidates:
            for split_data, cols, c, file_path in multivariate_candidates:
                self._append_series(
                    split_data,
                    cols,
                    file_path=file_path,
                    root_path=root_path,
                    text_encoder_name_or_path=text_encoder_name_or_path,
                    text_metadata_cache_dir=text_metadata_cache_dir,
                    text_encoder_local_files_only=text_encoder_local_files_only,
                )
        if self.selected_channel_count is not None:
            self.num_channels = self.selected_channel_count
        elif self.time_series_list:
            self.num_channels = self.time_series_list[0].shape[1]
        else:
            self.num_channels = 0

    def _append_series(self, split_data: np.ndarray, cols, *, file_path: str, root_path: str, text_encoder_name_or_path: str, text_metadata_cache_dir: str, text_encoder_local_files_only: bool):
        series_idx = len(self.time_series_list)
        self.time_series_list.append(split_data)
        self.channel_names_list.append(list(cols))
        self.positions_list.append(synthetic_channel_positions(split_data.shape[1]))
        metadata_embeddings = None
        stats_embeddings = None
        domain_name, dataset_name = infer_tslib_context(root_path, file_path)
        if self.domain_name_override:
            domain_name = self.domain_name_override
        if self.dataset_name_override:
            dataset_name = self.dataset_name_override
        if self.channel_metadata_mode in {"text", "text_stats_avg"}:
            metadata_embeddings = build_text_channel_metadata(
                channel_names=cols,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        elif self.channel_metadata_mode == "text_stats_joint":
            metadata_embeddings = build_joint_text_stats_channel_metadata(
                split_data,
                channel_names=cols,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            stats_embeddings = build_statistical_channel_metadata(
                split_data,
                channel_names=cols,
                domain=domain_name,
                dataset_name=dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        self.channel_text_embeddings_list.append(metadata_embeddings)
        self.channel_stats_embeddings_list.append(stats_embeddings)
        current_len = split_data.shape[0]
        num_windows = 0
        for start in range(0, current_len - self.seq_len + 1, self.stride):
            self.sample_indices.append((series_idx, start))
            num_windows += 1
        self.series_summaries.append(
            {
                "series_idx": series_idx,
                "file_name": os.path.basename(file_path),
                "file_path": file_path,
                "dataset_name": dataset_name,
                "domain_name": domain_name,
                "split_shape": tuple(split_data.shape),
                "num_channels": int(split_data.shape[1]),
                "num_timesteps": int(split_data.shape[0]),
                "num_windows": int(num_windows),
            }
        )

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        series_idx, start = self.sample_indices[idx]
        series_data = self.time_series_list[series_idx]
        window = torch.from_numpy(series_data[start:start + self.seq_len].T)
        return {
            "series": window,
            "channel_positions": self.positions_list[series_idx],
            "channel_mask": torch.ones(window.shape[0], dtype=torch.bool),
            "channel_names": self.channel_names_list[series_idx],
            "channel_text_embeddings": self.channel_text_embeddings_list[series_idx],
            "channel_stats_embeddings": self.channel_stats_embeddings_list[series_idx],
        }


class FileAwareBatchSampler(Sampler):
    def __init__(self, dataset: TSLDTimeSeriesPretrainDataset | TSLibTimeSeriesPretrainDataset, batch_size: int, shuffle: bool, drop_last: bool) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        grouped_indices: dict[int, list[int]] = defaultdict(list)
        for sample_idx, (series_idx, _) in enumerate(dataset.sample_indices):
            grouped_indices[series_idx].append(sample_idx)
        self.indices_by_series = dict(grouped_indices)

    def __iter__(self):
        series_groups = list(self.indices_by_series.values())
        if self.shuffle:
            random.shuffle(series_groups)
        batches: list[list[int]] = []
        for group_indices in series_groups:
            group_indices = list(group_indices)
            if self.shuffle:
                random.shuffle(group_indices)
            full_batch_limit = len(group_indices) if not self.drop_last else (len(group_indices) // self.batch_size) * self.batch_size
            for start in range(0, full_batch_limit, self.batch_size):
                batch = group_indices[start:start + self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    batches.append(batch)
            if not self.drop_last and full_batch_limit < len(group_indices):
                batches.append(group_indices[full_batch_limit:])
        if self.shuffle:
            random.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        total = 0
        for group_indices in self.indices_by_series.values():
            if self.drop_last:
                total += len(group_indices) // self.batch_size
            else:
                total += (len(group_indices) + self.batch_size - 1) // self.batch_size
        return total


def _build_tsld_loader(dataset: TSLDTimeSeriesPretrainDataset, *, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int) -> DataLoader:
    batch_sampler = FileAwareBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_with_positions)


def _build_tslib_loader(dataset: TSLibTimeSeriesPretrainDataset, *, batch_size: int, shuffle: bool, drop_last: bool, num_workers: int) -> DataLoader:
    batch_sampler = FileAwareBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_with_positions)


def _parse_lotsa_subset_names(path: str) -> list[str] | None:
    if not str(path).strip():
        return None
    subset_names = [item.strip() for item in str(path).split(",") if item.strip()]
    return subset_names or None


def _split_lotsa_sample_budget(max_samples: int | None, subset_count: int) -> list[int | None]:
    if max_samples is None:
        return [None] * subset_count
    if max_samples <= 0:
        raise ValueError("For dataset_type='lotsa', --max_files must be positive when provided.")
    base, remainder = divmod(max_samples, subset_count)
    budgets: list[int | None] = []
    for index in range(subset_count):
        budget = base + (1 if index < remainder else 0)
        budgets.append(budget)
    return budgets


def _resolve_lotsa_num_workers(requested_num_workers: int) -> int:
    # Our LOTSA loader is an IterableDataset over HF streaming/local datasets.
    # Using DataLoader workers here can duplicate or over-fetch iterator items,
    # which breaks the contract between __len__ and actual yielded batches.
    # Keep LOTSA single-process so batch counting and iteration stay identical.
    return 0


def get_pretrain_loaders(dataset_type: str, path: str, batch_size: int = 256, seq_len: int = 512, stride: int = 128, patch_size: int = 16, num_workers: int = 4, max_files: int | None = None, tsld_mode: str = "univariate", tslib_mode: str = "univariate", channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False, lotsa_dataset_path: str = "Salesforce/lotsa_data", lotsa_split_mode: str = "official", lotsa_sampling_mode: str = "official", lotsa_preprocessing_mode: str = "official", lotsa_sample_time_series: str = "proportional", lotsa_subset_sampling: str = "official", lotsa_min_patches: int = 2, lotsa_max_channel: int | None = None, lotsa_windows_per_series: int = 32):
    if dataset_type == "lotsa":
        effective_num_workers = _resolve_lotsa_num_workers(num_workers)
        subset_names = _parse_lotsa_subset_names(path)
        train_ds = LOTSABatchStreamingPretrainDataset(
            dataset_name=lotsa_dataset_path,
            subset_names=subset_names,
            batch_size=batch_size,
            seq_len=seq_len,
            stride=stride,
            patch_size=patch_size,
            mode="train",
            channel_metadata_mode=channel_metadata_mode,
            text_encoder_name_or_path=text_encoder_name_or_path,
            text_metadata_cache_dir=text_metadata_cache_dir,
            text_encoder_local_files_only=text_encoder_local_files_only,
            lotsa_split_mode=lotsa_split_mode,
            lotsa_sampling_mode=lotsa_sampling_mode,
            lotsa_preprocessing_mode=lotsa_preprocessing_mode,
            lotsa_sample_time_series=lotsa_sample_time_series,
            lotsa_subset_sampling=lotsa_subset_sampling,
            lotsa_min_patches=lotsa_min_patches,
            lotsa_max_channel=lotsa_max_channel,
            lotsa_windows_per_series=lotsa_windows_per_series,
            max_samples=max_files,
            skip_samples=0,
            shuffle_buffer_size=512,
        )
        try:
            val_ds = LOTSABatchStreamingPretrainDataset(
                dataset_name=lotsa_dataset_path,
                subset_names=subset_names,
                batch_size=batch_size,
                seq_len=seq_len,
                stride=stride,
                patch_size=patch_size,
                mode="val",
                channel_metadata_mode=channel_metadata_mode,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
                lotsa_split_mode=lotsa_split_mode,
                lotsa_sampling_mode=lotsa_sampling_mode,
                lotsa_preprocessing_mode=lotsa_preprocessing_mode,
                lotsa_sample_time_series="none",
                lotsa_subset_sampling=lotsa_subset_sampling,
                lotsa_min_patches=lotsa_min_patches,
                lotsa_max_channel=lotsa_max_channel,
                lotsa_windows_per_series=lotsa_windows_per_series,
                max_samples=max_files,
                skip_samples=0,
                shuffle_buffer_size=1,
            )
            val_length = len(val_ds)
        except ValueError:
            val_ds = None
            val_length = 0
        train_loader = DataLoader(train_ds, batch_size=None, num_workers=effective_num_workers)
        val_loader = None if val_ds is None or val_length == 0 else DataLoader(val_ds, batch_size=None, num_workers=effective_num_workers)
        return train_loader, val_loader

    if dataset_type == "tsld":
        train_selected_channel_count = None
        if tsld_mode == "multivariate":
            probe_train_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "train", max_files, tsld_mode=tsld_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
            probe_val_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "val", max_files, tsld_mode=tsld_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
            shared_channel_counts = sorted(set(probe_train_ds.observed_channel_counts) & set(probe_val_ds.observed_channel_counts))
            if shared_channel_counts:
                train_selected_channel_count = max(shared_channel_counts)
        train_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "train", max_files, tsld_mode=tsld_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, selected_channel_count=train_selected_channel_count)
        val_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "val", max_files, tsld_mode=tsld_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, selected_channel_count=train_selected_channel_count)
        if tsld_mode == "multivariate":
            train_loader = _build_tsld_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, num_workers=num_workers)
            val_loader = _build_tsld_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            return train_loader, val_loader
    elif dataset_type == "tslib":
        train_selected_channel_count = None
        if tslib_mode == "multivariate":
            probe_train_ds = TSLibTimeSeriesPretrainDataset(path, seq_len, stride, "train", max_files, tslib_mode=tslib_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
            probe_val_ds = TSLibTimeSeriesPretrainDataset(path, seq_len, stride, "val", max_files, tslib_mode=tslib_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
            shared_channel_counts = sorted(set(probe_train_ds.observed_channel_counts) & set(probe_val_ds.observed_channel_counts))
            if shared_channel_counts:
                train_selected_channel_count = max(shared_channel_counts)
        train_ds = TSLibTimeSeriesPretrainDataset(path, seq_len, stride, "train", max_files, tslib_mode=tslib_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, selected_channel_count=train_selected_channel_count)
        val_ds = TSLibTimeSeriesPretrainDataset(path, seq_len, stride, "val", max_files, tslib_mode=tslib_mode, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, selected_channel_count=train_selected_channel_count)
        if tslib_mode == "multivariate":
            train_loader = _build_tslib_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, num_workers=num_workers)
            val_loader = _build_tslib_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
            return train_loader, val_loader
    else:
        train_ds = CSVTimeSeriesPretrainDataset(path, seq_len, stride, "train", dataset_name=dataset_type, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
        val_ds = CSVTimeSeriesPretrainDataset(path, seq_len, stride, "val", dataset_name=dataset_type, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, num_workers=num_workers, collate_fn=collate_with_positions)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_with_positions)
    return train_loader, val_loader


def get_lotsa_pretrain_loader_groups(path: str, batch_size: int = 256, seq_len: int = 512, stride: int = 128, patch_size: int = 16, num_workers: int = 4, max_files: int | None = None, channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False, lotsa_dataset_path: str = "Salesforce/lotsa_data", lotsa_split_mode: str = "official", lotsa_sampling_mode: str = "official", lotsa_preprocessing_mode: str = "official", lotsa_sample_time_series: str = "proportional", lotsa_subset_sampling: str = "official", lotsa_min_patches: int = 2, lotsa_max_channel: int | None = None, lotsa_windows_per_series: int = 32):
    effective_num_workers = _resolve_lotsa_num_workers(num_workers)
    subset_names = _parse_lotsa_subset_names(path)
    if subset_names is None:
        probe_ds = LOTSABatchStreamingPretrainDataset(
            dataset_name=lotsa_dataset_path,
            subset_names=None,
            batch_size=batch_size,
            seq_len=seq_len,
            stride=stride,
            patch_size=patch_size,
            mode="train",
            channel_metadata_mode=channel_metadata_mode,
            text_encoder_name_or_path=text_encoder_name_or_path,
            text_metadata_cache_dir=text_metadata_cache_dir,
            text_encoder_local_files_only=text_encoder_local_files_only,
            lotsa_split_mode=lotsa_split_mode,
            lotsa_sampling_mode=lotsa_sampling_mode,
            lotsa_preprocessing_mode=lotsa_preprocessing_mode,
            lotsa_sample_time_series=lotsa_sample_time_series,
            lotsa_subset_sampling=lotsa_subset_sampling,
            lotsa_min_patches=lotsa_min_patches,
            lotsa_max_channel=lotsa_max_channel,
            lotsa_windows_per_series=lotsa_windows_per_series,
            max_samples=None,
            skip_samples=0,
            shuffle_buffer_size=512,
        )
        subset_names = probe_ds._resolve_subset_names()
    else:
        probe_ds = LOTSABatchStreamingPretrainDataset(
            dataset_name=lotsa_dataset_path,
            subset_names=subset_names,
            batch_size=batch_size,
            seq_len=seq_len,
            stride=stride,
            patch_size=patch_size,
            mode="train",
            channel_metadata_mode=channel_metadata_mode,
            text_encoder_name_or_path=text_encoder_name_or_path,
            text_metadata_cache_dir=text_metadata_cache_dir,
            text_encoder_local_files_only=text_encoder_local_files_only,
            lotsa_split_mode=lotsa_split_mode,
            lotsa_sampling_mode=lotsa_sampling_mode,
            lotsa_preprocessing_mode=lotsa_preprocessing_mode,
            lotsa_sample_time_series=lotsa_sample_time_series,
            lotsa_subset_sampling=lotsa_subset_sampling,
            lotsa_min_patches=lotsa_min_patches,
            lotsa_max_channel=lotsa_max_channel,
            lotsa_windows_per_series=lotsa_windows_per_series,
            max_samples=None,
            skip_samples=0,
            shuffle_buffer_size=512,
        )

    budgets = _split_lotsa_sample_budget(max_files, len(subset_names))
    subset_weights = (
        probe_ds._subset_sampling_weights(subset_names)
        if lotsa_subset_sampling == "official"
        else {subset_name: 1.0 / max(1, len(subset_names)) for subset_name in subset_names}
    )
    loader_groups = []
    for subset_name, subset_budget in zip(subset_names, budgets):
        if subset_budget is not None and subset_budget <= 0:
            if _LOTSA_DEBUG_ENABLED:
                print(f"[LOTSA-DEBUG] skip group subset={subset_name} reason=non_positive_budget budget={subset_budget}")
            continue
        train_ds = LOTSABatchStreamingPretrainDataset(
            dataset_name=lotsa_dataset_path,
            subset_names=[subset_name],
            batch_size=batch_size,
            seq_len=seq_len,
            stride=stride,
            patch_size=patch_size,
            mode="train",
            channel_metadata_mode=channel_metadata_mode,
            text_encoder_name_or_path=text_encoder_name_or_path,
            text_metadata_cache_dir=text_metadata_cache_dir,
            text_encoder_local_files_only=text_encoder_local_files_only,
            lotsa_split_mode=lotsa_split_mode,
            lotsa_sampling_mode=lotsa_sampling_mode,
            lotsa_preprocessing_mode=lotsa_preprocessing_mode,
            lotsa_sample_time_series=lotsa_sample_time_series,
            lotsa_subset_sampling=lotsa_subset_sampling,
            lotsa_min_patches=lotsa_min_patches,
            lotsa_max_channel=lotsa_max_channel,
            lotsa_windows_per_series=lotsa_windows_per_series,
            max_samples=subset_budget,
            skip_samples=0,
            shuffle_buffer_size=512,
        )
        try:
            val_ds = LOTSABatchStreamingPretrainDataset(
            dataset_name=lotsa_dataset_path,
                subset_names=[subset_name],
                batch_size=batch_size,
                seq_len=seq_len,
                stride=stride,
                patch_size=patch_size,
                mode="val",
                channel_metadata_mode=channel_metadata_mode,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
                lotsa_split_mode=lotsa_split_mode,
                lotsa_sampling_mode=lotsa_sampling_mode,
                lotsa_preprocessing_mode=lotsa_preprocessing_mode,
                lotsa_sample_time_series="none",
                lotsa_subset_sampling=lotsa_subset_sampling,
                lotsa_min_patches=lotsa_min_patches,
                lotsa_max_channel=lotsa_max_channel,
                lotsa_windows_per_series=lotsa_windows_per_series,
                max_samples=subset_budget,
                skip_samples=0,
                shuffle_buffer_size=1,
            )
            val_length = len(val_ds)
        except ValueError:
            val_ds = None
            val_length = 0
        train_length = len(train_ds)
        if _LOTSA_DEBUG_ENABLED:
            print(
                "[LOTSA-DEBUG] group subset="
                f"{subset_name} budget={subset_budget} train_batches={train_length} "
                f"train_samples={train_ds.num_samples} val_batches={val_length} "
                f"val_samples={0 if val_ds is None else val_ds.num_samples} "
                f"split_mode={lotsa_split_mode} sampling_mode={lotsa_sampling_mode} "
                f"preprocessing_mode={lotsa_preprocessing_mode} subset_sampling={lotsa_subset_sampling} "
                f"subset_weight={subset_weights.get(subset_name, 0.0):.6f} "
                f"max_channel={lotsa_max_channel} windows_per_series={lotsa_windows_per_series} "
                f"seq_len={seq_len} stride={stride}"
            )
        if train_length == 0:
            if _LOTSA_DEBUG_ENABLED:
                print(f"[LOTSA-DEBUG] drop group subset={subset_name} reason=empty_train_dataset")
            continue
        train_loader = DataLoader(train_ds, batch_size=None, num_workers=effective_num_workers)
        val_loader = None if val_ds is None or val_length == 0 else DataLoader(val_ds, batch_size=None, num_workers=effective_num_workers)
        loader_groups.append(
            {
                "group_name": subset_name,
                "channel_count": train_ds.num_channels,
                "subset_weight": float(subset_weights.get(subset_name, 0.0)),
                "train_loader": train_loader,
                "val_loader": val_loader,
            }
        )
    return loader_groups


def get_tsld_pretrain_loader_groups(path: str, batch_size: int = 256, seq_len: int = 512, stride: int = 128, num_workers: int = 4, max_files: int | None = None, channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False):
    probe_train_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "train", max_files, tsld_mode="multivariate", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
    probe_val_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "val", max_files, tsld_mode="multivariate", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
    shared_channel_counts = sorted(set(probe_train_ds.observed_channel_counts) & set(probe_val_ds.observed_channel_counts))
    loader_groups = []
    for channel_count in shared_channel_counts:
        train_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "train", max_files, tsld_mode="multivariate", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, selected_channel_count=channel_count)
        val_ds = TSLDTimeSeriesPretrainDataset(path, seq_len, stride, "val", max_files, tsld_mode="multivariate", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, selected_channel_count=channel_count)
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue
        train_loader = _build_tsld_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, num_workers=num_workers)
        val_loader = _build_tsld_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        loader_groups.append(
            {
                "group_name": f"C={channel_count}",
                "channel_count": channel_count,
                "train_loader": train_loader,
                "val_loader": val_loader,
            }
        )
    return loader_groups


def get_tslib_pretrain_loader_groups(path: str, batch_size: int = 256, seq_len: int = 512, stride: int = 128, num_workers: int = 4, max_files: int | None = None, channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False):
    csv_files = list_tslib_files(path, max_files=max_files)
    loader_groups = []
    for file_path in csv_files:
        domain_name, dataset_name = infer_tslib_context(path, file_path)
        train_ds = TSLibTimeSeriesPretrainDataset(path, seq_len, stride, "train", None, tslib_mode="multivariate", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, csv_files=[file_path], dataset_name_override=dataset_name, domain_name_override=domain_name)
        val_ds = TSLibTimeSeriesPretrainDataset(path, seq_len, stride, "val", None, tslib_mode="multivariate", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, csv_files=[file_path], dataset_name_override=dataset_name, domain_name_override=domain_name)
        if len(train_ds) == 0 or len(val_ds) == 0:
            continue
        channel_count = train_ds.num_channels
        train_loader = _build_tslib_loader(train_ds, batch_size=batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, num_workers=num_workers)
        val_loader = _build_tslib_loader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
        loader_groups.append(
            {
                "group_name": dataset_name,
                "channel_count": channel_count,
                "train_loader": train_loader,
                "val_loader": val_loader,
            }
        )
    return loader_groups
