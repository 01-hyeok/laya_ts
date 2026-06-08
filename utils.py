from __future__ import annotations

import os
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from .channel_metadata import (
    build_channel_descriptions,
    build_joint_channel_descriptions,
    build_statistical_channel_descriptions,
    encode_channel_descriptions,
)


VALID_TSLD_MODES = ("univariate", "multivariate")
VALID_TSLIB_MODES = ("univariate", "multivariate")
_TEXT_METADATA_PREVIEW_KEYS: set[tuple[str, str, tuple[str, ...]]] = set()
_STATS_METADATA_PREVIEW_KEYS: set[tuple[str, str, tuple[str, ...]]] = set()
_JOINT_METADATA_PREVIEW_KEYS: set[tuple[str, str, tuple[str, ...]]] = set()
_TSLD_PRETRAIN_SUPPORTED_EXTENSIONS = (".csv",)
_TSLIB_PRETRAIN_SUPPORTED_EXTENSIONS = (".csv", ".txt", ".npz")
_TSLIB_PRETRAIN_EXCLUDED_ROOT_DIRS = {"UCR", "UEA"}


def validate_tsld_mode(tsld_mode: str) -> str:
    mode = str(tsld_mode).strip().lower()
    if mode not in VALID_TSLD_MODES:
        raise ValueError(f"Unsupported tsld_mode: {tsld_mode}. Supported: {', '.join(VALID_TSLD_MODES)}")
    return mode


def validate_tslib_mode(tslib_mode: str) -> str:
    mode = str(tslib_mode).strip().lower()
    if mode not in VALID_TSLIB_MODES:
        raise ValueError(f"Unsupported tslib_mode: {tslib_mode}. Supported: {', '.join(VALID_TSLIB_MODES)}")
    return mode


def tsld_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]]


def tslib_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]]


def tsld_split_bounds(total_len: int, mode: str) -> Tuple[int, int, int]:
    val_len = int(total_len * 0.1)
    test_len = int(total_len * 0.2)
    train_len = total_len - val_len - test_len
    if mode == "train":
        return train_len, 0, train_len
    if mode == "val":
        return train_len, train_len, train_len + val_len
    return train_len, train_len + val_len, total_len


def tslib_split_bounds(total_len: int, mode: str) -> Tuple[int, int, int]:
    val_len = int(total_len * 0.1)
    test_len = int(total_len * 0.2)
    train_len = total_len - val_len - test_len
    if mode == "train":
        return train_len, 0, train_len
    if mode == "val":
        return train_len, train_len, train_len + val_len
    return train_len, train_len + val_len, total_len


def load_csv_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    for c in ["date", "Date", "timestamp", "Timestamp", "time", "Time"]:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
            break
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df.loc[df[col] < -9990.0, col] = np.nan
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear", limit_direction="both")
    df.dropna(inplace=True, axis=0)
    return df


def synthetic_channel_positions(num_channels: int) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, num_channels)
    y = torch.zeros_like(x)
    z = torch.linspace(1.0, 0.5, num_channels)
    return torch.stack([x, y, z], dim=-1)


def channel_names_for_count(num_channels: int) -> List[str]:
    return [f"channel_{idx+1}" for idx in range(num_channels)]


def infer_tsld_context(root_path: str, file_path: str) -> Tuple[str, str]:
    rel_path = os.path.relpath(file_path, root_path)
    parts = [part for part in os.path.dirname(rel_path).split(os.sep) if part not in {"", "."}]
    domain = parts[0] if parts else "tsld"
    dataset_name = parts[-1] if parts else os.path.splitext(os.path.basename(file_path))[0]
    return domain, dataset_name


def infer_tslib_context(root_path: str, file_path: str) -> Tuple[str, str]:
    rel_path = os.path.relpath(file_path, root_path)
    parts = [part for part in os.path.dirname(rel_path).split(os.sep) if part not in {"", "."}]
    domain = parts[0] if parts else "tslib"
    dataset_name = parts[-1] if parts else os.path.splitext(os.path.basename(file_path))[0]
    return domain, dataset_name


def infer_csv_context(dataset_name: str, data_path: str) -> Tuple[str, str]:
    base_name = str(dataset_name).strip() or os.path.splitext(os.path.basename(data_path))[0]
    pretty_name = base_name.replace("_", " ")
    return pretty_name, pretty_name


def load_tsld_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    cols = tsld_feature_columns(df)
    if cols:
        df = df.loc[:, cols]
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill().fillna(0)
    return df


def load_tslib_frame(path: str) -> pd.DataFrame:
    path_lower = str(path).lower()
    if path_lower.endswith(".csv"):
        df = pd.read_csv(path, low_memory=False)
        cols = tslib_feature_columns(df)
        if cols:
            df = df.loc[:, cols]
    elif path_lower.endswith(".txt"):
        rows: list[list[float]] = []
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                rows.append([float(piece) for piece in line.split(",")])
        if not rows:
            raise ValueError(f"tslib text file is empty: {path}")
        df = pd.DataFrame(np.asarray(rows, dtype=np.float32))
    elif path_lower.endswith(".npz"):
        payload = np.load(path, allow_pickle=True)
        if "data" not in payload:
            raise ValueError(f"tslib npz file does not contain a 'data' array: {path}")
        data = payload["data"]
        if data.ndim > 2:
            data = data[:, :, 0]
        df = pd.DataFrame(np.asarray(data, dtype=np.float32))
    else:
            raise ValueError(
            f"Unsupported tslib file extension for pretraining: {path}. "
            f"Supported: {', '.join(_TSLIB_PRETRAIN_SUPPORTED_EXTENSIONS)}"
        )

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.ffill().bfill().fillna(0)
    return df


def _maybe_log_text_metadata_preview(
    *,
    channel_names: Iterable[str],
    descriptions: List[str],
    domain: str,
    dataset_name: str,
) -> None:
    flag = os.environ.get("LAYA_TS_LOG_TEXT_METADATA_PREVIEW", "").strip().lower()
    if flag in {"", "0", "false", "no", "off"}:
        return
    channel_name_list = [str(name).strip() for name in channel_names]
    preview_key = (str(domain), str(dataset_name), tuple(channel_name_list))
    if preview_key in _TEXT_METADATA_PREVIEW_KEYS:
        return
    _TEXT_METADATA_PREVIEW_KEYS.add(preview_key)
    preview_count = max(1, min(2, len(descriptions)))
    print(
        f"Text metadata preview | dataset={dataset_name} | domain={domain} | channels={len(channel_name_list)}"
    )
    for idx in range(preview_count):
        channel_label = channel_name_list[idx] if idx < len(channel_name_list) else f"ch_{idx}"
        print(f"   - {channel_label}: {descriptions[idx]}")
    if len(descriptions) > preview_count:
        print(f"   - ... ({len(descriptions) - preview_count} more channel descriptions)")


def _maybe_log_stats_metadata_preview(
    *,
    channel_names: Iterable[str],
    descriptions: List[str],
    domain: str,
    dataset_name: str,
) -> None:
    flag = os.environ.get("LAYA_TS_LOG_TEXT_METADATA_PREVIEW", "").strip().lower()
    if flag in {"", "0", "false", "no", "off"}:
        return
    channel_name_list = [str(name).strip() for name in channel_names]
    preview_key = (str(domain), str(dataset_name), tuple(channel_name_list))
    if preview_key in _STATS_METADATA_PREVIEW_KEYS:
        return
    _STATS_METADATA_PREVIEW_KEYS.add(preview_key)
    if len(channel_name_list) == 0 or not descriptions:
        return
    preview_idx = 0
    channel_label = channel_name_list[preview_idx]
    print(
        f"Stats metadata preview | dataset={dataset_name} | domain={domain} | channels={len(channel_name_list)}"
    )
    print(f"   - {channel_label}: {descriptions[preview_idx]}")


def _maybe_log_joint_metadata_preview(
    *,
    channel_names: Iterable[str],
    descriptions: List[str],
    domain: str,
    dataset_name: str,
) -> None:
    flag = os.environ.get("LAYA_TS_LOG_TEXT_METADATA_PREVIEW", "").strip().lower()
    if flag in {"", "0", "false", "no", "off"}:
        return
    channel_name_list = [str(name).strip() for name in channel_names]
    preview_key = (str(domain), str(dataset_name), tuple(channel_name_list))
    if preview_key in _JOINT_METADATA_PREVIEW_KEYS:
        return
    _JOINT_METADATA_PREVIEW_KEYS.add(preview_key)
    if len(channel_name_list) == 0 or not descriptions:
        return
    preview_idx = 0
    channel_label = channel_name_list[preview_idx]
    print(
        f"Joint metadata preview | dataset={dataset_name} | domain={domain} | channels={len(channel_name_list)}"
    )
    print(f"   - {channel_label}: {descriptions[preview_idx]}")


def build_text_channel_metadata(
    *,
    channel_names: Iterable[str],
    domain: str,
    dataset_name: str,
    text_encoder_name_or_path: str,
    text_metadata_cache_dir: str,
    text_encoder_local_files_only: bool = False,
) -> torch.Tensor:
    channel_name_list = [str(name).strip() for name in channel_names]
    descriptions = build_channel_descriptions(channel_name_list, domain=domain, dataset_name=dataset_name)
    _maybe_log_text_metadata_preview(
        channel_names=channel_name_list,
        descriptions=descriptions,
        domain=domain,
        dataset_name=dataset_name,
    )
    return encode_channel_descriptions(
        descriptions,
        dataset_name=dataset_name,
        encoder_name_or_path=text_encoder_name_or_path,
        cache_dir=text_metadata_cache_dir,
        local_files_only=text_encoder_local_files_only,
    )


def _compute_channel_stats_records(raw: np.ndarray) -> list[dict[str, float]]:
    channel_mean = raw.mean(axis=0)
    channel_std = raw.std(axis=0)
    channel_min = raw.min(axis=0)
    channel_max = raw.max(axis=0)
    channel_median = np.median(raw, axis=0)
    channel_mean_abs = np.mean(np.abs(raw), axis=0)

    stats_records = []
    for idx in range(raw.shape[1]):
        stats_records.append(
            {
                "mean": float(channel_mean[idx]),
                "std": float(channel_std[idx]),
                "min": float(channel_min[idx]),
                "max": float(channel_max[idx]),
                "median": float(channel_median[idx]),
                "mean_abs": float(channel_mean_abs[idx]),
            }
        )
    return stats_records


def build_statistical_channel_metadata(
    values: np.ndarray | torch.Tensor,
    *,
    channel_names: Optional[Iterable[str]] = None,
    domain: Optional[str] = None,
    dataset_name: Optional[str] = None,
    text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    text_metadata_cache_dir: str = "./metadata_cache",
    text_encoder_local_files_only: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2:
        raise ValueError(f"Expected values shaped [T, C], got {tuple(raw.shape)}")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("Statistical metadata requires at least one timestep and one channel")

    if channel_names is None or domain is None or dataset_name is None:
        raise ValueError(
            "Statistical metadata now requires channel_names, domain, and dataset_name "
            "to build textualized statistical descriptions."
        )

    stats_records = _compute_channel_stats_records(raw)
    descriptions = build_statistical_channel_descriptions(
        channel_names=channel_names,
        domain=domain,
        dataset_name=dataset_name,
        stats=stats_records,
    )
    _maybe_log_stats_metadata_preview(
        channel_names=channel_names,
        descriptions=descriptions,
        domain=domain,
        dataset_name=dataset_name,
    )
    return encode_channel_descriptions(
        descriptions,
        dataset_name=f"{dataset_name}__stats",
        encoder_name_or_path=text_encoder_name_or_path,
        cache_dir=text_metadata_cache_dir,
        template_version="laya-ts-stats-v1",
        local_files_only=text_encoder_local_files_only,
    )


def build_joint_text_stats_channel_metadata(
    values: np.ndarray | torch.Tensor,
    *,
    channel_names: Optional[Iterable[str]] = None,
    domain: Optional[str] = None,
    dataset_name: Optional[str] = None,
    text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    text_metadata_cache_dir: str = "./metadata_cache",
    text_encoder_local_files_only: bool = False,
) -> torch.Tensor:
    raw = np.asarray(values, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw[:, None]
    if raw.ndim != 2:
        raise ValueError(f"Expected values shaped [T, C], got {tuple(raw.shape)}")
    if raw.shape[0] == 0 or raw.shape[1] == 0:
        raise ValueError("Joint metadata requires at least one timestep and one channel")
    if channel_names is None or domain is None or dataset_name is None:
        raise ValueError(
            "Joint metadata requires channel_names, domain, and dataset_name."
        )
    stats_records = _compute_channel_stats_records(raw)
    descriptions = build_joint_channel_descriptions(
        channel_names=channel_names,
        domain=domain,
        dataset_name=dataset_name,
        stats=stats_records,
    )
    _maybe_log_joint_metadata_preview(
        channel_names=channel_names,
        descriptions=descriptions,
        domain=domain,
        dataset_name=dataset_name,
    )
    return encode_channel_descriptions(
        descriptions,
        dataset_name=f"{dataset_name}__joint_stats",
        encoder_name_or_path=text_encoder_name_or_path,
        cache_dir=text_metadata_cache_dir,
        template_version="laya-ts-joint-stats-v1",
        local_files_only=text_encoder_local_files_only,
    )


def collate_with_positions(batch):
    if isinstance(batch[0], dict) and "series" in batch[0]:
        series = torch.stack([item["series"] for item in batch], dim=0)
        positions = torch.stack([item["channel_positions"] for item in batch], dim=0)
        mask = torch.stack([item["channel_mask"] for item in batch], dim=0)
        out = {
            "series": series,
            "channel_positions": positions,
            "channel_mask": mask,
            "channel_names": [item.get("channel_names") for item in batch],
        }
        if batch[0].get("channel_text_embeddings") is not None:
            out["channel_text_embeddings"] = torch.stack([item["channel_text_embeddings"] for item in batch], dim=0)
        if batch[0].get("channel_stats_embeddings") is not None:
            out["channel_stats_embeddings"] = torch.stack([item["channel_stats_embeddings"] for item in batch], dim=0)
        if "target" in batch[0]:
            out["target"] = torch.stack([item["target"] for item in batch], dim=0)
        if "label" in batch[0]:
            out["label"] = torch.stack([item["label"] for item in batch], dim=0)
        return out
    raise ValueError("Unsupported batch format for collate_with_positions")


def list_tslib_files(root_path: str, max_files: Optional[int] = None) -> List[str]:
    tslib_files: List[str] = []
    for current_root, _, file_names in os.walk(root_path):
        rel_root = os.path.relpath(current_root, root_path)
        rel_parts = {part for part in rel_root.split(os.sep) if part not in {"", "."}}
        if rel_parts & _TSLIB_PRETRAIN_EXCLUDED_ROOT_DIRS:
            continue
        for file_name in file_names:
            if file_name.startswith(".") or "hhh" in file_name:
                continue
            if not file_name.lower().endswith(_TSLIB_PRETRAIN_SUPPORTED_EXTENSIONS):
                continue
            tslib_files.append(os.path.join(current_root, file_name))
    tslib_files = sorted(tslib_files)
    if max_files:
        tslib_files = tslib_files[:max_files]
    return tslib_files


def list_tsld_csv_files(root_path: str, max_files: Optional[int] = None) -> List[str]:
    tsld_files: List[str] = []
    for current_root, _, file_names in os.walk(root_path):
        for file_name in file_names:
            if file_name.startswith(".") or "hhh" in file_name:
                continue
            if not file_name.lower().endswith(_TSLD_PRETRAIN_SUPPORTED_EXTENSIONS):
                continue
            tsld_files.append(os.path.join(current_root, file_name))
    tsld_files = sorted(tsld_files)
    if max_files:
        tsld_files = tsld_files[:max_files]
    return tsld_files
