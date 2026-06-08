from __future__ import annotations

import os
import re
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset

from .utils import (
    build_joint_text_stats_channel_metadata,
    build_statistical_channel_metadata,
    build_text_channel_metadata,
    collate_with_positions,
    synthetic_channel_positions,
)


def _parse_ts_file(path: str):
    data_started = False
    series_list, labels = [], []
    n_dims = 1
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.lower().startswith("@data"):
                data_started = True
                continue
            if line.lower().startswith("@"):
                if "numdimensions" in line.lower():
                    n_dims = int(re.search(r"\d+", line).group())
                continue
            if data_started:
                if ":" in line:
                    *dims_raw, label = line.rsplit(":", 1)
                    parts = (":".join(dims_raw)).split(":")
                    if len(parts) == n_dims + 1:
                        label = parts[-1].strip(); parts = parts[:-1]
                    dim_arrays = []
                    for p in parts:
                        p_clean = p.replace('?', 'NaN')
                        arr = np.fromstring(p_clean.strip(), sep=",")
                        if np.isnan(arr).any():
                            arr = pd.Series(arr).interpolate(limit_direction='both').fillna(0).values
                        dim_arrays.append(arr)
                    series_list.append(np.stack(dim_arrays, axis=0))
                else:
                    vals = line.replace('?', 'NaN').split(",")
                    label = vals[-1].strip()
                    arr = np.array([float(v) for v in vals[:-1]])
                    if np.isnan(arr).any():
                        arr = pd.Series(arr).interpolate(limit_direction='both').fillna(0).values
                    series_list.append(arr[np.newaxis, :])
                labels.append(label)
    max_len = max(s.shape[1] for s in series_list)
    padded = []
    for s in series_list:
        if s.shape[1] < max_len:
            s = np.pad(s, ((0, 0), (0, max_len - s.shape[1])))
        padded.append(s)
    return np.stack(padded, axis=0).astype(np.float32), np.array(labels)


def _parse_ucr_txt_file(path: str):
    series_list, labels = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            labels.append(tokens[0])
            series_list.append(np.array([float(v) for v in tokens[1:]], dtype=np.float32))
    max_len = max(len(s) for s in series_list)
    padded = np.zeros((len(series_list), max_len), dtype=np.float32)
    for i, s in enumerate(series_list):
        padded[i, :len(s)] = s
    return padded[:, np.newaxis, :], np.array(labels)


def _load_csv_classification(path: str):
    df = pd.read_csv(path, low_memory=False)
    y = df[df.columns[-1]].values.astype(str)
    X = df[df.columns[:-1]].values.astype(np.float32)
    return X[:, np.newaxis, :], y, [str(c) for c in df.columns[:-1]]


class TSClassificationDataset(Dataset):
    def __init__(self, data_root: str, seq_len: int = 512, mode: str = "train", scaler: StandardScaler | None = None, le: LabelEncoder | None = None, val_ratio: float = 0.1, channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False, channel_text_embeddings: torch.Tensor | None = None, channel_stats_embeddings: torch.Tensor | None = None):
        X_raw, y_raw, feature_names = self._load(data_root, mode, val_ratio)
        self.channel_metadata_mode = str(channel_metadata_mode).strip().lower()
        N, C, T = X_raw.shape
        train_split_flat = X_raw.transpose(0, 2, 1).reshape(-1, C)
        X_flat = X_raw.transpose(0, 2, 1).reshape(-1, C)
        if scaler is None:
            scaler = StandardScaler(); scaler.fit(X_flat)
        X_flat_norm = scaler.transform(X_flat)
        X_norm = X_flat_norm.reshape(N, T, C).transpose(0, 2, 1)
        if le is None:
            le = LabelEncoder(); le.fit(y_raw)
        if seq_len <= 0:
            seq_len = X_norm.shape[-1]
        if X_norm.shape[-1] >= seq_len:
            X_norm = X_norm[:, :, -seq_len:]
        else:
            X_norm = np.pad(X_norm, ((0, 0), (0, 0), (seq_len - X_norm.shape[-1], 0)))
        self.X = torch.from_numpy(X_norm).float()
        self.y = torch.from_numpy(le.transform(y_raw)).long()
        self.scaler = scaler
        self.le = le
        self.num_classes = len(le.classes_)
        self.feature_names = feature_names
        self.positions = synthetic_channel_positions(len(feature_names))
        self.channel_text_embeddings = None
        self.channel_stats_embeddings = None
        dataset_name = os.path.basename(os.path.normpath(data_root)) if os.path.isdir(data_root) else os.path.splitext(os.path.basename(data_root))[0]
        domain_name = os.path.basename(os.path.dirname(os.path.normpath(data_root))) if os.path.isdir(data_root) else dataset_name
        if self.channel_metadata_mode in {"text", "text_stats_avg"}:
            if channel_text_embeddings is not None:
                self.channel_text_embeddings = channel_text_embeddings
            else:
                self.channel_text_embeddings = build_text_channel_metadata(
                    channel_names=self.feature_names,
                    domain=domain_name,
                    dataset_name=dataset_name,
                    text_encoder_name_or_path=text_encoder_name_or_path,
                    text_metadata_cache_dir=text_metadata_cache_dir,
                    text_encoder_local_files_only=text_encoder_local_files_only,
                )
        elif self.channel_metadata_mode == "text_stats_joint":
            if channel_text_embeddings is not None:
                self.channel_text_embeddings = channel_text_embeddings
            else:
                self.channel_text_embeddings = build_joint_text_stats_channel_metadata(
                    train_split_flat,
                    channel_names=self.feature_names,
                    domain=domain_name,
                    dataset_name=dataset_name,
                    text_encoder_name_or_path=text_encoder_name_or_path,
                    text_metadata_cache_dir=text_metadata_cache_dir,
                    text_encoder_local_files_only=text_encoder_local_files_only,
                )
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            if channel_stats_embeddings is not None:
                self.channel_stats_embeddings = channel_stats_embeddings
            else:
                self.channel_stats_embeddings = build_statistical_channel_metadata(
                    train_split_flat,
                    channel_names=self.feature_names,
                    domain=domain_name,
                    dataset_name=dataset_name,
                    text_encoder_name_or_path=text_encoder_name_or_path,
                    text_metadata_cache_dir=text_metadata_cache_dir,
                    text_encoder_local_files_only=text_encoder_local_files_only,
                )

    def _load(self, data_root: str, mode: str, val_ratio: float):
        if os.path.isdir(data_root):
            dname = os.path.basename(data_root)
            train_path = _find_file(data_root, [f"{dname}_TRAIN.ts", "TRAIN.ts", f"{dname}_TRAIN.txt", "TRAIN.txt", f"{dname}_TRAIN.tsv", "TRAIN.tsv"])
            test_path = _find_file(data_root, [f"{dname}_TEST.ts", "TEST.ts", f"{dname}_TEST.txt", "TEST.txt", f"{dname}_TEST.tsv", "TEST.tsv"])
            def _parse(p):
                return _parse_ucr_txt_file(p) if os.path.splitext(p)[1].lower() in (".txt", ".tsv") else _parse_ts_file(p)
            X_train, y_train = _parse(train_path)
            X_test, y_test = _parse(test_path)
            feature_names = [f"channel_{idx+1}" for idx in range(X_train.shape[1])]
            if mode == "test":
                return X_test, y_test, feature_names
            N = len(X_train); n_val = max(1, int(N * val_ratio)); n_train = N - n_val
            rng = np.random.default_rng(42); indices = rng.permutation(N)
            X_train = X_train[indices]; y_train = y_train[indices]
            return (X_train[:n_train], y_train[:n_train], feature_names) if mode == "train" else (X_train[n_train:], y_train[n_train:], feature_names)
        if os.path.isfile(data_root) and data_root.endswith(".csv"):
            X_all, y_all, feature_names = _load_csv_classification(data_root)
            N = len(X_all); n_test = max(1, int(N * 0.2)); n_val = max(1, int(N * val_ratio)); n_train = N - n_val - n_test
            rng = np.random.default_rng(42); indices = rng.permutation(N)
            X_all = X_all[indices]; y_all = y_all[indices]
            if mode == "train":
                return X_all[:n_train], y_all[:n_train], feature_names
            if mode == "val":
                return X_all[n_train:n_train+n_val], y_all[n_train:n_train+n_val], feature_names
            return X_all[n_train+n_val:], y_all[n_train+n_val:], feature_names
        raise ValueError(f"Unsupported classification data_root: {data_root}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return {
            "series": self.X[idx],
            "label": self.y[idx],
            "channel_positions": self.positions,
            "channel_mask": torch.ones(self.X[idx].shape[0], dtype=torch.bool),
            "channel_names": self.feature_names,
            "channel_text_embeddings": self.channel_text_embeddings,
            "channel_stats_embeddings": self.channel_stats_embeddings,
        }


def _find_file(directory: str, candidates: list[str]) -> str:
    for name in candidates:
        p = os.path.join(directory, name)
        if os.path.exists(p):
            return p
    for ext in (".ts", ".txt", ".tsv"):
        for fname in os.listdir(directory):
            if fname.endswith(ext):
                return os.path.join(directory, fname)
    raise FileNotFoundError(f"No .ts/.txt/.tsv file found in {directory}")


def get_classification_loaders(data_root: str, seq_len: int = 512, batch_size: int = 32, num_workers: int = 0, val_ratio: float = 0.1, channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False):
    train_ds = TSClassificationDataset(data_root, seq_len, mode="train", val_ratio=val_ratio, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
    actual_seq_len = train_ds.X.shape[-1]
    shared_text_embeddings = train_ds.channel_text_embeddings if channel_metadata_mode in {"text", "text_stats_avg", "text_stats_joint"} else None
    shared_stats_embeddings = train_ds.channel_stats_embeddings if channel_metadata_mode in {"stats", "text_stats_avg"} else None
    val_ds = TSClassificationDataset(data_root, actual_seq_len, mode="val", scaler=train_ds.scaler, le=train_ds.le, val_ratio=val_ratio, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, channel_text_embeddings=shared_text_embeddings, channel_stats_embeddings=shared_stats_embeddings)
    test_ds = TSClassificationDataset(data_root, actual_seq_len, mode="test", scaler=train_ds.scaler, le=train_ds.le, val_ratio=val_ratio, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only, channel_text_embeddings=shared_text_embeddings, channel_stats_embeddings=shared_stats_embeddings)
    do_drop_last = len(train_ds) > batch_size
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=do_drop_last, num_workers=num_workers, collate_fn=collate_with_positions)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_with_positions)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_with_positions)
    return train_loader, val_loader, test_loader, train_ds.num_classes, train_ds.X.shape[1], actual_seq_len
