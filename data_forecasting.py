from __future__ import annotations

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .utils import (
    build_joint_text_stats_channel_metadata,
    build_statistical_channel_metadata,
    build_text_channel_metadata,
    collate_with_positions,
    infer_csv_context,
    load_csv_frame,
    synthetic_channel_positions,
)


class CSVForecastDataset(Dataset):
    def __init__(self, data_path: str, dataset_type: str = "Electricity", seq_len: int = 512, pred_len: int = 96, mode: str = "train", channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channel_metadata_mode = str(channel_metadata_mode).strip().lower()
        if dataset_type in {"ETTm1", "ETTm2"}:
            train_ratio, val_ratio = 12 / 20, 4 / 20
        else:
            train_ratio, val_ratio = 0.7, 0.1
        df = load_csv_frame(data_path)
        total_len = len(df)
        train_len = int(total_len * train_ratio)
        val_len = int(total_len * val_ratio)
        train_values_raw = df.values[:train_len].astype("float32", copy=False)
        scaler = StandardScaler()
        scaler.fit(df.values[:train_len])
        df_norm = scaler.transform(df.values)
        if mode == "train":
            start_idx, end_idx = 0, train_len
        elif mode == "val":
            start_idx, end_idx = train_len - seq_len, train_len + val_len
        else:
            start_idx, end_idx = train_len + val_len - seq_len, total_len
        self.data = torch.from_numpy(df_norm[start_idx:end_idx]).float()
        self.indices = [i for i in range(0, len(self.data) - seq_len - pred_len + 1)]
        self.feature_names = list(df.columns)
        self.positions = synthetic_channel_positions(len(self.feature_names))
        self.channel_text_embeddings = None
        self.channel_stats_embeddings = None
        domain_name, resolved_dataset_name = infer_csv_context(dataset_type, data_path)
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
                train_values_raw,
                channel_names=self.feature_names,
                domain=domain_name,
                dataset_name=resolved_dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        if self.channel_metadata_mode in {"stats", "text_stats_avg"}:
            self.channel_stats_embeddings = build_statistical_channel_metadata(
                train_values_raw,
                channel_names=self.feature_names,
                domain=domain_name,
                dataset_name=resolved_dataset_name,
                text_encoder_name_or_path=text_encoder_name_or_path,
                text_metadata_cache_dir=text_metadata_cache_dir,
                text_encoder_local_files_only=text_encoder_local_files_only,
            )
        self.scaler = scaler

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        context = self.data[start:start + self.seq_len].T
        target = self.data[start + self.seq_len:start + self.seq_len + self.pred_len].T
        return {
            "series": context,
            "target": target,
            "channel_positions": self.positions,
            "channel_mask": torch.ones(context.shape[0], dtype=torch.bool),
            "channel_names": self.feature_names,
            "channel_text_embeddings": self.channel_text_embeddings,
            "channel_stats_embeddings": self.channel_stats_embeddings,
        }


def get_forecasting_loaders(data_path: str, dataset_type: str, batch_size: int = 32, seq_len: int = 512, pred_len: int = 96, num_workers: int = 4, channel_metadata_mode: str = "onehot", text_encoder_name_or_path: str = "sentence-transformers/all-MiniLM-L6-v2", text_metadata_cache_dir: str = "./metadata_cache", text_encoder_local_files_only: bool = False):
    train_ds = CSVForecastDataset(data_path, dataset_type, seq_len, pred_len, "train", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
    val_ds = CSVForecastDataset(data_path, dataset_type, seq_len, pred_len, "val", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
    test_ds = CSVForecastDataset(data_path, dataset_type, seq_len, pred_len, "test", channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=text_encoder_name_or_path, text_metadata_cache_dir=text_metadata_cache_dir, text_encoder_local_files_only=text_encoder_local_files_only)
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, drop_last=len(train_ds) > batch_size, num_workers=num_workers, collate_fn=collate_with_positions)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_with_positions)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, drop_last=False, num_workers=num_workers, collate_fn=collate_with_positions)
    return train_loader, val_loader, test_loader, test_ds.scaler
