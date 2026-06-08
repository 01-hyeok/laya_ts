from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score

from laya_ts.config import LayaModelConfig

if __package__ in {None, ""}:
    from laya_ts.data_classification import get_classification_loaders
    from laya_ts.data_forecasting import get_forecasting_loaders
    from laya_ts.model import LayaTSEncoder, load_encoder_from_checkpoint_report, load_model_config_from_checkpoint
else:
    from .data_classification import get_classification_loaders
    from .data_forecasting import get_forecasting_loaders
    from .model import LayaTSEncoder, load_encoder_from_checkpoint_report, load_model_config_from_checkpoint


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EncoderProbe(nn.Module):
    def __init__(self, config: LayaModelConfig) -> None:
        super().__init__()
        self.encoder = LayaTSEncoder(config)


def _inverse_scale_batch(values: np.ndarray, scaler) -> np.ndarray:
    batch, channels, steps = values.shape
    flat = values.transpose(0, 2, 1).reshape(-1, channels)
    restored = scaler.inverse_transform(flat)
    return restored.reshape(batch, steps, channels).transpose(0, 2, 1)


def _pick_loader(task: str, split: str, loader_bundle):
    if task == "classification":
        train_loader, val_loader, test_loader, _, _, _ = loader_bundle
    else:
        train_loader, val_loader, test_loader, _ = loader_bundle
    if split == "train":
        return train_loader
    if split == "val":
        return val_loader
    return test_loader


def _classification_label_names(loader) -> list[str]:
    dataset = loader.dataset
    classes = getattr(getattr(dataset, "le", None), "classes_", None)
    if classes is None:
        return []
    return [str(value) for value in classes]


def _forecast_bucket_labels(
    *,
    targets: np.ndarray,
    scaler,
    mode: str,
    num_bins: int,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    restored = _inverse_scale_batch(targets, scaler)
    if mode == "future_mean":
        score = restored.mean(axis=(1, 2))
        prefix = "mean"
    elif mode == "future_volatility":
        score = restored.std(axis=(1, 2))
        prefix = "vol"
    else:
        score = restored[:, :, -1].mean(axis=1) - restored[:, :, 0].mean(axis=1)
        prefix = "slope"

    if score.shape[0] <= 1:
        return np.zeros(score.shape[0], dtype=np.int64), [f"{prefix}_q0"], score

    quantiles = np.linspace(0.0, 1.0, num_bins + 1)
    edges = np.quantile(score, quantiles)
    if np.unique(edges).shape[0] <= 1:
        return np.zeros(score.shape[0], dtype=np.int64), [f"{prefix}_q0"], score

    edges[0] = -np.inf
    edges[-1] = np.inf
    bucket_ids = np.digitize(score, edges[1:-1], right=False).astype(np.int64)
    label_names = [f"{prefix}_q{idx}" for idx in range(int(bucket_ids.max()) + 1)]
    return bucket_ids, label_names, score


def _scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    label_names: list[str],
    *,
    title: str,
    path: str,
) -> None:
    plt.figure(figsize=(10, 8))
    unique_labels = sorted(np.unique(labels).tolist())
    cmap = plt.get_cmap("tab20", max(1, len(unique_labels)))
    for color_idx, label_value in enumerate(unique_labels):
        mask = labels == label_value
        label_name = label_names[label_value] if 0 <= label_value < len(label_names) else str(label_value)
        plt.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.75,
            color=cmap(color_idx),
            label=f"{label_name} (n={int(mask.sum())})",
        )
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
    if len(unique_labels) <= 20:
        plt.legend(loc="best", fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _collect_classification_representations(
    *,
    encoder: LayaTSEncoder,
    loader,
    device: str,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    reps: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    collected = 0
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["series"].to(device)
            pos = batch["channel_positions"].to(device)
            mask = batch["channel_mask"].to(device)
            channel_text_embeddings = batch.get("channel_text_embeddings")
            if channel_text_embeddings is not None:
                channel_text_embeddings = channel_text_embeddings.to(device)
            channel_stats_embeddings = batch.get("channel_stats_embeddings")
            if channel_stats_embeddings is not None:
                channel_stats_embeddings = channel_stats_embeddings.to(device)
            features = encoder.forward_features(
                x,
                channel_positions=pos,
                channel_mask=mask,
                channel_text_embeddings=channel_text_embeddings,
                channel_stats_embeddings=channel_stats_embeddings,
            )
            batch_repr = features["mixed_repr"].detach().cpu().numpy()
            batch_labels = batch["label"].detach().cpu().numpy()
            reps.append(batch_repr)
            labels.append(batch_labels)
            collected += batch_repr.shape[0]
            if collected >= max_samples:
                break
    rep_array = np.concatenate(reps, axis=0)[:max_samples]
    label_array = np.concatenate(labels, axis=0)[:max_samples]
    return rep_array, label_array


def _collect_forecasting_representations(
    *,
    encoder: LayaTSEncoder,
    loader,
    scaler,
    device: str,
    max_samples: int,
    bucket_mode: str,
    num_bins: int,
) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    reps: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    collected = 0
    encoder.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["series"].to(device)
            pos = batch["channel_positions"].to(device)
            mask = batch["channel_mask"].to(device)
            channel_text_embeddings = batch.get("channel_text_embeddings")
            if channel_text_embeddings is not None:
                channel_text_embeddings = channel_text_embeddings.to(device)
            channel_stats_embeddings = batch.get("channel_stats_embeddings")
            if channel_stats_embeddings is not None:
                channel_stats_embeddings = channel_stats_embeddings.to(device)
            features = encoder.forward_features(
                x,
                channel_positions=pos,
                channel_mask=mask,
                channel_text_embeddings=channel_text_embeddings,
                channel_stats_embeddings=channel_stats_embeddings,
            )
            batch_repr = features["mixed_repr"].detach().cpu().numpy()
            reps.append(batch_repr)
            targets.append(batch["target"].detach().cpu().numpy())
            collected += batch_repr.shape[0]
            if collected >= max_samples:
                break
    rep_array = np.concatenate(reps, axis=0)[:max_samples]
    target_array = np.concatenate(targets, axis=0)[:max_samples]
    bucket_ids, bucket_names, bucket_scores = _forecast_bucket_labels(
        targets=target_array,
        scaler=scaler,
        mode=bucket_mode,
        num_bins=num_bins,
    )
    return rep_array, bucket_ids, bucket_names, bucket_scores


def _evaluate_cluster_metrics(representations: np.ndarray, labels: np.ndarray) -> dict[str, float | None]:
    unique = np.unique(labels)
    if representations.shape[0] < 3 or unique.shape[0] < 2:
        return {"silhouette": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(representations, labels)),
        "davies_bouldin": float(davies_bouldin_score(representations, labels)),
    }


def _build_model_config(
    *,
    checkpoint_cfg: LayaModelConfig,
    channel_metadata_mode: str,
    metadata_fusion_mode: str,
    channel_mixer_type: str,
    channel_mixer_relation_mode: str,
    onehot_channel_vocab_size: int,
    stats_metadata_dim: int,
) -> LayaModelConfig:
    config_dict = checkpoint_cfg.__dict__.copy()
    config_dict.update(
        {
            "channel_metadata_mode": channel_metadata_mode,
            "metadata_fusion_mode": metadata_fusion_mode,
            "channel_mixer_type": channel_mixer_type,
            "channel_mixer_relation_mode": channel_mixer_relation_mode,
            "onehot_channel_vocab_size": onehot_channel_vocab_size,
            "stats_metadata_dim": stats_metadata_dim,
        }
    )
    return LayaModelConfig(**config_dict)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Visualize frozen LayaTS encoder representations with PCA and t-SNE.")
    parser.add_argument("--task", type=str, required=True, choices=["classification", "forecasting"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channel_metadata_mode", type=str, default=None, choices=["onehot", "text", "stats", "text_stats_joint", "text_stats_avg", "none"])
    parser.add_argument("--metadata_fusion_mode", type=str, default=None, choices=["none", "add", "concat_kv", "attention_gate", "attention_suppress_gate"])
    parser.add_argument("--channel_mixer_type", type=str, default=None, choices=["mixer", "independent"])
    parser.add_argument("--channel_mixer_relation_mode", type=str, default=None, choices=["none", "laya_relation", "metadata_query_gate", "description_relation"])
    parser.add_argument("--stats_metadata_dim", type=int, default=None)
    parser.add_argument("--onehot_channel_vocab_size", type=int, default=0)
    parser.add_argument("--text_encoder_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text_metadata_cache_dir", type=str, default="./metadata_cache")
    parser.add_argument("--text_encoder_local_files_only", action="store_true")
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--tsne_perplexity", type=float, default=30.0)
    parser.add_argument("--tsne_learning_rate", type=float, default=200.0)
    parser.add_argument("--tsne_iterations", type=int, default=1500)
    parser.add_argument("--l2_normalize", action="store_true")
    parser.add_argument("--forecast_label_mode", type=str, default="future_slope", choices=["future_slope", "future_mean", "future_volatility"])
    parser.add_argument("--forecast_num_bins", type=int, default=4)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--dataset_type", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    args = parser.parse_args(argv)

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_cfg = load_model_config_from_checkpoint(args.checkpoint)
    channel_metadata_mode = args.channel_metadata_mode or checkpoint_cfg.channel_metadata_mode
    metadata_fusion_mode = args.metadata_fusion_mode or checkpoint_cfg.metadata_fusion_mode
    channel_mixer_type = args.channel_mixer_type or checkpoint_cfg.channel_mixer_type
    channel_mixer_relation_mode = args.channel_mixer_relation_mode or checkpoint_cfg.channel_mixer_relation_mode
    stats_metadata_dim = checkpoint_cfg.stats_metadata_dim if args.stats_metadata_dim is None else args.stats_metadata_dim

    if args.task == "classification":
        if not args.data_root:
            raise ValueError("--data_root is required for classification visualization")
        loader_bundle = get_classification_loaders(
            args.data_root,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            val_ratio=args.val_ratio,
            channel_metadata_mode=channel_metadata_mode,
            text_encoder_name_or_path=args.text_encoder_name_or_path,
            text_metadata_cache_dir=args.text_metadata_cache_dir,
            text_encoder_local_files_only=args.text_encoder_local_files_only,
        )
        loader = _pick_loader(args.task, args.split, loader_bundle)
        _, _, _, _, in_vars, _ = loader_bundle
        label_names = _classification_label_names(loader)
        onehot_vocab_size = max(checkpoint_cfg.onehot_channel_vocab_size, args.onehot_channel_vocab_size, in_vars)
        model_cfg = _build_model_config(
            checkpoint_cfg=checkpoint_cfg,
            channel_metadata_mode=channel_metadata_mode,
            metadata_fusion_mode=metadata_fusion_mode,
            channel_mixer_type=channel_mixer_type,
            channel_mixer_relation_mode=channel_mixer_relation_mode,
            onehot_channel_vocab_size=onehot_vocab_size,
            stats_metadata_dim=stats_metadata_dim,
        )
        probe = EncoderProbe(model_cfg).to(args.device)
        load_report = load_encoder_from_checkpoint_report(probe, args.checkpoint)
        representations, label_ids = _collect_classification_representations(
            encoder=probe.encoder,
            loader=loader,
            device=args.device,
            max_samples=args.max_samples,
        )
        score_values = None
        title_suffix = Path(args.data_root).name
    else:
        if not args.data_path or not args.dataset_type:
            raise ValueError("--data_path and --dataset_type are required for forecasting visualization")
        loader_bundle = get_forecasting_loaders(
            args.data_path,
            args.dataset_type,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            pred_len=args.pred_len,
            num_workers=args.num_workers,
            channel_metadata_mode=channel_metadata_mode,
            text_encoder_name_or_path=args.text_encoder_name_or_path,
            text_metadata_cache_dir=args.text_metadata_cache_dir,
            text_encoder_local_files_only=args.text_encoder_local_files_only,
        )
        loader = _pick_loader(args.task, args.split, loader_bundle)
        train_loader, _, _, scaler = loader_bundle
        first_batch = next(iter(train_loader))
        input_channels = int(first_batch["series"].shape[1])
        onehot_vocab_size = max(checkpoint_cfg.onehot_channel_vocab_size, args.onehot_channel_vocab_size, input_channels)
        model_cfg = _build_model_config(
            checkpoint_cfg=checkpoint_cfg,
            channel_metadata_mode=channel_metadata_mode,
            metadata_fusion_mode=metadata_fusion_mode,
            channel_mixer_type=channel_mixer_type,
            channel_mixer_relation_mode=channel_mixer_relation_mode,
            onehot_channel_vocab_size=onehot_vocab_size,
            stats_metadata_dim=stats_metadata_dim,
        )
        probe = EncoderProbe(model_cfg).to(args.device)
        load_report = load_encoder_from_checkpoint_report(probe, args.checkpoint)
        representations, label_ids, label_names, score_values = _collect_forecasting_representations(
            encoder=probe.encoder,
            loader=loader,
            scaler=scaler,
            device=args.device,
            max_samples=args.max_samples,
            bucket_mode=args.forecast_label_mode,
            num_bins=args.forecast_num_bins,
        )
        title_suffix = f"{args.dataset_type}-{args.forecast_label_mode}"

    if args.l2_normalize:
        norms = np.linalg.norm(representations, axis=1, keepdims=True)
        representations = representations / np.clip(norms, a_min=1e-12, a_max=None)

    pca_dim = min(args.pca_dim, representations.shape[0] - 1, representations.shape[1])
    if pca_dim < 2:
        raise ValueError("Need at least 2 effective PCA dimensions for visualization")
    pca = PCA(n_components=pca_dim, random_state=args.seed)
    pca_features = pca.fit_transform(representations)
    pca_coords = pca_features[:, :2]

    perplexity = min(args.tsne_perplexity, max(5.0, float((representations.shape[0] - 1) // 3)))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=args.tsne_learning_rate,
        n_iter=args.tsne_iterations,
        init="pca",
        random_state=args.seed,
    )
    tsne_coords = tsne.fit_transform(pca_features)

    metrics = _evaluate_cluster_metrics(representations, label_ids)
    label_hist = {label_names[idx] if idx < len(label_names) else str(idx): int(count) for idx, count in Counter(label_ids.tolist()).items()}

    torch.save(
        {
            "representations": torch.from_numpy(representations),
            "labels": torch.from_numpy(label_ids),
            "pca_features": torch.from_numpy(pca_features),
            "pca_coords": torch.from_numpy(pca_coords),
            "tsne_coords": torch.from_numpy(tsne_coords),
            "label_names": label_names,
            "forecast_scores": None if score_values is None else torch.from_numpy(score_values),
        },
        os.path.join(args.output_dir, "representation_payload.pt"),
    )

    _scatter_plot(
        pca_coords,
        label_ids,
        label_names,
        title=f"PCA | {title_suffix} | {args.split}",
        path=os.path.join(args.output_dir, "pca_scatter.png"),
    )
    _scatter_plot(
        tsne_coords,
        label_ids,
        label_names,
        title=f"t-SNE | {title_suffix} | {args.split}",
        path=os.path.join(args.output_dir, "tsne_scatter.png"),
    )

    summary = {
        "task": args.task,
        "split": args.split,
        "checkpoint": args.checkpoint,
        "num_samples": int(representations.shape[0]),
        "representation_dim": int(representations.shape[1]),
        "channel_metadata_mode": channel_metadata_mode,
        "metadata_fusion_mode": metadata_fusion_mode,
        "channel_mixer_type": channel_mixer_type,
        "channel_mixer_relation_mode": channel_mixer_relation_mode,
        "load_report": load_report,
        "pca_dim": int(pca_dim),
        "pca_explained_variance_ratio_2d": [float(value) for value in pca.explained_variance_ratio_[:2]],
        "tsne_perplexity": float(perplexity),
        "tsne_learning_rate": float(args.tsne_learning_rate),
        "tsne_iterations": int(args.tsne_iterations),
        "cluster_metrics": metrics,
        "label_histogram": label_hist,
    }
    if args.task == "forecasting":
        summary["forecast_label_mode"] = args.forecast_label_mode
        summary["forecast_num_bins"] = int(args.forecast_num_bins)
    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=True)

    print("Saved representation visualizations:")
    print(f" - {os.path.join(args.output_dir, 'pca_scatter.png')}")
    print(f" - {os.path.join(args.output_dir, 'tsne_scatter.png')}")
    print(f" - {os.path.join(args.output_dir, 'summary.json')}")
    print(f" - {os.path.join(args.output_dir, 'representation_payload.pt')}")
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"Matched encoder keys: {load_report['matched_keys']}/{load_report['total_encoder_keys']}")
    if metrics["silhouette"] is not None:
        print(f"Silhouette: {metrics['silhouette']:.6f}")
        print(f"Davies-Bouldin: {metrics['davies_bouldin']:.6f}")


if __name__ == "__main__":
    main()
