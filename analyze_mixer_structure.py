from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path
from typing import Optional

if __package__ in {None, ""}:
    WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch

from laya_ts.config import LayaModelConfig, TrainingConfig

if __package__ in {None, ""}:
    from laya_ts.data_pretrain import get_lotsa_pretrain_loader_groups, get_pretrain_loaders
    from laya_ts.model import LayaTSPretrainer, load_model_config_from_checkpoint
    from laya_ts.train_pretrain import move_batch_to_device
else:
    from .data_pretrain import get_lotsa_pretrain_loader_groups, get_pretrain_loaders
    from .model import LayaTSPretrainer, load_model_config_from_checkpoint
    from .train_pretrain import move_batch_to_device


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _normalize_dataset_key(name: str) -> str:
    key = str(name).strip().lower()
    mapping = {
        "electricity": "electricity",
        "tslib": "tslib",
        "tsld": "tsld",
        "lotsa": "lotsa",
    }
    if key not in mapping:
        raise ValueError(f"Unsupported --data value: {name}")
    return mapping[key]


def _resolve_existing_path(raw_path: str) -> Path:
    path = Path(raw_path)
    candidates = [path]
    if not path.is_absolute():
        candidates.append(WORKSPACE_ROOT / path)
        candidates.append(WORKSPACE_ROOT / "laya_ts" / path)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _resolve_data_path(args) -> str:
    data_path = Path(args.data_path)
    if data_path.is_absolute():
        return str(data_path)
    if args.root_path:
        return str((Path(args.root_path) / data_path).resolve())
    return str(data_path.resolve())


def _resolve_checkpoint_path(checkpoint_value: str) -> Path:
    checkpoint_path = _resolve_existing_path(checkpoint_value)
    if checkpoint_path.suffix not in {".pt", ".pth"}:
        raise ValueError(
            f"--checkpoint must point to a checkpoint file (.pt/.pth), got: {checkpoint_value}"
        )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return checkpoint_path


def _resolve_output_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return path.resolve()


def _extract_channel_names(batch) -> list[str]:
    raw_names = batch.get("channel_names")
    if raw_names is None:
        return []
    if isinstance(raw_names, (list, tuple)):
        if raw_names and isinstance(raw_names[0], (list, tuple)):
            return [str(name) for name in raw_names[0]]
        return [str(name) for name in raw_names]
    return []


def _ensure_labels(labels: list[str], count: int, prefix: str) -> list[str]:
    if len(labels) >= count:
        return labels[:count]
    return labels + [f"{prefix}{idx}" for idx in range(len(labels), count)]


def _load_pretrainer_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, object]:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "num_state_keys": len(state_dict),
    }


def _build_pretrain_loader(
    args,
    dataset_key: str,
    channel_metadata_mode: str,
    *,
    patch_size: int,
):
    batch_size = int(args.batch_size or TrainingConfig().batch_size)
    data_path = _resolve_data_path(args)
    if dataset_key == "lotsa":
        loader_groups = get_lotsa_pretrain_loader_groups(
            data_path,
            batch_size=batch_size,
            seq_len=args.seq_len,
            stride=args.seq_len,
            patch_size=patch_size,
            num_workers=0,
            max_files=None,
            channel_metadata_mode=channel_metadata_mode,
            text_encoder_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
            text_metadata_cache_dir="./metadata_cache",
            text_encoder_local_files_only=False,
        )
        if not loader_groups:
            raise ValueError("No LOTSA loader groups were created for analysis.")
        if args.split == "train":
            return loader_groups[0]["train_loader"]
        if args.split == "val":
            loader = loader_groups[0]["val_loader"]
            if loader is None:
                raise ValueError("Validation loader is not available for the selected LOTSA configuration.")
            return loader
        raise ValueError(f"Unsupported split for LOTSA: {args.split}")

    train_loader, val_loader = get_pretrain_loaders(
        dataset_key,
        data_path,
        batch_size=batch_size,
        seq_len=args.seq_len,
        stride=args.seq_len,
        patch_size=patch_size,
        num_workers=0,
        max_files=None,
        channel_metadata_mode=channel_metadata_mode,
        text_encoder_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        text_metadata_cache_dir="./metadata_cache",
        text_encoder_local_files_only=False,
    )
    if args.split == "train":
        return train_loader
    if args.split == "val":
        if val_loader is None:
            raise ValueError("Validation loader is not available for this run.")
        return val_loader
    raise ValueError(f"Unsupported split: {args.split}")


def _reduce_attention_to_qc(attention: torch.Tensor) -> tuple[torch.Tensor, int]:
    if attention.dim() < 2:
        raise ValueError(f"Expected attention with at least 2 dims, got {tuple(attention.shape)}")
    leading = attention.shape[:-2]
    reduced = attention.reshape(-1, attention.shape[-2], attention.shape[-1]).sum(dim=0)
    weight = int(np.prod(leading)) if leading else 1
    return reduced, weight


def _cosine_similarity_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.numel() == 0:
        return matrix.new_zeros((0, 0))
    normalized = torch.nn.functional.normalize(matrix, dim=-1, eps=1e-8)
    return normalized @ normalized.transpose(0, 1)


def _offdiag_mean(matrix: torch.Tensor) -> Optional[float]:
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] <= 1:
        return None
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    values = matrix[mask]
    if values.numel() == 0:
        return None
    return float(values.mean().cpu())


def _row_normalize(matrix: torch.Tensor) -> torch.Tensor:
    denom = matrix.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return matrix / denom


def _attention_entropy(attention_qc: torch.Tensor) -> torch.Tensor:
    probs = _row_normalize(attention_qc).clamp_min(1e-12)
    return -(probs * probs.log()).sum(dim=-1)


def _mean_topk_pre_similarity(
    pre_similarity: torch.Tensor,
    attention_qc: torch.Tensor,
    top_k: int,
) -> Optional[float]:
    if pre_similarity.numel() == 0 or attention_qc.numel() == 0:
        return None
    values: list[float] = []
    k = min(int(top_k), attention_qc.shape[-1])
    for query_idx in range(attention_qc.shape[0]):
        _, top_idx = torch.topk(attention_qc[query_idx], k=k)
        selected = pre_similarity.index_select(0, top_idx).index_select(1, top_idx)
        if selected.shape[0] <= 1:
            values.append(float(selected.mean().cpu()))
            continue
        mask = ~torch.eye(selected.shape[0], dtype=torch.bool, device=selected.device)
        values.append(float(selected[mask].mean().cpu()))
    if not values:
        return None
    return float(sum(values) / len(values))


def _write_matrix_csv(path: Path, matrix: torch.Tensor, row_labels: list[str], col_labels: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", *col_labels])
        data = matrix.detach().cpu().numpy()
        for idx, row in enumerate(data):
            writer.writerow([row_labels[idx], *[f"{float(value):.8f}" for value in row]])


def _save_heatmap(
    path: Path,
    matrix: torch.Tensor,
    *,
    title: str,
    xlabel: str,
    ylabel: str,
    x_labels: list[str],
    y_labels: list[str],
    cmap: str,
    figsize: tuple[float, float],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=figsize)
    plt.imshow(matrix.detach().cpu().numpy(), aspect="auto", cmap=cmap)
    plt.colorbar()
    if len(x_labels) <= 24:
        plt.xticks(np.arange(len(x_labels)), x_labels, rotation=90)
    if len(y_labels) <= 24:
        plt.yticks(np.arange(len(y_labels)), y_labels)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _pairwise_offdiag_values(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.dim() != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square similarity matrix, got {tuple(matrix.shape)}")
    if matrix.shape[0] <= 1:
        return matrix.new_empty((0,))
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def _save_histogram(
    path: Path,
    values: torch.Tensor,
    *,
    title: str,
    xlabel: str,
    bins: int = 40,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    array = values.detach().cpu().numpy() if values.numel() > 0 else np.array([], dtype=np.float32)
    plt.figure(figsize=(7, 5))
    plt.hist(array, bins=bins, range=(-1.0, 1.0), color="#2f5aa8", edgecolor="white")
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _save_overview_figure(
    path: Path,
    *,
    pre_similarity: torch.Tensor,
    pre_labels: list[str],
    attention_qc: torch.Tensor,
    query_labels: list[str],
    channel_labels: list[str],
    post_similarity: torch.Tensor,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(pre_similarity.detach().cpu().numpy(), aspect="auto", cmap="viridis")
    axes[0].set_title("Panel A. Pre-mixer channel similarity")
    axes[0].set_xlabel("channel id")
    axes[0].set_ylabel("channel id")
    if len(pre_labels) <= 16:
        axes[0].set_xticks(np.arange(len(pre_labels)), pre_labels, rotation=90)
        axes[0].set_yticks(np.arange(len(pre_labels)), pre_labels)
    fig.colorbar(axes[0].images[0], ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(attention_qc.detach().cpu().numpy(), aspect="auto", cmap="magma")
    axes[1].set_title("Panel B. Query-to-channel attention")
    axes[1].set_xlabel("channel id")
    axes[1].set_ylabel("query id")
    if len(channel_labels) <= 16:
        axes[1].set_xticks(np.arange(len(channel_labels)), channel_labels, rotation=90)
    if len(query_labels) <= 16:
        axes[1].set_yticks(np.arange(len(query_labels)), query_labels)
    fig.colorbar(axes[1].images[0], ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(post_similarity.detach().cpu().numpy(), aspect="auto", cmap="cividis")
    axes[2].set_title("Panel C. Post-mixer query similarity")
    axes[2].set_xlabel("query id")
    axes[2].set_ylabel("query id")
    if len(query_labels) <= 16:
        axes[2].set_xticks(np.arange(len(query_labels)), query_labels)
        axes[2].set_yticks(np.arange(len(query_labels)), query_labels)
    fig.colorbar(axes[2].images[0], ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _top1_unique_ratio(attention_qc: torch.Tensor) -> float:
    top1_indices = attention_qc.argmax(dim=-1)
    return float(top1_indices.unique().numel() / max(1, attention_qc.shape[0]))


def _build_mixer_batch_metrics(
    *,
    pre_similarity: torch.Tensor,
    attention_qc: torch.Tensor,
    post_similarity: torch.Tensor,
    top_k: int,
) -> dict[str, float | None]:
    entropies = _attention_entropy(attention_qc)
    attention_overlap = _cosine_similarity_matrix(attention_qc)
    post_similarity_offdiag = _pairwise_offdiag_values(post_similarity)
    return {
        "mean_attention_entropy": float(entropies.mean().cpu()),
        "top1_unique_ratio": _top1_unique_ratio(attention_qc),
        "mean_attention_overlap_offdiag": _offdiag_mean(attention_overlap),
        "mean_post_query_similarity_offdiag": _offdiag_mean(post_similarity),
        "std_post_query_similarity_offdiag": (
            float(post_similarity_offdiag.std(unbiased=False).cpu()) if post_similarity_offdiag.numel() > 0 else None
        ),
        "p90_post_query_similarity_offdiag": (
            float(torch.quantile(post_similarity_offdiag, 0.9).cpu()) if post_similarity_offdiag.numel() > 0 else None
        ),
        "collapse_ratio_post_query_similarity_ge_0_9": (
            float((post_similarity_offdiag >= 0.9).float().mean().cpu()) if post_similarity_offdiag.numel() > 0 else None
        ),
        "mean_topk_pre_similarity": _mean_topk_pre_similarity(pre_similarity, attention_qc, top_k),
    }


def _write_batch_metrics_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _mean_std_summary(values: list[float | None]) -> tuple[float | None, float | None]:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None, None
    if len(filtered) == 1:
        return filtered[0], 0.0
    array = np.asarray(filtered, dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=0))


def _write_topk_reports(
    *,
    output_dir: Path,
    attention_qc: torch.Tensor,
    channel_labels: list[str],
    top_k: int,
) -> dict[str, object]:
    query_labels = [f"q{idx}" for idx in range(attention_qc.shape[0])]
    entropies = _attention_entropy(attention_qc)
    csv_path = output_dir / "query_topk_channels.csv"
    txt_path = output_dir / "query_topk_channels.txt"
    rows = []
    with csv_path.open("w", encoding="utf-8", newline="") as csv_handle, txt_path.open("w", encoding="utf-8") as txt_handle:
        writer = csv.writer(csv_handle)
        writer.writerow(["query_id", "topk_channel_indices", "topk_channel_names", "topk_attention_weights", "attention_entropy", "max_attention_weight"])
        for query_idx, query_label in enumerate(query_labels):
            weights = attention_qc[query_idx]
            values, indices = torch.topk(weights, k=min(top_k, weights.shape[-1]))
            top_names = [channel_labels[int(idx)] for idx in indices.tolist()]
            top_indices = [int(idx) for idx in indices.tolist()]
            top_weights = [float(value) for value in values.tolist()]
            entropy = float(entropies[query_idx].cpu())
            max_weight = float(values[0].cpu())
            writer.writerow(
                [
                    query_label,
                    " ".join(str(idx) for idx in top_indices),
                    " | ".join(top_names),
                    " ".join(f"{weight:.8f}" for weight in top_weights),
                    f"{entropy:.8f}",
                    f"{max_weight:.8f}",
                ]
            )
            txt_handle.write(
                f"{query_label}: "
                f"indices={top_indices}, "
                f"names={top_names}, "
                f"weights={[round(weight, 6) for weight in top_weights]}, "
                f"entropy={entropy:.6f}, "
                f"max={max_weight:.6f}\n"
            )
            rows.append(
                {
                    "query_id": query_label,
                    "top_indices": top_indices,
                    "top_names": top_names,
                    "top_weights": top_weights,
                    "entropy": entropy,
                    "max_attention_weight": max_weight,
                }
            )
    return {"query_labels": query_labels, "rows": rows, "entropies": entropies}


def _write_report(
    path: Path,
    *,
    summary: dict[str, object],
    ci_only: bool,
) -> None:
    lines = [
        f"run_dir: {summary['run_dir']}",
        f"checkpoint_path: {summary['checkpoint_path']}",
        f"data: {summary['data']}",
        f"split: {summary['split']}",
        f"num_batches: {summary['num_batches']}",
        "",
        "pre-mixer channel similarity는 QueryChannelMixer에 들어가기 직전 channel representation들이 서로 얼마나 유사한지 보여준다.",
    ]
    if ci_only:
        lines.extend(
            [
                "",
                "CI checkpoint가 감지되어 query-to-channel attention과 post-mixer query similarity는 unavailable로 처리했다.",
            ]
        )
    else:
        lines.extend(
            [
                "query-to-channel attention은 각 query가 어떤 channel subset을 주로 읽는지 보여준다.",
                "post-mixer query similarity는 mixer 이후 query representation들이 서로 분화되었는지, 아니면 유사하게 collapse되었는지 보여준다.",
                "",
                f"mean_attention_entropy: {summary['mean_attention_entropy']}",
                f"top1_unique_ratio: {summary['top1_unique_ratio']}",
                f"mean_attention_overlap_offdiag: {summary['mean_attention_overlap_offdiag']}",
                f"mean_post_query_similarity_offdiag: {summary['mean_post_query_similarity_offdiag']}",
                f"std_post_query_similarity_offdiag: {summary['std_post_query_similarity_offdiag']}",
                f"p90_post_query_similarity_offdiag: {summary['p90_post_query_similarity_offdiag']}",
                f"collapse_ratio_post_query_similarity_ge_0_9: {summary['collapse_ratio_post_query_similarity_ge_0_9']}",
                f"mean_topk_pre_similarity: {summary['mean_topk_pre_similarity']}",
            ]
        )
    lines.extend(
        [
            "",
            "Attention이 sharp하다고 해서 반드시 downstream에서 중요하다는 뜻은 아니다. 이 분석은 Channel Mixer가 representation을 어떻게 압축하는지 보여주는 구조 분석이다. Downstream에서 실제로 쓰이는지는 별도의 linear probing ablation 분석으로 확인해야 한다.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Post-hoc structure analysis for LayaTS Channel Mixer checkpoints")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--data", type=str, default="Electricity")
    parser.add_argument("--root_path", type=str, default="../Dataset/long_term_forecast/electricity")
    parser.add_argument("--data_path", type=str, default="electricity.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--num_batches", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="analysis/mixer_structure")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args(argv)

    if args.num_batches <= 0:
        raise ValueError(f"--num_batches must be positive, got {args.num_batches}")
    if args.top_k <= 0:
        raise ValueError(f"--top_k must be positive, got {args.top_k}")
    if args.seq_len <= 0:
        raise ValueError(f"--seq_len must be positive, got {args.seq_len}")

    set_seed(args.seed)

    run_dir = None if args.run_dir is None else _resolve_existing_path(args.run_dir)
    output_dir = _resolve_output_dir(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Checkpoint path: {args.checkpoint}")
    checkpoint_path = _resolve_checkpoint_path(args.checkpoint)
    print(f"[OK] Loaded checkpoint: {checkpoint_path}")

    checkpoint_cfg = load_model_config_from_checkpoint(str(checkpoint_path))
    dataset_key = _normalize_dataset_key(args.data)
    model_cfg = LayaModelConfig(**checkpoint_cfg.__dict__)
    model = LayaTSPretrainer(model_cfg).to(args.device)
    load_report = _load_pretrainer_checkpoint(model, checkpoint_path)
    model.eval()

    loader = _build_pretrain_loader(
        args,
        dataset_key,
        checkpoint_cfg.channel_metadata_mode,
        patch_size=max(1, checkpoint_cfg.patch_size),
    )
    print(f"[OK] Built validation loader: data={args.data}, split={args.split}")
    print(f"[OK] Detected mixer module: {type(model.encoder.channel_mixer).__name__ if model.encoder.channel_mixer is not None else 'None (CI)'}")

    pre_channel_sum: Optional[torch.Tensor] = None
    pre_channel_weight = 0
    post_query_sum: Optional[torch.Tensor] = None
    post_query_weight = 0
    attention_sum: Optional[torch.Tensor] = None
    attention_weight = 0
    batch_metric_rows: list[dict[str, object]] = []
    channel_names: list[str] = []
    num_channels: Optional[int] = None
    num_queries: Optional[int] = None
    num_patches: Optional[int] = None
    embedding_dim: Optional[int] = None
    processed_batches = 0
    captured_pre = False
    captured_attn = False
    captured_post = False

    with torch.no_grad():
        for batch in loader:
            if processed_batches >= args.num_batches:
                break
            if not channel_names:
                channel_names = _extract_channel_names(batch)
            batch_on_device = move_batch_to_device(batch, args.device)
            outputs = model(
                batch_on_device["series"],
                channel_positions=batch_on_device["channel_positions"],
                channel_mask=batch_on_device["channel_mask"],
                channel_text_embeddings=batch_on_device["channel_text_embeddings"],
                channel_stats_embeddings=batch_on_device["channel_stats_embeddings"],
                return_aux=True,
                patch_mask_seed=0,
            )
            features = outputs["full_features"]

            channel_tokens = features["channel_tokens"].detach().cpu()
            if channel_tokens.dim() != 4:
                raise ValueError(f"Expected pre-mixer channel tokens [B, C, N, D], got {tuple(channel_tokens.shape)}")
            if pre_channel_sum is None:
                pre_channel_sum = channel_tokens.sum(dim=(0, 2))
            else:
                pre_channel_sum += channel_tokens.sum(dim=(0, 2))
            pre_channel_weight += int(channel_tokens.shape[0] * channel_tokens.shape[2])
            num_channels = channel_tokens.shape[1]
            num_patches = channel_tokens.shape[2]
            embedding_dim = channel_tokens.shape[3]
            captured_pre = True

            affinity = features.get("channel_affinity")
            if affinity is not None:
                affinity_cpu = affinity.detach().cpu()
                reduced_attention, reduced_weight = _reduce_attention_to_qc(affinity_cpu)
                attention_sum = reduced_attention if attention_sum is None else attention_sum + reduced_attention
                attention_weight += reduced_weight
                num_queries = reduced_attention.shape[0]
                captured_attn = True

            latent_tokens = features.get("channel_mixer_latent_tokens")
            if latent_tokens is not None:
                latent_tokens = latent_tokens.detach().cpu()
                if latent_tokens.dim() == 4:
                    post_sum = latent_tokens.sum(dim=(0, 2))
                    weight = int(latent_tokens.shape[0] * latent_tokens.shape[2])
                elif latent_tokens.dim() == 3:
                    post_sum = latent_tokens.sum(dim=0)
                    weight = int(latent_tokens.shape[0])
                else:
                    raise ValueError(
                        "Expected post-mixer latent tokens with shape [B, Q, N, D] or [B, Q, D], "
                        f"got {tuple(latent_tokens.shape)}"
                    )
                post_query_sum = post_sum if post_query_sum is None else post_query_sum + post_sum
                post_query_weight += weight
                num_queries = latent_tokens.shape[1]
                embedding_dim = latent_tokens.shape[-1]
                captured_post = True

            if affinity is not None and latent_tokens is not None:
                batch_pre_repr = channel_tokens.sum(dim=(0, 2)) / float(channel_tokens.shape[0] * channel_tokens.shape[2])
                batch_pre_similarity = _cosine_similarity_matrix(batch_pre_repr)
                batch_attention_qc = _row_normalize(reduced_attention / float(reduced_weight))
                if latent_tokens.dim() == 4:
                    batch_post_repr = latent_tokens.sum(dim=(0, 2)) / float(latent_tokens.shape[0] * latent_tokens.shape[2])
                else:
                    batch_post_repr = latent_tokens.sum(dim=0) / float(latent_tokens.shape[0])
                batch_post_similarity = _cosine_similarity_matrix(batch_post_repr)
                batch_metrics = _build_mixer_batch_metrics(
                    pre_similarity=batch_pre_similarity,
                    attention_qc=batch_attention_qc,
                    post_similarity=batch_post_similarity,
                    top_k=args.top_k,
                )
                batch_metric_rows.append(
                    {
                        "batch_index": processed_batches,
                        **batch_metrics,
                    }
                )

            processed_batches += 1

    if processed_batches == 0 or pre_channel_sum is None or pre_channel_weight == 0:
        raise RuntimeError("No batches were processed for mixer structure analysis.")

    pre_channel_repr = pre_channel_sum / float(pre_channel_weight)
    pre_similarity = _cosine_similarity_matrix(pre_channel_repr)
    channel_labels = _ensure_labels(channel_names, pre_similarity.shape[0], "ch_")
    ci_only = not (captured_attn and captured_post and attention_sum is not None and post_query_sum is not None)

    if ci_only:
        print("[INFO] CI model detected. Query-to-channel attention and post-query similarity are not available.")
        print("[INFO] Saving channel similarity only.")
        pre_png = output_dir / "ci_channel_similarity.png"
        pre_csv = output_dir / "ci_channel_similarity.csv"
        _save_heatmap(
            pre_png,
            pre_similarity,
            title=f"{args.data} pre-mixer channel similarity",
            xlabel="channel id",
            ylabel="channel id",
            x_labels=channel_labels,
            y_labels=channel_labels,
            cmap="viridis",
            figsize=(10, 8),
        )
        _write_matrix_csv(pre_csv, pre_similarity, channel_labels, channel_labels)
        summary = {
            "run_dir": None if run_dir is None else str(run_dir),
            "checkpoint_path": str(checkpoint_path),
            "data": args.data,
            "split": args.split,
            "num_batches": processed_batches,
            "num_channels": pre_similarity.shape[0],
            "num_queries": None,
            "num_patches": num_patches,
            "embedding_dim": embedding_dim,
            "mean_attention_entropy": None,
            "top1_unique_ratio": None,
            "mean_attention_overlap_offdiag": None,
            "mean_post_query_similarity_offdiag": None,
            "mean_topk_pre_similarity": None,
            "batch_metric_count": 0,
            "load_report": load_report,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        _write_report(output_dir / "mixer_structure_report.txt", summary=summary, ci_only=True)
        print("[OK] Saved summary.json")
        print("[OK] Saved report: mixer_structure_report.txt")
        return

    attention_qc = _row_normalize(attention_sum / float(attention_weight))
    post_query_repr = post_query_sum / float(post_query_weight)
    post_similarity = _cosine_similarity_matrix(post_query_repr)
    query_report = _write_topk_reports(
        output_dir=output_dir,
        attention_qc=attention_qc,
        channel_labels=channel_labels,
        top_k=args.top_k,
    )
    query_labels = query_report["query_labels"]
    entropies = query_report["entropies"]

    pre_png = output_dir / "pre_mixer_channel_similarity.png"
    pre_csv = output_dir / "pre_mixer_channel_similarity.csv"
    attn_png = output_dir / "query_channel_attention.png"
    attn_csv = output_dir / "query_channel_attention.csv"
    post_png = output_dir / "post_mixer_query_similarity.png"
    post_csv = output_dir / "post_mixer_query_similarity.csv"
    post_hist_png = output_dir / "post_mixer_query_similarity_histogram.png"
    post_hist_csv = output_dir / "post_mixer_query_similarity_histogram_values.csv"

    _save_heatmap(
        pre_png,
        pre_similarity,
        title=f"{args.data} pre-mixer channel similarity",
        xlabel="channel id",
        ylabel="channel id",
        x_labels=channel_labels,
        y_labels=channel_labels,
        cmap="viridis",
        figsize=(10, 8),
    )
    _write_matrix_csv(pre_csv, pre_similarity, channel_labels, channel_labels)
    print(f"[OK] Captured pre-mixer tokens: {tuple(pre_channel_repr.shape)}")

    _save_heatmap(
        attn_png,
        attention_qc,
        title=f"{args.data} query-to-channel attention",
        xlabel="channel id",
        ylabel="query id",
        x_labels=channel_labels,
        y_labels=query_labels,
        cmap="magma",
        figsize=(12, 5),
    )
    _write_matrix_csv(attn_csv, attention_qc, query_labels, channel_labels)
    print(f"[OK] Captured mixer attention: {tuple(attention_qc.shape)}")

    _save_heatmap(
        post_png,
        post_similarity,
        title=f"{args.data} post-mixer query similarity",
        xlabel="query id",
        ylabel="query id",
        x_labels=query_labels,
        y_labels=query_labels,
        cmap="cividis",
        figsize=(6, 5),
    )
    _write_matrix_csv(post_csv, post_similarity, query_labels, query_labels)
    print(f"[OK] Captured post-mixer tokens: {tuple(post_query_repr.shape)}")

    post_similarity_offdiag = _pairwise_offdiag_values(post_similarity)
    _save_histogram(
        post_hist_png,
        post_similarity_offdiag,
        title=f"{args.data} post-mixer query cosine histogram",
        xlabel="pairwise cosine similarity",
    )
    with post_hist_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["pairwise_cosine_similarity"])
        for value in post_similarity_offdiag.detach().cpu().tolist():
            writer.writerow([f"{float(value):.8f}"])
    print("[OK] Saved post-mixer query cosine histogram")

    _save_overview_figure(
        output_dir / "mixer_structure_overview.png",
        pre_similarity=pre_similarity,
        pre_labels=channel_labels,
        attention_qc=attention_qc,
        query_labels=query_labels,
        channel_labels=channel_labels,
        post_similarity=post_similarity,
    )
    print("[OK] Saved mixer_structure_overview.png")

    attention_overlap = _cosine_similarity_matrix(attention_qc)
    top1_indices = attention_qc.argmax(dim=-1)
    batch_metric_means: dict[str, float | None] = {}
    batch_metric_stds: dict[str, float | None] = {}
    if batch_metric_rows:
        _write_batch_metrics_csv(output_dir / "batch_metrics.csv", batch_metric_rows)
        metric_names = [
            "mean_attention_entropy",
            "top1_unique_ratio",
            "mean_attention_overlap_offdiag",
            "mean_post_query_similarity_offdiag",
            "std_post_query_similarity_offdiag",
            "p90_post_query_similarity_offdiag",
            "collapse_ratio_post_query_similarity_ge_0_9",
            "mean_topk_pre_similarity",
        ]
        for metric_name in metric_names:
            mean_value, std_value = _mean_std_summary([row.get(metric_name) for row in batch_metric_rows])
            batch_metric_means[f"{metric_name}_batch_mean"] = mean_value
            batch_metric_stds[f"{metric_name}_batch_std"] = std_value
    summary = {
        "run_dir": None if run_dir is None else str(run_dir),
        "checkpoint_path": str(checkpoint_path),
        "data": args.data,
        "split": args.split,
        "num_batches": processed_batches,
        "num_channels": pre_similarity.shape[0],
        "num_queries": attention_qc.shape[0],
        "num_patches": num_patches,
        "embedding_dim": embedding_dim,
        "mean_attention_entropy": float(entropies.mean().cpu()),
        "top1_unique_ratio": float(top1_indices.unique().numel() / max(1, attention_qc.shape[0])),
        "mean_attention_overlap_offdiag": _offdiag_mean(attention_overlap),
        "mean_post_query_similarity_offdiag": _offdiag_mean(post_similarity),
        "std_post_query_similarity_offdiag": (
            float(post_similarity_offdiag.std(unbiased=False).cpu()) if post_similarity_offdiag.numel() > 0 else None
        ),
        "p90_post_query_similarity_offdiag": (
            float(torch.quantile(post_similarity_offdiag, 0.9).cpu()) if post_similarity_offdiag.numel() > 0 else None
        ),
        "collapse_ratio_post_query_similarity_ge_0_9": (
            float((post_similarity_offdiag >= 0.9).float().mean().cpu()) if post_similarity_offdiag.numel() > 0 else None
        ),
        "mean_topk_pre_similarity": _mean_topk_pre_similarity(pre_similarity, attention_qc, args.top_k),
        "batch_metric_count": len(batch_metric_rows),
        **batch_metric_means,
        **batch_metric_stds,
        "load_report": load_report,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(output_dir / "mixer_structure_report.txt", summary=summary, ci_only=False)
    print("[OK] Saved summary.json")
    print("[OK] Saved report: mixer_structure_report.txt")


if __name__ == "__main__":
    main()
