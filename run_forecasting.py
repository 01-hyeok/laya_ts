from __future__ import annotations

import argparse
import math
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
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from laya_ts.config import LayaModelConfig

if __package__ in {None, ""}:
    from laya_ts.data_forecasting import get_forecasting_loaders
    from laya_ts.model import (
        LayaTSForecaster,
        infer_temporal_patchifier_num_patches,
        load_encoder_from_checkpoint_report,
        load_model_config_from_checkpoint,
        summarize_metadata_usage,
    )
else:
    from .data_forecasting import get_forecasting_loaders
    from .model import (
        LayaTSForecaster,
        infer_temporal_patchifier_num_patches,
        load_encoder_from_checkpoint_report,
        load_model_config_from_checkpoint,
        summarize_metadata_usage,
    )


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _parse_int_list(raw_value: Optional[str]) -> tuple[int, ...] | None:
    if raw_value in {None, "", "none", "None"}:
        return None
    values = tuple(int(piece.strip()) for piece in str(raw_value).split(",") if piece.strip())
    if not values:
        raise ValueError("Expected at least one integer value")
    if any(value <= 0 for value in values):
        raise ValueError(f"All kernel sizes must be positive, got {values}")
    return values


def _add_bool_optional_arg(parser: argparse.ArgumentParser, option: str, *, default=None) -> None:
    dest = option.lstrip("-").replace("-", "_")
    parser.add_argument(option, dest=dest, action="store_true")
    parser.add_argument(f"--no-{option[2:]}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def _relation_adapter_checkpoint_status(load_report: dict[str, object]) -> tuple[bool, bool]:
    matched_key_names = [str(key) for key in load_report.get("matched_key_names", [])]
    skipped_keys = [str(key) for key in load_report.get("skipped_keys", [])]
    missing_keys = [str(key) for key in load_report.get("missing_keys", [])]
    adapter_pretrained = any(key.startswith("relation_adapter.") for key in matched_key_names)
    adapter_missing = any(key.startswith("relation_adapter.") for key in skipped_keys + missing_keys)
    return adapter_pretrained, adapter_missing


def _mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _infer_pretrain_dataset(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).name.lower()
    if "_tslib_" in name:
        return "tslib"
    if "_tsld_" in name:
        return "tsld"
    if "_electricity_" in name:
        return "electricity"
    return "unknown"


def _build_best_forecasting_checkpoint_path(dataset_type: str, pred_len: int, checkpoint_path: str) -> str:
    checkpoint_parent = Path(checkpoint_path).resolve().parent.name
    if checkpoint_parent.startswith("checkpoints"):
        checkpoint_parent = Path(checkpoint_path).resolve().stem
    checkpoint_parent = checkpoint_parent.strip() or "forecasting_model"
    dataset_name = str(dataset_type).strip().lower()
    checkpoint_dir = Path("./checkpoints") / f"forecasting_{dataset_name}_{pred_len}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return str(checkpoint_dir / f"{checkpoint_parent}_best.pt")


def _inverse_scale_batch(values: np.ndarray, scaler) -> np.ndarray:
    batch, channels, steps = values.shape
    flat = values.transpose(0, 2, 1).reshape(-1, channels)
    restored = scaler.inverse_transform(flat)
    return restored.reshape(batch, steps, channels).transpose(0, 2, 1)


def _parse_plot_channels(raw_value: Optional[str], total_channels: int) -> list[int]:
    if not raw_value:
        return [max(0, total_channels - 1)]
    indices = []
    for chunk in raw_value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        idx = int(chunk)
        if idx < 0 or idx >= total_channels:
            raise ValueError(f"plot channel index {idx} is out of range for {total_channels} channels")
        indices.append(idx)
    return indices or [max(0, total_channels - 1)]


def _format_metadata_usage(metadata_usage: dict[str, float]) -> str:
    if not metadata_usage:
        return "Meta: n/a"
    parts = []
    if "adapter_scale_mean" in metadata_usage:
        parts.append(f"AdapterScaleμ:{metadata_usage['adapter_scale_mean']:.4f}")
    if "adapter_metadata_scale_mean" in metadata_usage:
        parts.append(f"AdapterMetaScaleμ:{metadata_usage['adapter_metadata_scale_mean']:.4f}")
    if "adapter_gate_mean" in metadata_usage:
        parts.append(f"AdapterGateμ:{metadata_usage['adapter_gate_mean']:.4f}")
    if "adapter_bias_mean_abs" in metadata_usage:
        parts.append(f"AdapterBias|B|μ:{metadata_usage['adapter_bias_mean_abs']:.4f}")
    if "adapter_metadata_present" in metadata_usage:
        parts.append(f"AdapterMetaOn:{metadata_usage['adapter_metadata_present']:.2f}")
    if "adapter_metadata_nonzero_fraction" in metadata_usage:
        parts.append(f"AdapterMetaNZ%:{100.0 * metadata_usage['adapter_metadata_nonzero_fraction']:.1f}")
    if "adapter_delta_ratio" in metadata_usage:
        parts.append(f"AdapterΔ/Input:{100.0 * metadata_usage['adapter_delta_ratio']:.2f}%")
    if "relation_scale_mean" in metadata_usage:
        parts.append(f"Scaleμ:{metadata_usage['relation_scale_mean']:.4f}")
    if "signal_score_mean_abs" in metadata_usage:
        parts.append(f"Signal|S|μ:{metadata_usage['signal_score_mean_abs']:.4f}")
    if "score_delta_mean_abs" in metadata_usage:
        parts.append(f"Delta|S|μ:{metadata_usage['score_delta_mean_abs']:.4f}")
    if "score_delta_ratio" in metadata_usage:
        parts.append(f"Delta/Signal:{100.0 * metadata_usage['score_delta_ratio']:.2f}%")
    if "relation_gate_mean" in metadata_usage:
        parts.append(f"Gateμ:{metadata_usage['relation_gate_mean']:.4f}")
    if "relation_gate_sparsity" in metadata_usage:
        parts.append(f"Gate0%:{100.0 * metadata_usage['relation_gate_sparsity']:.1f}")
    if "metadata_norm_mean" in metadata_usage:
        parts.append(f"MetaNormμ:{metadata_usage['metadata_norm_mean']:.4f}")
    return "Meta: " + ", ".join(parts)


def _save_forecasting_visuals(
    *,
    contexts: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    channel_names: list[str],
    output_dir: str,
    dataset_name: str,
    pred_len: int,
    max_samples: int,
    plot_channels: list[int],
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    saved = 0
    total_samples = min(max_samples, contexts.shape[0])
    context_len = contexts.shape[-1]
    future_x = np.arange(context_len, context_len + pred_len)
    for sample_idx in range(total_samples):
        for channel_idx in plot_channels:
            plt.figure(figsize=(10, 4))
            plt.plot(np.arange(context_len), contexts[sample_idx, channel_idx], label="Context", linewidth=1.8)
            plt.plot(future_x, y_true[sample_idx, channel_idx], label="GroundTruth", linewidth=2.0)
            plt.plot(future_x, y_pred[sample_idx, channel_idx], label="Prediction", linewidth=2.0)
            channel_name = channel_names[channel_idx] if channel_idx < len(channel_names) else f"ch_{channel_idx}"
            plt.title(f"{dataset_name} | sample={sample_idx} | channel={channel_name}")
            plt.legend()
            plt.tight_layout()
            file_name = f"sample_{sample_idx:03d}_channel_{channel_idx:03d}.png"
            plt.savefig(os.path.join(output_dir, file_name), bbox_inches="tight")
            plt.close()
            saved += 1
    return saved


def _save_attention_maps(
    *,
    features: dict,
    channel_names: list[str],
    output_dir: str,
    sample_offset: int,
    max_samples: int,
    dataset_name: str,
) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _sparse_ticks(labels: list[str], limit: int = 24):
        if not labels:
            return [], []
        if len(labels) <= limit:
            ticks = np.arange(len(labels))
            return ticks, labels
        step = max(1, int(np.ceil(len(labels) / limit)))
        ticks = np.arange(0, len(labels), step)
        shown = [labels[i] for i in ticks]
        return ticks, shown

    def _write_topk_lines(path: str, rows: torch.Tensor, labels: list[str], prefix: str, topk: int = 12):
        with open(path, "w", encoding="utf-8") as handle:
            for row_idx in range(rows.shape[0]):
                top_vals, top_idx = torch.topk(rows[row_idx], k=min(topk, rows.shape[1]))
                pairs = [f"{labels[int(idx)] if labels else int(idx)}={float(val):.4f}" for val, idx in zip(top_vals, top_idx)]
                handle.write(f"{prefix}{row_idx}: " + ", ".join(pairs) + "\n")

    def _selected_columns(matrix: torch.Tensor, topk: int = 24) -> torch.Tensor:
        importance = matrix.mean(dim=0)
        _, idx = torch.topk(importance, k=min(topk, matrix.shape[-1]))
        return idx.sort().values

    os.makedirs(output_dir, exist_ok=True)
    saved = 0

    relation = features.get("relation_affinity")
    if relation is not None:
        relation = relation.detach().cpu()
    mixer = features.get("channel_affinity")
    if mixer is not None:
        mixer = mixer.detach().cpu()

    if relation is None and mixer is None:
        return 0

    batch_size = relation.shape[0] if relation is not None else mixer.shape[0]
    for local_idx in range(min(max_samples, batch_size)):
        global_idx = sample_offset + local_idx
        sample_dir = os.path.join(output_dir, f"sample_{global_idx:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        payload = {}

        if relation is not None:
            sample_relation = relation[local_idx]
            payload["relation_affinity"] = sample_relation
            torch.save(sample_relation, os.path.join(sample_dir, "relation_affinity.pt"))
            reduced_relation = sample_relation.mean(dim=0) if sample_relation.dim() == 4 else sample_relation
            for head_idx in range(reduced_relation.shape[0]):
                head_matrix = reduced_relation[head_idx]
                labels = channel_names[: head_matrix.shape[-1]]
                plt.figure(figsize=(12, 10))
                plt.imshow(head_matrix.numpy(), aspect="auto", cmap="viridis")
                plt.colorbar()
                if labels:
                    ticks, shown = _sparse_ticks(labels)
                    plt.xticks(ticks, shown, rotation=90)
                    plt.yticks(ticks, shown)
                plt.title(f"{dataset_name} relation head={head_idx} sample={global_idx} (patch-avg)")
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f"relation_head_{head_idx:02d}.png"), bbox_inches="tight")
                plt.close()
                chosen = _selected_columns(head_matrix)
                reduced = head_matrix.index_select(0, chosen).index_select(1, chosen)
                reduced_labels = [labels[int(i)] for i in chosen.tolist()] if labels else [str(int(i)) for i in chosen.tolist()]
                plt.figure(figsize=(10, 8))
                plt.imshow(reduced.numpy(), aspect="auto", cmap="viridis")
                plt.colorbar()
                rticks = np.arange(len(reduced_labels))
                plt.xticks(rticks, reduced_labels, rotation=90)
                plt.yticks(rticks, reduced_labels)
                plt.title(f"{dataset_name} relation head={head_idx} sample={global_idx} top-channels (patch-avg)")
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f"relation_head_{head_idx:02d}_topk.png"), bbox_inches="tight")
                plt.close()
                _write_topk_lines(os.path.join(sample_dir, f"relation_head_{head_idx:02d}_topk.txt"), head_matrix, labels, prefix="channel_")

        if mixer is not None:
            sample_mixer = mixer[local_idx].mean(dim=0)
            payload["channel_affinity"] = sample_mixer
            torch.save(sample_mixer, os.path.join(sample_dir, "channel_affinity.pt"))
            for head_idx in range(sample_mixer.shape[0]):
                head_matrix = sample_mixer[head_idx]
                labels = channel_names[: head_matrix.shape[-1]]
                plt.figure(figsize=(12, 5))
                plt.imshow(head_matrix.numpy(), aspect="auto", cmap="magma")
                plt.colorbar()
                if labels:
                    ticks, shown = _sparse_ticks(labels)
                    plt.xticks(ticks, shown, rotation=90)
                plt.yticks(np.arange(head_matrix.shape[-2]), [f"q{i}" for i in range(head_matrix.shape[-2])])
                plt.title(f"{dataset_name} mixer head={head_idx} sample={global_idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f"mixer_head_{head_idx:02d}.png"), bbox_inches="tight")
                plt.close()
                chosen = _selected_columns(head_matrix)
                reduced = head_matrix.index_select(1, chosen)
                reduced_labels = [labels[int(i)] for i in chosen.tolist()] if labels else [str(int(i)) for i in chosen.tolist()]
                plt.figure(figsize=(10, 5))
                plt.imshow(reduced.numpy(), aspect="auto", cmap="magma")
                plt.colorbar()
                plt.xticks(np.arange(len(reduced_labels)), reduced_labels, rotation=90)
                plt.yticks(np.arange(head_matrix.shape[-2]), [f"q{i}" for i in range(head_matrix.shape[-2])])
                plt.title(f"{dataset_name} mixer head={head_idx} sample={global_idx} top-channels")
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f"mixer_head_{head_idx:02d}_topk.png"), bbox_inches="tight")
                plt.close()
                _write_topk_lines(os.path.join(sample_dir, f"mixer_head_{head_idx:02d}_topk.txt"), head_matrix, labels, prefix="q")

        if payload:
            torch.save(payload, os.path.join(sample_dir, "affinities.pt"))
            saved += 1

    return saved


def main(argv=None):
    p = argparse.ArgumentParser(description="Forecasting downstream with standalone Laya on time-series data")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--dataset_type", type=str, default="Electricity")
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--channel_metadata_mode", type=str, default=None, choices=["onehot", "text", "stats", "text_stats_joint", "text_stats_avg", "none"])
    p.add_argument("--metadata_fusion_mode", type=str, default=None, choices=["none", "add", "concat_kv", "attention_gate", "attention_suppress_gate"])
    p.add_argument("--onehot_channel_vocab_size", type=int, default=None)
    p.add_argument("--channel_mixer_type", type=str, default=None, choices=["mixer", "independent", "ci_adapter"])
    p.add_argument("--channel_mixer_relation_mode", type=str, default=None, choices=["none", "laya_relation", "metadata_query_gate", "metadata_query_bias", "description_relation"])
    p.add_argument("--channel_mixer_relation_scale_init", type=float, default=None)
    _add_bool_optional_arg(p, "--use_relation_adapter", default=None)
    p.add_argument("--relation_num_heads", type=int, default=None)
    p.add_argument("--relation_dropout", type=float, default=None)
    p.add_argument("--relation_scale_init", type=float, default=None)
    _add_bool_optional_arg(p, "--use_metadata_bias", default=None)
    _add_bool_optional_arg(p, "--use_metadata_gate", default=None)
    p.add_argument("--metadata_scale_init", type=float, default=None)
    p.add_argument("--metadata_dropout", type=float, default=None)
    p.add_argument("--relation_adapter_position", type=str, default=None, choices=["post_encoder"])
    p.add_argument("--description_relation_num_latents", type=int, default=None)
    p.add_argument("--description_relation_metric", type=str, default=None, choices=["projected_dot", "cosine"])
    p.add_argument("--description_relation_lambda_init", type=float, default=None)
    p.add_argument("--description_relation_gamma_init", type=float, default=None)
    p.add_argument("--use_channel_relation_block", action="store_true")
    p.add_argument("--channel_relation_heads", type=int, default=None)
    p.add_argument("--channel_relation_gate_scale_init", type=float, default=None)
    p.add_argument("--channel_relation_residual_scale_init", type=float, default=None)
    p.add_argument("--encoder_variant", type=str, default=None, choices=["default"])
    p.add_argument("--temporal_patchifier_mode", type=str, default=None, choices=["fixed", "multiscale", "charm_like"])
    p.add_argument("--charm_kernel_sizes", type=str, default=None)
    p.add_argument("--charm_stride", type=int, default=None)
    p.add_argument("--charm_patchifier_dropout", type=float, default=None)
    p.add_argument("--charm_scale_gate_source", type=str, default=None, choices=["learned", "text"])
    p.add_argument("--charm_scale_gate_temperature", type=float, default=None)
    p.add_argument("--charm_patchifier_fusion", type=str, default=None, choices=["replace", "residual"])
    p.add_argument("--charm_patchifier_residual_init", type=float, default=None)
    p.add_argument("--text_encoder_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--text_metadata_cache_dir", type=str, default="./metadata_cache")
    p.add_argument("--text_encoder_local_files_only", action="store_true")
    p.add_argument("--stats_metadata_dim", type=int, default=None)
    p.add_argument("--require_relation_adapter_checkpoint", action="store_true")
    p.add_argument("--allow_random_init_relation_adapter", action="store_true")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="laya_ts/runs/forecasting")
    p.add_argument("--save_test_plots", action="store_true")
    p.add_argument("--num_test_plots", type=int, default=5)
    p.add_argument("--plot_channels", type=str, default=None, help="Comma-separated channel indices to visualize at test time")
    p.add_argument("--save_attention_maps", action="store_true")
    p.add_argument("--num_attention_map_samples", type=int, default=3)
    _add_bool_optional_arg(p, "--use_revin", default=True)
    _add_bool_optional_arg(p, "--revin_affine", default=False)
    _add_bool_optional_arg(p, "--revin_subtract_last", default=False)
    p.add_argument("--revin_eps", type=float, default=1e-5)
    args = p.parse_args(argv)
    if args.stats_metadata_dim is not None and args.stats_metadata_dim <= 0:
        raise ValueError(f"--stats_metadata_dim must be positive, got {args.stats_metadata_dim}")
    if args.revin_eps <= 0:
        raise ValueError(f"--revin_eps must be positive, got {args.revin_eps}")
    set_seed(args.seed)
    checkpoint_cfg = load_model_config_from_checkpoint(args.checkpoint)
    channel_metadata_mode = args.channel_metadata_mode or checkpoint_cfg.channel_metadata_mode
    metadata_fusion_mode = args.metadata_fusion_mode or checkpoint_cfg.metadata_fusion_mode
    if channel_metadata_mode == "coordinates":
        raise ValueError("laya_ts no longer supports channel_metadata_mode='coordinates'. Use one of: onehot, text, stats, text_stats_joint, text_stats_avg, none.")
    requested_channel_mixer_type = args.channel_mixer_type or checkpoint_cfg.channel_mixer_type
    use_relation_adapter = checkpoint_cfg.use_relation_adapter
    if args.use_relation_adapter is not None:
        use_relation_adapter = args.use_relation_adapter
    if str(requested_channel_mixer_type).strip().lower().replace("-", "_") == "ci_adapter":
        channel_mixer_type = "independent"
        use_relation_adapter = True
    else:
        channel_mixer_type = requested_channel_mixer_type
    train_loader, val_loader, test_loader, scaler = get_forecasting_loaders(args.data_path, args.dataset_type, batch_size=args.batch_size, seq_len=args.seq_len, pred_len=args.pred_len, num_workers=args.num_workers, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=args.text_encoder_name_or_path, text_metadata_cache_dir=args.text_metadata_cache_dir, text_encoder_local_files_only=args.text_encoder_local_files_only)
    first_batch = next(iter(train_loader))
    input_channels = first_batch["series"].shape[1]
    out_channels = first_batch["target"].shape[1]
    channel_names = list(first_batch.get("channel_names", []))
    pretrain_dataset = _infer_pretrain_dataset(args.checkpoint)
    onehot_vocab_size = checkpoint_cfg.onehot_channel_vocab_size
    if channel_metadata_mode == "onehot":
        onehot_vocab_size = max(onehot_vocab_size, args.onehot_channel_vocab_size or 0, input_channels)
    relation_heads = checkpoint_cfg.channel_relation_heads
    if args.channel_relation_heads is not None:
        relation_heads = args.channel_relation_heads
    relation_gate_scale = checkpoint_cfg.channel_relation_gate_scale_init
    if args.channel_relation_gate_scale_init is not None:
        relation_gate_scale = args.channel_relation_gate_scale_init
    relation_residual_scale = checkpoint_cfg.channel_relation_residual_scale_init
    if args.channel_relation_residual_scale_init is not None:
        relation_residual_scale = args.channel_relation_residual_scale_init
    channel_mixer_relation_mode = args.channel_mixer_relation_mode or checkpoint_cfg.channel_mixer_relation_mode
    channel_mixer_relation_scale_init = (
        checkpoint_cfg.channel_mixer_relation_scale_init
        if args.channel_mixer_relation_scale_init is None
        else args.channel_mixer_relation_scale_init
    )
    relation_num_heads = (
        checkpoint_cfg.relation_num_heads
        if args.relation_num_heads is None
        else args.relation_num_heads
    )
    relation_dropout = (
        checkpoint_cfg.relation_dropout
        if args.relation_dropout is None
        else args.relation_dropout
    )
    relation_scale_init = (
        checkpoint_cfg.relation_scale_init
        if args.relation_scale_init is None
        else args.relation_scale_init
    )
    use_metadata_bias = (
        checkpoint_cfg.use_metadata_bias
        if args.use_metadata_bias is None
        else args.use_metadata_bias
    )
    use_metadata_gate = (
        checkpoint_cfg.use_metadata_gate
        if args.use_metadata_gate is None
        else args.use_metadata_gate
    )
    metadata_scale_init = (
        checkpoint_cfg.metadata_scale_init
        if args.metadata_scale_init is None
        else args.metadata_scale_init
    )
    metadata_dropout = (
        checkpoint_cfg.metadata_dropout
        if args.metadata_dropout is None
        else args.metadata_dropout
    )
    relation_adapter_position = (
        checkpoint_cfg.relation_adapter_position
        if args.relation_adapter_position is None
        else args.relation_adapter_position
    )
    description_relation_num_latents = (
        checkpoint_cfg.description_relation_num_latents
        if args.description_relation_num_latents is None
        else args.description_relation_num_latents
    )
    description_relation_metric = args.description_relation_metric or checkpoint_cfg.description_relation_metric
    description_relation_lambda_init = (
        checkpoint_cfg.description_relation_lambda_init
        if args.description_relation_lambda_init is None
        else args.description_relation_lambda_init
    )
    description_relation_gamma_init = (
        checkpoint_cfg.description_relation_gamma_init
        if args.description_relation_gamma_init is None
        else args.description_relation_gamma_init
    )
    temporal_patchifier_mode = args.temporal_patchifier_mode or checkpoint_cfg.temporal_patchifier_mode
    charm_kernel_sizes = _parse_int_list(args.charm_kernel_sizes) or checkpoint_cfg.charm_kernel_sizes
    charm_stride = checkpoint_cfg.charm_stride if args.charm_stride is None else args.charm_stride
    charm_patchifier_dropout = checkpoint_cfg.charm_patchifier_dropout if args.charm_patchifier_dropout is None else args.charm_patchifier_dropout
    charm_scale_gate_source = args.charm_scale_gate_source or checkpoint_cfg.charm_scale_gate_source
    charm_scale_gate_temperature = checkpoint_cfg.charm_scale_gate_temperature if args.charm_scale_gate_temperature is None else args.charm_scale_gate_temperature
    charm_patchifier_fusion = args.charm_patchifier_fusion or checkpoint_cfg.charm_patchifier_fusion
    charm_patchifier_residual_init = checkpoint_cfg.charm_patchifier_residual_init if args.charm_patchifier_residual_init is None else args.charm_patchifier_residual_init
    encoder_variant = args.encoder_variant or checkpoint_cfg.encoder_variant
    stats_metadata_dim = checkpoint_cfg.stats_metadata_dim if args.stats_metadata_dim is None else args.stats_metadata_dim
    model_cfg = LayaModelConfig(
        **{
            **checkpoint_cfg.__dict__,
            "channel_metadata_mode": channel_metadata_mode,
            "metadata_fusion_mode": metadata_fusion_mode,
            "channel_mixer_type": channel_mixer_type,
            "channel_mixer_relation_mode": channel_mixer_relation_mode,
            "channel_mixer_relation_scale_init": channel_mixer_relation_scale_init,
            "use_relation_adapter": use_relation_adapter,
            "relation_num_heads": relation_num_heads,
            "relation_dropout": relation_dropout,
            "relation_scale_init": relation_scale_init,
            "use_metadata_bias": use_metadata_bias,
            "use_metadata_gate": use_metadata_gate,
            "metadata_scale_init": metadata_scale_init,
            "metadata_dropout": metadata_dropout,
            "relation_adapter_position": relation_adapter_position,
            "description_relation_num_latents": description_relation_num_latents,
            "description_relation_metric": description_relation_metric,
            "description_relation_lambda_init": description_relation_lambda_init,
            "description_relation_gamma_init": description_relation_gamma_init,
            "onehot_channel_vocab_size": onehot_vocab_size,
            "use_channel_relation_block": args.use_channel_relation_block or checkpoint_cfg.use_channel_relation_block,
            "channel_relation_heads": relation_heads,
            "channel_relation_gate_scale_init": relation_gate_scale,
            "channel_relation_residual_scale_init": relation_residual_scale,
            "encoder_variant": encoder_variant,
            "temporal_patchifier_mode": temporal_patchifier_mode,
            "charm_kernel_sizes": charm_kernel_sizes,
            "charm_stride": charm_stride,
            "charm_patchifier_dropout": charm_patchifier_dropout,
            "charm_scale_gate_source": charm_scale_gate_source,
            "charm_scale_gate_temperature": charm_scale_gate_temperature,
            "charm_patchifier_fusion": charm_patchifier_fusion,
            "charm_patchifier_residual_init": charm_patchifier_residual_init,
            "stats_metadata_dim": stats_metadata_dim,
        }
    )
    num_patches = infer_temporal_patchifier_num_patches(model_cfg, first_batch["series"].shape[-1])
    model = LayaTSForecaster(
        model_cfg,
        pred_len=args.pred_len,
        out_channels=out_channels,
        num_patches=num_patches,
        use_revin=args.use_revin,
        revin_affine=args.revin_affine,
        revin_subtract_last=args.revin_subtract_last,
        revin_eps=args.revin_eps,
    ).to(args.device)
    load_report = load_encoder_from_checkpoint_report(model, args.checkpoint)
    skipped_encoder_keys = list(load_report["skipped_keys"])
    skipped_onehot_projector = any(key.startswith("channel_id_projector.") for key in skipped_encoder_keys)
    skipped_relation_adapter = any(key.startswith("relation_adapter.") for key in skipped_encoder_keys)
    adapter_pretrained, adapter_missing = _relation_adapter_checkpoint_status(load_report)
    print("=" * 50)
    print("🚀 Architecture: LAYA")
    print(f"📊 Target Dataset: {args.dataset_type} (Channels: {input_channels})")
    print(f"📊 Pretrain Info: {pretrain_dataset} (Detected Channels: {input_channels})")
    print(f"📊 Prediction Length: {args.pred_len}")
    print("=" * 50)
    print(f"✅ Loaded pretrained encoder from {args.checkpoint}")
    print(f"   - {load_report['matched_keys']}/{load_report['total_encoder_keys']} keys matched.")
    if load_report["missing_keys"]:
        print(f"   ⚠️ Missing keys (not loaded): {load_report['missing_keys'][:3]}...")
    if load_report["unexpected_keys"]:
        print(f"   ℹ️ Unexpected keys ignored: {load_report['unexpected_keys'][:3]}...")
    if skipped_encoder_keys:
        print(f"   ℹ️ Shape-mismatched keys skipped: {skipped_encoder_keys[:3]}...")
    if model_cfg.use_relation_adapter:
        if args.require_relation_adapter_checkpoint and not adapter_pretrained:
            raise ValueError(
                "Relation adapter is enabled, but the checkpoint does not contain compatible relation_adapter weights."
            )
        if adapter_missing and not adapter_pretrained and not args.allow_random_init_relation_adapter:
            raise ValueError(
                "Relation adapter is enabled, but the checkpoint does not provide compatible relation_adapter weights. "
                "Use a pretrained ci_adapter checkpoint, or pass --allow_random_init_relation_adapter explicitly."
            )
    for p_ in model.encoder.parameters():
        p_.requires_grad = False
    finetune_params = list(model.head.parameters())
    if skipped_onehot_projector and model.encoder.channel_id_projector is not None:
        for p_ in model.encoder.channel_id_projector.parameters():
            p_.requires_grad = True
        finetune_params.extend(model.encoder.channel_id_projector.parameters())
        print(
            f"Onehot vocab resized from checkpoint value {checkpoint_cfg.onehot_channel_vocab_size} "
            f"to {model_cfg.onehot_channel_vocab_size}; reinitializing channel_id_projector."
        )
    if skipped_relation_adapter and model.encoder.relation_adapter is not None:
        for p_ in model.encoder.relation_adapter.parameters():
            p_.requires_grad = True
        finetune_params.extend(model.encoder.relation_adapter.parameters())
        print("Relation adapter parameters were not restored from checkpoint; enabling them for downstream training.")
    print(
        f"ℹ️ Channel Adapter: {'enabled' if model_cfg.use_relation_adapter else 'disabled'}"
    )
    if model_cfg.use_relation_adapter:
        print(
            f"   - position: {model_cfg.relation_adapter_position}, heads: {model_cfg.relation_num_heads}, "
            f"dropout: {model_cfg.relation_dropout}, metadata_dropout: {model_cfg.metadata_dropout}"
        )
        print(f"   - adapter_pretrained: {adapter_pretrained}")
    print(
        f"📝 metadata: mode={channel_metadata_mode}, "
        f"fusion={model_cfg.metadata_fusion_mode}, relation={model_cfg.channel_mixer_relation_mode}"
    )
    print(
        f"ℹ️ RevIN: {'enabled' if args.use_revin else 'disabled'} | "
        f"affine={args.revin_affine} | subtract_last={args.revin_subtract_last} | eps={args.revin_eps}"
    )
    print(f"ℹ️ Metadata Fusion Mode: {model_cfg.metadata_fusion_mode}")
    print(f"ℹ️ Channel Mixer Relation Mode: {model_cfg.channel_mixer_relation_mode}")
    if model_cfg.metadata_fusion_mode in {"attention_gate", "attention_suppress_gate"}:
        relation_label = (
            "suppression relation"
            if model_cfg.metadata_fusion_mode == "attention_suppress_gate"
            else "description relation"
        )
        print(
            f"🔗 {relation_label}: metric={model_cfg.description_relation_metric}, "
            f"lambda_init={model_cfg.description_relation_lambda_init}, "
            f"gamma_init={model_cfg.description_relation_gamma_init}"
        )
        print(f"   - relation_metric: {model_cfg.description_relation_metric}")
        print(f"   - lambda_init: {model_cfg.description_relation_lambda_init}")
        print(f"   - gamma_init: {model_cfg.description_relation_gamma_init}")
    print(f"ℹ️ Encoder Variant: {model_cfg.encoder_variant}")
    print(f"ℹ️ Channel Relation Block: {'enabled' if model_cfg.use_channel_relation_block else 'disabled'}")
    if model_cfg.use_channel_relation_block:
        print(f"   - relation heads: {model_cfg.channel_relation_heads}")
        print(f"   - gate scale init: {model_cfg.channel_relation_gate_scale_init}")
        print(f"   - residual scale init: {model_cfg.channel_relation_residual_scale_init}")
    head_label = "CI" if channel_mixer_type == "independent" else "Mixer"
    print(
        f"✅ Probing Head [{head_label}, channel-wise]: "
        f"[B*C, {num_patches * model.probe_token_dim}] -> [B, {out_channels}, {args.pred_len}]"
    )
    optimizer = torch.optim.AdamW(finetune_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    best_state = None; best_val = float("inf")
    best_checkpoint_path = _build_best_forecasting_checkpoint_path(args.dataset_type, args.pred_len, args.checkpoint)
    steps_per_epoch = len(train_loader)
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); train_loss = 0.0
            train_metadata_usage = {}
            for batch_idx, batch in enumerate(train_loader):
                x = batch["series"].to(args.device); y = batch["target"].to(args.device)
                pos = batch["channel_positions"].to(args.device); mask = batch["channel_mask"].to(args.device)
                channel_text_embeddings = batch.get("channel_text_embeddings")
                if channel_text_embeddings is not None:
                    channel_text_embeddings = channel_text_embeddings.to(args.device)
                channel_stats_embeddings = batch.get("channel_stats_embeddings")
                if channel_stats_embeddings is not None:
                    channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                if batch_idx == 0:
                    pred, features = model(
                        x,
                        pos,
                        mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                        return_features=True,
                    )
                    train_metadata_usage = summarize_metadata_usage(features)
                else:
                    pred = model(
                        x,
                        pos,
                        mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                    )
                loss = criterion(pred, y); loss.backward(); optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= max(1, len(train_loader.dataset))
            writer.add_scalar("train/mse", train_loss, epoch)
            for key, value in train_metadata_usage.items():
                writer.add_scalar(f"train_meta/{key}", value, epoch)
            model.eval(); losses=[]
            val_metadata_usage = {}
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    x = batch["series"].to(args.device); y = batch["target"].to(args.device)
                    pos = batch["channel_positions"].to(args.device); mask = batch["channel_mask"].to(args.device)
                    channel_text_embeddings = batch.get("channel_text_embeddings")
                    if channel_text_embeddings is not None:
                        channel_text_embeddings = channel_text_embeddings.to(args.device)
                    channel_stats_embeddings = batch.get("channel_stats_embeddings")
                    if channel_stats_embeddings is not None:
                        channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                    if batch_idx == 0:
                        pred, features = model(
                            x,
                            pos,
                            mask,
                            channel_text_embeddings=channel_text_embeddings,
                            channel_stats_embeddings=channel_stats_embeddings,
                            return_features=True,
                        )
                        val_metadata_usage = summarize_metadata_usage(features)
                    else:
                        pred = model(
                            x,
                            pos,
                            mask,
                            channel_text_embeddings=channel_text_embeddings,
                            channel_stats_embeddings=channel_stats_embeddings,
                        )
                    losses.append(criterion(pred, y).item())
            val_loss = float(np.mean(losses)) if losses else float("inf")
            writer.add_scalar("val/mse", val_loss, epoch)
            for key, value in val_metadata_usage.items():
                writer.add_scalar(f"val_meta/{key}", value, epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{args.epochs} | Steps: {steps_per_epoch} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"Train {_format_metadata_usage(train_metadata_usage)} | "
                f"Val {_format_metadata_usage(val_metadata_usage)}"
            )
            if val_loss <= best_val:
                best_val = val_loss; best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": best_state,
                        "model_config": model_cfg.__dict__,
                        "dataset_type": args.dataset_type,
                        "data_path": args.data_path,
                        "seq_len": args.seq_len,
                        "pred_len": args.pred_len,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "checkpoint": args.checkpoint,
                        "channel_metadata_mode": channel_metadata_mode,
                        "metadata_fusion_mode": metadata_fusion_mode,
                        "channel_mixer_type": channel_mixer_type,
                        "channel_mixer_relation_mode": channel_mixer_relation_mode,
                        "channel_mixer_relation_scale_init": channel_mixer_relation_scale_init,
                        "text_encoder_name_or_path": args.text_encoder_name_or_path,
                        "text_metadata_cache_dir": args.text_metadata_cache_dir,
                        "text_encoder_local_files_only": args.text_encoder_local_files_only,
                        "use_revin": args.use_revin,
                        "revin_affine": args.revin_affine,
                        "revin_subtract_last": args.revin_subtract_last,
                        "revin_eps": args.revin_eps,
                        "adapter_pretrained": adapter_pretrained,
                        "best_val_mse": val_loss,
                    },
                    best_checkpoint_path,
                )
                print(f"   -> Updated best validation state: val_loss={val_loss:.6f} at epoch {epoch}")
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval(); preds=[]; ys=[]; contexts=[]; attention_saved = 0; processed_test_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["series"].to(args.device); y = batch["target"].to(args.device)
                pos = batch["channel_positions"].to(args.device); mask = batch["channel_mask"].to(args.device)
                channel_text_embeddings = batch.get("channel_text_embeddings")
                if channel_text_embeddings is not None:
                    channel_text_embeddings = channel_text_embeddings.to(args.device)
                channel_stats_embeddings = batch.get("channel_stats_embeddings")
                if channel_stats_embeddings is not None:
                    channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                if args.save_attention_maps and attention_saved < args.num_attention_map_samples:
                    pred, features = model(
                        x,
                        pos,
                        mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                        return_features=True,
                    )
                    attention_dir = os.path.join(args.log_dir, "attention_maps")
                    newly_saved = _save_attention_maps(
                        features=features,
                        channel_names=channel_names,
                        output_dir=attention_dir,
                        sample_offset=processed_test_samples,
                        max_samples=args.num_attention_map_samples - attention_saved,
                        dataset_name=args.dataset_type,
                    )
                    attention_saved += newly_saved
                else:
                    pred = model(
                        x,
                        pos,
                        mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                    )
                contexts.append(batch["series"].cpu().numpy())
                preds.append(pred.cpu().numpy()); ys.append(y.cpu().numpy())
                processed_test_samples += x.shape[0]
        context_series = np.concatenate(contexts, axis=0)
        y_true = np.concatenate(ys, axis=0); y_pred = np.concatenate(preds, axis=0)
        print({"test_mse": _mse(y_true, y_pred), "test_mae": _mae(y_true, y_pred)})
        print(f"Saved best forecasting checkpoint under {best_checkpoint_path}")
        if args.save_test_plots:
            restored_context = _inverse_scale_batch(context_series, scaler)
            restored_true = _inverse_scale_batch(y_true, scaler)
            restored_pred = _inverse_scale_batch(y_pred, scaler)
            plot_channels = _parse_plot_channels(args.plot_channels, restored_true.shape[1])
            plot_dir = os.path.join(args.log_dir, "test_plots")
            saved = _save_forecasting_visuals(
                contexts=restored_context,
                y_true=restored_true,
                y_pred=restored_pred,
                channel_names=channel_names,
                output_dir=plot_dir,
                dataset_name=args.dataset_type,
                pred_len=args.pred_len,
                max_samples=args.num_test_plots,
                plot_channels=plot_channels,
            )
            print(f"Saved {saved} test forecast plot(s) under {plot_dir}")
        if args.save_attention_maps:
            print(f"Saved attention maps for {attention_saved} sample(s) under {os.path.join(args.log_dir, 'attention_maps')}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
