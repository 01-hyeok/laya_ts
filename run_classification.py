from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

if __package__ in {None, ""}:
    WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.tensorboard import SummaryWriter

from laya_ts.config import LayaModelConfig

if __package__ in {None, ""}:
    from laya_ts.data_classification import get_classification_loaders
    from laya_ts.model import (
        LayaTSClassifier,
        load_encoder_from_checkpoint_report,
        load_model_config_from_checkpoint,
        summarize_metadata_usage,
    )
else:
    from .data_classification import get_classification_loaders
    from .model import (
        LayaTSClassifier,
        load_encoder_from_checkpoint_report,
        load_model_config_from_checkpoint,
        summarize_metadata_usage,
    )


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _parse_int_list(raw_value: str | None) -> tuple[int, ...] | None:
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


def _infer_pretrain_dataset(checkpoint_path: str) -> str:
    name = Path(checkpoint_path).name.lower()
    if "_tslib_" in name:
        return "tslib"
    if "_tsld_" in name:
        return "tsld"
    if "_electricity_" in name:
        return "electricity"
    return "unknown"


def _extract_channel_names(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        if not value:
            return []
        first = value[0]
        if isinstance(first, (list, tuple)):
            return [str(item) for item in first]
        return [str(item) for item in value]
    return [str(value)]


def _format_metadata_usage(metadata_usage: dict[str, float]) -> str:
    if not metadata_usage:
        return "Meta: n/a"
    parts = []
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

    def _resolve_labels(raw_labels: list[str], expected_len: int, *, label_prefix: str, affinity_name: str) -> list[str]:
        labels = [str(value) for value in raw_labels[:expected_len]]
        if len(labels) < expected_len:
            print(
                f"[attention_maps] Warning: {dataset_name} {affinity_name} labels length "
                f"({len(labels)}) does not match attention width ({expected_len}); "
                f"using fallback numeric labels for missing entries."
            )
            labels.extend(f"{label_prefix}{idx + 1}" for idx in range(len(labels), expected_len))
        return labels

    def _write_topk_lines(path: str, rows: torch.Tensor, labels: list[str], prefix: str, topk: int = 12):
        with open(path, "w", encoding="utf-8") as handle:
            for row_idx in range(rows.shape[0]):
                top_vals, top_idx = torch.topk(rows[row_idx], k=min(topk, rows.shape[1]))
                pairs = [f"{labels[int(idx)]}={float(val):.4f}" for val, idx in zip(top_vals, top_idx)]
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
                labels = _resolve_labels(
                    channel_names,
                    head_matrix.shape[-1],
                    label_prefix="channel_",
                    affinity_name="relation",
                )
                plt.figure(figsize=(12, 10))
                plt.imshow(head_matrix.numpy(), aspect="auto", cmap="viridis")
                plt.colorbar()
                ticks, shown = _sparse_ticks(labels)
                plt.xticks(ticks, shown, rotation=90)
                plt.yticks(ticks, shown)
                plt.title(f"{dataset_name} relation head={head_idx} sample={global_idx} (patch-avg)")
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f"relation_head_{head_idx:02d}.png"), bbox_inches="tight")
                plt.close()
                chosen = _selected_columns(head_matrix)
                reduced = head_matrix.index_select(0, chosen).index_select(1, chosen)
                reduced_labels = [labels[int(i)] for i in chosen.tolist()]
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
                labels = _resolve_labels(
                    channel_names,
                    head_matrix.shape[-1],
                    label_prefix="channel_",
                    affinity_name="mixer",
                )
                plt.figure(figsize=(12, 5))
                plt.imshow(head_matrix.numpy(), aspect="auto", cmap="magma")
                plt.colorbar()
                ticks, shown = _sparse_ticks(labels)
                plt.xticks(ticks, shown, rotation=90)
                plt.yticks(np.arange(head_matrix.shape[-2]), [f"q{i}" for i in range(head_matrix.shape[-2])])
                plt.title(f"{dataset_name} mixer head={head_idx} sample={global_idx}")
                plt.tight_layout()
                plt.savefig(os.path.join(sample_dir, f"mixer_head_{head_idx:02d}.png"), bbox_inches="tight")
                plt.close()
                chosen = _selected_columns(head_matrix)
                reduced = head_matrix.index_select(1, chosen)
                reduced_labels = [labels[int(i)] for i in chosen.tolist()]
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
    p = argparse.ArgumentParser(description="Classification downstream with standalone Laya on time-series data")
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--channel_metadata_mode", type=str, default=None, choices=["onehot", "text", "stats", "text_stats_joint", "text_stats_avg", "none"])
    p.add_argument("--metadata_fusion_mode", type=str, default=None, choices=["none", "add", "concat_kv", "attention_gate", "attention_suppress_gate"])
    p.add_argument("--onehot_channel_vocab_size", type=int, default=None)
    p.add_argument("--channel_mixer_type", type=str, default=None, choices=["mixer", "independent"])
    p.add_argument("--channel_mixer_relation_mode", type=str, default=None, choices=["none", "laya_relation", "metadata_query_gate", "metadata_query_bias", "description_relation"])
    p.add_argument("--channel_mixer_relation_scale_init", type=float, default=None)
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
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="laya_ts/runs/classification")
    _add_bool_optional_arg(p, "--use_tensorboard", default=False)
    p.add_argument("--save_attention_maps", action="store_true")
    p.add_argument("--num_attention_map_samples", type=int, default=3)
    args = p.parse_args(argv)
    if args.stats_metadata_dim is not None and args.stats_metadata_dim <= 0:
        raise ValueError(f"--stats_metadata_dim must be positive, got {args.stats_metadata_dim}")
    set_seed(args.seed)
    checkpoint_cfg = load_model_config_from_checkpoint(args.checkpoint)
    channel_metadata_mode = args.channel_metadata_mode or checkpoint_cfg.channel_metadata_mode
    metadata_fusion_mode = args.metadata_fusion_mode or checkpoint_cfg.metadata_fusion_mode
    if channel_metadata_mode == "coordinates":
        raise ValueError("laya_ts no longer supports channel_metadata_mode='coordinates'. Use one of: onehot, text, stats, text_stats_joint, text_stats_avg, none.")
    channel_mixer_type = args.channel_mixer_type or checkpoint_cfg.channel_mixer_type
    train_loader, val_loader, test_loader, num_classes, in_vars, actual_seq_len = get_classification_loaders(args.data_root, seq_len=args.seq_len, batch_size=args.batch_size, num_workers=args.num_workers, val_ratio=args.val_ratio, channel_metadata_mode=channel_metadata_mode, text_encoder_name_or_path=args.text_encoder_name_or_path, text_metadata_cache_dir=args.text_metadata_cache_dir, text_encoder_local_files_only=args.text_encoder_local_files_only)
    first_batch = next(iter(train_loader))
    channel_names = _extract_channel_names(first_batch.get("channel_names"))
    onehot_vocab_size = checkpoint_cfg.onehot_channel_vocab_size
    if channel_metadata_mode == "onehot":
        onehot_vocab_size = max(onehot_vocab_size, args.onehot_channel_vocab_size or 0, in_vars)
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
    model = LayaTSClassifier(model_cfg, num_classes=num_classes).to(args.device)
    load_report = load_encoder_from_checkpoint_report(model, args.checkpoint)
    skipped_encoder_keys = list(load_report["skipped_keys"])
    skipped_onehot_projector = any(key.startswith("channel_id_projector.") for key in skipped_encoder_keys)
    pretrain_dataset = _infer_pretrain_dataset(args.checkpoint)
    head_input = model.encoder.config.embed_dim if channel_mixer_type == "independent" else model.encoder.config.embed_dim
    print("=" * 50)
    print("🚀 Architecture: LAYA")
    print(f"📊 Target Dataset: {Path(args.data_root).name} (Channels: {in_vars})")
    print(f"📊 Pretrain Info: {pretrain_dataset}")
    print(f"📊 Classes: {num_classes} | Sequence Length: {actual_seq_len}")
    print("=" * 50)
    print(f"✅ Loaded pretrained encoder from {args.checkpoint}")
    print(f"   - {load_report['matched_keys']}/{load_report['total_encoder_keys']} keys matched.")
    if load_report["missing_keys"]:
        print(f"   ⚠️ Missing keys (not loaded): {load_report['missing_keys'][:3]}...")
    if load_report["unexpected_keys"]:
        print(f"   ℹ️ Unexpected keys ignored: {load_report['unexpected_keys'][:3]}...")
    if skipped_encoder_keys:
        print(f"   ℹ️ Shape-mismatched keys skipped: {skipped_encoder_keys[:3]}...")
    for p_ in model.encoder.parameters():
        p_.requires_grad = False
    finetune_params = list(model.head.parameters()) + list(model.norm.parameters())
    if skipped_onehot_projector and model.encoder.channel_id_projector is not None:
        for p_ in model.encoder.channel_id_projector.parameters():
            p_.requires_grad = True
        finetune_params.extend(model.encoder.channel_id_projector.parameters())
        print(
            f"Onehot vocab resized from checkpoint value {checkpoint_cfg.onehot_channel_vocab_size} "
            f"to {model_cfg.onehot_channel_vocab_size}; reinitializing channel_id_projector."
        )
    print("✅ Encoder is FROZEN (Linear Probing mode).")
    print(
        f"📝 metadata: mode={channel_metadata_mode}, "
        f"fusion={model_cfg.metadata_fusion_mode}, relation={model_cfg.channel_mixer_relation_mode}"
    )
    print(f"ℹ️  Metadata Fusion Mode: {model_cfg.metadata_fusion_mode}")
    print(f"ℹ️  Channel Mixer Relation Mode: {model_cfg.channel_mixer_relation_mode}")
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
    print(f"ℹ️  Encoder Variant: {model_cfg.encoder_variant}")
    print(f"ℹ️  Channel Adapter: disabled for Laya downstream ({'ci' if channel_mixer_type == 'independent' else 'mixer'})")
    print(f"ℹ️  Channel Relation Block: {'enabled' if model_cfg.use_channel_relation_block else 'disabled'}")
    if model_cfg.use_channel_relation_block:
        print(f"   - relation heads: {model_cfg.channel_relation_heads}")
        print(f"   - gate scale init: {model_cfg.channel_relation_gate_scale_init}")
        print(f"   - residual scale init: {model_cfg.channel_relation_residual_scale_init}")
    print(f"✅ ClassificationHead: [B, {head_input}] → [B, {num_classes}]")
    optimizer = torch.optim.AdamW(finetune_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    if args.use_tensorboard or args.save_attention_maps:
        os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir) if args.use_tensorboard else None
    best_state = None; best_val_loss = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            model.train(); train_loss = 0.0; train_preds=[]; train_ys=[]
            train_metadata_usage = {}
            for batch_idx, batch in enumerate(train_loader):
                x = batch["series"].to(args.device); y = batch["label"].to(args.device)
                pos = batch["channel_positions"].to(args.device); mask = batch["channel_mask"].to(args.device)
                channel_text_embeddings = batch.get("channel_text_embeddings")
                if channel_text_embeddings is not None:
                    channel_text_embeddings = channel_text_embeddings.to(args.device)
                channel_stats_embeddings = batch.get("channel_stats_embeddings")
                if channel_stats_embeddings is not None:
                    channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                if batch_idx == 0:
                    logits, features = model(
                        x,
                        pos,
                        mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                        return_features=True,
                    )
                    train_metadata_usage = summarize_metadata_usage(features)
                else:
                    logits = model(
                        x,
                        pos,
                        mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                    )
                loss = criterion(logits, y); loss.backward(); optimizer.step()
                train_loss += loss.item() * x.size(0)
                train_preds.append(torch.argmax(logits, dim=1).detach().cpu().numpy()); train_ys.append(y.detach().cpu().numpy())
            train_loss /= max(1, len(train_loader.dataset))
            train_y_true = np.concatenate(train_ys); train_y_pred = np.concatenate(train_preds)
            train_acc = accuracy_score(train_y_true, train_y_pred); train_f1 = f1_score(train_y_true, train_y_pred, average="macro")
            if writer is not None:
                writer.add_scalar("train/loss", train_loss, epoch)
                writer.add_scalar("train/accuracy", train_acc, epoch)
                writer.add_scalar("train/f1_macro", train_f1, epoch)
                for key, value in train_metadata_usage.items():
                    writer.add_scalar(f"train_meta/{key}", value, epoch)
            model.eval(); preds=[]; ys=[]; val_losses=[]
            val_metadata_usage = {}
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    x = batch["series"].to(args.device); y = batch["label"].to(args.device)
                    pos = batch["channel_positions"].to(args.device); mask = batch["channel_mask"].to(args.device)
                    channel_text_embeddings = batch.get("channel_text_embeddings")
                    if channel_text_embeddings is not None:
                        channel_text_embeddings = channel_text_embeddings.to(args.device)
                    channel_stats_embeddings = batch.get("channel_stats_embeddings")
                    if channel_stats_embeddings is not None:
                        channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                    if batch_idx == 0:
                        logits, features = model(
                            x,
                            pos,
                            mask,
                            channel_text_embeddings=channel_text_embeddings,
                            channel_stats_embeddings=channel_stats_embeddings,
                            return_features=True,
                        )
                        val_metadata_usage = summarize_metadata_usage(features)
                    else:
                        logits = model(
                            x,
                            pos,
                            mask,
                            channel_text_embeddings=channel_text_embeddings,
                            channel_stats_embeddings=channel_stats_embeddings,
                        )
                    val_losses.append(criterion(logits, y).item() * x.size(0))
                    preds.append(torch.argmax(logits, dim=1).cpu().numpy()); ys.append(y.cpu().numpy())
            y_true = np.concatenate(ys); y_pred = np.concatenate(preds)
            val_loss = float(np.sum(val_losses) / max(1, len(val_loader.dataset)))
            val_acc = accuracy_score(y_true, y_pred); val_f1 = f1_score(y_true, y_pred, average="macro")
            if writer is not None:
                writer.add_scalar("val/loss", val_loss, epoch)
                writer.add_scalar("val/accuracy", val_acc, epoch); writer.add_scalar("val/f1_macro", val_f1, epoch)
                for key, value in val_metadata_usage.items():
                    writer.add_scalar(f"val_meta/{key}", value, epoch)
            print(
                f"Epoch {epoch:>3}/{args.epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} | "
                f"Train {_format_metadata_usage(train_metadata_usage)} | "
                f"Val {_format_metadata_usage(val_metadata_usage)}"
            )
            if val_loss <= best_val_loss:
                previous_best = best_val_loss
                best_val_loss = val_loss; best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
                if math.isinf(previous_best):
                    print(f"Validation Loss decreased (inf --> {val_loss:.6f}). Saving best model...")
                else:
                    print(f"Validation Loss decreased ({previous_best:.6f} --> {val_loss:.6f}). Saving best model...")
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval(); preds=[]; ys=[]; attention_saved = 0; processed_test_samples = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch["series"].to(args.device); y = batch["label"].to(args.device)
                pos = batch["channel_positions"].to(args.device); mask = batch["channel_mask"].to(args.device)
                channel_text_embeddings = batch.get("channel_text_embeddings")
                if channel_text_embeddings is not None:
                    channel_text_embeddings = channel_text_embeddings.to(args.device)
                channel_stats_embeddings = batch.get("channel_stats_embeddings")
                if channel_stats_embeddings is not None:
                    channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                if args.save_attention_maps and attention_saved < args.num_attention_map_samples:
                    batch_channel_names = _extract_channel_names(batch.get("channel_names")) or channel_names
                    logits, features = model(
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
                        channel_names=batch_channel_names,
                        output_dir=attention_dir,
                        sample_offset=processed_test_samples,
                        max_samples=args.num_attention_map_samples - attention_saved,
                        dataset_name=Path(args.data_root).name,
                    )
                    attention_saved += newly_saved
                else:
                    logits = model(
                        x,
                        pos,
                        mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                    )
                preds.append(torch.argmax(logits, dim=1).cpu().numpy()); ys.append(y.cpu().numpy())
                processed_test_samples += x.shape[0]
        y_true = np.concatenate(ys); y_pred = np.concatenate(preds)
        test_acc = float(accuracy_score(y_true, y_pred))
        test_f1 = float(f1_score(y_true, y_pred, average="macro"))
        print()
        print(f"Classification Test Result ({Path(args.data_root).name})")
        print(f"Accuracy : {test_acc:.6f}")
        print(f"F1 (macro): {test_f1:.6f}")
        if args.save_attention_maps:
            print(f"Saved attention maps for {attention_saved} sample(s) under {os.path.join(args.log_dir, 'attention_maps')}")
    finally:
        if writer is not None:
            writer.close()


if __name__ == "__main__":
    main()
