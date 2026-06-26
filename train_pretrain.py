from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
    if str(WORKSPACE_ROOT) not in sys.path:
        sys.path.insert(0, str(WORKSPACE_ROOT))

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from laya_ts.config import LayaModelConfig, TrainingConfig, normalize_variant_name

if __package__ in {None, ""}:
    from laya_ts.config import PretrainConfig
    from laya_ts.data_pretrain import (
        get_lotsa_pretrain_loader_groups,
        get_pretrain_loaders,
        get_tsld_pretrain_loader_groups,
        get_tslib_pretrain_loader_groups,
    )
    from laya_ts.model import LayaTSPretrainer
else:
    from .config import PretrainConfig
    from .data_pretrain import (
        get_lotsa_pretrain_loader_groups,
        get_pretrain_loaders,
        get_tsld_pretrain_loader_groups,
        get_tslib_pretrain_loader_groups,
    )
    from .model import LayaTSPretrainer


def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


def _parse_int_list(raw_value: str) -> tuple[int, ...]:
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


def build_scheduler(optimizer: AdamW, training: TrainingConfig, total_steps: int, warmup_steps: int) -> LambdaLR:
    min_ratio = training.min_lr / training.lr
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = min(1.0, (step - warmup_steps) / float(max(1, total_steps - warmup_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def _dataset_sample_count(dataset) -> int:
    num_samples = getattr(dataset, "num_samples", None)
    if num_samples is not None:
        return int(num_samples)
    return len(dataset)


def _accumulate_metadata_usage(
    accumulator: dict[str, float],
    metadata_usage: dict[str, float] | None,
    *,
    weight: int,
) -> None:
    if not metadata_usage:
        return
    for key, value in metadata_usage.items():
        accumulator[key] = accumulator.get(key, 0.0) + (float(value) * weight)


def _average_metadata_usage(accumulator: dict[str, float], total_weight: int) -> dict[str, float]:
    if total_weight <= 0:
        return {}
    return {key: value / float(total_weight) for key, value in accumulator.items()}


def _extract_scale_weights(metadata_usage: dict[str, float] | None) -> dict[str, float]:
    if not metadata_usage:
        return {}
    scale_weights: dict[str, float] = {}
    for key, value in metadata_usage.items():
        if key.startswith("scale_weight_p"):
            scale_weights[key[len("scale_weight_"):]] = float(value)
    return scale_weights


def _accumulate_named_scale_weights(
    accumulator: dict[str, float],
    scale_weights: dict[str, float] | None,
    *,
    weight: int,
) -> None:
    if not scale_weights:
        return
    for scale_name, value in scale_weights.items():
        accumulator[scale_name] = accumulator.get(scale_name, 0.0) + (float(value) * weight)


def _average_named_scale_weights(
    accumulator: dict[str, float],
    total_weight: int,
) -> dict[str, float]:
    if total_weight <= 0:
        return {}
    return {scale_name: value / float(total_weight) for scale_name, value in accumulator.items()}


def _accumulate_dataset_scale_weights(
    accumulator: dict[str, dict[str, float] | int],
    dataset_name: str | None,
    scale_weights: dict[str, float] | None,
    *,
    weight: int,
) -> None:
    if not dataset_name or not scale_weights:
        return
    bucket = accumulator.setdefault(dataset_name, {"total_weight": 0})
    bucket["total_weight"] = int(bucket.get("total_weight", 0)) + weight
    for scale_name, value in scale_weights.items():
        bucket[scale_name] = float(bucket.get(scale_name, 0.0)) + (float(value) * weight)


def _average_dataset_scale_weights(
    accumulator: dict[str, dict[str, float] | int],
) -> dict[str, dict[str, float]]:
    averaged: dict[str, dict[str, float]] = {}
    for dataset_name, bucket in accumulator.items():
        total_weight = int(bucket.get("total_weight", 0))
        if total_weight <= 0:
            continue
        averaged[dataset_name] = {
            scale_name: float(value) / float(total_weight)
            for scale_name, value in bucket.items()
            if scale_name != "total_weight"
        }
    return averaged


def _resolve_dataset_log_name(batch: dict[str, object] | None, fallback_name: str | None = None) -> str | None:
    if batch is not None:
        for field_name in ("subset_name", "dataset_name"):
            raw_value = batch.get(field_name)
            if isinstance(raw_value, str) and raw_value.strip():
                return raw_value.strip()
            if isinstance(raw_value, (list, tuple)) and raw_value:
                first_value = raw_value[0]
                if isinstance(first_value, str) and first_value.strip():
                    return first_value.strip()
    if fallback_name is None:
        return None
    fallback = str(fallback_name).strip()
    return fallback or None


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
    if "metadata_norm_std" in metadata_usage:
        parts.append(f"MetaNormσ:{metadata_usage['metadata_norm_std']:.4f}")
    if "metadata_norm_ratio" in metadata_usage:
        parts.append(f"MetaNormMax/Min:{metadata_usage['metadata_norm_ratio']:.2f}")
    for scale_name in ("p4", "p8", "p16", "p32"):
        key = f"scale_weight_{scale_name}"
        if key in metadata_usage:
            parts.append(f"{scale_name}μ:{metadata_usage[key]:.4f}")
    return "Meta: " + ", ".join(parts)


def _infer_loader_channel_count(loader) -> int:
    dataset_channels = getattr(loader.dataset, "num_channels", None)
    if dataset_channels is not None:
        return int(dataset_channels)
    probe_batch = next(iter(loader))
    return int(probe_batch["series"].shape[1])


def _iter_interleaved_loader_groups(loader_groups, *, epoch: int, seed: int, target_steps: int | None = None, subset_sampling: str = "exhaustive"):
    if subset_sampling in {"uniform", "official"} and target_steps is not None and target_steps > 0:
        rng = random.Random(seed + epoch)
        states = [
            {
                "group_name": group["group_name"],
                "subset_weight": float(group.get("subset_weight", 0.0)),
                "loader": group["train_loader"],
                "iterator": iter(group["train_loader"]),
            }
            for group in loader_groups
        ]
        if subset_sampling == "official":
            weights = [max(0.0, state["subset_weight"]) for state in states]
            weight_total = sum(weights)
            if weight_total > 0.0:
                cumulative_weights: list[float] | None = []
                running_total = 0.0
                for weight in weights:
                    running_total += weight
                    cumulative_weights.append(running_total)
            else:
                cumulative_weights = None
        else:
            cumulative_weights = None
        for _ in range(target_steps):
            if cumulative_weights is None:
                state = states[rng.randrange(len(states))]
            else:
                draw = rng.random() * cumulative_weights[-1]
                state = states[0]
                for index, threshold in enumerate(cumulative_weights):
                    if draw <= threshold:
                        state = states[index]
                        break
            try:
                batch = next(state["iterator"])
            except StopIteration:
                state["iterator"] = iter(state["loader"])
                batch = next(state["iterator"])
            yield state["group_name"], batch
        return

    active = [
        {
            "group_name": group["group_name"],
            "loader": group["train_loader"],
            "iterator": iter(group["train_loader"]),
        }
        for group in loader_groups
    ]
    rng = random.Random(seed + epoch)
    while active:
        rng.shuffle(active)
        next_active = []
        for state in active:
            try:
                batch = next(state["iterator"])
            except StopIteration:
                continue
            yield state["group_name"], batch
            next_active.append(state)
        active = next_active


def _sanitize_metric_name(name: str) -> str:
    return str(name).replace("/", "_").replace(" ", "_")


def move_batch_to_device(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor | None]:
    channel_text_embeddings = batch.get("channel_text_embeddings")
    channel_stats_embeddings = batch.get("channel_stats_embeddings")
    non_blocking = torch.cuda.is_available() and str(device).startswith("cuda")
    return {
        "series": batch["series"].to(device, non_blocking=non_blocking),
        "channel_positions": batch["channel_positions"].to(device, non_blocking=non_blocking),
        "channel_mask": batch["channel_mask"].to(device, non_blocking=non_blocking),
        "channel_text_embeddings": None if channel_text_embeddings is None else channel_text_embeddings.to(device, non_blocking=non_blocking),
        "channel_stats_embeddings": None if channel_stats_embeddings is None else channel_stats_embeddings.to(device, non_blocking=non_blocking),
    }


def _ensure_finite_batch(batch: dict[str, torch.Tensor | None], *, epoch: int, step: int) -> None:
    for key in ("series", "channel_positions", "channel_text_embeddings", "channel_stats_embeddings"):
        value = batch.get(key)
        if value is not None and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite batch tensor detected for '{key}' at epoch={epoch}, step={step}.")


def _ensure_finite_outputs(outputs: dict[str, torch.Tensor], *, epoch: int, step: int) -> None:
    for key in ("loss", "pred_loss", "sigreg_loss", "query_loss", "mixed_repr", "pred_tokens", "target_tokens"):
        value = outputs.get(key)
        if value is not None and not torch.isfinite(value).all():
            raise ValueError(f"Non-finite model output detected for '{key}' at epoch={epoch}, step={step}.")


def _print_series_summaries(title: str, summaries: Iterable[dict[str, object]]) -> None:
    summaries = list(summaries)
    if not summaries:
        return
    if title:
        print(title)
    for summary in summaries:
        split_t, split_c = summary["split_shape"]
        print(
            f"   - {summary['file_name']}: split_shape=[T={split_t}, C={split_c}] "
            f"windows={summary['num_windows']}"
        )


def _save_attention_maps(
    *,
    features: dict,
    channel_names: list[str],
    output_dir: str,
    epoch: int,
    prefix: str,
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

    sample_dir = os.path.join(output_dir, f"epoch_{epoch:03d}_{prefix}")
    os.makedirs(sample_dir, exist_ok=True)
    payload = {}
    if relation is not None:
        sample_relation = relation[0]
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
            plt.title(f"{prefix} relation head={head_idx} epoch={epoch} (patch-avg)")
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
            plt.title(f"{prefix} relation head={head_idx} epoch={epoch} top-channels (patch-avg)")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f"relation_head_{head_idx:02d}_topk.png"), bbox_inches="tight")
            plt.close()
            _write_topk_lines(os.path.join(sample_dir, f"relation_head_{head_idx:02d}_topk.txt"), head_matrix, labels, prefix="channel_")
    if mixer is not None:
        sample_mixer = mixer[0].mean(dim=0)
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
            plt.title(f"{prefix} mixer head={head_idx} epoch={epoch}")
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
            plt.title(f"{prefix} mixer head={head_idx} epoch={epoch} top-channels")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, f"mixer_head_{head_idx:02d}_topk.png"), bbox_inches="tight")
            plt.close()
            _write_topk_lines(os.path.join(sample_dir, f"mixer_head_{head_idx:02d}_topk.txt"), head_matrix, labels, prefix="q")
    if payload:
        torch.save(payload, os.path.join(sample_dir, "affinities.pt"))
        saved += 1
    return saved


def _representation_stats(repr_tensor: torch.Tensor) -> dict[str, float]:
    if repr_tensor.dim() != 2:
        raise ValueError(f"Expected representation tensor [B, D], got {tuple(repr_tensor.shape)}")

    repr_cpu = repr_tensor.detach().float().cpu()
    norms = repr_cpu.norm(dim=-1)
    feature_var_per_dim = repr_cpu.var(dim=0, unbiased=False)
    feature_var = feature_var_per_dim.mean()
    feature_var_min = feature_var_per_dim.min() if feature_var_per_dim.numel() > 0 else repr_cpu.new_tensor(0.0)
    feature_var_max = feature_var_per_dim.max() if feature_var_per_dim.numel() > 0 else repr_cpu.new_tensor(0.0)
    low_variance_eps = 1e-8
    dead_dim_count = int((feature_var_per_dim <= low_variance_eps).sum().item()) if feature_var_per_dim.numel() > 0 else 0
    dead_dim_fraction = float(dead_dim_count / max(1, feature_var_per_dim.numel()))
    norm_ratio = float(norms.max() / norms.clamp_min(1e-12).min())

    centered = repr_cpu - repr_cpu.mean(dim=0, keepdim=True)
    max_rank = min(centered.shape[0], centered.shape[1])
    if max_rank > 0:
        singular_values = torch.linalg.svdvals(centered)
        singular_sum = singular_values.sum()
        if singular_values.numel() > 0 and float(singular_sum) > 0.0:
            singular_probs = singular_values / singular_sum
            singular_entropy = -torch.sum(singular_probs * singular_probs.clamp_min(1e-12).log())
            effective_rank = float(torch.exp(singular_entropy))
            stable_rank = float((singular_values.square().sum() / singular_values.max().square().clamp_min(1e-12)))
            singular_top1_fraction = float(singular_values[0] / singular_sum)
        else:
            effective_rank = 0.0
            stable_rank = 0.0
            singular_top1_fraction = 0.0
        effective_rank_ratio = float(effective_rank / max_rank)
    else:
        effective_rank = 0.0
        effective_rank_ratio = 0.0
        stable_rank = 0.0
        singular_top1_fraction = 0.0

    if repr_cpu.shape[0] > 1:
        pairwise_l2 = torch.pdist(repr_cpu, p=2)
        normalized = torch.nn.functional.normalize(repr_cpu, dim=-1)
        cosine_sim = normalized @ normalized.transpose(0, 1)
        upper = torch.triu_indices(cosine_sim.shape[0], cosine_sim.shape[1], offset=1)
        pairwise_cos_sim = cosine_sim[upper[0], upper[1]]
        pairwise_cos = 1.0 - cosine_sim[upper[0], upper[1]]
        pairwise_l2_mean = float(pairwise_l2.mean()) if pairwise_l2.numel() > 0 else 0.0
        pairwise_l2_std = float(pairwise_l2.std(unbiased=False)) if pairwise_l2.numel() > 0 else 0.0
        pairwise_cos_sim_mean = float(pairwise_cos_sim.mean()) if pairwise_cos_sim.numel() > 0 else 0.0
        pairwise_cos_sim_std = float(pairwise_cos_sim.std(unbiased=False)) if pairwise_cos_sim.numel() > 0 else 0.0
        pairwise_cos_mean = float(pairwise_cos.mean()) if pairwise_cos.numel() > 0 else 0.0
        pairwise_cos_std = float(pairwise_cos.std(unbiased=False)) if pairwise_cos.numel() > 0 else 0.0
    else:
        pairwise_l2_mean = 0.0
        pairwise_l2_std = 0.0
        pairwise_cos_sim_mean = 1.0
        pairwise_cos_sim_std = 0.0
        pairwise_cos_mean = 0.0
        pairwise_cos_std = 0.0

    return {
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std(unbiased=False)),
        "norm_ratio": norm_ratio,
        "feature_var_mean": float(feature_var),
        "feature_var_min": float(feature_var_min),
        "feature_var_max": float(feature_var_max),
        "feature_var_ratio": float(feature_var_max / feature_var_min.clamp_min(low_variance_eps)),
        "dead_dim_count": float(dead_dim_count),
        "dead_dim_fraction": dead_dim_fraction,
        "effective_rank": effective_rank,
        "effective_rank_ratio": effective_rank_ratio,
        "stable_rank": stable_rank,
        "singular_top1_fraction": singular_top1_fraction,
        "pairwise_l2_mean": pairwise_l2_mean,
        "pairwise_l2_std": pairwise_l2_std,
        "pairwise_cosine_similarity_mean": pairwise_cos_sim_mean,
        "pairwise_cosine_similarity_std": pairwise_cos_sim_std,
        "pairwise_cosine_distance_mean": pairwise_cos_mean,
        "pairwise_cosine_distance_std": pairwise_cos_std,
    }


def _empty_representation_stats() -> dict[str, float]:
    return {
        "norm_mean": 0.0,
        "norm_std": 0.0,
        "norm_ratio": 0.0,
        "feature_var_mean": 0.0,
        "feature_var_min": 0.0,
        "feature_var_max": 0.0,
        "feature_var_ratio": 0.0,
        "dead_dim_count": 0.0,
        "dead_dim_fraction": 0.0,
        "effective_rank": 0.0,
        "effective_rank_ratio": 0.0,
        "stable_rank": 0.0,
        "singular_top1_fraction": 0.0,
        "pairwise_l2_mean": 0.0,
        "pairwise_l2_std": 0.0,
        "pairwise_cosine_similarity_mean": 0.0,
        "pairwise_cosine_similarity_std": 0.0,
        "pairwise_cosine_distance_mean": 0.0,
        "pairwise_cosine_distance_std": 0.0,
    }


def _prediction_alignment_stats(pred_tokens: torch.Tensor, target_tokens: torch.Tensor, patch_mask: torch.Tensor) -> dict[str, float]:
    pred_masked = pred_tokens[patch_mask]
    target_masked = target_tokens[patch_mask]
    if pred_masked.numel() == 0 or target_masked.numel() == 0:
        return {
            "cosine_similarity_mean": 0.0,
            "cosine_similarity_std": 0.0,
            "cosine_distance_mean": 0.0,
            "cosine_distance_std": 0.0,
            "l2_mean": 0.0,
            "l2_std": 0.0,
        }

    cosine = torch.nn.functional.cosine_similarity(pred_masked.detach().float(), target_masked.detach().float(), dim=-1)
    distances = (pred_masked.detach().float() - target_masked.detach().float()).norm(dim=-1)
    return {
        "cosine_similarity_mean": float(cosine.mean().cpu()),
        "cosine_similarity_std": float(cosine.std(unbiased=False).cpu()),
        "cosine_distance_mean": float((1.0 - cosine).mean().cpu()),
        "cosine_distance_std": float((1.0 - cosine).std(unbiased=False).cpu()),
        "l2_mean": float(distances.mean().cpu()),
        "l2_std": float(distances.std(unbiased=False).cpu()),
    }


def _empty_prediction_alignment_stats() -> dict[str, float]:
    return {
        "cosine_similarity_mean": 0.0,
        "cosine_similarity_std": 0.0,
        "cosine_distance_mean": 0.0,
        "cosine_distance_std": 0.0,
        "l2_mean": 0.0,
        "l2_std": 0.0,
    }


def _evaluate_validation(
    *,
    model: torch.nn.Module,
    train_cfg: TrainingConfig,
    val_loader,
    loader_groups,
    args,
    epoch: int,
    global_step: int,
):
    total_val_samples = 0
    val_loss = 0.0
    val_pred_loss = 0.0
    val_sigreg_loss = 0.0
    val_query_loss = 0.0
    attention_batch = None
    val_repr_stats = _empty_representation_stats()
    val_align_stats = _empty_prediction_alignment_stats()
    val_metadata_usage: dict[str, float] = {}
    val_scale_weight_totals: dict[str, float] = {}
    val_scale_weight_by_dataset: dict[str, dict[str, float] | int] = {}
    per_dataset_metrics = []

    iter_val_loaders = (
        [{"group_name": "val", "val_loader": val_loader}]
        if loader_groups is None
        else [{"group_name": group["group_name"], "val_loader": group["val_loader"]} for group in loader_groups]
    )

    with torch.no_grad():
        for group in iter_val_loaders:
            if group["val_loader"] is None:
                continue
            dataset_samples = 0
            dataset_loss = 0.0
            dataset_pred_loss = 0.0
            dataset_sigreg_loss = 0.0
            dataset_query_loss = 0.0
            dataset_scale_weight_totals: dict[str, float] = {}

            for batch in group["val_loader"]:
                batch_on_device = move_batch_to_device(batch, train_cfg.device)
                _ensure_finite_batch(batch_on_device, epoch=epoch, step=global_step)
                batch_size = int(batch_on_device["series"].shape[0])
                need_attention = args.save_attention_maps and attention_batch is None
                outputs = model(
                    batch_on_device["series"],
                    channel_positions=batch_on_device["channel_positions"],
                    channel_mask=batch_on_device["channel_mask"],
                    channel_text_embeddings=batch_on_device["channel_text_embeddings"],
                    channel_stats_embeddings=batch_on_device["channel_stats_embeddings"],
                    return_aux=need_attention,
                    patch_mask_seed=0,
                )
                _ensure_finite_outputs(outputs, epoch=epoch, step=global_step)

                dataset_samples += batch_size
                total_val_samples += batch_size
                batch_loss = outputs["loss"].item()
                batch_pred_loss = outputs["pred_loss"].item()
                batch_sigreg_loss = outputs["sigreg_loss"].item()
                batch_query_loss = outputs["query_loss"].item()
                dataset_loss += batch_loss * batch_size
                dataset_pred_loss += batch_pred_loss * batch_size
                dataset_sigreg_loss += batch_sigreg_loss * batch_size
                dataset_query_loss += batch_query_loss * batch_size
                val_loss += batch_loss * batch_size
                val_pred_loss += batch_pred_loss * batch_size
                val_sigreg_loss += batch_sigreg_loss * batch_size
                val_query_loss += batch_query_loss * batch_size
                repr_stats = _representation_stats(outputs["mixed_repr"])
                align_stats = _prediction_alignment_stats(outputs["pred_tokens"], outputs["target_tokens"], outputs["patch_mask"])
                for key, value in repr_stats.items():
                    val_repr_stats[key] += value * batch_size
                for key, value in align_stats.items():
                    val_align_stats[key] += value * batch_size
                _accumulate_metadata_usage(
                    val_metadata_usage,
                    outputs.get("metadata_usage"),
                    weight=batch_size,
                )
                scale_weights = _extract_scale_weights(outputs.get("metadata_usage"))
                _accumulate_named_scale_weights(
                    val_scale_weight_totals,
                    scale_weights,
                    weight=batch_size,
                )
                dataset_log_name = _resolve_dataset_log_name(batch, group["group_name"])
                _accumulate_named_scale_weights(
                    dataset_scale_weight_totals,
                    scale_weights,
                    weight=batch_size,
                )
                _accumulate_dataset_scale_weights(
                    val_scale_weight_by_dataset,
                    dataset_log_name,
                    scale_weights,
                    weight=batch_size,
                )
                if need_attention:
                    raw_names = batch.get("channel_names", [])
                    channel_names = list(raw_names[0]) if raw_names else []
                    attention_batch = (outputs["full_features"], channel_names)

            if dataset_samples > 0:
                per_dataset_metrics.append(
                    {
                        "group_name": group["group_name"],
                        "num_samples": dataset_samples,
                        "loss": dataset_loss / dataset_samples,
                        "pred_loss": dataset_pred_loss / dataset_samples,
                        "sigreg_loss": dataset_sigreg_loss / dataset_samples,
                        "query_loss": dataset_query_loss / dataset_samples,
                        "scale_weights": _average_named_scale_weights(
                            dataset_scale_weight_totals,
                            dataset_samples,
                        ),
                    }
                )

    avg_val_loss = val_loss / max(1, total_val_samples)
    avg_val_pred_loss = val_pred_loss / max(1, total_val_samples)
    avg_val_sigreg_loss = val_sigreg_loss / max(1, total_val_samples)
    avg_val_query_loss = val_query_loss / max(1, total_val_samples)
    avg_val_repr_stats = {key: value / max(1, total_val_samples) for key, value in val_repr_stats.items()}
    avg_val_align_stats = {key: value / max(1, total_val_samples) for key, value in val_align_stats.items()}
    avg_val_metadata_usage = _average_metadata_usage(val_metadata_usage, total_val_samples)
    avg_val_scale_weights = _average_named_scale_weights(val_scale_weight_totals, total_val_samples)
    avg_val_scale_weights_by_dataset = _average_dataset_scale_weights(val_scale_weight_by_dataset)

    if per_dataset_metrics:
        macro_val_loss = float(sum(metric["loss"] for metric in per_dataset_metrics) / len(per_dataset_metrics))
        macro_val_pred_loss = float(sum(metric["pred_loss"] for metric in per_dataset_metrics) / len(per_dataset_metrics))
        macro_val_sigreg_loss = float(sum(metric["sigreg_loss"] for metric in per_dataset_metrics) / len(per_dataset_metrics))
        macro_val_query_loss = float(sum(metric["query_loss"] for metric in per_dataset_metrics) / len(per_dataset_metrics))
    else:
        macro_val_loss = avg_val_loss
        macro_val_pred_loss = avg_val_pred_loss
        macro_val_sigreg_loss = avg_val_sigreg_loss
        macro_val_query_loss = avg_val_query_loss

    return {
        "micro_loss": avg_val_loss,
        "micro_pred_loss": avg_val_pred_loss,
        "micro_sigreg_loss": avg_val_sigreg_loss,
        "micro_query_loss": avg_val_query_loss,
        "macro_loss": macro_val_loss,
        "macro_pred_loss": macro_val_pred_loss,
        "macro_sigreg_loss": macro_val_sigreg_loss,
        "macro_query_loss": macro_val_query_loss,
        "total_val_samples": total_val_samples,
        "repr_stats": avg_val_repr_stats,
        "align_stats": avg_val_align_stats,
        "metadata_usage": avg_val_metadata_usage,
        "scale_weights": avg_val_scale_weights,
        "scale_weights_by_dataset": avg_val_scale_weights_by_dataset,
        "per_dataset_metrics": per_dataset_metrics,
        "attention_batch": attention_batch,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="Pretrain standalone Laya on time-series data")
    p.add_argument(
        "--dataset_type",
        type=str,
        default="tslib",
        choices=["tsld", "tslib", "electricity", "lotsa", "ETTm1", "ETTm2", "weather"],
    )
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--patch_size", type=int, default=PretrainConfig().patch_size)
    p.add_argument("--model_id", type=str, default=LayaModelConfig().model_id)
    p.add_argument("--patchifier_mode", type=str, default=LayaModelConfig().patchifier_mode, choices=["single", "multiscale", "trend_seasonal"])
    p.add_argument("--multiscale_patch_sizes", type=str, default=",".join(str(value) for value in LayaModelConfig().multiscale_patch_sizes))
    p.add_argument("--multiscale_base_patch", type=int, default=LayaModelConfig().multiscale_base_patch)
    p.add_argument("--multiscale_gate_temperature", type=float, default=LayaModelConfig().multiscale_gate_temperature)
    p.add_argument("--trend_seasonal_kernel", type=int, default=LayaModelConfig().trend_seasonal_kernel)
    p.add_argument("--trend_seasonal_gate_temperature", type=float, default=LayaModelConfig().trend_seasonal_gate_temperature)
    p.add_argument("--d_model", type=int, default=LayaModelConfig().embed_dim)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--tsld_mode", type=str, default=getattr(PretrainConfig(), "tsld_mode", "univariate"), choices=["univariate", "multivariate"])
    p.add_argument("--tslib_mode", type=str, default=PretrainConfig().tslib_mode, choices=["univariate", "multivariate"])
    p.add_argument("--max_files", type=int, default=None)
    p.add_argument("--variant", type=str, default="s", choices=["s", "b", "laya-s", "laya-b"])
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--epochs", type=int, default=PretrainConfig().epochs)
    p.add_argument("--warmup_epochs", type=int, default=PretrainConfig().warmup_epochs)
    p.add_argument("--n_heads", type=int, default=LayaModelConfig().num_heads)
    p.add_argument("--n_layers", type=int, default=LayaModelConfig().depth)
    p.add_argument("--proj_dim", type=int, default=LayaModelConfig().proj_dim)
    p.add_argument("--predictor_depth", type=int, default=LayaModelConfig().predictor_depth)
    p.add_argument("--predictor_heads", type=int, default=LayaModelConfig().predictor_heads)
    p.add_argument("--onehot_channel_vocab_size", type=int, default=PretrainConfig().onehot_channel_vocab_size)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--val_interval", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=5e-2)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--channel_metadata_mode", type=str, default="onehot", choices=["onehot", "text", "stats", "text_stats_joint", "text_stats_avg", "none"])
    p.add_argument("--metadata_fusion_mode", type=str, default=LayaModelConfig().metadata_fusion_mode, choices=["none", "add", "concat_kv", "attention_gate", "attention_suppress_gate"])
    p.add_argument("--channel_mixer_type", type=str, default="mixer", choices=["mixer", "independent", "ci_adapter"])
    p.add_argument("--channel_mixer_relation_mode", type=str, default=LayaModelConfig().channel_mixer_relation_mode, choices=["none", "laya_relation", "metadata_query_gate", "metadata_query_bias", "description_relation"])
    p.add_argument("--channel_mixer_relation_scale_init", type=float, default=LayaModelConfig().channel_mixer_relation_scale_init)
    _add_bool_optional_arg(p, "--use_relation_adapter", default=LayaModelConfig().use_relation_adapter)
    p.add_argument("--relation_num_heads", type=int, default=LayaModelConfig().relation_num_heads)
    p.add_argument("--relation_dropout", type=float, default=LayaModelConfig().relation_dropout)
    p.add_argument("--relation_scale_init", type=float, default=LayaModelConfig().relation_scale_init)
    _add_bool_optional_arg(p, "--use_metadata_bias", default=LayaModelConfig().use_metadata_bias)
    _add_bool_optional_arg(p, "--use_metadata_gate", default=LayaModelConfig().use_metadata_gate)
    p.add_argument("--metadata_scale_init", type=float, default=LayaModelConfig().metadata_scale_init)
    p.add_argument("--metadata_dropout", type=float, default=LayaModelConfig().metadata_dropout)
    p.add_argument("--relation_adapter_position", type=str, default=LayaModelConfig().relation_adapter_position, choices=["post_encoder"])
    p.add_argument("--description_relation_num_latents", type=int, default=LayaModelConfig().description_relation_num_latents)
    p.add_argument("--description_relation_metric", type=str, default=LayaModelConfig().description_relation_metric, choices=["projected_dot", "cosine"])
    p.add_argument("--description_relation_lambda_init", type=float, default=LayaModelConfig().description_relation_lambda_init)
    p.add_argument("--description_relation_gamma_init", type=float, default=LayaModelConfig().description_relation_gamma_init)
    p.add_argument("--use_channel_relation_block", action="store_true")
    p.add_argument("--channel_relation_heads", type=int, default=1)
    p.add_argument("--channel_relation_gate_scale_init", type=float, default=0.01)
    p.add_argument("--channel_relation_residual_scale_init", type=float, default=0.05)
    p.add_argument("--encoder_variant", type=str, default=LayaModelConfig().encoder_variant, choices=["default"])
    p.add_argument("--temporal_patchifier_mode", type=str, default=LayaModelConfig().temporal_patchifier_mode, choices=["fixed", "multiscale", "charm_like"])
    p.add_argument("--charm_kernel_sizes", type=str, default=",".join(str(value) for value in LayaModelConfig().charm_kernel_sizes))
    p.add_argument("--charm_stride", type=int, default=0, help="0 uses patch_size for CHARM-like patchification")
    p.add_argument("--charm_patchifier_dropout", type=float, default=LayaModelConfig().charm_patchifier_dropout)
    p.add_argument("--charm_scale_gate_source", type=str, default=LayaModelConfig().charm_scale_gate_source, choices=["learned", "text"])
    p.add_argument("--charm_scale_gate_temperature", type=float, default=LayaModelConfig().charm_scale_gate_temperature)
    p.add_argument("--charm_patchifier_fusion", type=str, default=LayaModelConfig().charm_patchifier_fusion, choices=["replace", "residual"])
    p.add_argument("--charm_patchifier_residual_init", type=float, default=LayaModelConfig().charm_patchifier_residual_init)
    p.add_argument("--text_encoder_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--text_metadata_cache_dir", type=str, default="./metadata_cache")
    p.add_argument("--text_encoder_local_files_only", action="store_true")
    p.add_argument("--lotsa_dataset_path", type=str, default="Salesforce/lotsa_data")
    p.add_argument("--lotsa_split_mode", type=str, default="temporal_90_10", choices=["official", "temporal_90_10", "temporal_70_10_20"])
    p.add_argument("--lotsa_sampling_mode", type=str, default="official", choices=["official", "sliding_window", "hierarchical"])
    p.add_argument("--lotsa_preprocessing_mode", type=str, default="official", choices=["official", "standardize"])
    p.add_argument("--lotsa_sample_time_series", type=str, default="proportional", choices=["none", "uniform", "proportional"])
    p.add_argument("--lotsa_subset_sampling", type=str, default="exhaustive", choices=["exhaustive", "uniform", "official"])
    p.add_argument("--lotsa_min_patches", type=int, default=2)
    p.add_argument("--lotsa_max_channel", "--lotsa_max_dim", dest="lotsa_max_channel", type=int, default=None)
    p.add_argument("--lotsa_windows_per_series", type=int, default=32)
    p.add_argument("--text_metadata_dim", type=int, default=384)
    p.add_argument("--stats_metadata_dim", type=int, default=LayaModelConfig().stats_metadata_dim)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--log_dir", type=str, default="./runs")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--save_attention_maps", action="store_true")
    p.add_argument("--attention_map_tag", type=str, default=None)
    args = p.parse_args(argv)

    if args.epochs <= 0:
        raise ValueError(f"--epochs must be positive, got {args.epochs}")
    if args.patch_size <= 0:
        raise ValueError(f"--patch_size must be positive, got {args.patch_size}")
    if args.patch_size > args.seq_len:
        raise ValueError(f"--patch_size must be <= --seq_len, got patch_size={args.patch_size}, seq_len={args.seq_len}")
    if args.multiscale_base_patch <= 0:
        raise ValueError(
            f"--multiscale_base_patch must be positive, got {args.multiscale_base_patch}"
        )
    if args.multiscale_gate_temperature <= 0:
        raise ValueError(
            "--multiscale_gate_temperature must be positive, "
            f"got {args.multiscale_gate_temperature}"
        )
    if args.trend_seasonal_kernel <= 0:
        raise ValueError(
            f"--trend_seasonal_kernel must be positive, got {args.trend_seasonal_kernel}"
        )
    if args.trend_seasonal_gate_temperature <= 0:
        raise ValueError(
            "--trend_seasonal_gate_temperature must be positive, "
            f"got {args.trend_seasonal_gate_temperature}"
        )
    if args.d_model <= 0:
        raise ValueError(f"--d_model must be positive, got {args.d_model}")
    if args.n_heads <= 0:
        raise ValueError(f"--n_heads must be positive, got {args.n_heads}")
    if args.n_layers <= 0:
        raise ValueError(f"--n_layers must be positive, got {args.n_layers}")
    if args.proj_dim <= 0:
        raise ValueError(f"--proj_dim must be positive, got {args.proj_dim}")
    if args.predictor_depth <= 0:
        raise ValueError(f"--predictor_depth must be positive, got {args.predictor_depth}")
    if args.predictor_heads <= 0:
        raise ValueError(f"--predictor_heads must be positive, got {args.predictor_heads}")
    if args.channel_mixer_relation_scale_init < 0:
        raise ValueError(
            f"--channel_mixer_relation_scale_init must be non-negative, got {args.channel_mixer_relation_scale_init}"
        )
    if args.description_relation_num_latents <= 0:
        raise ValueError(
            f"--description_relation_num_latents must be positive, got {args.description_relation_num_latents}"
        )
    if args.channel_relation_heads <= 0:
        raise ValueError(f"--channel_relation_heads must be positive, got {args.channel_relation_heads}")
    if args.channel_relation_gate_scale_init < 0:
        raise ValueError(f"--channel_relation_gate_scale_init must be non-negative, got {args.channel_relation_gate_scale_init}")
    if args.channel_relation_residual_scale_init < 0:
        raise ValueError(f"--channel_relation_residual_scale_init must be non-negative, got {args.channel_relation_residual_scale_init}")
    if args.charm_stride < 0:
        raise ValueError(f"--charm_stride must be non-negative, got {args.charm_stride}")
    if not 0.0 <= args.charm_patchifier_dropout <= 1.0:
        raise ValueError(
            f"--charm_patchifier_dropout must be in [0, 1], got {args.charm_patchifier_dropout}"
        )
    if args.charm_scale_gate_temperature <= 0:
        raise ValueError(
            f"--charm_scale_gate_temperature must be positive, got {args.charm_scale_gate_temperature}"
        )
    if args.onehot_channel_vocab_size <= 0:
        raise ValueError(f"--onehot_channel_vocab_size must be positive, got {args.onehot_channel_vocab_size}")
    if args.warmup_epochs < 0:
        raise ValueError(f"--warmup_epochs must be non-negative, got {args.warmup_epochs}")
    if args.steps is None and args.warmup_epochs >= args.epochs:
        raise ValueError(f"--warmup_epochs must be smaller than --epochs, got warmup={args.warmup_epochs}, epochs={args.epochs}")
    if args.save_every <= 0:
        raise ValueError(f"--save_every must be positive, got {args.save_every}")
    if args.steps is not None and args.steps <= 0:
        raise ValueError(f"--steps must be positive when provided, got {args.steps}")
    if args.val_interval is not None and args.val_interval <= 0:
        raise ValueError(f"--val_interval must be positive when provided, got {args.val_interval}")
    if args.stats_metadata_dim <= 0:
        raise ValueError(f"--stats_metadata_dim must be positive, got {args.stats_metadata_dim}")
    if args.lotsa_min_patches <= 0:
        raise ValueError(f"--lotsa_min_patches must be positive, got {args.lotsa_min_patches}")
    if args.lotsa_max_channel is not None and args.lotsa_max_channel <= 0:
        raise ValueError(f"--lotsa_max_channel must be positive, got {args.lotsa_max_channel}")
    if args.lotsa_windows_per_series <= 0:
        raise ValueError(f"--lotsa_windows_per_series must be positive, got {args.lotsa_windows_per_series}")

    charm_kernel_sizes = _parse_int_list(args.charm_kernel_sizes)
    multiscale_patch_sizes = _parse_int_list(args.multiscale_patch_sizes)
    charm_stride = args.charm_stride or args.patch_size
    encoder_variant = args.encoder_variant

    set_seed(args.seed)
    variant_name = normalize_variant_name(args.variant)
    train_cfg = TrainingConfig(lr=args.lr, min_lr=args.min_lr, weight_decay=args.weight_decay, num_workers=args.num_workers, device=args.device, batch_size=args.batch_size or TrainingConfig().batch_size)
    loader_groups = None
    loader_group_kind = None
    if args.dataset_type == "tsld" and args.tsld_mode == "multivariate":
        loader_groups = get_tsld_pretrain_loader_groups(args.data_path, batch_size=train_cfg.batch_size, seq_len=args.seq_len, stride=args.stride, num_workers=args.num_workers, max_files=args.max_files, channel_metadata_mode=args.channel_metadata_mode, text_encoder_name_or_path=args.text_encoder_name_or_path, text_metadata_cache_dir=args.text_metadata_cache_dir, text_encoder_local_files_only=args.text_encoder_local_files_only)
        if not loader_groups:
            raise ValueError("No multivariate tsld loader groups were created; check the dataset path and preprocessing configuration.")
        loader_group_kind = "tsld"
        train_loader = loader_groups[0]["train_loader"]
        val_loader = loader_groups[0]["val_loader"]
        steps_per_epoch = sum(len(group["train_loader"]) for group in loader_groups)
        train_sample_count = sum(len(group["train_loader"].dataset) for group in loader_groups)
        val_sample_count = sum(len(group["val_loader"].dataset) for group in loader_groups)
        val_steps = sum(len(group["val_loader"]) for group in loader_groups)
        shared_channel_counts = [group["channel_count"] for group in loader_groups]
        channel_count = max(shared_channel_counts)
    elif args.dataset_type == "tslib" and args.tslib_mode == "multivariate":
        loader_groups = get_tslib_pretrain_loader_groups(args.data_path, batch_size=train_cfg.batch_size, seq_len=args.seq_len, stride=args.stride, num_workers=args.num_workers, max_files=args.max_files, channel_metadata_mode=args.channel_metadata_mode, text_encoder_name_or_path=args.text_encoder_name_or_path, text_metadata_cache_dir=args.text_metadata_cache_dir, text_encoder_local_files_only=args.text_encoder_local_files_only)
        if not loader_groups:
            raise ValueError("No multivariate tslib loader groups were created; check the dataset path and preprocessing configuration.")
        loader_group_kind = "tslib"
        train_loader = loader_groups[0]["train_loader"]
        val_loader = loader_groups[0]["val_loader"]
        steps_per_epoch = sum(len(group["train_loader"]) for group in loader_groups)
        train_sample_count = sum(len(group["train_loader"].dataset) for group in loader_groups)
        val_sample_count = sum(len(group["val_loader"].dataset) for group in loader_groups)
        val_steps = sum(len(group["val_loader"]) for group in loader_groups)
        shared_channel_counts = [group["channel_count"] for group in loader_groups]
        channel_count = max(shared_channel_counts)
    elif args.dataset_type == "lotsa":
        loader_groups = get_lotsa_pretrain_loader_groups(args.data_path, batch_size=train_cfg.batch_size, seq_len=args.seq_len, stride=args.stride, patch_size=args.patch_size, num_workers=args.num_workers, max_files=args.max_files, channel_metadata_mode=args.channel_metadata_mode, text_encoder_name_or_path=args.text_encoder_name_or_path, text_metadata_cache_dir=args.text_metadata_cache_dir, text_encoder_local_files_only=args.text_encoder_local_files_only, lotsa_dataset_path=args.lotsa_dataset_path, lotsa_split_mode=args.lotsa_split_mode, lotsa_sampling_mode=args.lotsa_sampling_mode, lotsa_preprocessing_mode=args.lotsa_preprocessing_mode, lotsa_sample_time_series=args.lotsa_sample_time_series, lotsa_subset_sampling=args.lotsa_subset_sampling, lotsa_min_patches=args.lotsa_min_patches, lotsa_max_channel=args.lotsa_max_channel, lotsa_windows_per_series=args.lotsa_windows_per_series)
        if not loader_groups:
            raise ValueError("No LOTSA loader groups were created; check the subset list and preprocessing configuration.")
        loader_group_kind = "lotsa"
        train_loader = loader_groups[0]["train_loader"]
        val_loader = loader_groups[0]["val_loader"]
        steps_per_epoch = sum(len(group["train_loader"]) for group in loader_groups)
        train_sample_count = sum(_dataset_sample_count(group["train_loader"].dataset) for group in loader_groups)
        val_sample_count = sum(_dataset_sample_count(group["val_loader"].dataset) for group in loader_groups if group["val_loader"] is not None)
        val_steps = sum(len(group["val_loader"]) for group in loader_groups if group["val_loader"] is not None)
        shared_channel_counts = [group["channel_count"] for group in loader_groups]
        channel_count = max(shared_channel_counts)
    else:
        train_loader, val_loader = get_pretrain_loaders(args.dataset_type, args.data_path, batch_size=train_cfg.batch_size, seq_len=args.seq_len, stride=args.stride, patch_size=args.patch_size, num_workers=args.num_workers, max_files=args.max_files, tsld_mode=args.tsld_mode, tslib_mode=args.tslib_mode, channel_metadata_mode=args.channel_metadata_mode, text_encoder_name_or_path=args.text_encoder_name_or_path, text_metadata_cache_dir=args.text_metadata_cache_dir, text_encoder_local_files_only=args.text_encoder_local_files_only, lotsa_dataset_path=args.lotsa_dataset_path, lotsa_split_mode=args.lotsa_split_mode, lotsa_sampling_mode=args.lotsa_sampling_mode, lotsa_preprocessing_mode=args.lotsa_preprocessing_mode, lotsa_sample_time_series=args.lotsa_sample_time_series, lotsa_subset_sampling=args.lotsa_subset_sampling, lotsa_min_patches=args.lotsa_min_patches, lotsa_max_channel=args.lotsa_max_channel, lotsa_windows_per_series=args.lotsa_windows_per_series)
        steps_per_epoch = len(train_loader)
        if steps_per_epoch == 0:
            raise ValueError("train_loader is empty; check the dataset path and preprocessing configuration.")
        train_sample_count = _dataset_sample_count(train_loader.dataset)
        val_sample_count = 0 if val_loader is None else _dataset_sample_count(val_loader.dataset)
        val_steps = 0 if val_loader is None else len(val_loader)
        shared_channel_counts = None
        channel_count = _infer_loader_channel_count(train_loader)
    has_validation = (
        any(group["val_loader"] is not None for group in loader_groups)
        if loader_groups is not None
        else val_loader is not None
    )
    if args.dataset_type == "lotsa":
        total_steps = args.steps if args.steps is not None else args.epochs * steps_per_epoch
        total_epochs = max(1, math.ceil(total_steps / max(1, steps_per_epoch)))
        warmup_steps = max(1, int(math.ceil(total_steps * 0.05)))
        val_interval = args.val_interval if args.val_interval is not None else steps_per_epoch
    else:
        total_epochs = args.epochs
        total_steps = total_epochs * steps_per_epoch
        warmup_steps = args.warmup_epochs * steps_per_epoch
        val_interval = args.val_interval
    onehot_vocab_size = args.onehot_channel_vocab_size if args.channel_metadata_mode == "onehot" else 0
    if args.channel_metadata_mode == "onehot" and channel_count > onehot_vocab_size:
        raise ValueError(
            f"Input channels {channel_count} exceed configured onehot vocab size {onehot_vocab_size}. "
            "Increase --onehot_channel_vocab_size."
        )
    requested_channel_mixer_type = args.channel_mixer_type
    use_relation_adapter = bool(args.use_relation_adapter or requested_channel_mixer_type == "ci_adapter")
    channel_mixer_type = "independent" if requested_channel_mixer_type == "ci_adapter" else requested_channel_mixer_type
    model_cfg = LayaModelConfig(
        variant=args.variant,
        model_id=args.model_id,
        patch_size=args.patch_size,
        patchifier_mode=args.patchifier_mode,
        multiscale_patch_sizes=multiscale_patch_sizes,
        multiscale_base_patch=args.multiscale_base_patch,
        multiscale_gate_temperature=args.multiscale_gate_temperature,
        trend_seasonal_kernel=args.trend_seasonal_kernel,
        trend_seasonal_gate_temperature=args.trend_seasonal_gate_temperature,
        embed_dim=args.d_model,
        depth=args.n_layers,
        num_heads=args.n_heads,
        proj_dim=args.proj_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        channel_metadata_mode=args.channel_metadata_mode,
        metadata_fusion_mode=args.metadata_fusion_mode,
        channel_mixer_type=channel_mixer_type,
        channel_mixer_relation_mode=args.channel_mixer_relation_mode,
        channel_mixer_relation_scale_init=args.channel_mixer_relation_scale_init,
        use_relation_adapter=use_relation_adapter,
        relation_num_heads=args.relation_num_heads,
        relation_dropout=args.relation_dropout,
        relation_scale_init=args.relation_scale_init,
        use_metadata_bias=args.use_metadata_bias,
        use_metadata_gate=args.use_metadata_gate,
        metadata_scale_init=args.metadata_scale_init,
        metadata_dropout=args.metadata_dropout,
        relation_adapter_position=args.relation_adapter_position,
        description_relation_num_latents=args.description_relation_num_latents,
        description_relation_metric=args.description_relation_metric,
        description_relation_lambda_init=args.description_relation_lambda_init,
        description_relation_gamma_init=args.description_relation_gamma_init,
        use_channel_relation_block=args.use_channel_relation_block,
        channel_relation_heads=args.channel_relation_heads,
        channel_relation_gate_scale_init=args.channel_relation_gate_scale_init,
        channel_relation_residual_scale_init=args.channel_relation_residual_scale_init,
        encoder_variant=encoder_variant,
        temporal_patchifier_mode=args.temporal_patchifier_mode,
        charm_kernel_sizes=charm_kernel_sizes,
        charm_stride=charm_stride,
        charm_patchifier_dropout=args.charm_patchifier_dropout,
        charm_scale_gate_source=args.charm_scale_gate_source,
        charm_scale_gate_temperature=args.charm_scale_gate_temperature,
        charm_patchifier_fusion=args.charm_patchifier_fusion,
        charm_patchifier_residual_init=args.charm_patchifier_residual_init,
        onehot_channel_vocab_size=onehot_vocab_size,
        text_metadata_dim=args.text_metadata_dim,
        stats_metadata_dim=args.stats_metadata_dim,
    )
    model = LayaTSPretrainer(model_cfg).to(train_cfg.device)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    optimizer = AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = build_scheduler(optimizer, train_cfg, total_steps, warmup_steps)
    os.makedirs(args.save_dir, exist_ok=True); os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(os.path.join(args.log_dir, f"laya_ts_{variant_name}_{args.dataset_type}"))
    print("📊 Dataset Statistics:")
    print(f"   - Sequence Length (T): {args.seq_len}")
    print(f"   - Train Samples: {train_sample_count}, Steps: {steps_per_epoch}")
    print(f"   - Val Samples: {val_sample_count}, Steps: {val_steps}")
    if loader_group_kind == "tsld":
        print(f"   - Multivariate channel groups: {shared_channel_counts}")
        print(f"🗂️ {loader_group_kind} Train File Sampling Summary:")
        for group in loader_groups:
            print(f"   Group {group['group_name']} (file-aware batches)")
            _print_series_summaries("", group["train_loader"].dataset.series_summaries)
        print(f"🗂️ {loader_group_kind} Val File Sampling Summary:")
        for group in loader_groups:
            print(f"   Group {group['group_name']}")
            _print_series_summaries("", group["val_loader"].dataset.series_summaries)
    elif loader_group_kind == "tslib":
        print(f"   - Multivariate dataset groups: {[group['group_name'] for group in loader_groups]}")
        print(f"   - Dataset channel counts: {shared_channel_counts}")
        print(f"🗂️ {loader_group_kind} Train File Sampling Summary:")
        for group in loader_groups:
            print(f"   Group {group['group_name']} (file-aware batches)")
            _print_series_summaries("", group["train_loader"].dataset.series_summaries)
        print(f"🗂️ {loader_group_kind} Val File Sampling Summary:")
        for group in loader_groups:
            print(f"   Group {group['group_name']}")
            _print_series_summaries("", group["val_loader"].dataset.series_summaries)
    elif loader_group_kind == "lotsa":
        print(f"   - LOTSA subset groups: {[group['group_name'] for group in loader_groups]}")
        print(f"   - LOTSA channel counts: {shared_channel_counts}")
        print(f"   - LOTSA split mode: {args.lotsa_split_mode}")
        print(f"   - LOTSA sampling mode: {args.lotsa_sampling_mode}")
        print(f"   - LOTSA preprocessing mode: {args.lotsa_preprocessing_mode}")
        print(f"   - LOTSA sample_time_series: {args.lotsa_sample_time_series}")
        print(f"   - LOTSA subset_sampling: {args.lotsa_subset_sampling}")
        print(f"   - LOTSA min_patches: {args.lotsa_min_patches}")
        print(f"   - LOTSA max_channel: {'none' if args.lotsa_max_channel is None else args.lotsa_max_channel}")
        print(f"   - LOTSA windows_per_series: {args.lotsa_windows_per_series}")
        if (
            args.lotsa_split_mode == "official"
            and args.lotsa_sampling_mode == "official"
            and args.lotsa_preprocessing_mode == "official"
            and args.lotsa_sample_time_series == "proportional"
            and args.lotsa_subset_sampling == "official"
        ):
            print("   - LOTSA sampler note: official-like LOTSA data protocol with the LAYA objective")
        else:
            print("   - LOTSA sampler note: custom JEPA sampler over LOTSA, not an official LOTSA dataset protocol")
    elif args.dataset_type == "tsld":
        _print_series_summaries("🗂️ tsld Train File Sampling Summary:", train_loader.dataset.series_summaries)
        _print_series_summaries("🗂️ tsld Val File Sampling Summary:", val_loader.dataset.series_summaries)
    elif args.dataset_type == "tslib":
        _print_series_summaries("🗂️ tslib Train File Sampling Summary:", train_loader.dataset.series_summaries)
        _print_series_summaries("🗂️ tslib Val File Sampling Summary:", val_loader.dataset.series_summaries)
    print("🔧 Encoder Hyperparameters:")
    print(f"   - d_model: {model_cfg.embed_dim}")
    print(f"   - n_heads: {model_cfg.num_heads}")
    print(f"   - n_layers: {model_cfg.depth}")
    print(f"   - proj_dim: {model_cfg.proj_dim}")
    print(f"   - predictor_depth: {model_cfg.predictor_depth}")
    print(f"   - predictor_heads: {model_cfg.predictor_heads}")
    print(f"   - patch_size: {model_cfg.patch_size}")
    print(f"   - model_id: {model_cfg.model_id or 'none'}")
    print(f"   - patchifier_mode: {model_cfg.patchifier_mode}")
    if model_cfg.patchifier_mode == "multiscale" or model_cfg.model_id == "laya_ci_multiscale":
        print(f"   - multiscale_patch_sizes: {model_cfg.multiscale_patch_sizes}")
        print(f"   - multiscale_base_patch: {model_cfg.multiscale_base_patch}")
        print(f"   - multiscale_gate_temperature: {model_cfg.multiscale_gate_temperature}")
    if model_cfg.patchifier_mode == "trend_seasonal" or model_cfg.model_id == "laya_ci_decomp":
        print(f"   - trend_seasonal_kernel: {model_cfg.trend_seasonal_kernel}")
        print(f"   - trend_seasonal_gate_temperature: {model_cfg.trend_seasonal_gate_temperature}")
    print(f"   - metadata_fusion_mode: {model_cfg.metadata_fusion_mode}")
    print(f"   - channel_mixer_relation_mode: {model_cfg.channel_mixer_relation_mode}")
    if model_cfg.channel_mixer_relation_mode in {"laya_relation", "metadata_query_gate"}:
        print(f"   - channel_mixer_relation_scale_init: {model_cfg.channel_mixer_relation_scale_init}")
    print(f"   - use_relation_adapter: {model_cfg.use_relation_adapter}")
    if model_cfg.use_relation_adapter:
        print(f"   - relation_adapter_position: {model_cfg.relation_adapter_position}")
        print(f"   - relation_num_heads: {model_cfg.relation_num_heads}")
        print(f"   - relation_dropout: {model_cfg.relation_dropout}")
        print(f"   - relation_scale_init: {model_cfg.relation_scale_init}")
        print(f"   - use_metadata_bias: {model_cfg.use_metadata_bias}")
        print(f"   - use_metadata_gate: {model_cfg.use_metadata_gate}")
        print(f"   - metadata_scale_init: {model_cfg.metadata_scale_init}")
        print(f"   - metadata_dropout: {model_cfg.metadata_dropout}")
    if model_cfg.metadata_fusion_mode in {"attention_gate", "attention_suppress_gate"}:
        print(f"   - description_relation_metric: {model_cfg.description_relation_metric}")
        print(f"   - description_relation_lambda_init: {model_cfg.description_relation_lambda_init}")
        print(f"   - description_relation_gamma_init: {model_cfg.description_relation_gamma_init}")
    print(f"   - encoder_variant: {model_cfg.encoder_variant}")
    print(f"   - temporal_patchifier_mode: {model_cfg.temporal_patchifier_mode}")
    if model_cfg.temporal_patchifier_mode != "fixed":
        print(f"   - charm_kernel_sizes: {model_cfg.charm_kernel_sizes}")
        print(f"   - charm_stride: {model_cfg.charm_stride}")
        print(f"   - charm_scale_gate_source: {model_cfg.charm_scale_gate_source}")
        print(f"   - charm_patchifier_fusion: {model_cfg.charm_patchifier_fusion}")
    print(f"   - use_channel_relation_block: {model_cfg.use_channel_relation_block}")
    if model_cfg.use_channel_relation_block:
        print(f"   - channel_relation_heads: {model_cfg.channel_relation_heads}")
        print(f"   - channel_relation_gate_scale_init: {model_cfg.channel_relation_gate_scale_init}")
        print(f"   - channel_relation_residual_scale_init: {model_cfg.channel_relation_residual_scale_init}")
    if args.channel_metadata_mode == "onehot":
        print(f"   - onehot_channel_vocab_size: {model_cfg.onehot_channel_vocab_size}")
    print("🧮 Parameter Counts:")
    print(f"   - Total Parameters: {total_params:,}")
    print(f"   - Trainable Parameters: {trainable_params:,}")
    print("⏱️ Schedule:")
    print(f"   - total_steps: {total_steps}")
    print(f"   - warmup_steps: {warmup_steps}")
    print(f"   - derived_epochs: {total_epochs}")
    if val_interval is not None:
        print(f"   - val_interval: {val_interval}")
    attention_saved = 0
    try:
        global_step = 0
        best_val_loss = float("inf")
        best_epoch = 0
        best_global_step = 0
        last_validation_metrics = None
        for epoch in range(1, total_epochs + 1):
            model.train()
            epoch_loss = 0.0
            epoch_pred_loss = 0.0
            epoch_sigreg_loss = 0.0
            epoch_query_loss = 0.0
            epoch_repr_stats = _empty_representation_stats()
            epoch_align_stats = _empty_prediction_alignment_stats()
            epoch_metadata_usage: dict[str, float] = {}
            epoch_scale_weight_totals: dict[str, float] = {}
            epoch_scale_weight_by_dataset: dict[str, dict[str, float] | int] = {}

            epoch_steps = 0
            total_train_samples = 0
            if loader_group_kind == "lotsa":
                batch_stream = _iter_interleaved_loader_groups(
                    loader_groups,
                    epoch=epoch,
                    seed=args.seed,
                    target_steps=steps_per_epoch,
                    subset_sampling=args.lotsa_subset_sampling,
                )
            else:
                iter_loaders = [(None, train_loader)] if loader_groups is None else [(group["group_name"], group["train_loader"]) for group in loader_groups]
                if loader_groups is not None:
                    random.Random(args.seed + epoch).shuffle(iter_loaders)
                batch_stream = (
                    (group_name, batch)
                    for group_name, active_train_loader in iter_loaders
                    for batch in active_train_loader
                )

            for group_name, batch in batch_stream:
                if global_step >= total_steps:
                    break
                batch_on_device = move_batch_to_device(batch, train_cfg.device)
                _ensure_finite_batch(batch_on_device, epoch=epoch, step=global_step + 1)
                batch_size = int(batch_on_device["series"].shape[0])
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    batch_on_device["series"],
                    channel_positions=batch_on_device["channel_positions"],
                    channel_mask=batch_on_device["channel_mask"],
                    channel_text_embeddings=batch_on_device["channel_text_embeddings"],
                    channel_stats_embeddings=batch_on_device["channel_stats_embeddings"],
                )
                _ensure_finite_outputs(outputs, epoch=epoch, step=global_step + 1)
                outputs["loss"].backward()
                optimizer.step(); scheduler.step(); global_step += 1
                epoch_steps += 1
                total_train_samples += batch_size

                loss_value = outputs["loss"].item()
                pred_loss_value = outputs["pred_loss"].item()
                sigreg_loss_value = outputs["sigreg_loss"].item()
                query_loss_value = outputs["query_loss"].item()
                repr_stats = _representation_stats(outputs["mixed_repr"])
                align_stats = _prediction_alignment_stats(outputs["pred_tokens"], outputs["target_tokens"], outputs["patch_mask"])
                epoch_loss += loss_value * batch_size
                epoch_pred_loss += pred_loss_value * batch_size
                epoch_sigreg_loss += sigreg_loss_value * batch_size
                epoch_query_loss += query_loss_value * batch_size
                for key, value in repr_stats.items():
                    epoch_repr_stats[key] += value * batch_size
                for key, value in align_stats.items():
                    epoch_align_stats[key] += value * batch_size
                _accumulate_metadata_usage(
                    epoch_metadata_usage,
                    outputs.get("metadata_usage"),
                    weight=batch_size,
                )
                scale_weights = _extract_scale_weights(outputs.get("metadata_usage"))
                _accumulate_named_scale_weights(
                    epoch_scale_weight_totals,
                    scale_weights,
                    weight=batch_size,
                )
                dataset_log_name = _resolve_dataset_log_name(batch, group_name)
                _accumulate_dataset_scale_weights(
                    epoch_scale_weight_by_dataset,
                    dataset_log_name,
                    scale_weights,
                    weight=batch_size,
                )

                if global_step % args.log_every == 0 or global_step == 1:
                    writer.add_scalar("train/loss", loss_value, global_step)
                    writer.add_scalar("train/pred_loss", pred_loss_value, global_step)
                    writer.add_scalar("train/sigreg_loss", sigreg_loss_value, global_step)
                    writer.add_scalar("train/query_loss", query_loss_value, global_step)
                    writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], global_step)
                    for key, value in repr_stats.items():
                        writer.add_scalar(f"train_repr/{key}", value, global_step)
                    for key, value in align_stats.items():
                        writer.add_scalar(f"train_align/{key}", value, global_step)
                    for key, value in (outputs.get("metadata_usage") or {}).items():
                        writer.add_scalar(f"train_meta/{key}", value, global_step)
                    for scale_name, value in scale_weights.items():
                        writer.add_scalar(f"scale_weight/{scale_name}", value, global_step)

                should_validate_now = (
                    args.dataset_type == "lotsa"
                    and has_validation
                    and (global_step % val_interval == 0 or global_step >= total_steps)
                )
                if should_validate_now:
                    model.eval()
                    val_metrics = _evaluate_validation(
                        model=model,
                        train_cfg=train_cfg,
                        val_loader=val_loader,
                        loader_groups=loader_groups,
                        args=args,
                        epoch=epoch,
                        global_step=global_step,
                    )
                    last_validation_metrics = val_metrics
                    writer.add_scalar("val_step/macro_loss", val_metrics["macro_loss"], global_step)
                    writer.add_scalar("val_step/macro_pred_loss", val_metrics["macro_pred_loss"], global_step)
                    writer.add_scalar("val_step/macro_sigreg_loss", val_metrics["macro_sigreg_loss"], global_step)
                    writer.add_scalar("val_step/macro_query_loss", val_metrics["macro_query_loss"], global_step)
                    writer.add_scalar("val_step/micro_loss", val_metrics["micro_loss"], global_step)
                    writer.add_scalar("val_step/micro_pred_loss", val_metrics["micro_pred_loss"], global_step)
                    writer.add_scalar("val_step/micro_sigreg_loss", val_metrics["micro_sigreg_loss"], global_step)
                    writer.add_scalar("val_step/micro_query_loss", val_metrics["micro_query_loss"], global_step)
                    for key, value in val_metrics["repr_stats"].items():
                        writer.add_scalar(f"val_step/repr_{key}", value, global_step)
                    for key, value in val_metrics["align_stats"].items():
                        writer.add_scalar(f"val_step/align_{key}", value, global_step)
                    for key, value in val_metrics["metadata_usage"].items():
                        writer.add_scalar(f"val_step/meta_{key}", value, global_step)
                    for scale_name, value in val_metrics["scale_weights"].items():
                        writer.add_scalar(f"scale_weight/{scale_name}", value, global_step)
                    for dataset_metric in val_metrics["per_dataset_metrics"]:
                        metric_name = _sanitize_metric_name(dataset_metric["group_name"])
                        writer.add_scalar(f"val_dataset/{metric_name}/loss", dataset_metric["loss"], global_step)
                        writer.add_scalar(f"val_dataset/{metric_name}/pred_loss", dataset_metric["pred_loss"], global_step)
                        writer.add_scalar(f"val_dataset/{metric_name}/sigreg_loss", dataset_metric["sigreg_loss"], global_step)
                        writer.add_scalar(f"val_dataset/{metric_name}/query_loss", dataset_metric["query_loss"], global_step)
                        for scale_name, value in dataset_metric.get("scale_weights", {}).items():
                            writer.add_scalar(
                                f"scale_weight_by_dataset/{metric_name}/{scale_name}",
                                value,
                                global_step,
                            )

                    current_lr = optimizer.param_groups[0]["lr"]
                    is_best = val_metrics["macro_loss"] < best_val_loss
                    if is_best:
                        best_val_loss = val_metrics["macro_loss"]
                        best_epoch = epoch
                        best_global_step = global_step
                    writer.add_scalar("val_step/best_macro_loss", best_val_loss, global_step)
                    per_dataset_summary = " | ".join(
                        f"{metric['group_name']}={metric['loss']:.4f}"
                        for metric in val_metrics["per_dataset_metrics"]
                    )
                    print(
                        f"Validation @ step {global_step}/{total_steps} | "
                        f"Epoch {epoch}/{total_epochs} | "
                        f"LR: {current_lr:.6f} | "
                        f"Macro Val: {val_metrics['macro_loss']:.6f} "
                        f"(P:{val_metrics['macro_pred_loss']:.4f}, S:{val_metrics['macro_sigreg_loss']:.4f}, Q:{val_metrics['macro_query_loss']:.4f}) | "
                        f"Micro Val: {val_metrics['micro_loss']:.6f} "
                        f"(P:{val_metrics['micro_pred_loss']:.4f}, S:{val_metrics['micro_sigreg_loss']:.4f}, Q:{val_metrics['micro_query_loss']:.4f}) | "
                        f"{_format_metadata_usage(val_metrics['metadata_usage'])}"
                    )
                    if per_dataset_summary:
                        print(f"   -> Per-dataset val loss: {per_dataset_summary}")

                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "model_config": model.encoder.get_config(),
                        "epoch": epoch,
                        "epochs": total_epochs,
                        "schedule_mode": "steps",
                        "warmup_epochs": args.warmup_epochs,
                        "warmup_ratio": 0.05,
                        "global_step": global_step,
                        "steps_per_epoch": steps_per_epoch,
                        "actual_steps_per_epoch": epoch_steps,
                        "warmup_steps": warmup_steps,
                        "total_steps": total_steps,
                        "val_interval": val_interval,
                        "best_epoch": best_epoch,
                        "best_global_step": best_global_step,
                        "best_val_loss": best_val_loss,
                        "val_macro_loss": val_metrics["macro_loss"],
                        "val_macro_pred_loss": val_metrics["macro_pred_loss"],
                        "val_macro_sigreg_loss": val_metrics["macro_sigreg_loss"],
                        "val_macro_query_loss": val_metrics["macro_query_loss"],
                        "val_micro_loss": val_metrics["micro_loss"],
                        "val_micro_pred_loss": val_metrics["micro_pred_loss"],
                        "val_micro_sigreg_loss": val_metrics["micro_sigreg_loss"],
                        "val_micro_query_loss": val_metrics["micro_query_loss"],
                        "val_losses_by_dataset": {
                            metric["group_name"]: metric["loss"]
                            for metric in val_metrics["per_dataset_metrics"]
                        },
                    }
                    ckpt_path = os.path.join(args.save_dir, f"laya_ts_{args.dataset_type}_{args.variant}_epoch_{epoch}_step_{global_step}.pt")
                    best_ckpt_path = os.path.join(args.save_dir, f"laya_ts_{args.dataset_type}_{args.variant}_best.pt")
                    last_ckpt_path = os.path.join(args.save_dir, f"laya_ts_{args.dataset_type}_{args.variant}_last.pt")
                    torch.save(checkpoint, ckpt_path)
                    torch.save(checkpoint, last_ckpt_path)
                    if is_best:
                        torch.save(checkpoint, best_ckpt_path)
                        print(f"   -> Saved improved checkpoint: macro_val_loss={val_metrics['macro_loss']:.6f} at step {global_step}")
                    else:
                        print(f"   -> Saved last checkpoint at step {global_step}")

                    if args.save_attention_maps and val_metrics["attention_batch"] is not None:
                        features, channel_names = val_metrics["attention_batch"]
                        attention_dir = os.path.join(args.log_dir, "attention_maps")
                        if args.attention_map_tag:
                            attention_dir = os.path.join(attention_dir, args.attention_map_tag)
                        attention_saved += _save_attention_maps(
                            features=features,
                            channel_names=channel_names,
                            output_dir=attention_dir,
                            epoch=global_step,
                            prefix="val_step",
                        )
                    model.train()

            if epoch_steps == 0:
                raise ValueError("No training batches were produced for this epoch.")
            avg_loss = epoch_loss / max(1, total_train_samples)
            avg_pred_loss = epoch_pred_loss / max(1, total_train_samples)
            avg_sigreg_loss = epoch_sigreg_loss / max(1, total_train_samples)
            avg_query_loss = epoch_query_loss / max(1, total_train_samples)
            avg_repr_stats = {key: value / max(1, total_train_samples) for key, value in epoch_repr_stats.items()}
            avg_align_stats = {key: value / max(1, total_train_samples) for key, value in epoch_align_stats.items()}
            avg_metadata_usage = _average_metadata_usage(epoch_metadata_usage, total_train_samples)
            avg_epoch_scale_weights = _average_named_scale_weights(epoch_scale_weight_totals, total_train_samples)
            avg_epoch_scale_weights_by_dataset = _average_dataset_scale_weights(epoch_scale_weight_by_dataset)
            writer.add_scalar("epoch/train_loss", avg_loss, epoch)
            writer.add_scalar("epoch/train_pred_loss", avg_pred_loss, epoch)
            writer.add_scalar("epoch/train_sigreg_loss", avg_sigreg_loss, epoch)
            writer.add_scalar("epoch/train_query_loss", avg_query_loss, epoch)
            for key, value in avg_repr_stats.items():
                writer.add_scalar(f"epoch/train_repr_{key}", value, epoch)
            for key, value in avg_align_stats.items():
                writer.add_scalar(f"epoch/train_align_{key}", value, epoch)
            for key, value in avg_metadata_usage.items():
                writer.add_scalar(f"epoch/train_meta_{key}", value, epoch)
            for scale_name, value in avg_epoch_scale_weights.items():
                writer.add_scalar(f"scale_weight_epoch/{scale_name}", value, epoch)
            for dataset_name, dataset_scale_weights in avg_epoch_scale_weights_by_dataset.items():
                dataset_metric_name = _sanitize_metric_name(dataset_name)
                for scale_name, value in dataset_scale_weights.items():
                    writer.add_scalar(
                        f"scale_weight_epoch/{dataset_metric_name}/{scale_name}",
                        value,
                        epoch,
                    )
            if args.dataset_type == "lotsa":
                print(
                    f"Epoch {epoch}/{total_epochs} | "
                    f"Steps: {epoch_steps} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
                    f"Train Loss: {avg_loss:.6f} (P:{avg_pred_loss:.4f}, S:{avg_sigreg_loss:.4f}, Q:{avg_query_loss:.4f}) | "
                    f"Repr L2μ: {avg_repr_stats['pairwise_l2_mean']:.4f} | "
                    f"Repr Cosμ: {avg_repr_stats['pairwise_cosine_similarity_mean']:.4f} | "
                    f"Pred→Target Cosμ: {avg_align_stats['cosine_similarity_mean']:.4f} | "
                    f"Repr NormRatio: {avg_repr_stats['norm_ratio']:.4f} | "
                    f"{_format_metadata_usage(avg_metadata_usage)}"
                )
                if not has_validation and (epoch % args.save_every == 0 or global_step >= total_steps):
                    checkpoint = {
                        "model_state_dict": model.state_dict(),
                        "model_config": model.encoder.get_config(),
                        "epoch": epoch,
                        "epochs": total_epochs,
                        "schedule_mode": "steps",
                        "warmup_epochs": args.warmup_epochs,
                        "warmup_ratio": 0.05,
                        "global_step": global_step,
                        "steps_per_epoch": steps_per_epoch,
                        "actual_steps_per_epoch": epoch_steps,
                        "warmup_steps": warmup_steps,
                        "total_steps": total_steps,
                        "val_interval": val_interval,
                        "best_epoch": best_epoch,
                        "best_val_loss": None,
                        "val_available": False,
                    }
                    last_ckpt_path = os.path.join(args.save_dir, f"laya_ts_{args.dataset_type}_{args.variant}_last.pt")
                    torch.save(checkpoint, last_ckpt_path)
                    print(f"   -> Saved last checkpoint without validation at step {global_step}")
                if global_step >= total_steps:
                    break
                continue

            model.eval()
            val_metrics = _evaluate_validation(
                model=model,
                train_cfg=train_cfg,
                val_loader=val_loader,
                loader_groups=loader_groups,
                args=args,
                epoch=epoch,
                global_step=global_step,
            )
            last_validation_metrics = val_metrics
            avg_val_loss = val_metrics["micro_loss"]
            avg_val_pred_loss = val_metrics["micro_pred_loss"]
            avg_val_sigreg_loss = val_metrics["micro_sigreg_loss"]
            avg_val_query_loss = val_metrics["micro_query_loss"]
            avg_val_repr_stats = val_metrics["repr_stats"]
            avg_val_align_stats = val_metrics["align_stats"]
            avg_val_metadata_usage = val_metrics["metadata_usage"]
            avg_val_scale_weights = val_metrics["scale_weights"]
            avg_val_scale_weights_by_dataset = val_metrics["scale_weights_by_dataset"]
            attention_batch = val_metrics["attention_batch"]

            writer.add_scalar("epoch/loss", avg_loss, epoch)
            writer.add_scalar("epoch/pred_loss", avg_pred_loss, epoch)
            writer.add_scalar("epoch/sigreg_loss", avg_sigreg_loss, epoch)
            writer.add_scalar("epoch/query_loss", avg_query_loss, epoch)
            writer.add_scalar("epoch/val_loss", avg_val_loss, epoch)
            writer.add_scalar("epoch/val_pred_loss", avg_val_pred_loss, epoch)
            writer.add_scalar("epoch/val_sigreg_loss", avg_val_sigreg_loss, epoch)
            writer.add_scalar("epoch/val_query_loss", avg_val_query_loss, epoch)
            for key, value in avg_val_repr_stats.items():
                writer.add_scalar(f"epoch/val_repr_{key}", value, epoch)
            for key, value in avg_val_align_stats.items():
                writer.add_scalar(f"epoch/val_align_{key}", value, epoch)
            for key, value in avg_val_metadata_usage.items():
                writer.add_scalar(f"epoch/val_meta_{key}", value, epoch)
            for scale_name, value in avg_val_scale_weights.items():
                writer.add_scalar(f"scale_weight_epoch/val_{scale_name}", value, epoch)
            for dataset_name, dataset_scale_weights in avg_val_scale_weights_by_dataset.items():
                dataset_metric_name = _sanitize_metric_name(dataset_name)
                for scale_name, value in dataset_scale_weights.items():
                    writer.add_scalar(
                        f"scale_weight_epoch/val_{dataset_metric_name}/{scale_name}",
                        value,
                        epoch,
                    )
            writer.add_scalars("compare/loss", {"train": avg_loss, "val": avg_val_loss}, epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                best_epoch = epoch
            writer.add_scalar("epoch/best_val_loss", best_val_loss, epoch)
            print(
                f"Epoch {epoch}/{total_epochs} | "
                f"Steps: {epoch_steps} | "
                f"LR: {current_lr:.6f} | "
                f"Train Loss: {avg_loss:.6f} (P:{avg_pred_loss:.4f}, S:{avg_sigreg_loss:.4f}, Q:{avg_query_loss:.4f}) | "
                f"Val Loss: {avg_val_loss:.6f} (P:{avg_val_pred_loss:.4f}, S:{avg_val_sigreg_loss:.4f}, Q:{avg_val_query_loss:.4f}) | "
                f"Repr L2μ: {avg_repr_stats['pairwise_l2_mean']:.4f} Val L2μ: {avg_val_repr_stats['pairwise_l2_mean']:.4f} | "
                f"Repr Cosμ: {avg_repr_stats['pairwise_cosine_similarity_mean']:.4f} Val Cosμ: {avg_val_repr_stats['pairwise_cosine_similarity_mean']:.4f} | "
                f"Pred→Target Cosμ: {avg_align_stats['cosine_similarity_mean']:.4f} Val Cosμ: {avg_val_align_stats['cosine_similarity_mean']:.4f} | "
                f"Repr NormRatio: {avg_repr_stats['norm_ratio']:.4f} Val NormRatio: {avg_val_repr_stats['norm_ratio']:.4f} | "
                f"Train {_format_metadata_usage(avg_metadata_usage)} | "
                f"Val {_format_metadata_usage(avg_val_metadata_usage)}"
            )

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "model_config": model.encoder.get_config(),
                "epoch": epoch,
                "epochs": total_epochs,
                "schedule_mode": "steps" if args.dataset_type == "lotsa" else "epochs",
                "warmup_epochs": args.warmup_epochs,
                "warmup_ratio": 0.05 if args.dataset_type == "lotsa" else None,
                "global_step": global_step,
                "steps_per_epoch": steps_per_epoch,
                "actual_steps_per_epoch": epoch_steps,
                "warmup_steps": warmup_steps,
                "total_steps": total_steps,
                "val_interval": val_interval,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "val_loss": avg_val_loss,
                "val_pred_loss": avg_val_pred_loss,
                "val_sigreg_loss": avg_val_sigreg_loss,
                "val_query_loss": avg_val_query_loss,
            }
            ckpt_path = os.path.join(args.save_dir, f"laya_ts_{args.dataset_type}_{args.variant}_epoch_{epoch}.pt")
            best_ckpt_path = os.path.join(args.save_dir, f"laya_ts_{args.dataset_type}_{args.variant}_best.pt")
            last_ckpt_path = os.path.join(args.save_dir, f"laya_ts_{args.dataset_type}_{args.variant}_last.pt")
            torch.save(checkpoint, ckpt_path)
            torch.save(checkpoint, last_ckpt_path)
            if is_best:
                torch.save(checkpoint, best_ckpt_path)
                print(f"   -> Saved improved checkpoint: val_loss={avg_val_loss:.6f} at epoch {epoch}")
            else:
                print(f"   -> Saved last checkpoint at epoch {epoch}")

            if args.save_attention_maps and attention_batch is not None:
                features, channel_names = attention_batch
                attention_dir = os.path.join(args.log_dir, "attention_maps")
                if args.attention_map_tag:
                    attention_dir = os.path.join(attention_dir, args.attention_map_tag)
                attention_saved += _save_attention_maps(
                    features=features,
                    channel_names=channel_names,
                    output_dir=attention_dir,
                    epoch=epoch,
                    prefix="val",
                )
        if args.save_attention_maps:
            print(f"Saved attention maps for {attention_saved} epoch snapshot(s) under {os.path.join(args.log_dir, 'attention_maps')}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
