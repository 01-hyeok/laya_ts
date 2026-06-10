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
import torch.nn as nn

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


def _parse_nonnegative_int_list(raw_value: Optional[str]) -> tuple[int, ...] | None:
    if raw_value in {None, "", "none", "None"}:
        return None
    values = tuple(int(piece.strip()) for piece in str(raw_value).split(",") if piece.strip())
    if not values:
        raise ValueError("Expected at least one integer value")
    if any(value < 0 for value in values):
        raise ValueError(f"All values must be non-negative, got {values}")
    return values


def _add_bool_optional_arg(parser: argparse.ArgumentParser, option: str, *, default=None) -> None:
    dest = option.lstrip("-").replace("-", "_")
    parser.add_argument(option, dest=dest, action="store_true")
    parser.add_argument(f"--no-{option[2:]}", dest=dest, action="store_false")
    parser.set_defaults(**{dest: default})


def _mse(y_true, y_pred):
    return float(np.mean((y_true - y_pred) ** 2))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return path.resolve()


def _resolve_output_dir(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return path.resolve()


def _resolve_device(args) -> str:
    if args.device:
        return args.device
    if args.use_gpu and torch.cuda.is_available():
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_forecasting_data_path(args) -> str:
    if args.data_path:
        data_path = Path(args.data_path)
        if data_path.is_absolute():
            return str(data_path)
        root = Path(args.root_path) if args.root_path else Path(".")
        return str((root / data_path).resolve())

    dataset = str(args.data)
    if dataset in {"ETTm1", "ETTm2"}:
        return str((Path(args.root_path) / "ETT-small" / f"{dataset}.csv").resolve())
    if dataset == "Electricity":
        electricity_root = Path("../Dataset/long_term_forecast/electricity")
        return str((electricity_root / "electricity.csv").resolve())
    return str((Path(args.root_path) / dataset / f"{dataset}.csv").resolve())


def _resolve_pretrain_checkpoint(pretrain_run_dir: str, checkpoint_value: str) -> Path:
    checkpoint_raw = Path(checkpoint_value)
    if checkpoint_raw.suffix in {".pt", ".pth"}:
        checkpoint_path = _resolve_path(checkpoint_value)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Pretrain checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    run_dir = _resolve_path(pretrain_run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"checkpoints directory not found under pretrain run dir: {checkpoints_dir}")

    mode = str(checkpoint_value).strip().lower()
    patterns_by_mode = {
        "best": ["best*.pt", "*best*.pt", "checkpoint_best*.pt", "model_best*.pt"],
        "last": ["last*.pt", "*last*.pt", "checkpoint_last*.pt", "model_last*.pt"],
        "final": ["final*.pt", "*final*.pt", "checkpoint_final*.pt", "model_final*.pt"],
    }
    if mode not in patterns_by_mode:
        raise ValueError(
            f"Unsupported --pretrain_checkpoint value: {checkpoint_value}. "
            "Use best/final/last or a direct .pt/.pth path."
        )

    for pattern in patterns_by_mode[mode]:
        matches = sorted(checkpoints_dir.glob(pattern))
        if matches:
            return matches[0].resolve()

    raise FileNotFoundError(
        f"Could not find a '{mode}' checkpoint under {checkpoints_dir} "
        f"with patterns {patterns_by_mode[mode]}"
    )


def _resolve_forecasting_checkpoint(raw_path: str) -> Path:
    checkpoint_path = _resolve_path(raw_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Forecasting checkpoint not found: {checkpoint_path}")
    if checkpoint_path.suffix not in {".pt", ".pth"}:
        raise ValueError(f"Forecasting checkpoint must be a .pt/.pth file, got: {checkpoint_path}")
    return checkpoint_path


def _detect_representation_type(
    requested: str,
    checkpoint_cfg: LayaModelConfig,
    pretrain_run_dir: str | None,
    checkpoint_path: Path,
) -> str:
    if requested in {"ci", "mixer"}:
        return requested

    mixer_type = str(checkpoint_cfg.channel_mixer_type).strip().lower()
    if mixer_type == "independent":
        return "ci"
    if mixer_type == "mixer":
        return "mixer"

    run_name = None if pretrain_run_dir is None else Path(pretrain_run_dir).name.lower()
    checkpoint_name = checkpoint_path.as_posix().lower()
    if run_name is not None:
        if "_ci_" in run_name or run_name.endswith("_ci_none"):
            return "ci"
        if "_mixer_" in run_name or "metadata_query_gate" in run_name or "mixer_concat" in run_name:
            return "mixer"
    if "ci_none" in checkpoint_name:
        return "ci"
    if "_mixer_" in checkpoint_name or "metadata_query_gate" in checkpoint_name or "mixer_concat" in checkpoint_name:
        return "mixer"

    raise ValueError(
        "Could not auto-detect representation_type from checkpoint config, run dir, or checkpoint path. "
        "Please pass --representation_type ci or --representation_type mixer explicitly."
    )


class ForecastingAblationProbe(LayaTSForecaster):
    def __init__(
        self,
        config: LayaModelConfig,
        *,
        pred_len: int,
        out_channels: int,
        representation_type: str,
        num_patches: int,
        use_revin: bool = False,
        revin_affine: bool = True,
        revin_subtract_last: bool = False,
        revin_eps: float = 1e-5,
    ) -> None:
        super().__init__(
            config,
            pred_len=pred_len,
            out_channels=out_channels,
            num_patches=num_patches,
            use_revin=use_revin,
            revin_affine=revin_affine,
            revin_subtract_last=revin_subtract_last,
            revin_eps=revin_eps,
        )
        self.pred_len = pred_len
        self.out_channels = out_channels
        self.representation_type = representation_type
        self.unit_type = "channel" if representation_type == "ci" else "query"
        self.num_units: int | None = None
        self.num_patches = num_patches
        self._active_ablation: str = "zero"
        self._active_ablate_unit_ids: tuple[int, ...] = ()
        self._last_unit_info: dict[str, object] | None = None

    def _apply_ci_ablation(
        self,
        tokens: torch.Tensor,
        *,
        ablation: str,
        ablate_unit_ids: tuple[int, ...],
    ) -> torch.Tensor:
        if not ablate_unit_ids:
            return tokens
        invalid = [unit_id for unit_id in ablate_unit_ids if unit_id < 0 or unit_id >= tokens.shape[1]]
        if invalid:
            raise IndexError(f"channel ablate_unit_id values {invalid} out of range for {tokens.shape[1]} channels")
        tokens = tokens.clone()
        if ablation in {"zero", "zero_renorm"}:
            tokens[:, list(ablate_unit_ids)] = 0.0
            return tokens
        if ablation == "mean":
            replacement = tokens.mean(dim=1, keepdim=True)
            for unit_id in ablate_unit_ids:
                tokens[:, unit_id:unit_id + 1] = replacement
            return tokens
        raise ValueError(f"Unsupported ablation mode: {ablation}")

    def _apply_mixer_affinity_ablation(
        self,
        affinity: torch.Tensor,
        *,
        ablation: str,
        ablate_unit_ids: tuple[int, ...],
    ) -> torch.Tensor:
        if not ablate_unit_ids:
            return affinity
        invalid = [unit_id for unit_id in ablate_unit_ids if unit_id < 0 or unit_id >= affinity.shape[-2]]
        if invalid:
            raise IndexError(f"query ablate_unit_id values {invalid} out of range for {affinity.shape[-2]} queries")
        affinity = affinity.clone()
        if ablation == "zero":
            affinity[:, :, :, list(ablate_unit_ids), :] = 0.0
            return affinity
        if ablation == "zero_renorm":
            affinity[:, :, :, list(ablate_unit_ids), :] = 0.0
            remaining_queries = affinity.shape[-2] - len(ablate_unit_ids)
            if remaining_queries <= 0:
                return affinity
            scale = affinity.shape[-2] / float(remaining_queries)
            return affinity * scale
        if ablation == "mean":
            replacement = affinity.mean(dim=3, keepdim=True)
            for unit_id in ablate_unit_ids:
                affinity[:, :, :, unit_id:unit_id + 1, :] = replacement
            return affinity
        raise ValueError(f"Unsupported ablation mode: {ablation}")

    def _resolve_channelwise_probe_tokens(self, features: dict[str, torch.Tensor]) -> torch.Tensor:
        ablation = self._active_ablation
        ablate_unit_ids = self._active_ablate_unit_ids
        if self.representation_type == "ci":
            tokens = features.get("independent_tokens")
            if tokens is None:
                raise ValueError("CI representation expects features['independent_tokens'] to be available.")
            self._last_unit_info = {
                "unit_type": "channel",
                "num_units": int(tokens.shape[1]),
                "ablation_target": "independent_tokens",
            }
            self.num_units = int(tokens.shape[1])
            if not ablate_unit_ids:
                return super()._resolve_channelwise_probe_tokens(features)
            return self._apply_ci_ablation(tokens, ablation=ablation, ablate_unit_ids=ablate_unit_ids)

        channel_tokens = features.get("channel_mixer_refined_tokens")
        if channel_tokens is None:
            channel_tokens = features.get("channel_tokens")
        if channel_tokens is None:
            raise ValueError("Mixer representation expects channel tokens to be available.")

        affinity = features.get("channel_affinity")
        if affinity is None:
            raise ValueError("Mixer representation expects features['channel_affinity'] to be available.")
        self._last_unit_info = {
            "unit_type": "query",
            "num_units": int(affinity.shape[-2]),
            "ablation_target": "channel_affinity",
        }
        self.num_units = int(affinity.shape[-2])
        if not ablate_unit_ids:
            return super()._resolve_channelwise_probe_tokens(features)
        affinity = self._apply_mixer_affinity_ablation(
            affinity,
            ablation=ablation,
            ablate_unit_ids=ablate_unit_ids,
        )

        channel_importance = affinity.mean(dim=(2, 3))
        channel_importance = channel_importance * channel_tokens.shape[1]
        channel_tokens = channel_tokens * channel_importance.permute(0, 2, 1).unsqueeze(-1)
        return channel_tokens

    def forward(
        self,
        x: torch.Tensor,
        channel_positions: Optional[torch.Tensor],
        channel_mask: Optional[torch.Tensor] = None,
        channel_text_embeddings: Optional[torch.Tensor] = None,
        channel_stats_embeddings: Optional[torch.Tensor] = None,
        *,
        return_features: bool = False,
        ablation: str = "zero",
        ablate_unit_id: Optional[int] = None,
        ablate_unit_ids: Optional[tuple[int, ...]] = None,
    ):
        prev_ablation = self._active_ablation
        prev_unit_ids = self._active_ablate_unit_ids
        prev_unit_info = self._last_unit_info
        self._active_ablation = ablation
        if ablate_unit_ids is not None:
            self._active_ablate_unit_ids = tuple(sorted(set(int(unit_id) for unit_id in ablate_unit_ids)))
        elif ablate_unit_id is not None:
            self._active_ablate_unit_ids = (int(ablate_unit_id),)
        else:
            self._active_ablate_unit_ids = ()
        self._last_unit_info = None
        try:
            outputs = super().forward(
                x,
                channel_positions,
                channel_mask=channel_mask,
                channel_text_embeddings=channel_text_embeddings,
                channel_stats_embeddings=channel_stats_embeddings,
                return_features=return_features,
            )
            unit_info = self._last_unit_info
            if unit_info is None:
                raise RuntimeError("Unit info was not populated during forecasting forward pass.")
            if return_features:
                pred, features = outputs
                return pred, features, unit_info
            return outputs
        finally:
            self._active_ablation = prev_ablation
            self._active_ablate_unit_ids = prev_unit_ids
            self._last_unit_info = prev_unit_info


def _batch_to_device(batch, device: str):
    x = batch["series"].to(device)
    y = batch["target"].to(device)
    pos = batch["channel_positions"].to(device)
    mask = batch["channel_mask"].to(device)
    channel_text_embeddings = batch.get("channel_text_embeddings")
    if channel_text_embeddings is not None:
        channel_text_embeddings = channel_text_embeddings.to(device)
    channel_stats_embeddings = batch.get("channel_stats_embeddings")
    if channel_stats_embeddings is not None:
        channel_stats_embeddings = channel_stats_embeddings.to(device)
    return x, y, pos, mask, channel_text_embeddings, channel_stats_embeddings


def _evaluate(
    model: ForecastingAblationProbe,
    loader,
    *,
    device: str,
    criterion,
    ablation: str = "zero",
    ablate_unit_id: Optional[int] = None,
    ablate_unit_ids: Optional[tuple[int, ...]] = None,
) -> tuple[float, float, np.ndarray, np.ndarray, dict[str, float]]:
    model.eval()
    losses = []
    preds = []
    ys = []
    metadata_usage = {}
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x, y, pos, mask, text_meta, stats_meta = _batch_to_device(batch, device)
            outputs = model(
                x,
                pos,
                mask,
                channel_text_embeddings=text_meta,
                channel_stats_embeddings=stats_meta,
                return_features=(batch_idx == 0),
                ablation=ablation,
                ablate_unit_id=ablate_unit_id,
                ablate_unit_ids=ablate_unit_ids,
            )
            if batch_idx == 0:
                pred, features, _ = outputs
                metadata_usage = summarize_metadata_usage(features)
            else:
                pred = outputs
            losses.append(criterion(pred, y).item())
            preds.append(pred.cpu().numpy())
            ys.append(y.cpu().numpy())
    y_true = np.concatenate(ys, axis=0)
    y_pred = np.concatenate(preds, axis=0)
    return float(np.mean(losses)) if losses else float("inf"), _mae(y_true, y_pred), y_true, y_pred, metadata_usage


def _save_train_log(path: Path, rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_mse", "val_mse", "val_mae", "lr"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_ablation_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "unit_type",
                "unit_id",
                "baseline_mse",
                "mse_without_unit",
                "delta_mse",
                "baseline_mae",
                "mae_without_unit",
                "delta_mae",
                "delta_mse_ratio_pct",
                "delta_mae_ratio_pct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_group_ablation_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "group_type",
                "group_size",
                "group_label",
                "unit_ids",
                "baseline_mse",
                "mse_without_group",
                "delta_mse",
                "delta_mse_ratio_pct",
                "baseline_mae",
                "mae_without_group",
                "delta_mae",
                "delta_mae_ratio_pct",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _save_delta_plot(path: Path, rows: list[dict[str, object]], unit_type: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(row["unit_id"]) for row in rows]
    y = [float(row["delta_mse"]) for row in rows]
    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.xlabel("unit_id")
    plt.ylabel("delta_mse")
    title = "Channel ablation effect on downstream MSE" if unit_type == "channel" else "Query ablation effect on downstream MSE"
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _save_group_delta_plot(path: Path, rows: list[dict[str, object]], unit_type: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["group_label"]) for row in rows]
    values = [float(row["delta_mse_ratio_pct"]) for row in rows]
    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(len(labels)), values)
    plt.xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    plt.ylabel("delta_mse_ratio_pct")
    plt.xlabel("group_label")
    plt.title(f"Grouped {unit_type} ablation effect on downstream MSE")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def _build_group_row(
    *,
    group_type: str,
    group_size: int,
    group_label: str,
    unit_ids: tuple[int, ...],
    baseline_mse: float,
    mse_without_group: float,
    baseline_mae: float,
    mae_without_group: float,
) -> dict[str, object]:
    delta_mse = float(mse_without_group - baseline_mse)
    delta_mae = float(mae_without_group - baseline_mae)
    return {
        "group_type": group_type,
        "group_size": int(group_size),
        "group_label": group_label,
        "unit_ids": " ".join(str(unit_id) for unit_id in unit_ids),
        "baseline_mse": baseline_mse,
        "mse_without_group": mse_without_group,
        "delta_mse": delta_mse,
        "delta_mse_ratio_pct": 100.0 * delta_mse / max(abs(baseline_mse), 1e-12),
        "baseline_mae": baseline_mae,
        "mae_without_group": mae_without_group,
        "delta_mae": delta_mae,
        "delta_mae_ratio_pct": 100.0 * delta_mae / max(abs(baseline_mae), 1e-12),
    }


def _write_report(path: Path, *, summary: dict[str, object], unit_type: str) -> None:
    lines = [
        f"forecasting_checkpoint_path: {summary.get('forecasting_checkpoint_path')}",
        f"pretrain_run_dir: {summary['pretrain_run_dir']}",
        f"pretrain_checkpoint_path: {summary['pretrain_checkpoint_path']}",
        f"data: {summary['data']}",
        f"pred_len: {summary['pred_len']}",
        f"representation_type: {summary['representation_type']}",
        f"unit_type: {unit_type}",
        f"ablation_target: {summary['ablation_target']}",
        f"best_val_mse: {summary['best_val_mse']}",
        f"test_mse: {summary['test_mse']}",
        "",
        f"Most important {unit_type} by ablation: {summary['most_important_unit_by_ablation']}",
        f"Max delta MSE: {summary['max_delta_mse']}",
        f"Mean delta MSE: {summary['mean_delta_mse']}",
        f"Max delta MSE ratio (%): {summary.get('max_delta_mse_ratio_pct')}",
        f"Mean delta MSE ratio (%): {summary.get('mean_delta_mse_ratio_pct')}",
        "",
        "이 결과는 run_forecasting.py와 같은 channel-wise forecasting head 기준으로 해석해야 한다. 따라서 delta MSE는 latent 자체의 직접 readout 중요도가 아니라, 최종 forecasting 경로 안에서 그 unit이 얼마나 기여하는지를 뜻한다.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Linear probe then ablate pretrained representation units")
    parser.add_argument("--forecasting_checkpoint", type=str, default=None)
    parser.add_argument("--pretrain_run_dir", type=str, default=None)
    parser.add_argument("--pretrain_checkpoint", type=str, default="best")
    parser.add_argument("--representation_type", type=str, default="auto", choices=["auto", "ci", "mixer"])
    parser.add_argument("--data", type=str, default="ETTm1")
    parser.add_argument("--root_path", type=str, default="../Dataset/Time-Series-Library_dataset")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--channel_metadata_mode", type=str, default=None, choices=["onehot", "text", "stats", "text_stats_joint", "text_stats_avg", "none"])
    parser.add_argument("--metadata_fusion_mode", type=str, default=None, choices=["none", "add", "concat_kv", "attention_gate", "attention_suppress_gate"])
    parser.add_argument("--onehot_channel_vocab_size", type=int, default=None)
    parser.add_argument("--channel_mixer_type", type=str, default=None, choices=["mixer", "independent"])
    parser.add_argument("--channel_mixer_relation_mode", type=str, default=None, choices=["none", "laya_relation", "metadata_query_gate", "metadata_query_bias", "description_relation"])
    parser.add_argument("--channel_mixer_relation_scale_init", type=float, default=None)
    parser.add_argument("--description_relation_num_latents", type=int, default=None)
    parser.add_argument("--description_relation_metric", type=str, default=None, choices=["projected_dot", "cosine"])
    parser.add_argument("--description_relation_lambda_init", type=float, default=None)
    parser.add_argument("--description_relation_gamma_init", type=float, default=None)
    parser.add_argument("--use_channel_relation_block", action="store_true")
    parser.add_argument("--channel_relation_heads", type=int, default=None)
    parser.add_argument("--channel_relation_gate_scale_init", type=float, default=None)
    parser.add_argument("--channel_relation_residual_scale_init", type=float, default=None)
    parser.add_argument("--encoder_variant", type=str, default=None, choices=["default"])
    parser.add_argument("--temporal_patchifier_mode", type=str, default=None, choices=["fixed", "multiscale", "charm_like"])
    parser.add_argument("--charm_kernel_sizes", type=str, default=None)
    parser.add_argument("--charm_stride", type=int, default=None)
    parser.add_argument("--charm_patchifier_dropout", type=float, default=None)
    parser.add_argument("--charm_scale_gate_source", type=str, default=None, choices=["learned", "text"])
    parser.add_argument("--charm_scale_gate_temperature", type=float, default=None)
    parser.add_argument("--charm_patchifier_fusion", type=str, default=None, choices=["replace", "residual"])
    parser.add_argument("--charm_patchifier_residual_init", type=float, default=None)
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--train_epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ablation", type=str, default="zero", choices=["zero", "zero_renorm", "mean"])
    parser.add_argument("--topk_group_sizes", type=str, default="1,2,4")
    parser.add_argument("--random_group_repeats", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="analysis/linear_probe_ablation")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_gpu", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--text_encoder_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--text_metadata_cache_dir", type=str, default="./metadata_cache")
    parser.add_argument("--text_encoder_local_files_only", action="store_true")
    parser.add_argument("--stats_metadata_dim", type=int, default=None)
    _add_bool_optional_arg(parser, "--use_revin", default=True)
    _add_bool_optional_arg(parser, "--revin_affine", default=True)
    _add_bool_optional_arg(parser, "--revin_subtract_last", default=False)
    parser.add_argument("--revin_eps", type=float, default=1e-5)
    args = parser.parse_args(argv)
    if args.stats_metadata_dim is not None and args.stats_metadata_dim <= 0:
        raise ValueError(f"--stats_metadata_dim must be positive, got {args.stats_metadata_dim}")
    if args.random_group_repeats < 0:
        raise ValueError(f"--random_group_repeats must be non-negative, got {args.random_group_repeats}")
    if args.revin_eps <= 0:
        raise ValueError(f"--revin_eps must be positive, got {args.revin_eps}")
    topk_group_sizes = _parse_nonnegative_int_list(args.topk_group_sizes) or ()

    set_seed(args.seed)
    device = _resolve_device(args)
    output_dir = _resolve_output_dir(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    os.makedirs(checkpoints_dir, exist_ok=True)

    print("[INFO] Using forecasting-checkpoint-compatible flow")

    forecasting_checkpoint_path: Path | None = None
    forecasting_payload: dict[str, object] | None = None
    if args.forecasting_checkpoint:
        forecasting_checkpoint_path = _resolve_forecasting_checkpoint(args.forecasting_checkpoint)
        forecasting_payload = torch.load(str(forecasting_checkpoint_path), map_location="cpu", weights_only=True)
        raw_model_config = forecasting_payload.get("model_config")
        if raw_model_config is None:
            raise ValueError(
                f"Forecasting checkpoint {forecasting_checkpoint_path} does not contain 'model_config'."
            )
        checkpoint_cfg = LayaModelConfig(**raw_model_config)
        checkpoint_path = _resolve_path(str(forecasting_payload.get("checkpoint", "")))
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Pretrain checkpoint recorded in forecasting checkpoint does not exist: {checkpoint_path}"
            )
        effective_data = str(forecasting_payload.get("dataset_type", args.data))
        effective_data_path = forecasting_payload.get("data_path") or args.data_path
        effective_seq_len = int(forecasting_payload.get("seq_len", args.seq_len))
        effective_pred_len = int(forecasting_payload.get("pred_len", args.pred_len))
        effective_text_encoder_name = str(
            forecasting_payload.get("text_encoder_name_or_path", args.text_encoder_name_or_path)
        )
        effective_text_metadata_cache_dir = str(
            forecasting_payload.get("text_metadata_cache_dir", args.text_metadata_cache_dir)
        )
        effective_text_encoder_local_files_only = bool(
            forecasting_payload.get("text_encoder_local_files_only", args.text_encoder_local_files_only)
        )
        effective_use_revin = bool(forecasting_payload.get("use_revin", False))
        effective_revin_affine = bool(forecasting_payload.get("revin_affine", True))
        effective_revin_subtract_last = bool(forecasting_payload.get("revin_subtract_last", False))
        effective_revin_eps = float(forecasting_payload.get("revin_eps", 1e-5))
        channel_metadata_mode = checkpoint_cfg.channel_metadata_mode
        metadata_fusion_mode = checkpoint_cfg.metadata_fusion_mode
        channel_mixer_type = checkpoint_cfg.channel_mixer_type
        representation_type = _detect_representation_type(
            args.representation_type,
            checkpoint_cfg,
            args.pretrain_run_dir,
            checkpoint_path,
        )
        print(f"[OK] Loaded forecasting checkpoint from: {forecasting_checkpoint_path}")
        print(f"[OK] Loaded pretrained checkpoint from forecasting payload: {checkpoint_path}")
        print(f"[OK] Detected representation_type: {representation_type}")
    else:
        print(f"[INFO] Pretrain checkpoint mode: {args.pretrain_checkpoint}")
        checkpoint_path = _resolve_pretrain_checkpoint(args.pretrain_run_dir or ".", args.pretrain_checkpoint)
        checkpoint_cfg = load_model_config_from_checkpoint(str(checkpoint_path))
        channel_metadata_mode = args.channel_metadata_mode or checkpoint_cfg.channel_metadata_mode
        metadata_fusion_mode = args.metadata_fusion_mode or checkpoint_cfg.metadata_fusion_mode
        if channel_metadata_mode == "coordinates":
            raise ValueError("laya_ts no longer supports channel_metadata_mode='coordinates'. Use one of: onehot, text, stats, text_stats_joint, text_stats_avg, none.")
        channel_mixer_type = args.channel_mixer_type or checkpoint_cfg.channel_mixer_type
        representation_type = _detect_representation_type(
            args.representation_type,
            checkpoint_cfg,
            args.pretrain_run_dir,
            checkpoint_path,
        )
        effective_data = args.data
        effective_data_path = args.data_path
        effective_seq_len = args.seq_len
        effective_pred_len = args.pred_len
        effective_text_encoder_name = args.text_encoder_name_or_path
        effective_text_metadata_cache_dir = args.text_metadata_cache_dir
        effective_text_encoder_local_files_only = args.text_encoder_local_files_only
        effective_use_revin = args.use_revin
        effective_revin_affine = args.revin_affine
        effective_revin_subtract_last = args.revin_subtract_last
        effective_revin_eps = args.revin_eps
        print(f"[OK] Loaded pretrained checkpoint from: {checkpoint_path}")
        print(f"[OK] Detected representation_type: {representation_type}")

    data_path = _resolve_forecasting_data_path(args)
    if effective_data_path:
        data_path = _resolve_path(str(effective_data_path)).as_posix()
    train_loader, val_loader, test_loader, _ = get_forecasting_loaders(
        data_path,
        effective_data,
        batch_size=args.batch_size,
        seq_len=effective_seq_len,
        pred_len=effective_pred_len,
        num_workers=args.num_workers,
        channel_metadata_mode=channel_metadata_mode,
        text_encoder_name_or_path=effective_text_encoder_name,
        text_metadata_cache_dir=effective_text_metadata_cache_dir,
        text_encoder_local_files_only=effective_text_encoder_local_files_only,
    )
    print(f"[OK] Built dataloaders: data={effective_data}, pred_len={effective_pred_len}")

    first_batch = next(iter(train_loader))
    input_channels = int(first_batch["series"].shape[1])
    out_channels = int(first_batch["target"].shape[1])
    onehot_vocab_size = checkpoint_cfg.onehot_channel_vocab_size
    if channel_metadata_mode == "onehot":
        onehot_vocab_size = max(onehot_vocab_size, args.onehot_channel_vocab_size or 0, input_channels)
    relation_heads = checkpoint_cfg.channel_relation_heads if args.channel_relation_heads is None else args.channel_relation_heads
    relation_gate_scale = checkpoint_cfg.channel_relation_gate_scale_init if args.channel_relation_gate_scale_init is None else args.channel_relation_gate_scale_init
    relation_residual_scale = checkpoint_cfg.channel_relation_residual_scale_init if args.channel_relation_residual_scale_init is None else args.channel_relation_residual_scale_init
    channel_mixer_relation_mode = args.channel_mixer_relation_mode or checkpoint_cfg.channel_mixer_relation_mode
    channel_mixer_relation_scale_init = checkpoint_cfg.channel_mixer_relation_scale_init if args.channel_mixer_relation_scale_init is None else args.channel_mixer_relation_scale_init
    description_relation_num_latents = checkpoint_cfg.description_relation_num_latents if args.description_relation_num_latents is None else args.description_relation_num_latents
    description_relation_metric = args.description_relation_metric or checkpoint_cfg.description_relation_metric
    description_relation_lambda_init = checkpoint_cfg.description_relation_lambda_init if args.description_relation_lambda_init is None else args.description_relation_lambda_init
    description_relation_gamma_init = checkpoint_cfg.description_relation_gamma_init if args.description_relation_gamma_init is None else args.description_relation_gamma_init
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
    num_patches = infer_temporal_patchifier_num_patches(model_cfg, first_batch["series"].shape[-1])
    model = ForecastingAblationProbe(
        model_cfg,
        pred_len=effective_pred_len,
        out_channels=out_channels,
        representation_type=representation_type,
        num_patches=num_patches,
        use_revin=effective_use_revin,
        revin_affine=effective_revin_affine,
        revin_subtract_last=effective_revin_subtract_last,
        revin_eps=effective_revin_eps,
    ).to(device)
    if forecasting_payload is not None:
        model.load_state_dict(forecasting_payload["model_state_dict"])
        print("[OK] Loaded forecasting model state directly from forecasting checkpoint")
    else:
        load_report = load_encoder_from_checkpoint_report(model, str(checkpoint_path))
        if load_report["missing_keys"] or load_report["unexpected_keys"]:
            print(f"[INFO] Encoder load report: {load_report}")
        model.encoder.requires_grad_(False)
        finetune_params = list(model.head.parameters())
        skipped_encoder_keys = list(load_report.get("skipped_keys", []))
        skipped_onehot_projector = any(key.startswith("channel_id_projector.") for key in skipped_encoder_keys)
        if skipped_onehot_projector and model.encoder.channel_id_projector is not None:
            for p_ in model.encoder.channel_id_projector.parameters():
                p_.requires_grad = True
            finetune_params.extend(model.encoder.channel_id_projector.parameters())
            print(
                f"[INFO] Onehot vocab resized from checkpoint value {checkpoint_cfg.onehot_channel_vocab_size}; "
                "reinitializing channel_id_projector."
            )
    model.encoder.requires_grad_(False)

    with torch.no_grad():
        x, _, pos, mask, text_meta, stats_meta = _batch_to_device(first_batch, device)
        _, _, unit_info = model(
            x,
            pos,
            mask,
            channel_text_embeddings=text_meta,
            channel_stats_embeddings=stats_meta,
            return_features=True,
        )
    unit_type = str(unit_info["unit_type"])
    print(f"[OK] Unit type: {unit_type}")
    print(f"[OK] Ablation target: {unit_info['ablation_target']}")
    print(
        f"[OK] RevIN: {'enabled' if effective_use_revin else 'disabled'} | "
        f"affine={effective_revin_affine} | subtract_last={effective_revin_subtract_last} | eps={effective_revin_eps}"
    )

    criterion = nn.MSELoss()
    best_checkpoint_path = checkpoints_dir / "best_linear_probe.pt"
    best_epoch = None
    if forecasting_payload is None:
        optimizer = torch.optim.AdamW(finetune_params, lr=args.learning_rate, weight_decay=args.weight_decay)
        train_log_rows: list[dict[str, float]] = []
        best_val_mse = float("inf")
        best_val_mae = float("inf")

        print("[OK] Started linear probing")
        for epoch in range(1, args.train_epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_metadata_usage = {}
            for batch_idx, batch in enumerate(train_loader):
                x, y, pos, mask, text_meta, stats_meta = _batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(
                    x,
                    pos,
                    mask,
                    channel_text_embeddings=text_meta,
                    channel_stats_embeddings=stats_meta,
                    return_features=(batch_idx == 0),
                )
                if batch_idx == 0:
                    pred, features, _ = outputs
                    train_metadata_usage = summarize_metadata_usage(features)
                else:
                    pred = outputs
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss_sum += loss.item() * x.size(0)

            train_mse = train_loss_sum / max(1, len(train_loader.dataset))
            val_mse, val_mae, _, _, val_metadata_usage = _evaluate(model, val_loader, device=device, criterion=criterion)
            current_lr = optimizer.param_groups[0]["lr"]
            train_log_rows.append(
                {
                    "epoch": epoch,
                    "train_mse": train_mse,
                    "val_mse": val_mse,
                    "val_mae": val_mae,
                    "lr": current_lr,
                }
            )
            print(
                f"Epoch {epoch}/{args.train_epochs} | LR: {current_lr:.6f} | "
                f"Train MSE: {train_mse:.6f} | Val MSE: {val_mse:.6f} | Val MAE: {val_mae:.6f}"
            )
            if train_metadata_usage:
                print(f"[INFO] train metadata usage: {train_metadata_usage}")
            if val_metadata_usage:
                print(f"[INFO] val metadata usage: {val_metadata_usage}")

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_val_mae = val_mae
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "representation_type": representation_type,
                        "unit_type": unit_type,
                        "best_val_mse": best_val_mse,
                        "best_val_mae": best_val_mae,
                        "pretrain_checkpoint_path": str(checkpoint_path),
                        "use_revin": effective_use_revin,
                        "revin_affine": effective_revin_affine,
                        "revin_subtract_last": effective_revin_subtract_last,
                        "revin_eps": effective_revin_eps,
                    },
                    best_checkpoint_path,
                )
                print(f"[OK] Saved best linear checkpoint: epoch={epoch}, val_mse={val_mse:.6f}")

        _save_train_log(output_dir / "train_log.csv", train_log_rows)

        best_ckpt = torch.load(str(best_checkpoint_path), map_location="cpu", weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        model.to(device)
        print("[OK] Reloaded best linear checkpoint")
    else:
        print("[OK] Skipping linear probe training; using saved forecasting checkpoint as baseline")
        best_val_mse = float(forecasting_payload.get("best_val_mse", float("nan")))
        best_val_mae = float("nan")

    val_mse, val_mae, _, _, val_metadata_usage = _evaluate(model, val_loader, device=device, criterion=criterion)
    test_mse, test_mae, _, _, test_metadata_usage = _evaluate(model, test_loader, device=device, criterion=criterion)
    if forecasting_payload is not None:
        best_val_mse = val_mse if not np.isfinite(best_val_mse) else best_val_mse
        best_val_mae = val_mae if not np.isfinite(best_val_mae) else best_val_mae
    print(f"[OK] Baseline test MSE/MAE: mse={test_mse:.6f}, mae={test_mae:.6f}")
    if val_metadata_usage:
        print(f"[INFO] baseline val metadata usage: {val_metadata_usage}")
    if test_metadata_usage:
        print(f"[INFO] test metadata usage: {test_metadata_usage}")

    baseline_metrics = {
        "forecasting_checkpoint_path": None if forecasting_checkpoint_path is None else str(forecasting_checkpoint_path),
        "pretrain_run_dir": None if args.pretrain_run_dir is None else str(_resolve_path(args.pretrain_run_dir)),
        "pretrain_checkpoint_path": str(checkpoint_path),
        "data": effective_data,
        "pred_len": effective_pred_len,
        "seq_len": effective_seq_len,
        "label_len": args.label_len,
        "representation_type": representation_type,
        "unit_type": unit_type,
        "best_val_mse": best_val_mse,
        "best_val_mae": best_val_mae,
        "val_mse": val_mse,
        "val_mae": val_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "ablation_target": str(unit_info["ablation_target"]),
        "num_units": int(model.num_units if model.num_units is not None else unit_info["num_units"]),
    }
    (output_dir / "baseline_metrics.json").write_text(json.dumps(baseline_metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if model.num_units is None:
        raise RuntimeError("Probe head was not initialized with unit count.")
    ablation_rows: list[dict[str, object]] = []
    for unit_id in range(model.num_units):
        ablated_mse, ablated_mae, _, _, _ = _evaluate(
            model,
            test_loader,
            device=device,
            criterion=criterion,
            ablation=args.ablation,
            ablate_unit_id=unit_id,
        )
        ablation_rows.append(
            {
                "unit_type": unit_type,
                "unit_id": unit_id,
                "baseline_mse": test_mse,
                "mse_without_unit": ablated_mse,
                "delta_mse": ablated_mse - test_mse,
                "baseline_mae": test_mae,
                "mae_without_unit": ablated_mae,
                "delta_mae": ablated_mae - test_mae,
                "delta_mse_ratio_pct": 100.0 * (ablated_mse - test_mse) / max(abs(test_mse), 1e-12),
                "delta_mae_ratio_pct": 100.0 * (ablated_mae - test_mae) / max(abs(test_mae), 1e-12),
            }
        )
    print("[OK] Unit ablation completed")

    _save_ablation_csv(output_dir / "unit_ablation_results.csv", ablation_rows)
    _save_delta_plot(output_dir / "unit_ablation_delta_mse.png", ablation_rows, unit_type)
    print("[OK] Saved unit_ablation_results.csv")
    print("[OK] Saved unit_ablation_delta_mse.png")

    ranked_rows = sorted(ablation_rows, key=lambda row: float(row["delta_mse"]), reverse=True)
    group_ablation_rows: list[dict[str, object]] = []
    valid_group_sizes = [size for size in topk_group_sizes if size > 0 and size <= len(ranked_rows)]
    rng = random.Random(args.seed)
    for group_size in valid_group_sizes:
        topk_unit_ids = tuple(int(row["unit_id"]) for row in ranked_rows[:group_size])
        topk_mse, topk_mae, _, _, _ = _evaluate(
            model,
            test_loader,
            device=device,
            criterion=criterion,
            ablation=args.ablation,
            ablate_unit_ids=topk_unit_ids,
        )
        group_ablation_rows.append(
            _build_group_row(
                group_type="topk",
                group_size=group_size,
                group_label=f"top{group_size}",
                unit_ids=topk_unit_ids,
                baseline_mse=test_mse,
                mse_without_group=topk_mse,
                baseline_mae=test_mae,
                mae_without_group=topk_mae,
            )
        )

        if args.random_group_repeats > 0:
            random_rows_for_size: list[dict[str, object]] = []
            for repeat_idx in range(args.random_group_repeats):
                sampled_ids = tuple(sorted(rng.sample(range(model.num_units), k=group_size)))
                sampled_mse, sampled_mae, _, _, _ = _evaluate(
                    model,
                    test_loader,
                    device=device,
                    criterion=criterion,
                    ablation=args.ablation,
                    ablate_unit_ids=sampled_ids,
                )
                row = _build_group_row(
                    group_type="random",
                    group_size=group_size,
                    group_label=f"random{group_size}_rep{repeat_idx + 1}",
                    unit_ids=sampled_ids,
                    baseline_mse=test_mse,
                    mse_without_group=sampled_mse,
                    baseline_mae=test_mae,
                    mae_without_group=sampled_mae,
                )
                random_rows_for_size.append(row)
                group_ablation_rows.append(row)
            mean_random_delta_mse = float(np.mean([float(row["delta_mse"]) for row in random_rows_for_size]))
            mean_random_delta_mae = float(np.mean([float(row["delta_mae"]) for row in random_rows_for_size]))
            group_ablation_rows.append(
                _build_group_row(
                    group_type="random_mean",
                    group_size=group_size,
                    group_label=f"random{group_size}_mean",
                    unit_ids=(),
                    baseline_mse=test_mse,
                    mse_without_group=test_mse + mean_random_delta_mse,
                    baseline_mae=test_mae,
                    mae_without_group=test_mae + mean_random_delta_mae,
                )
            )

    if group_ablation_rows:
        _save_group_ablation_csv(output_dir / "group_ablation_results.csv", group_ablation_rows)
        group_plot_rows = [row for row in group_ablation_rows if row["group_type"] in {"topk", "random_mean"}]
        _save_group_delta_plot(output_dir / "group_ablation_delta_mse_ratio.png", group_plot_rows, unit_type)
        print("[OK] Saved group_ablation_results.csv")
        print("[OK] Saved group_ablation_delta_mse_ratio.png")

    most_important_row = max(ablation_rows, key=lambda row: float(row["delta_mse"]))
    topk_summary: dict[str, object] = {}
    for row in group_ablation_rows:
        if row["group_type"] != "topk":
            continue
        group_size = int(row["group_size"])
        random_mean_row = next(
            (
                candidate
                for candidate in group_ablation_rows
                if candidate["group_type"] == "random_mean" and int(candidate["group_size"]) == group_size
            ),
            None,
        )
        topk_summary[f"top{group_size}"] = {
            "unit_ids": [int(piece) for piece in str(row["unit_ids"]).split()] if row["unit_ids"] else [],
            "delta_mse": float(row["delta_mse"]),
            "delta_mse_ratio_pct": float(row["delta_mse_ratio_pct"]),
            "random_mean_delta_mse": None if random_mean_row is None else float(random_mean_row["delta_mse"]),
            "random_mean_delta_mse_ratio_pct": (
                None if random_mean_row is None else float(random_mean_row["delta_mse_ratio_pct"])
            ),
        }
    summary = {
        "forecasting_checkpoint_path": None if forecasting_checkpoint_path is None else str(forecasting_checkpoint_path),
        "pretrain_run_dir": None if args.pretrain_run_dir is None else str(_resolve_path(args.pretrain_run_dir)),
        "pretrain_checkpoint_path": str(checkpoint_path),
        "data": effective_data,
        "pred_len": effective_pred_len,
        "representation_type": representation_type,
        "unit_type": unit_type,
        "ablation_target": str(unit_info["ablation_target"]),
        "best_linear_checkpoint": None if forecasting_payload is not None else str(best_checkpoint_path),
        "best_val_mse": best_val_mse,
        "val_mse": val_mse,
        "test_mse": test_mse,
        "most_important_unit_by_ablation": int(most_important_row["unit_id"]),
        "max_delta_mse": float(most_important_row["delta_mse"]),
        "mean_delta_mse": float(np.mean([float(row["delta_mse"]) for row in ablation_rows])),
        "max_delta_mse_ratio_pct": float(most_important_row["delta_mse_ratio_pct"]),
        "mean_delta_mse_ratio_pct": float(np.mean([float(row["delta_mse_ratio_pct"]) for row in ablation_rows])),
        "topk_group_ablation": topk_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_report(output_dir / "unit_usage_report.txt", summary=summary, unit_type=unit_type)
    print(f"[OK] Saved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
