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
    from laya_ts.model import LayaTSForecaster, summarize_metadata_usage
else:
    from .data_forecasting import get_forecasting_loaders
    from .model import LayaTSForecaster, summarize_metadata_usage


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
        raise ValueError(f"All values must be positive, got {values}")
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


def _format_metadata_usage(metadata_usage: dict[str, float]) -> str:
    if not metadata_usage:
        return "Meta: n/a"
    return "Meta: " + ", ".join(f"{key}={value:.4f}" for key, value in sorted(metadata_usage.items()))


def _num_patches_from_config(config: LayaModelConfig, seq_len: int) -> int:
    if config.patchifier_mode == "multiscale":
        return math.ceil(seq_len / config.multiscale_base_patch)
    return math.ceil(seq_len / config.patch_size)


def main(argv=None):
    p = argparse.ArgumentParser(description="Random encoder linear probing for forecasting with Laya-TS")
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--dataset_type", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--pred_len", type=int, default=96)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--d_model", type=int, default=LayaModelConfig().embed_dim)
    p.add_argument("--patch_size", type=int, default=LayaModelConfig().patch_size)
    p.add_argument("--n_heads", type=int, default=LayaModelConfig().num_heads)
    p.add_argument("--n_layers", type=int, default=LayaModelConfig().depth)
    p.add_argument("--proj_dim", type=int, default=LayaModelConfig().proj_dim)
    p.add_argument("--predictor_depth", type=int, default=LayaModelConfig().predictor_depth)
    p.add_argument("--predictor_heads", type=int, default=LayaModelConfig().predictor_heads)
    p.add_argument("--patchifier_mode", type=str, default=LayaModelConfig().patchifier_mode, choices=["single", "multiscale"])
    p.add_argument("--multiscale_patch_sizes", type=str, default=",".join(str(v) for v in LayaModelConfig().multiscale_patch_sizes))
    p.add_argument("--multiscale_base_patch", type=int, default=LayaModelConfig().multiscale_base_patch)
    p.add_argument("--multiscale_gate_temperature", type=float, default=LayaModelConfig().multiscale_gate_temperature)
    p.add_argument("--channel_metadata_mode", type=str, default="none", choices=["onehot", "text", "stats", "text_stats_joint", "text_stats_avg", "none"])
    p.add_argument("--metadata_fusion_mode", type=str, default=LayaModelConfig().metadata_fusion_mode, choices=["none", "add", "concat_kv", "attention_gate", "attention_suppress_gate"])
    p.add_argument("--onehot_channel_vocab_size", type=int, default=0)
    p.add_argument("--channel_mixer_type", type=str, default="independent", choices=["mixer", "independent", "ci_adapter"])
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
    p.add_argument("--channel_relation_heads", type=int, default=LayaModelConfig().channel_relation_heads)
    p.add_argument("--channel_relation_gate_scale_init", type=float, default=LayaModelConfig().channel_relation_gate_scale_init)
    p.add_argument("--channel_relation_residual_scale_init", type=float, default=LayaModelConfig().channel_relation_residual_scale_init)
    p.add_argument("--encoder_variant", type=str, default=LayaModelConfig().encoder_variant, choices=["default"])
    p.add_argument("--temporal_patchifier_mode", type=str, default=LayaModelConfig().temporal_patchifier_mode, choices=["fixed", "multiscale", "charm_like"])
    p.add_argument("--charm_kernel_sizes", type=str, default=",".join(str(v) for v in LayaModelConfig().charm_kernel_sizes))
    p.add_argument("--charm_stride", type=int, default=LayaModelConfig().charm_stride)
    p.add_argument("--charm_patchifier_dropout", type=float, default=LayaModelConfig().charm_patchifier_dropout)
    p.add_argument("--charm_scale_gate_source", type=str, default=LayaModelConfig().charm_scale_gate_source, choices=["learned", "text"])
    p.add_argument("--charm_scale_gate_temperature", type=float, default=LayaModelConfig().charm_scale_gate_temperature)
    p.add_argument("--charm_patchifier_fusion", type=str, default=LayaModelConfig().charm_patchifier_fusion, choices=["replace", "residual"])
    p.add_argument("--charm_patchifier_residual_init", type=float, default=LayaModelConfig().charm_patchifier_residual_init)
    p.add_argument("--text_encoder_name_or_path", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--text_metadata_cache_dir", type=str, default="./metadata_cache")
    p.add_argument("--text_encoder_local_files_only", action="store_true")
    p.add_argument("--stats_metadata_dim", type=int, default=LayaModelConfig().stats_metadata_dim)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", type=str, default="laya_ts/runs/random_forecasting")
    _add_bool_optional_arg(p, "--use_revin", default=True)
    _add_bool_optional_arg(p, "--revin_affine", default=False)
    _add_bool_optional_arg(p, "--revin_subtract_last", default=False)
    p.add_argument("--revin_eps", type=float, default=1e-5)
    args = p.parse_args(argv)

    set_seed(args.seed)
    train_loader, val_loader, test_loader, scaler = get_forecasting_loaders(
        args.data_path,
        args.dataset_type,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        num_workers=args.num_workers,
        channel_metadata_mode=args.channel_metadata_mode,
        text_encoder_name_or_path=args.text_encoder_name_or_path,
        text_metadata_cache_dir=args.text_metadata_cache_dir,
        text_encoder_local_files_only=args.text_encoder_local_files_only,
    )
    first_batch = next(iter(train_loader))
    input_channels = first_batch["series"].shape[1]
    out_channels = first_batch["target"].shape[1]
    onehot_vocab_size = args.onehot_channel_vocab_size
    if args.channel_metadata_mode == "onehot":
        onehot_vocab_size = max(onehot_vocab_size, input_channels)
    model_cfg = LayaModelConfig(
        patch_size=args.patch_size,
        embed_dim=args.d_model,
        depth=args.n_layers,
        num_heads=args.n_heads,
        proj_dim=args.proj_dim,
        predictor_depth=args.predictor_depth,
        predictor_heads=args.predictor_heads,
        patchifier_mode=args.patchifier_mode,
        multiscale_patch_sizes=_parse_int_list(args.multiscale_patch_sizes) or LayaModelConfig().multiscale_patch_sizes,
        multiscale_base_patch=args.multiscale_base_patch,
        multiscale_gate_temperature=args.multiscale_gate_temperature,
        channel_metadata_mode=args.channel_metadata_mode,
        metadata_fusion_mode=args.metadata_fusion_mode,
        channel_mixer_type=args.channel_mixer_type,
        channel_mixer_relation_mode=args.channel_mixer_relation_mode,
        channel_mixer_relation_scale_init=args.channel_mixer_relation_scale_init,
        use_relation_adapter=args.use_relation_adapter,
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
        onehot_channel_vocab_size=onehot_vocab_size,
        use_channel_relation_block=args.use_channel_relation_block,
        channel_relation_heads=args.channel_relation_heads,
        channel_relation_gate_scale_init=args.channel_relation_gate_scale_init,
        channel_relation_residual_scale_init=args.channel_relation_residual_scale_init,
        encoder_variant=args.encoder_variant,
        temporal_patchifier_mode=args.temporal_patchifier_mode,
        charm_kernel_sizes=_parse_int_list(args.charm_kernel_sizes) or LayaModelConfig().charm_kernel_sizes,
        charm_stride=args.charm_stride,
        charm_patchifier_dropout=args.charm_patchifier_dropout,
        charm_scale_gate_source=args.charm_scale_gate_source,
        charm_scale_gate_temperature=args.charm_scale_gate_temperature,
        charm_patchifier_fusion=args.charm_patchifier_fusion,
        charm_patchifier_residual_init=args.charm_patchifier_residual_init,
        stats_metadata_dim=args.stats_metadata_dim,
    )
    num_patches = _num_patches_from_config(model_cfg, first_batch["series"].shape[-1])
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

    for parameter in model.encoder.parameters():
        parameter.requires_grad = False
    finetune_params = list(model.head.parameters())
    print("=" * 50)
    print("🚀 Architecture: LAYA Random Encoder")
    print(f"📊 Target Dataset: {args.dataset_type} (Channels: {input_channels})")
    print(f"📊 Prediction Length: {args.pred_len}")
    print("=" * 50)
    print("✅ Encoder is RANDOMLY initialized and FROZEN (Linear Probing mode).")
    optimizer = torch.optim.AdamW(finetune_params, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()
    os.makedirs(args.log_dir, exist_ok=True)
    writer = SummaryWriter(args.log_dir)
    best_state = None
    best_val = float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss = 0.0
            train_metadata_usage = {}
            for batch_idx, batch in enumerate(train_loader):
                x = batch["series"].to(args.device)
                y = batch["target"].to(args.device)
                pos = batch["channel_positions"].to(args.device)
                mask = batch["channel_mask"].to(args.device)
                channel_text_embeddings = batch.get("channel_text_embeddings")
                if channel_text_embeddings is not None:
                    channel_text_embeddings = channel_text_embeddings.to(args.device)
                channel_stats_embeddings = batch.get("channel_stats_embeddings")
                if channel_stats_embeddings is not None:
                    channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                optimizer.zero_grad(set_to_none=True)
                if batch_idx == 0:
                    pred, features = model(
                        x, pos, mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                        return_features=True,
                    )
                    train_metadata_usage = summarize_metadata_usage(features)
                else:
                    pred = model(
                        x, pos, mask,
                        channel_text_embeddings=channel_text_embeddings,
                        channel_stats_embeddings=channel_stats_embeddings,
                    )
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * x.size(0)
            train_loss /= max(1, len(train_loader.dataset))
            writer.add_scalar("train/mse", train_loss, epoch)
            for key, value in train_metadata_usage.items():
                writer.add_scalar(f"train_meta/{key}", value, epoch)

            model.eval()
            val_losses = []
            val_metadata_usage = {}
            with torch.no_grad():
                for batch_idx, batch in enumerate(val_loader):
                    x = batch["series"].to(args.device)
                    y = batch["target"].to(args.device)
                    pos = batch["channel_positions"].to(args.device)
                    mask = batch["channel_mask"].to(args.device)
                    channel_text_embeddings = batch.get("channel_text_embeddings")
                    if channel_text_embeddings is not None:
                        channel_text_embeddings = channel_text_embeddings.to(args.device)
                    channel_stats_embeddings = batch.get("channel_stats_embeddings")
                    if channel_stats_embeddings is not None:
                        channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                    if batch_idx == 0:
                        pred, features = model(
                            x, pos, mask,
                            channel_text_embeddings=channel_text_embeddings,
                            channel_stats_embeddings=channel_stats_embeddings,
                            return_features=True,
                        )
                        val_metadata_usage = summarize_metadata_usage(features)
                    else:
                        pred = model(
                            x, pos, mask,
                            channel_text_embeddings=channel_text_embeddings,
                            channel_stats_embeddings=channel_stats_embeddings,
                        )
                    val_losses.append(criterion(pred, y).item() * x.size(0))
            val_loss = float(np.sum(val_losses) / max(1, len(val_loader.dataset)))
            writer.add_scalar("val/mse", val_loss, epoch)
            for key, value in val_metadata_usage.items():
                writer.add_scalar(f"val_meta/{key}", value, epoch)
            print(
                f"Epoch {epoch:>3}/{args.epochs} | Train MSE: {train_loss:.4f} | "
                f"Val MSE: {val_loss:.4f} | Train {_format_metadata_usage(train_metadata_usage)} | "
                f"Val {_format_metadata_usage(val_metadata_usage)}"
            )
            if val_loss <= best_val:
                best_val = val_loss
                best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        trues, preds = [], []
        with torch.no_grad():
            for batch in test_loader:
                x = batch["series"].to(args.device)
                y = batch["target"].to(args.device)
                pos = batch["channel_positions"].to(args.device)
                mask = batch["channel_mask"].to(args.device)
                channel_text_embeddings = batch.get("channel_text_embeddings")
                if channel_text_embeddings is not None:
                    channel_text_embeddings = channel_text_embeddings.to(args.device)
                channel_stats_embeddings = batch.get("channel_stats_embeddings")
                if channel_stats_embeddings is not None:
                    channel_stats_embeddings = channel_stats_embeddings.to(args.device)
                pred = model(
                    x, pos, mask,
                    channel_text_embeddings=channel_text_embeddings,
                    channel_stats_embeddings=channel_stats_embeddings,
                )
                trues.append(y.cpu().numpy())
                preds.append(pred.cpu().numpy())
        y_true = np.concatenate(trues, axis=0)
        y_pred = np.concatenate(preds, axis=0)
        print()
        print(f"Random Encoder Forecasting Result ({args.dataset_type}, pred_len={args.pred_len})")
        print(f"MSE : {_mse(y_true, y_pred):.6f}")
        print(f"MAE : {_mae(y_true, y_pred):.6f}")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
