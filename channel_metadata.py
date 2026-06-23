from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch.utils.data import get_worker_info

_ENCODER_CACHE: dict[tuple[str, bool, str], tuple[object, object]] = {}


def build_channel_descriptions(
    channel_names: Iterable[str],
    *,
    domain: str,
    dataset_name: str,
) -> list[str]:
    pretty_domain = str(domain).strip().replace("_", " ")
    pretty_dataset = str(dataset_name).strip().replace("_", " ")
    descriptions = []
    for channel_name in channel_names:
        pretty_channel = str(channel_name).strip().replace("_", " ")
        descriptions.append(
            f"In the {pretty_domain} domain, this channel belongs to the {pretty_dataset} dataset and represents the feature {pretty_channel}."
        )
    return descriptions


def build_statistical_channel_descriptions(
    channel_names: Iterable[str],
    *,
    domain: str,
    dataset_name: str,
    stats: list[dict[str, float]],
) -> list[str]:
    channel_name_list = [str(name).strip() for name in channel_names]
    if len(channel_name_list) != len(stats):
        raise ValueError(
            f"Expected the same number of channel names and stats entries, got "
            f"{len(channel_name_list)} names and {len(stats)} stats."
        )
    descriptions = []
    for channel_stats in stats:
        descriptions.append(
            f"Channel statistics over the training split: mean value is {channel_stats['mean']:.3f}, "
            f"standard deviation is {channel_stats['std']:.3f}, minimum value is {channel_stats['min']:.3f}, "
            f"maximum value is {channel_stats['max']:.3f}, median value is {channel_stats['median']:.3f}, "
            f"and mean absolute value is {channel_stats['mean_abs']:.3f}."
        )
    return descriptions


def build_joint_channel_descriptions(
    channel_names: Iterable[str],
    *,
    domain: str,
    dataset_name: str,
    stats: list[dict[str, float]],
) -> list[str]:
    text_descriptions = build_channel_descriptions(
        channel_names,
        domain=domain,
        dataset_name=dataset_name,
    )
    stats_descriptions = build_statistical_channel_descriptions(
        channel_names,
        domain=domain,
        dataset_name=dataset_name,
        stats=stats,
    )
    if len(text_descriptions) != len(stats_descriptions):
        raise ValueError(
            f"Expected same number of text and stats descriptions, got "
            f"{len(text_descriptions)} and {len(stats_descriptions)}."
        )
    return [
        f"{text_desc} {stats_desc}"
        for text_desc, stats_desc in zip(text_descriptions, stats_descriptions)
    ]


def _stable_payload(descriptions: list[str], dataset_name: str, encoder_name: str, template_version: str) -> str:
    payload = {
        "descriptions": descriptions,
        "dataset_name": dataset_name,
        "encoder_name": encoder_name,
        "template_version": template_version,
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True)


def _cache_path(cache_dir: str, descriptions: list[str], dataset_name: str, encoder_name: str, template_version: str) -> Path:
    payload = _stable_payload(descriptions, dataset_name, encoder_name, template_version)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    safe_dataset = (dataset_name or "dataset").replace(os.sep, "_").replace(" ", "_")
    safe_encoder = encoder_name.replace("/", "__")
    return Path(cache_dir) / f"{safe_dataset}__{safe_encoder}__{template_version}__{digest}.pt"


def _mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_state.dtype)
    summed = (last_hidden_state * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def _get_encoder_components(
    encoder_name_or_path: str,
    *,
    local_files_only: bool,
    resolved_device: str,
):
    cache_key = (encoder_name_or_path, bool(local_files_only), resolved_device)
    cached = _ENCODER_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "transformers is required for textual channel metadata. Install it with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(encoder_name_or_path, local_files_only=local_files_only)
    model = AutoModel.from_pretrained(encoder_name_or_path, local_files_only=local_files_only)
    model.eval()
    model.to(resolved_device)
    cached = (tokenizer, model)
    _ENCODER_CACHE[cache_key] = cached
    return cached


def encode_channel_descriptions(
    descriptions: Iterable[str],
    dataset_name: str,
    encoder_name_or_path: str,
    cache_dir: str,
    template_version: str = "laya-ts-v2",
    local_files_only: bool = False,
    device: str | None = None,
) -> torch.Tensor:
    description_list = [str(desc).strip() for desc in descriptions]
    if not description_list:
        raise ValueError("Channel descriptions must not be empty.")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cache_path = _cache_path(cache_dir, description_list, dataset_name, encoder_name_or_path, template_version)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    torch_load_kwargs = {"map_location": "cpu"}
    if "weights_only" in inspect.signature(torch.load).parameters:
        torch_load_kwargs["weights_only"] = True

    if cache_path.exists():
        cached = torch.load(cache_path, **torch_load_kwargs)
        cached_embeddings = cached.get("embeddings")
        if (
            cached.get("descriptions") == description_list
            and torch.is_tensor(cached_embeddings)
            and torch.isfinite(cached_embeddings).all()
        ):
            return cached_embeddings.float()

    worker_info = get_worker_info()
    running_in_dataloader_worker = worker_info is not None
    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if running_in_dataloader_worker and resolved_device.startswith("cuda"):
        resolved_device = "cpu"
    tokenizer, model = _get_encoder_components(
        encoder_name_or_path,
        local_files_only=local_files_only,
        resolved_device=resolved_device,
    )

    with torch.no_grad():
        encoded = tokenizer(description_list, padding=True, truncation=True, return_tensors="pt")
        encoded = {key: value.to(resolved_device) for key, value in encoded.items()}
        outputs = model(**encoded)
        pooled = _mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        embeddings = F.normalize(pooled, dim=-1).detach().cpu().float()

    torch.save(
        {
            "encoder_name": encoder_name_or_path,
            "dataset_name": dataset_name,
            "template_version": template_version,
            "descriptions": description_list,
            "embeddings": embeddings,
        },
        cache_path,
    )
    return embeddings
