import pytest
import torch

from laya.config import LayaModelConfig
from laya_ts.model import LayaTSEncoder


def test_laya_ts_encoder_supports_attention_gate_metadata_fusion():
    config = LayaModelConfig(
        embed_dim=48,
        depth=2,
        num_heads=2,
        patch_size=8,
        num_queries=4,
        channel_mixer_dim=16,
        channel_mixer_heads=2,
        proj_dim=16,
        channel_mixer_type="mixer",
        channel_metadata_mode="text",
        metadata_fusion_mode="attention_gate",
        text_metadata_dim=12,
        channel_mixer_relation_mode="none",
        description_relation_metric="cosine",
        description_relation_lambda_init=0.0,
        description_relation_gamma_init=1.0,
    )
    encoder = LayaTSEncoder(config)
    x = torch.randn(2, 5, 32)
    text_embeddings_a = torch.randn(2, 5, 12)
    text_embeddings_b = torch.randn(2, 5, 12)

    outputs_a = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 5, dtype=torch.bool),
        channel_text_embeddings=text_embeddings_a,
    )
    outputs_b = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 5, dtype=torch.bool),
        channel_text_embeddings=text_embeddings_b,
    )

    assert outputs_a["mixed_tokens"].shape == (2, 4, 48)
    assert outputs_a["mixed_repr"].shape == (2, 48)
    assert outputs_a["channel_affinity"] is not None
    assert outputs_a["channel_affinity"].shape == (2, 4, 2, 4, 5)
    assert outputs_a["channel_mixer_relation_scores"] is not None
    assert outputs_a["channel_mixer_relation_scores"].shape == (2, 2, 5, 5)
    assert outputs_a["channel_mixer_refined_tokens"] is not None
    assert outputs_a["channel_mixer_refined_tokens"].shape == (2, 5, 4, 16)
    assert outputs_a["channel_mixer_refiner_attention"] is not None
    assert outputs_a["channel_mixer_refiner_attention"].shape == (2, 4, 2, 5, 5)
    assert outputs_a["channel_mixer_latent_tokens"] is None
    assert torch.allclose(outputs_a["mixed_tokens"], outputs_b["mixed_tokens"])
    assert torch.allclose(outputs_a["channel_tokens"], outputs_b["channel_tokens"])
    assert torch.isfinite(outputs_a["mixed_tokens"]).all()
    assert torch.isfinite(outputs_a["channel_mixer_relation_scores"]).all()


def test_laya_ts_encoder_supports_charm_style_attention_suppression():
    config = LayaModelConfig(
        embed_dim=48,
        depth=2,
        num_heads=2,
        patch_size=8,
        num_queries=4,
        channel_mixer_dim=16,
        channel_mixer_heads=2,
        proj_dim=16,
        channel_mixer_type="mixer",
        channel_metadata_mode="text",
        metadata_fusion_mode="attention_suppress_gate",
        text_metadata_dim=12,
        channel_mixer_relation_mode="none",
        description_relation_metric="cosine",
        description_relation_lambda_init=1.0,
        description_relation_gamma_init=1.0,
    )
    encoder = LayaTSEncoder(config)
    x = torch.randn(2, 5, 32)
    text_embeddings_a = torch.randn(2, 5, 12)
    text_embeddings_b = text_embeddings_a.roll(shifts=1, dims=1)

    outputs_a = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 5, dtype=torch.bool),
        channel_text_embeddings=text_embeddings_a,
    )
    outputs_b = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 5, dtype=torch.bool),
        channel_text_embeddings=text_embeddings_b,
    )

    assert outputs_a["channel_mixer_relation_scores"] is not None
    assert outputs_a["channel_mixer_relation_scores"].shape == (2, 2, 5, 5)
    assert outputs_a["channel_mixer_relation_threshold"] is not None
    assert outputs_a["channel_mixer_relation_threshold"].shape == (2, 2, 5, 5)
    assert outputs_a["channel_mixer_relation_gate"] is not None
    assert outputs_a["channel_mixer_relation_gate"].shape == (2, 2, 5, 5)
    assert outputs_a["channel_mixer_refined_tokens"] is not None
    assert outputs_a["channel_mixer_refiner_attention"] is not None
    assert torch.isfinite(outputs_a["channel_mixer_relation_scores"]).all()
    assert torch.isfinite(outputs_a["channel_mixer_relation_threshold"]).all()
    assert torch.isfinite(outputs_a["channel_mixer_relation_gate"]).all()
    diag = outputs_a["channel_mixer_relation_gate"].diagonal(dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag))
    assert not torch.allclose(outputs_a["mixed_tokens"], outputs_b["mixed_tokens"])


def test_attention_gate_requires_text_metadata_mode():
    with pytest.raises(ValueError, match="channel_metadata_mode='text'"):
        LayaTSEncoder(
            LayaModelConfig(
                embed_dim=32,
                depth=1,
                num_heads=2,
                patch_size=8,
                num_queries=2,
                channel_mixer_dim=16,
                proj_dim=16,
                channel_mixer_type="mixer",
                channel_metadata_mode="onehot",
                metadata_fusion_mode="attention_gate",
                onehot_channel_vocab_size=8,
                channel_mixer_relation_mode="none",
            )
        )
