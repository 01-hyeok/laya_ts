import torch
import pytest

from laya_ts.config import LayaModelConfig
from laya_ts.model import LayaTSEncoder, MetadataGuidedInterChannelAdapter


def test_relation_adapter_is_identity_for_single_channel():
    adapter = MetadataGuidedInterChannelAdapter(
        token_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    z = torch.randn(2, 1, 5, 32)

    z_out, aux = adapter(z, metadata=None, channel_mask=torch.ones(2, 1, dtype=torch.bool))

    assert torch.allclose(z_out, z)
    assert aux["relation_adapter_attention"] is None


def test_relation_adapter_accepts_missing_metadata_and_preserves_shape():
    adapter = MetadataGuidedInterChannelAdapter(
        token_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    z = torch.randn(2, 3, 5, 32)

    z_out, aux = adapter(z, metadata=None, channel_mask=torch.ones(2, 3, dtype=torch.bool))

    assert z_out.shape == (2, 3, 5, 32)
    assert aux["relation_adapter_attention"] is not None
    assert aux["relation_adapter_attention"].shape == (2, 5, 4, 3, 3)
    assert torch.isfinite(z_out).all()


def test_relation_adapter_accepts_channel_and_batch_metadata_shapes():
    adapter = MetadataGuidedInterChannelAdapter(
        token_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    z = torch.randn(2, 3, 5, 32)
    channel_mask = torch.ones(2, 3, dtype=torch.bool)

    channel_metadata = torch.randn(3, 32)
    batch_metadata = torch.randn(2, 3, 32)

    out_channel, _ = adapter(z, metadata=channel_metadata, channel_mask=channel_mask)
    out_batch, _ = adapter(z, metadata=batch_metadata, channel_mask=channel_mask)

    assert out_channel.shape == z.shape
    assert out_batch.shape == z.shape
    assert torch.isfinite(out_channel).all()
    assert torch.isfinite(out_batch).all()


def test_relation_adapter_rejects_unprojected_metadata_width():
    adapter = MetadataGuidedInterChannelAdapter(
        token_dim=32,
        num_heads=4,
        dropout=0.0,
    )
    z = torch.randn(2, 3, 5, 32)

    with pytest.raises(ValueError, match="projected metadata width"):
        adapter(z, metadata=torch.randn(3, 7), channel_mask=torch.ones(2, 3, dtype=torch.bool))


def test_ci_encoder_skips_relation_adapter_when_disabled():
    config = LayaModelConfig(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=8,
        proj_dim=16,
        channel_mixer_type="independent",
        channel_metadata_mode="none",
        metadata_fusion_mode="none",
        use_relation_adapter=False,
    )
    encoder = LayaTSEncoder(config)
    x = torch.randn(2, 3, 32)

    outputs = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 3, dtype=torch.bool),
    )

    assert outputs["independent_tokens"].shape == (2, 3, 4, 32)
    assert outputs["mixed_tokens"].shape == (6, 4, 32)
    assert outputs["relation_adapter_attention"] is None
    assert outputs["relation_adapter_scale"] is None


def test_ci_encoder_supports_ci_adapter_mode_with_text_metadata():
    config = LayaModelConfig(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=8,
        proj_dim=16,
        channel_mixer_type="ci_adapter",
        channel_metadata_mode="text",
        metadata_fusion_mode="none",
        text_metadata_dim=9,
        relation_num_heads=4,
        relation_dropout=0.0,
        metadata_dropout=0.0,
    )
    encoder = LayaTSEncoder(config)
    x = torch.randn(2, 3, 32)
    text_embeddings = torch.randn(3, 9)

    outputs = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 3, dtype=torch.bool),
        channel_text_embeddings=text_embeddings,
    )

    assert outputs["independent_tokens"].shape == (2, 3, 4, 32)
    assert outputs["mixed_tokens"].shape == (6, 4, 32)
    assert outputs["relation_adapter_attention"] is not None
    assert outputs["relation_adapter_attention"].shape == (2, 4, 4, 3, 3)
    assert outputs["relation_adapter_scale"] is not None
    assert torch.isfinite(outputs["mixed_tokens"]).all()


def test_ci_encoder_metadata_changes_adapter_outputs_when_enabled():
    config = LayaModelConfig(
        embed_dim=32,
        depth=2,
        num_heads=4,
        patch_size=8,
        proj_dim=16,
        channel_mixer_type="ci_adapter",
        channel_metadata_mode="text",
        metadata_fusion_mode="none",
        text_metadata_dim=9,
        relation_num_heads=4,
        relation_dropout=0.0,
        relation_scale_init=1.0,
        metadata_scale_init=1.0,
        metadata_dropout=0.0,
        use_metadata_bias=True,
        use_metadata_gate=True,
    )
    encoder = LayaTSEncoder(config)
    x = torch.randn(2, 3, 32)
    text_embeddings = torch.randn(3, 9)

    outputs_without_metadata = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 3, dtype=torch.bool),
        channel_text_embeddings=None,
    )
    outputs_with_metadata = encoder.forward_features(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 3, dtype=torch.bool),
        channel_text_embeddings=text_embeddings,
    )

    assert outputs_without_metadata["relation_adapter_metadata_present"].item() == 0.0
    assert outputs_with_metadata["relation_adapter_metadata_present"].item() == 1.0
    assert not torch.allclose(
        outputs_without_metadata["mixed_tokens"],
        outputs_with_metadata["mixed_tokens"],
    )
