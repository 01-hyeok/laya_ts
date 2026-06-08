import torch

from laya.config import LayaModelConfig
from laya_ts.model import LayaTSEncoder


def test_laya_ts_encoder_supports_patchwise_relation_gate_with_text_metadata():
    config = LayaModelConfig(
        embed_dim=64,
        depth=2,
        num_heads=2,
        patch_size=25,
        num_queries=4,
        channel_mixer_dim=16,
        proj_dim=16,
        channel_mixer_type="mixer",
        channel_metadata_mode="text",
        text_metadata_dim=8,
        use_channel_relation_block=True,
        channel_relation_heads=1,
    )
    encoder = LayaTSEncoder(config)
    x = torch.randn(2, 3, 250)
    text_embeddings = torch.randn(2, 3, 8)
    outputs = encoder(
        x,
        channel_positions=None,
        channel_mask=torch.ones(2, 3, dtype=torch.bool),
        channel_text_embeddings=text_embeddings,
    )
    assert outputs["relation_affinity"] is not None
    assert outputs["relation_affinity"].shape == (2, 10, 1, 3, 3)
    assert outputs["relation_gate"] is not None
    assert outputs["relation_gate"].shape == (2, 1, 3, 3)
    assert torch.isfinite(outputs["relation_affinity"]).all()
    assert torch.isfinite(outputs["relation_gate"]).all()


def test_laya_ts_encoder_supports_patchwise_relation_gate_in_independent_mode():
    config = LayaModelConfig(
        embed_dim=64,
        depth=2,
        num_heads=2,
        patch_size=25,
        num_queries=4,
        channel_mixer_dim=16,
        proj_dim=16,
        channel_mixer_type="independent",
        channel_metadata_mode="onehot",
        onehot_channel_vocab_size=4,
        use_channel_relation_block=True,
        channel_relation_heads=1,
    )
    encoder = LayaTSEncoder(config)
    x = torch.randn(2, 4, 250)
    outputs = encoder(x, channel_positions=None, channel_mask=torch.ones(2, 4, dtype=torch.bool))
    assert outputs["independent_tokens"].shape == (2, 4, 10, 64)
    assert outputs["relation_affinity"] is not None
    assert outputs["relation_affinity"].shape == (2, 10, 1, 4, 4)
    assert outputs["relation_gate"] is not None
    assert outputs["relation_gate"].shape == (2, 1, 4, 4)
