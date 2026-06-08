import torch

from laya.config import LayaModelConfig
from laya_ts.model import LayaTSPretrainer


def test_laya_ts_pretrainer_uses_repeatable_seeded_validation_mask():
    config = LayaModelConfig(
        embed_dim=32,
        depth=1,
        num_heads=2,
        patch_size=8,
        num_queries=2,
        channel_mixer_dim=16,
        proj_dim=16,
        channel_mixer_type="mixer",
        channel_metadata_mode="text",
        metadata_fusion_mode="attention_gate",
        text_metadata_dim=10,
    )
    model = LayaTSPretrainer(config)
    x = torch.randn(2, 4, 32)
    text_embeddings = torch.randn(2, 4, 10)
    channel_mask = torch.ones(2, 4, dtype=torch.bool)

    outputs_a = model(
        x,
        channel_positions=None,
        channel_mask=channel_mask,
        channel_text_embeddings=text_embeddings,
        patch_mask_seed=0,
    )
    outputs_b = model(
        x,
        channel_positions=None,
        channel_mask=channel_mask,
        channel_text_embeddings=text_embeddings,
        patch_mask_seed=0,
    )

    assert torch.equal(outputs_a["patch_mask"], outputs_b["patch_mask"])
    assert torch.allclose(outputs_a["loss"], outputs_b["loss"])
