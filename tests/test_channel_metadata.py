import torch

from laya_ts.channel_metadata import build_channel_descriptions
from laya_ts.utils import collate_with_positions


def test_build_channel_descriptions_matches_requested_template():
    descriptions = build_channel_descriptions(
        ["load_1", "load_2"],
        domain="electricity",
        dataset_name="electricity",
    )
    assert descriptions == [
        "In the electricity domain, this channel belongs to the electricity dataset and represents the feature load 1.",
        "In the electricity domain, this channel belongs to the electricity dataset and represents the feature load 2.",
    ]


def test_collate_with_positions_stacks_optional_text_embeddings():
    batch = [
        {
            "series": torch.randn(2, 8),
            "channel_positions": torch.randn(2, 3),
            "channel_mask": torch.tensor([True, True]),
            "channel_names": ["a", "b"],
            "channel_text_embeddings": torch.randn(2, 4),
        },
        {
            "series": torch.randn(2, 8),
            "channel_positions": torch.randn(2, 3),
            "channel_mask": torch.tensor([True, True]),
            "channel_names": ["a", "b"],
            "channel_text_embeddings": torch.randn(2, 4),
        },
    ]
    out = collate_with_positions(batch)
    assert out["channel_text_embeddings"].shape == (2, 2, 4)
