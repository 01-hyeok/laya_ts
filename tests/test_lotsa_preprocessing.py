import numpy as np

from laya_ts.data_pretrain import LOTSABatchStreamingPretrainDataset


def test_lotsa_normalization_uses_observed_values_only():
    train_reference = np.array(
        [
            [1.0, 2.0, np.nan, 4.0],
            [np.inf, 10.0, 10.0, 10.0],
        ],
        dtype=np.float32,
    )
    split = np.array(
        [
            [1.0, np.nan, 4.0, 7.0],
            [10.0, 10.0, np.nan, 10.0],
        ],
        dtype=np.float32,
    )

    normalized, observed = LOTSABatchStreamingPretrainDataset._normalize_split_with_observed_values(
        split,
        train_reference,
    )

    assert observed.dtype == np.bool_
    assert np.isfinite(normalized).all()
    assert normalized[0, 1] == 0.0
    assert normalized[1, 2] == 0.0
    assert observed[0, 1] == np.bool_(False)
    assert observed[1, 2] == np.bool_(False)


def test_lotsa_official_zero_impute_preserves_observed_values():
    split = np.array(
        [
            [1.0, np.nan, 4.0],
            [np.inf, -2.0, 3.0],
        ],
        dtype=np.float32,
    )

    imputed, observed = LOTSABatchStreamingPretrainDataset._zero_impute_with_observed_mask(split)

    assert observed.dtype == np.bool_
    assert np.isfinite(imputed).all()
    assert imputed[0, 0] == np.float32(1.0)
    assert imputed[0, 1] == np.float32(0.0)
    assert imputed[1, 0] == np.float32(0.0)
    assert observed[0, 1] == np.bool_(False)
    assert observed[1, 0] == np.bool_(False)


def test_lotsa_channel_name_resolution_prefers_record_metadata():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["beijing_air_quality"],
        batch_size=2,
        channel_metadata_mode="text",
    )
    record = {
        "metadata": {
            "feature_names": ["PM2.5", "PM10", "SO2"],
        }
    }

    names = dataset._channel_names("beijing_air_quality", record, 3)

    assert names == ["PM2.5", "PM10", "SO2"]


def test_lotsa_channel_name_resolution_ignores_non_name_static_categories():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["Q-TRAFFIC"],
        batch_size=2,
        channel_metadata_mode="text",
    )
    record = {
        "feat_static_cat": [0],
    }

    names = dataset._channel_names("Q-TRAFFIC", record, 1)

    assert names == ["target"]


def test_lotsa_beijing_uses_known_feature_names_when_metadata_is_missing():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["beijing_air_quality"],
        batch_size=2,
        channel_metadata_mode="text",
    )

    names = dataset._channel_names("beijing_air_quality", {}, 11)

    assert names == [
        "PM2.5",
        "PM10",
        "SO2",
        "NO2",
        "CO",
        "O3",
        "TEMP",
        "PRES",
        "DEWP",
        "RAIN",
        "WSPM",
    ]


def test_lotsa_australian_electricity_uses_state_name_for_univariate_series():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["australian_electricity_demand"],
        batch_size=2,
        channel_metadata_mode="text",
    )
    record = {"feat_static_cat": [0]}

    channel_names = dataset._channel_names("australian_electricity_demand", record, 1)
    dataset_name = dataset._metadata_dataset_name("australian_electricity_demand", record)

    assert channel_names == ["Victoria electricity demand"]
    assert dataset_name == "australian_electricity_demand state Victoria"


def test_lotsa_q_traffic_uses_road_segment_identity():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["Q-TRAFFIC"],
        batch_size=2,
        channel_metadata_mode="text",
    )
    record = {"road_segment_id": "R1024"}

    channel_names = dataset._channel_names("Q-TRAFFIC", record, 1)
    dataset_name = dataset._metadata_dataset_name("Q-TRAFFIC", record)

    assert channel_names == ["road segment R1024 traffic speed"]
    assert dataset_name == "Q-TRAFFIC road segment R1024"


def test_lotsa_train_mode_shuffles_subset_order_across_iterations():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["a", "b", "c"],
        batch_size=2,
        mode="train",
        random_seed=7,
    )

    order1 = dataset._iter_subset_names_for_epoch()
    order2 = dataset._iter_subset_names_for_epoch()

    assert sorted(order1) == ["a", "b", "c"]
    assert sorted(order2) == ["a", "b", "c"]
    assert order1 != order2


def test_lotsa_val_mode_keeps_subset_order_fixed():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["a", "b", "c"],
        batch_size=2,
        mode="val",
        random_seed=7,
    )

    order1 = dataset._iter_subset_names_for_epoch()
    order2 = dataset._iter_subset_names_for_epoch()

    assert order1 == ["a", "b", "c"]
    assert order2 == ["a", "b", "c"]


class _SplitProbeLotsaDataset(LOTSABatchStreamingPretrainDataset):
    def __init__(self, *args, available_splits: tuple[str, ...], **kwargs):
        self._available_splits = available_splits
        super().__init__(*args, **kwargs)

    def _available_hf_split_names(self, subset_name: str) -> tuple[str, ...]:
        return self._available_splits


def test_lotsa_official_val_prefers_validation_split():
    dataset = _SplitProbeLotsaDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["beijing_air_quality"],
        batch_size=2,
        mode="val",
        lotsa_split_mode="official",
        available_splits=("train", "validation", "test"),
    )

    split_name = dataset._resolve_hf_split_name("beijing_air_quality")

    assert split_name == "validation"


def test_lotsa_official_val_falls_back_to_test_when_validation_missing():
    dataset = _SplitProbeLotsaDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["beijing_air_quality"],
        batch_size=2,
        mode="val",
        lotsa_split_mode="official",
        available_splits=("train", "test"),
    )

    split_name = dataset._resolve_hf_split_name("beijing_air_quality")

    assert split_name == "test"


def test_lotsa_official_val_raises_without_heldout_split():
    dataset = _SplitProbeLotsaDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["beijing_air_quality"],
        batch_size=2,
        mode="val",
        lotsa_split_mode="official",
        available_splits=("train",),
    )

    try:
        dataset._resolve_hf_split_name("beijing_air_quality")
    except ValueError as exc:
        assert "official split mode requested" in str(exc)
    else:
        raise AssertionError("Expected official LOTSA val split resolution to fail without a held-out split.")


def test_lotsa_official_window_count_counts_valid_random_crop_series():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["beijing_air_quality"],
        batch_size=2,
        mode="val",
        seq_len=16,
        patch_size=4,
        stride=4,
        lotsa_split_mode="official",
        lotsa_sampling_mode="official",
    )

    assert dataset._split_window_count(20) == 1


def test_lotsa_official_patch_crop_bounds_follow_patch_size():
    dataset = LOTSABatchStreamingPretrainDataset(
        dataset_name="Salesforce/lotsa_data",
        subset_names=["beijing_air_quality"],
        batch_size=2,
        mode="train",
        seq_len=16,
        patch_size=4,
        lotsa_sampling_mode="official",
        lotsa_min_patches=2,
    )

    rng = np.random.default_rng(0)
    bounds = dataset._official_crop_bounds(25, rng=rng)

    assert bounds is not None
    start, stop = bounds
    assert 0 <= start < stop <= 25
    assert (stop - start) % 4 == 0
    assert (stop - start) >= 8
    assert (stop - start) <= 16
