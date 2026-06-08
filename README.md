# laya_ts

`laya_ts` reuses the standalone `laya` model **as-is** for time-series work and borrows
dataset loading / split conventions from `lejepa`.

## What is reused from `laya`

- `LayaEncoder`
- `LayaPretrainer`
- `LayaModelConfig`
- latent prediction objective and SIGReg

## What is adapted for time series

- tslib pretraining dataset loading
- Electricity CSV pretraining dataset loading
- forecasting dataset (`seq_len -> pred_len`)
- classification dataset (UCR/UEA `.ts` and generic CSV)
- synthetic per-channel positions derived from feature index order

## Pretraining

### tslib

```bash
bash laya_ts/scripts/pretrain_tslib_laya_mixer_text.sh
```

Pretraining now runs by `epoch` rather than a fixed `step` budget. By default the
script saves both per-epoch checkpoints such as
`checkpoints/tslib_laya_mixer_text/laya_ts_tslib_s_epoch_100.pt` and a rolling
`checkpoints/tslib_laya_mixer_text/laya_ts_tslib_s_latest.pt`.
The default schedule is `100` total epochs with `10` warmup epochs.

TensorBoard logs are written by default under:

```bash
laya_ts/runs/pretrain_tslib_laya_mixer_text
```

Inspect them with:

```bash
tensorboard --logdir laya_ts/runs
```

### Electricity

```bash
bash laya_ts/scripts/pretrain_electricity_laya_mixer.sh /path/to/electricity.csv
```

Description relation-aware mixer 실험은 다음 스크립트를 사용합니다.

```bash
bash laya_ts/scripts/pretrain_electricity_laya_description_relation_text.sh
```

## Downstream forecasting

```bash
bash laya_ts/scripts/forecasting_electricity_laya_mixer.sh
```

Description relation-aware mixer 기반 forecasting:

```bash
bash laya_ts/scripts/forecasting_electricity_laya_description_relation_text.sh
```

For tslib-pretrained checkpoints:

```bash
bash laya_ts/scripts/forecasting_tslib_laya_mixer_text.sh
```

## Downstream classification

```bash
bash laya_ts/scripts/classification_electricity_laya_mixer.sh
```

Description relation-aware mixer 기반 classification:

```bash
bash laya_ts/scripts/classification_electricity_laya_description_relation_text.sh
```

For tslib-pretrained checkpoints:

```bash
bash laya_ts/scripts/classification_tslib_laya_mixer_text.sh
```

## Notes

- The core Laya model is not replaced by `lejepa`'s old `model_ts_laya.py`.
- `lejepa` is used only as a reference for dataset splits, normalization, and task conventions.
