# Electricity Pretrain 모드 종합 분석 보고서

## 1. 보고서 목적

이 보고서는 `analysis/`에 저장된 결과를 종합하여 Electricity로 사전학습한 각 LayaTS 모드를 다음 다섯 관점에서 평가한다.

1. **Pretraining objective 적합도**: validation loss와 prediction-target alignment
2. **Representation 건강도**: representation similarity와 feature variance
3. **Mixer 구조**: query-to-channel attention의 분화와 post-mixer query collapse
4. **Downstream 기능 기여**: ETTm1-96 forecasting checkpoint에서 unit ablation 효과
5. **Transfer 성능**: forecasting 및 classification 결과

핵심 질문은 단순히 “어느 모델의 loss가 가장 낮은가?”가 아니다. 이 보고서는 다음을 구분한다.

- objective를 잘 푼 모델
- 표현 공간이 건강한 모델
- query가 구조적으로 분화된 모델
- query가 실제 forecast 경로에서 사용되는 모델
- forecasting 또는 classification으로 잘 전이되는 모델

---

## 2. 사용한 자료와 비교 범위

### 2.1 핵심 13개 모드

다음 모드는 pretrain, mixer structure 또는 CI 기준선, ETTm1-96 ablation, forecasting, classification 근거를 함께 비교할 수 있다.

- `ci_none`
- `mixer_stats`
- `mixer_text`
- `mixer_text_stats_avg`
- `mixer_text_stats_joint`
- `mixer_concat_stats`
- `mixer_concat_text`
- `mixer_concat_text_stats_avg`
- `mixer_concat_text_stats_joint`
- `metadata_query_gate_stats`
- `metadata_query_gate_text`
- `metadata_query_gate_text_stats_avg`
- `metadata_query_gate_text_stats_joint`

### 2.2 추가 downstream 전용 모드

다음 모드는 forecasting 또는 classification CSV에는 있지만, 현재 `analysis/`에 같은 수준의 mixer structure와 ablation 결과가 없다.

- `metadata_query_bias_stats`, `metadata_query_bias_text`
- `description_relation_text`
- `description_suppression_stats`
- `description_suppression_text`
- `description_suppression_text_stats_avg`
- `description_suppression_text_stats_joint`
- `relation_text`는 classification에서 EthanolConcentration 한 건만 존재

따라서 이 모드들은 downstream 결과만 평가하며 내부 표현에 대한 강한 결론은 내리지 않는다.

### 2.3 중요한 시간축 차이

`forecasting_electricity_results_0609.csv`는 **2026-06-09 04:29 UTC**에 생성되었다. 반면 현재 ETTm1-96 forecasting checkpoint들은 **14:36~14:57 UTC**에 다시 생성되었고, ablation 결과는 **14:59~15:08 UTC**에 평가되었다.

따라서 다음처럼 값이 일치하지 않는 경우가 있다.

| 모드 | 오전 forecasting CSV ETTm1-96 MSE | 오후 ablation baseline MSE |
|---|---:|---:|
| `ci_none` | 0.325 | 0.3253 |
| `metadata_query_gate_stats` | 0.385 | 0.4036 |
| `mixer_stats` | 0.626 | 0.5262 |
| `mixer_concat_stats` | 0.909 | 0.6473 |
| `mixer_text_stats_joint` | 0.485 | 0.4636 |
| `mixer_concat_text_stats_joint` | 0.588 | 0.6077 |

이 보고서에서는:

- **전체 데이터셋·예측 길이 경향**은 오전 forecasting CSV를 사용한다.
- **ETTm1-96 unit importance**는 더 최근 checkpoint를 사용한 ablation 결과를 사용한다.
- 두 값을 같은 checkpoint의 반복 측정값으로 간주하지 않는다.

---

## 3. 지표 해석 가이드

### 3.1 Pretrain 지표

#### `best_val_loss`

pretraining validation objective의 최솟값이다. 낮을수록 objective fit은 좋지만, representation collapse와 독립적인 지표이므로 단독 순위로 사용하면 안 된다.

#### `val_align_cos`

predicted token과 target token의 cosine alignment다. 높을수록 pretraining target을 방향적으로 잘 복원했다는 뜻이다.

- `0.90 이상`: 매우 강함
- `0.80대`: 강함
- `0.70대`: 보통
- `0.50 이하`: 약함

#### `val_repr_cos`

서로 다른 representation 간 평균 cosine similarity다. 너무 높으면 샘플 표현이 같은 방향으로 뭉친 collapse 가능성이 있다.

- `0.99 이상`: 강한 collapse 경고
- `0.95~0.99`: 과도한 집중 가능성
- `0.85~0.95`: 비교적 건강
- 낮은 값은 다양성을 뜻하지만, objective alignment가 함께 낮다면 단순한 불안정성일 수도 있다.

#### `val_repr_var`

feature 차원별 평균 분산이다. 너무 낮으면 표현 공간을 거의 사용하지 못하는 상태다.

- `0.02 이상`: 다양성이 비교적 살아 있음
- `0.005 이하`: collapse 경고
- `0.001 수준`: 매우 강한 collapse 의심

### 3.2 Mixer structure 지표

현재 structure 결과는 모든 모드가 **Electricity validation 1 batch, 321 channels, 16 queries** 기준이다. 동일 조건 비교에는 유효하지만 전체 validation 분포로 일반화할 때 주의가 필요하다.

#### `mean_attention_entropy`

query attention이 채널에 얼마나 넓게 퍼졌는지 나타낸다.

- 낮음: 소수 채널에 집중
- 높음: 많은 채널에 분산
- 높거나 낮은 것 자체가 정답은 아니다. `top1_unique_ratio`, overlap, downstream ablation과 함께 봐야 한다.

321개 채널에서 완전 균등분포의 entropy는 약 `ln(321)=5.77`이다.

#### `top1_unique_ratio`

16개 query의 top-1 채널 중 고유 채널 수를 16으로 나눈 값이다.

- `1.0`: 모든 query가 서로 다른 top-1 채널 선택
- 낮은 값: 여러 query가 같은 채널을 반복 선택

#### `mean_attention_overlap_offdiag`

query별 attention vector 사이 cosine similarity의 off-diagonal 평균이다.

- 낮을수록 query가 서로 다른 채널 분포를 읽음
- 높을수록 query들이 비슷한 채널을 읽음

#### Post-query similarity 지표

- `mean_post_query_similarity_offdiag`: query 쌍 평균 cosine similarity
- `std`: query 쌍 간 편차
- `p90`: 상위 10% query 쌍의 similarity 기준점
- `collapse_ratio_ge_0.9`: cosine similarity가 0.9 이상인 query 쌍 비율

평균만 낮아도 안심할 수 없다. 평균이 낮고 `p90`과 collapse ratio가 높다면, 서로 반대 방향인 query와 거의 동일한 query가 함께 존재하는 **양극화된 구조**일 수 있다.

#### `mean_topk_pre_similarity`

각 query가 attention으로 선택한 top-k 채널들이 mixer 이전부터 서로 얼마나 비슷했는지 나타낸다.

- 높음: query가 이미 유사한 채널 묶음을 모음
- 낮음: query가 서로 다른 pre-mixer 채널들을 결합
- 낮다고 항상 좋은 것은 아니며, 이질적 채널 통합인지 무질서한 선택인지 downstream 근거가 필요하다.

### 3.3 Downstream ablation 지표

#### CI 모드

`independent_tokens`에서 채널 하나를 0으로 만든다. 각 output channel의 representation 자체를 제거하므로 성능 저하가 크게 나오는 것이 자연스럽다.

#### Mixer 모드

`channel_affinity[:, :, :, query_id, :]`를 0으로 만든 뒤, 남은 query 평균으로 channel importance를 다시 계산한다. 이는 post-mixer latent query 자체를 제거하는 실험이 아니라 **query가 channel weighting 경로에 미치는 기여**를 제거하는 실험이다.

따라서 CI의 `delta MSE`와 mixer의 `delta MSE` 크기를 직접 비교해서 “CI channel이 query보다 중요하다”고 결론내리면 안 된다.

현재 결과는 기본값인 `zero` ablation이다. 제거한 query 뒤에 남은 15개 query를 다시 정규화하지 않고 원래 16개 query 축으로 평균하므로, 절대 delta에는 다음 두 효과가 함께 들어간다.

- 해당 query가 선택하던 고유 channel pattern의 손실
- 전체 query affinity mass가 약 `1/16` 줄어드는 공통 scaling 효과

따라서 mixer ablation은 절대 delta 자체보다 **같은 모드 안에서 query 간 상대 순위**, max와 mean의 차이, 여러 query에 효과가 분산되는지를 중심으로 해석하는 것이 안전하다.

#### 해석

- `max_delta_mse`: 가장 중요한 단일 unit 제거 효과
- `mean_delta_mse`: 전체 unit 제거 효과 평균
- `delta / baseline`: baseline 난이도를 보정한 상대적 성능 저하
- max만 크고 mean이 작으면 특정 query에 기능이 집중된 구조
- max와 mean이 모두 크면 여러 query가 고르게 forecast 경로에 기여하는 구조

### 3.4 Downstream 성능 지표

- Forecasting: MSE와 MAE 모두 낮을수록 좋다.
- Classification: accuracy와 macro F1이 높을수록 좋다.
- 불균형 데이터에서는 accuracy보다 macro F1이 더 중요하다.
- 현재 결과는 seed 반복의 평균과 표준편차가 없으므로 작은 차이는 확정적인 우열로 해석하지 않는다.

---

## 4. 전체 결과 요약

### 4.1 Pretrain 표현 지표

| 모드 | best val loss | align cos | repr cos | repr var | 핵심 판정 |
|---|---:|---:|---:|---:|---|
| `ci_none` | 0.1885 | **0.9221** | 0.9541 | 0.0169 | objective와 표현 균형이 가장 안정적 |
| `mixer_stats` | 0.3907 | 0.8733 | 0.8505 | 0.0684 | 표현 다양성이 건강함 |
| `mixer_text` | 0.0600 | 0.7812 | 0.9964 | 0.00264 | low loss지만 global collapse 경고 |
| `mixer_text_stats_avg` | 0.2672 | 0.7724 | 0.9950 | 0.00359 | global collapse 경고 |
| `mixer_text_stats_joint` | 0.3029 | **0.4632** | 0.9662 | 0.0296 | 다양성은 있으나 alignment가 매우 약함 |
| `mixer_concat_stats` | 0.2675 | 0.7927 | 0.9797 | 0.0170 | best 이후 final val loss 악화, 불안정 |
| `mixer_concat_text` | **0.0384** | 0.8015 | **0.9995** | **0.00035** | objective 최상, representation collapse 최강 |
| `mixer_concat_text_stats_avg` | 0.2093 | 0.7703 | 0.9671 | 0.0334 | 비교적 균형적인 concat 표현 |
| `mixer_concat_text_stats_joint` | 0.1970 | 0.8723 | 0.9915 | 0.00593 | alignment는 좋지만 약한 collapse 위험 |
| `metadata_query_gate_stats` | 0.2910 | 0.8402 | 0.9697 | 0.0237 | objective·다양성의 균형이 좋음 |
| `metadata_query_gate_text` | 0.4403 | 0.7117 | 0.8721 | 0.0710 | 다양성은 건강, objective는 약함 |
| `metadata_query_gate_text_stats_avg` | 0.4737 | 0.6694 | 0.9767 | 0.0164 | objective와 다양성 모두 중하위 |
| `metadata_query_gate_text_stats_joint` | 0.3751 | 0.7503 | **0.7032** | **0.1395** | 가장 풍부한 표현, objective는 중간 |

핵심적으로 `mixer_concat_text`는 loss만 보면 최고지만 representation learner로는 가장 위험하다. 반대로 `metadata_query_gate_text_stats_joint`는 loss 1등은 아니지만 표현 공간은 가장 풍부하다.

### 4.2 Mixer structure 지표

| 모드 | entropy | top1 unique | attn overlap | post mean | post std | post p90 | collapse >=.9 | top-k pre sim |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mixer_stats` | 3.1879 | 0.4375 | 0.2174 | 0.8098 | 0.1887 | 0.9875 | 0.4083 | **0.9315** |
| `mixer_text` | 2.1688 | **1.0000** | **0.0058** | 0.0521 | 0.4162 | 0.6399 | **0.0000** | 0.8289 |
| `mixer_text_stats_avg` | 2.7232 | 0.8125 | 0.0607 | 0.2849 | 0.4039 | 0.7912 | 0.0500 | 0.8015 |
| `mixer_text_stats_joint` | 2.0665 | 0.8125 | 0.0505 | 0.5515 | 0.2076 | 0.8202 | 0.0250 | 0.8575 |
| `mixer_concat_stats` | 2.5478 | **0.3125** | 0.3221 | **0.8903** | 0.0764 | 0.9877 | **0.4750** | 0.5327 |
| `mixer_concat_text` | 2.5541 | **1.0000** | 0.0146 | 0.1000 | 0.3818 | 0.5386 | 0.0167 | **0.4979** |
| `mixer_concat_text_stats_avg` | 3.4332 | **1.0000** | 0.0753 | 0.1495 | 0.4238 | 0.7856 | 0.0333 | 0.6780 |
| `mixer_concat_text_stats_joint` | **1.8072** | 0.8750 | 0.0338 | 0.5115 | 0.2411 | 0.8508 | 0.0500 | 0.6092 |
| `metadata_query_gate_stats` | **4.3824** | **0.3125** | **0.4602** | 0.5420 | 0.5494 | 0.9990 | 0.4667 | 0.8744 |
| `metadata_query_gate_text` | 2.5099 | 0.8750 | 0.0699 | **0.0368** | **0.9287** | 0.9981 | 0.4250 | 0.6299 |
| `metadata_query_gate_text_stats_avg` | 2.6740 | 0.8125 | 0.0826 | 0.0809 | 0.8690 | 0.9921 | 0.3833 | 0.5823 |
| `metadata_query_gate_text_stats_joint` | 2.6878 | 0.6250 | 0.1337 | 0.0698 | 0.8764 | **0.9956** | 0.3917 | 0.7101 |

구조적으로 가장 명확한 query 분화는 `mixer_text`다. 모든 query의 top-1 채널이 다르고 attention overlap이 거의 0이며, post-query collapse 쌍도 없다.

반대로 `mixer_concat_stats`, `mixer_stats`, `metadata_query_gate_stats`는 query들이 같은 채널을 반복해서 읽고 post-query 유사도도 높다. 특히 `metadata_query_gate_stats`는 entropy가 높고 top-k weight가 약 `0.004~0.006` 수준인 query가 있어, 일부 query가 거의 균등한 분포에 가깝다.

`metadata_query_gate_text*` 계열은 post-query 평균만 보면 매우 낮지만 `p90≈0.99`, collapse ratio `0.38~0.43`이다. 즉 전체가 균일하게 잘 분리된 것이 아니라, 반대 방향 query와 거의 동일한 query cluster가 공존하는 양극화 구조로 해석해야 한다.

### 4.3 최신 ETTm1-96 downstream ablation

| 모드 | baseline MSE | 중요 unit | max delta | mean delta | max 상대 증가 | mean 상대 증가 |
|---|---:|---:|---:|---:|---:|---:|
| `ci_none` | **0.3253** | channel 2 | **0.2278** | **0.1334** | 70.04% | 40.99% |
| `metadata_query_gate_stats` | 0.4036 | q3 | 0.0112 | 0.0067 | 2.76% | 1.67% |
| `mixer_concat_text` | 0.4525 | q6 | 0.0245 | 0.0165 | 5.42% | **3.64%** |
| `mixer_text_stats_joint` | 0.4636 | q8 | 0.0399 | 0.0103 | 8.60% | 2.21% |
| `mixer_text_stats_avg` | 0.4696 | q5 | 0.0362 | 0.0068 | 7.71% | 1.44% |
| `metadata_query_gate_text_stats_avg` | 0.4821 | q3 | 0.0301 | 0.0154 | 6.24% | 3.20% |
| `mixer_text` | 0.4891 | q13 | 0.0492 | 0.0127 | **10.06%** | 2.59% |
| `metadata_query_gate_text` | 0.4931 | q10 | 0.0190 | 0.0134 | 3.85% | 2.71% |
| `metadata_query_gate_text_stats_joint` | 0.4936 | q15 | 0.0216 | 0.0155 | 4.37% | 3.14% |
| `mixer_stats` | 0.5262 | q3 | 0.0361 | 0.0128 | 6.86% | 2.43% |
| `mixer_concat_text_stats_avg` | 0.5549 | q13 | 0.0319 | **0.0172** | 5.75% | 3.10% |
| `mixer_concat_text_stats_joint` | 0.6077 | q9 | **0.0543** | 0.0156 | 8.94% | 2.57% |
| `mixer_concat_stats` | 0.6473 | q7 | 0.0300 | **0.0024** | 4.63% | **0.37%** |

해석상 중요한 패턴은 다음과 같다.

- `mixer_text`는 구조 분화가 가장 명확하고 가장 중요한 query 제거 시 상대 MSE가 10.06% 증가한다. 구조적 specialization과 기능적 specialization이 가장 잘 일치한다.
- `mixer_concat_text`는 mixer 모드 중 mean 상대 증가가 가장 높아 여러 query가 비교적 고르게 forecast 경로에 참여한다.
- `mixer_concat_text_stats_joint`는 q9 의존도가 강하지만 baseline 자체가 좋지 않다. 중요한 query가 존재한다는 것과 좋은 모델이라는 것은 별개다.
- `metadata_query_gate_stats`는 baseline은 mixer 중 가장 좋지만 query ablation 영향은 가장 약한 축이다. 성능이 개별 query specialization보다 공유된 broad weighting 또는 channel token 자체에서 나올 가능성이 크다.
- `mixer_concat_stats`는 mean ablation 효과가 0.37%뿐이다. 구조 collapse와 낮은 기능적 query 활용이 함께 나타난다.

### 4.4 Forecasting 전체 평균

각 모드별로 ETTm1, ETTm2, Weather와 prediction length 96/192/336/720, 총 12개 결과를 단순 평균했다.

| 순위 | 모드 | 평균 MSE | 평균 MAE | 판정 |
|---:|---|---:|---:|---|
| 1 | `ci_none` | **0.3027** | **0.3623** | 모든 데이터셋에서 가장 강한 기준선 |
| 2 | `metadata_query_gate_stats` | 0.4233 | 0.4684 | mixer 계열 최상 |
| 3 | `mixer_text_stats_avg` | 0.5223 | 0.5319 | plain mixer 최상 |
| 4 | `mixer_concat_text` | 0.5415 | 0.5428 | loss/collapse 경고에도 forecasting은 비교적 강함 |
| 5 | `metadata_query_gate_text_stats_avg` | 0.5590 | 0.5612 | 중상위 |
| 6 | `mixer_text` | 0.5789 | 0.5726 | 구조 분화는 강하지만 성능은 중간 |
| 7 | `metadata_query_gate_text_stats_joint` | 0.5965 | 0.5822 | classification 대비 forecasting은 약함 |
| 8 | `mixer_concat_text_stats_avg` | 0.6085 | 0.5833 | 평균적 |
| 9 | `mixer_text_stats_joint` | 0.6505 | 0.6069 | 약함 |
| 10 | `mixer_concat_text_stats_joint` | 0.6507 | 0.5803 | MSE가 약함 |
| 11 | `metadata_query_gate_text` | 0.6548 | 0.6073 | classification 대비 forecasting이 약함 |
| 12 | `mixer_stats` | 0.7306 | 0.6462 | pretrain 건강도 대비 transfer 실패 |
| 13 | `mixer_concat_stats` | **0.9125** | **0.7112** | 핵심 모드 중 최하 |

데이터셋별로도 `ci_none`이 ETTm1 `0.4010`, ETTm2 `0.2180`, Weather `0.2890`으로 모두 1위다. `metadata_query_gate_stats`도 각각 `0.4568`, `0.3300`, `0.4830`으로 일관되게 mixer 1위다.

### 4.5 Classification 전체 평균

14개 공통 classification 데이터셋의 단순 평균이다. `relation_text`는 한 데이터셋만 있어 평균 비교에서 제외한다.

| 순위 | 모드 | 평균 accuracy | 평균 macro F1 |
|---:|---|---:|---:|
| 1 | `metadata_query_gate_text_stats_joint` | **0.5424** | **0.4814** |
| 2 | `metadata_query_gate_text` | 0.5324 | 0.4729 |
| 3 | `metadata_query_gate_text_stats_avg` | 0.5330 | 0.4676 |
| 4 | `ci_none` | 0.5409 | 0.4600 |
| 5 | `mixer_stats` | 0.4951 | 0.4051 |
| 6 | `metadata_query_gate_stats` | 0.4962 | 0.4023 |
| 7 | `mixer_concat_text_stats_joint` | 0.4919 | 0.3817 |
| 8 | `mixer_text_stats_avg` | 0.4803 | 0.3739 |
| 9 | `mixer_concat_text_stats_avg` | 0.4674 | 0.3603 |
| 10 | `mixer_concat_stats` | 0.4661 | 0.3526 |
| 11 | `mixer_text_stats_joint` | 0.4688 | 0.3511 |
| 12 | `mixer_concat_text` | 0.4634 | 0.3466 |
| 13 | `mixer_text` | **0.4544** | **0.3337** |

`metadata_query_gate_text_stats_joint`는 ECG200, FordA, FordB에서 accuracy와 F1 모두 최고이며 전체 macro F1도 1위다. `metadata_query_gate_text_stats_avg`는 JapaneseVowels와 UWaveGestureLibrary에서 최고이고, `ci_none`은 ECG5000, FaceDetection, PEMS-SF에서 최고다.

즉 forecasting에는 stats 기반 metadata가 유리하고, heterogeneous classification transfer에는 text 또는 text+stats metadata가 유리한 경향이 명확하다.

---

## 5. 모드별 종합 평가

### 5.1 `ci_none`

**근거**

- Pretrain: align `0.9221`로 최고, repr cos `0.9541`, var `0.0169`
- Forecasting 평균: MSE `0.3027`, MAE `0.3623`로 압도적 1위
- Classification 평균 F1: `0.4600`, 전체 4위
- ETTm1-96: MSE `0.3253`
- channel ablation: channel 2 제거 시 `+0.2278 MSE`, 평균 `+0.1334`

**해석**

가장 강하고 안전한 baseline이다. 채널 독립 인코딩이 Electricity pretrain에서 얻은 temporal pattern을 ETTm1/2와 Weather로 가장 안정적으로 전이한다. Mixer가 제공하는 cross-channel 압축보다 원래 채널별 정보를 보존하는 것이 현재 forecasting head에는 더 유리하다.

큰 channel ablation 효과는 각 출력 채널의 token을 직접 제거하기 때문에 예상 가능한 결과다. mixer query ablation과 수치 크기를 직접 비교해서는 안 된다.

**판정: 전체 forecasting 최우선 모델이자 기준선**

### 5.2 `mixer_stats`

**근거**

- Pretrain: align `0.8733`, repr cos `0.8505`, var `0.0684`로 건강
- Structure: top1 unique `0.4375`, overlap `0.2174`, post mean `0.8098`, collapse `0.4083`
- Ablation: q3 최대 `+6.86%`, 평균 `+2.43%`
- Forecasting 평균 MSE `0.7306`
- Classification F1 `0.4051`

**해석**

pretrain scalar만 보면 매우 좋은 representation learner지만, 최신 구조 분석에서는 query 중복과 post-query collapse가 뚜렷하다. 높은 `mean_topk_pre_similarity=0.9315`는 query가 이미 유사한 채널 묶음을 반복해서 모으는 경향을 보여준다.

즉 sample-level representation diversity는 살아 있어도 query 역할 분화는 약할 수 있다. 이 차이가 forecasting transfer 부진을 설명하는 유력한 근거다.

**판정: pretrain 건강도는 좋지만 downstream 효율이 낮음**

### 5.3 `mixer_text`

**근거**

- Pretrain: loss `0.0600`, repr cos `0.9964`, var `0.00264`로 global collapse 경고
- Structure: top1 unique `1.0`, overlap `0.0058`, post mean `0.0521`, collapse `0`
- Ablation: q13 최대 `+10.06%`, mixer 중 상대 최대
- Forecasting 평균 MSE `0.5789`
- Classification F1 `0.3337`, 핵심 모드 최하

**해석**

가장 흥미로운 모순을 보인다. 샘플 representation은 collapse 성향이 강하지만 query 역할은 가장 선명하게 분화되어 있고 실제 forecast weighting에도 중요한 query가 존재한다.

따라서 “query specialization”과 “좋은 global representation”은 별개다. Text metadata가 query routing에는 강한 구조를 만들었지만, classification에 필요한 sample discrimination은 충분히 만들지 못했다.

**판정: query specialization 연구에는 최적, 범용 transfer 모델로는 부적합**

### 5.4 `mixer_text_stats_avg`

**근거**

- Pretrain: repr cos `0.9950`, var `0.00359`로 collapse 경고
- Structure: top1 unique `0.8125`, overlap `0.0607`, collapse `0.05`
- Ablation: q5 최대 `+7.71%`
- Forecasting 평균 MSE `0.5223`, plain mixer 중 1위
- Classification F1 `0.3739`

**해석**

Text와 stats 평균 결합이 text-only보다 query 분화를 약간 완화하면서 forecasting 성능을 높였다. 하지만 pretrain global representation collapse는 여전히 강하고 classification transfer도 제한적이다.

**판정: plain mixer forecasting을 써야 한다면 가장 실용적인 선택**

### 5.5 `mixer_text_stats_joint`

**근거**

- Pretrain align `0.4632`로 매우 약함
- Structure: overlap `0.0505`, post mean `0.5515`, collapse `0.025`
- Ablation: q8 최대 `+8.60%`
- Forecasting 평균 MSE `0.6505`
- Classification F1 `0.3511`

**해석**

query attention은 어느 정도 분화되어 있고 특정 query도 중요하지만, pretraining target alignment가 약해 downstream 성능으로 연결되지 않는다. 구조적 다양성만으로 objective 학습 실패를 보상하지 못한 사례다.

**판정: 중요한 query는 있으나 전체 모델 품질은 낮음**

### 5.6 `mixer_concat_stats`

**근거**

- Pretrain best loss `0.2675`지만 final val loss `1.0063`으로 악화
- Structure: top1 unique `0.3125`, overlap `0.3221`, post mean `0.8903`, collapse `0.475`
- Ablation: 평균 상대 증가 `0.37%`로 최저
- Forecasting 평균 MSE `0.9125`, 핵심 모드 최하
- Classification F1 `0.3526`

**해석**

학습 불안정, query 중복, post-query collapse, 낮은 기능적 query 활용, 낮은 transfer 성능이 모두 같은 방향을 가리킨다. q7 하나는 영향을 주지만 대부분 query는 forecast weighting에 거의 기여하지 않는다.

**판정: 현재 구성에서는 제거 또는 재설계 우선 대상**

### 5.7 `mixer_concat_text`

**근거**

- Pretrain loss `0.0384`로 최저지만 repr cos `0.9995`, var `0.00035`
- Structure: top1 unique `1.0`, overlap `0.0146`, collapse `0.0167`
- Ablation: max `5.42%`, mean `3.64%`
- Forecasting 평균 MSE `0.5415`, 전체 4위
- Classification F1 `0.3466`

**해석**

objective fit과 query routing은 좋지만 global representation은 거의 collapse했다. Forecasting head는 분화된 channel weighting을 활용해 비교적 좋은 성능을 내지만, classification에서는 sample-level 정보 부족이 드러난다.

`mean_topk_pre_similarity=0.4979`가 가장 낮아 query가 이질적인 채널들을 결합하는 성향도 강하다.

**판정: forecasting 전용으로는 유효, 범용 representation 주장은 위험**

### 5.8 `mixer_concat_text_stats_avg`

**근거**

- Pretrain repr cos `0.9671`, var `0.0334`로 concat 계열 중 건강
- Structure: top1 unique `1.0`, post mean `0.1495`, collapse `0.0333`
- Ablation: mean delta `0.0172`, 상대 `3.10%`
- Forecasting 평균 MSE `0.6085`
- Classification F1 `0.3603`

**해석**

표현 건강도, query 분화, 기능적 활용은 균형적이지만 최종 성능은 중간에 머문다. 구조적으로 건강하다는 것만으로 task-relevant representation이 보장되지 않는 사례다.

**판정: 균형적이나 뚜렷한 downstream 우위 없음**

### 5.9 `mixer_concat_text_stats_joint`

**근거**

- Pretrain align `0.8723`, repr cos `0.9915`
- Structure entropy `1.8072`, 일부 query는 top weight `0.99` 수준으로 매우 sharp
- Post mean `0.5115`, collapse `0.05`
- Ablation: q9 최대 `+8.94%`
- Forecasting 평균 MSE `0.6507`
- Classification F1 `0.3817`

**해석**

특정 채널과 query에 역할이 집중된 구조다. q9 제거 효과도 크지만 baseline 성능은 약하다. Sharp attention과 높은 unit importance는 specialization의 증거이지 좋은 일반화의 증거는 아니다.

**판정: 강한 국소 specialization, 약한 전체 성능**

### 5.10 `metadata_query_gate_stats`

**근거**

- Pretrain align `0.8402`, repr var `0.0237`
- 기존 pretrain metadata score delta ratio 약 `0.9283`
- Structure entropy `4.3824`, top1 unique `0.3125`, overlap `0.4602`, collapse `0.4667`
- Ablation: max `2.76%`, mean `1.67%`
- Forecasting 평균 MSE `0.4233`, mixer 전체 1위
- Classification F1 `0.4023`

**해석**

Forecasting 성능은 가장 강하지만 최신 구조에서는 개별 query specialization이 약하다. Metadata가 score를 강하게 바꾸더라도 그 결과가 날카로운 분업일 필요는 없다. 현재 결과는 stats metadata가 여러 query에 걸쳐 broad channel prior를 제공하고, forecasting이 그 공유된 weighting을 활용하는 구조와 잘 맞는다.

즉 이 모델의 장점은 “해석 가능한 독립 query 전문가”보다 “forecasting에 유용한 전역적 channel bias”에 가깝다.

**판정: mixer 기반 forecasting 최우선 모델**

### 5.11 `metadata_query_gate_text`

**근거**

- Pretrain repr cos `0.8721`, var `0.0710`으로 건강
- Structure top1 unique `0.875`, overlap `0.0699`
- Post mean `0.0368`이지만 p90 `0.9981`, collapse `0.425`
- Ablation 평균 상대 증가 `2.71%`
- Forecasting 평균 MSE `0.6548`
- Classification F1 `0.4729`, 전체 2위

**해석**

Text metadata는 forecasting보다 classification transfer에 훨씬 유리하다. Query attention은 분화되어 있지만 post-query 공간은 일부 동일 cluster와 반대 방향 cluster가 공존한다.

**판정: classification 중심의 metadata 모델**

### 5.12 `metadata_query_gate_text_stats_avg`

**근거**

- Pretrain align `0.6694`, repr cos `0.9767`
- Structure top1 unique `0.8125`, collapse `0.3833`
- Ablation q3 최대 `+6.24%`, 평균 `+3.20%`
- Forecasting 평균 MSE `0.5590`
- Classification F1 `0.4676`, 전체 3위

**해석**

Forecasting과 classification 사이에서 가장 타협적인 metadata gate 모델이다. Stats-only보다 forecasting은 약하지만 text-only보다 개선되고, classification은 상위권을 유지한다.

**판정: 두 downstream task를 함께 고려할 때 균형형 후보**

### 5.13 `metadata_query_gate_text_stats_joint`

**근거**

- Pretrain repr cos `0.7032`, var `0.1395`로 가장 풍부
- Structure top1 unique `0.625`, overlap `0.1337`
- Post mean `0.0698`이지만 p90 `0.9956`, collapse `0.3917`
- Ablation 평균 상대 증가 `3.14%`
- Forecasting 평균 MSE `0.5965`
- Classification accuracy `0.5424`, F1 `0.4814`로 1위

**해석**

가장 풍부한 sample representation이 classification transfer 1위로 이어진다. 반면 query들이 완전히 독립적인 것은 아니고, 일부 post-query cluster는 collapse되어 있다. 풍부한 global representation과 완전한 query 분화는 동일 개념이 아님을 보여준다.

**판정: 범용 classification representation 최우선 모델**

---

## 6. 추가 downstream 모드 평가

### 6.1 Forecasting 평균

| 모드 | 평균 MSE | 평균 MAE | 해석 |
|---|---:|---:|---|
| `metadata_query_bias_stats` | 0.5969 | 0.5406 | Weather는 강하지만 ETTm1/2는 약함 |
| `description_suppression_stats` | 0.6300 | 0.5983 | 중하위 |
| `description_suppression_text` | 0.6300 | 0.5983 | stats와 결과가 완전히 동일 |
| `description_suppression_text_stats_joint` | 0.6300 | 0.5983 | 위 두 모드와 결과가 완전히 동일 |
| `metadata_query_bias_text` | 0.6450 | 0.5783 | stats bias보다 약함 |
| `description_suppression_text_stats_avg` | 0.6673 | 0.6209 | suppression 계열 중 최하 |
| `description_relation_text` | 0.8408 | 0.6672 | 매우 약함 |

`metadata_query_bias_stats`는 Weather 평균 MSE `0.3728`로 `metadata_query_gate_stats`의 `0.4830`보다 좋지만, ETTm1/2에서는 크게 뒤진다. 데이터셋 특화 효과로 보는 것이 안전하다.

`description_suppression_stats`, `text`, `text_stats_joint`는 forecasting 12개 값뿐 아니라 classification 14개 값도 완전히 동일하다. 우연으로 보기 어려우므로 다음 가능성을 확인해야 한다.

- 같은 checkpoint 또는 log가 여러 mode 이름으로 집계됨
- metadata mode가 실제 모델 경로에 반영되지 않음
- 결과 수집 스크립트가 동일 파일을 반복 참조함

현재 자료만으로 세 모드가 진짜 동등하다고 결론내리면 안 된다.

### 6.2 Classification 평균

| 모드 | 평균 accuracy | 평균 macro F1 |
|---|---:|---:|
| `description_suppression_stats` | 0.5203 | 0.4541 |
| `description_suppression_text` | 0.5203 | 0.4541 |
| `description_suppression_text_stats_joint` | 0.5203 | 0.4541 |
| `description_suppression_text_stats_avg` | 0.5102 | 0.4253 |

`relation_text`는 EthanolConcentration 한 건(`accuracy=0.312`, `F1=0.285`)만 존재하므로 전체 평균 순위에 포함할 수 없다.

---

## 7. 핵심 결론

### 7.1 Task별 최적 선택

| 목적 | 추천 모드 | 근거 |
|---|---|---|
| Forecasting 전체 성능 | **`ci_none`** | 모든 forecasting 데이터셋 평균 1위 |
| Mixer를 반드시 사용할 때 forecasting | **`metadata_query_gate_stats`** | mixer 평균 MSE 1위, 일관된 데이터셋 성능 |
| Plain mixer forecasting | **`mixer_text_stats_avg`** | plain mixer 평균 MSE 1위 |
| Classification transfer | **`metadata_query_gate_text_stats_joint`** | 평균 macro F1 1위, 표현 다양성 최고 |
| Forecasting·classification 균형 | **`metadata_query_gate_text_stats_avg`** | 두 task 모두 중상위 |
| Query specialization 연구 | **`mixer_text`** | attention overlap 최저, collapse 0, 단일 query ablation 효과 최대 |
| 제거·재설계 우선 | **`mixer_concat_stats`** | 구조 collapse, 낮은 평균 ablation 효과, 최하 forecasting |

### 7.2 연구적으로 가장 중요한 관찰

1. **낮은 pretrain loss는 좋은 representation을 보장하지 않는다.**  
   `mixer_concat_text`는 loss가 가장 낮지만 repr cos `0.9995`, var `0.00035`다.

2. **Sample representation collapse와 query specialization은 동시에 존재할 수 있다.**  
   `mixer_text`와 `mixer_concat_text`는 global collapse 경고가 있지만 query attention과 post-query 구조는 잘 분화된다.

3. **구조적 specialization은 downstream 성능을 보장하지 않는다.**  
   `mixer_text`는 query 구조가 가장 깨끗하지만 forecasting은 CI와 metadata gate stats보다 약하고 classification은 최하위다.

4. **Stats metadata와 text metadata는 다른 downstream inductive bias를 만든다.**  
   Stats는 forecasting에 유리하고, text 또는 text+stats joint는 classification에 유리하다.

5. **좋은 baseline과 큰 ablation 효과도 별개다.**  
   `mixer_concat_text_stats_joint`는 중요한 q9를 갖지만 baseline 성능이 낮다. 반대로 `metadata_query_gate_stats`는 query별 ablation 효과가 작아도 forecasting이 강하다.

6. **Post-query 평균 similarity만 보면 잘못 해석할 수 있다.**  
   Metadata gate text 계열은 평균이 낮지만 p90과 collapse ratio가 매우 높아 query 공간이 양극화되어 있다.

---

## 8. 결과 사용 시 주의사항

1. Mixer structure는 현재 1개 validation batch에 기반한다. 여러 batch에서 평균과 분산을 구하기 전까지 query 구조를 전체 데이터의 고정 특성으로 단정하지 않는다.
2. Pretrain 기존 attention snapshot과 새 structure 분석은 checkpoint, batch, aggregation 방식이 다를 수 있다. 예를 들어 기존 보고서의 `mixer_stats top1 unique=1.0`과 새 결과 `0.4375`는 직접 모순이라기보다 측정 조건 차이를 먼저 의심해야 한다.
3. Ablation은 ETTm1-96 한 설정만 평가한다. query 중요도가 다른 dataset과 horizon에서도 유지되는지는 아직 검증되지 않았다.
4. Mixer ablation은 latent query 직접 제거가 아니라 affinity 기반 channel weighting 효과 제거다.
5. Zero query ablation은 남은 query를 재정규화하지 않아 query-specific 손실과 공통 affinity scaling 효과가 섞인다.
6. Structure similarity는 각 sample의 cosine을 평균한 값이 아니라 batch와 patch를 먼저 평균한 대표 vector들의 cosine이다. Histogram도 16개 평균 query vector 사이 off-diagonal 값이므로 sample-level 분포가 아니다.
7. 모든 결과가 단일 seed라면 작은 차이는 학습 변동일 수 있다. 논문 수준의 순위를 위해서는 반복 실험의 평균과 표준편차가 필요하다.
8. Classification의 accuracy와 F1 간 차이가 큰 데이터셋이 있으므로 macro F1을 우선 해석한다.
9. Channel 이름 대부분이 숫자 index이므로 top-k 결과는 specialization의 존재는 보여주지만 도메인 의미를 직접 설명하지는 못한다.

---

## 9. 근거 자료

### 전체 결과

- [Pretrain 임시 분석](./pretrain_electricity_runs_temp_report_0609.md)
- [Forecasting 결과](./forecasting_electricity_results_0609.csv)
- [Classification 결과](./classification_electricity_results_0609.csv)

### 대표 구조 시각화

- [`mixer_text` overview](./mixer_structure/electricity_mixer_text/mixer_structure_overview.png)
- [`mixer_text` post-query histogram](./mixer_structure/electricity_mixer_text/post_mixer_query_similarity_histogram.png)
- [`metadata_query_gate_stats` overview](./mixer_structure/electricity_metadata_query_gate_stats/mixer_structure_overview.png)
- [`metadata_query_gate_text_stats_joint` histogram](./mixer_structure/electricity_metadata_query_gate_text_stats_joint/post_mixer_query_similarity_histogram.png)
- [`mixer_concat_stats` overview](./mixer_structure/electricity_mixer_concat_stats/mixer_structure_overview.png)

### 대표 downstream ablation

- [`ci_none` ablation](./linear_probe_ablation/ci_none_ETTm1_96/unit_ablation_delta_mse.png)
- [`mixer_text` ablation](./linear_probe_ablation/mixer_text_ETTm1_96/unit_ablation_delta_mse.png)
- [`metadata_query_gate_stats` ablation](./linear_probe_ablation/metadata_query_gate_stats_ETTm1_96/unit_ablation_delta_mse.png)
- [`metadata_query_gate_text_stats_joint` ablation](./linear_probe_ablation/metadata_query_gate_text_stats_joint_ETTm1_96/unit_ablation_delta_mse.png)
- [`mixer_concat_stats` ablation](./linear_probe_ablation/mixer_concat_stats_ETTm1_96/unit_ablation_delta_mse.png)

---

## 10. 최종 판정

현재 증거로 가장 강하게 말할 수 있는 결론은 다음과 같다.

> **Forecasting에서는 channel-independent baseline이 가장 강하며, mixer가 필요하다면 stats 기반 metadata query gate가 최선이다. Classification에서는 text와 stats를 joint로 사용하는 metadata query gate가 가장 좋은 transfer representation을 제공한다.**

Mixer의 가치를 단일 점수로 평가하면 안 된다. `mixer_text`는 가장 분명한 query specialization을, `metadata_query_gate_stats`는 가장 좋은 mixer forecasting을, `metadata_query_gate_text_stats_joint`는 가장 좋은 classification transfer를 제공한다. 반면 `mixer_concat_stats`는 구조와 기능, downstream 성능이 모두 약해 현재 실험군에서 가장 명확한 실패 사례다.
