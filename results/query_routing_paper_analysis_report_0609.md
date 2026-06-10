# Query-to-Channel Routing 논문용 분석 보고서

## 1. 연구 질문

이 분석은 LayaTS Channel Mixer의 downstream 성능 차이가 query-to-channel routing 구조와 관련되는지를 확인하기 위해 수행되었다.

주요 가설은 다음과 같다.

> Query가 지나치게 적은 채널에 집중하거나 여러 query가 동일한 채널을 반복 선택하면 유효한 query specialization이 감소하고, 그 결과 downstream forecasting 성능이 저하될 수 있다.

이 가설은 세 단계로 나누어 검토한다.

1. **Routing concentration**: 각 query가 채널을 얼마나 좁거나 넓게 선택하는가?
2. **Query redundancy**: 여러 query가 동일한 채널 분포와 post-mixer 표현을 만드는가?
3. **Functional contribution**: 구조적으로 구분되는 query가 실제 forecasting prediction에 필요한가?

단일 지표만으로는 인과관계를 주장할 수 없다. 따라서 attention 구조, post-query geometry, single/group ablation, random-control ablation, baseline forecasting 성능을 함께 해석한다.

---

## 2. 실험 범위와 분석 단위

### 2.1 Mixer structure

- Dataset: Electricity validation split
- Channel 수: 321
- Query 수: 16
- Checkpoint: 각 pretrain mode의 best checkpoint
- 분석 파일: `analysis/mixer_structure/<mode>/`

스크립트에는 `NUM_BATCHES=32`가 지정됐지만 현재 validation loader가 하나의 batch만 제공하여 모든 mode의 `batch_metric_count=1`이다.

따라서 structure 결과는 다음 수준의 증거로 취급한다.

> 동일한 Electricity validation batch에 대한 mode 간 통제 비교

다음 수준으로는 해석하지 않는다.

> 전체 validation distribution에서 안정적으로 재현되는 평균과 표준편차

`batch_std=0`은 안정성이 높다는 뜻이 아니라 관측 batch가 하나뿐이라는 뜻이다.

### 2.2 Functional ablation

- Dataset: ETTm1
- Input length: 512
- Prediction length: 96
- Checkpoint: 각 mode의 forecasting best checkpoint
- 평가 범위: 전체 test loader
- 기본 ablation: `zero`
- Random group repetitions: 16
- 분석 파일: `analysis/linear_probe_ablation/<mode>_ETTm1_96/`

Mixer ablation은 post-mixer latent query를 직접 제거하지 않는다. 특정 query의 `channel_affinity`를 0으로 만든 뒤 query 평균 channel importance를 다시 계산한다.

따라서 이 분석의 정확한 의미는 다음과 같다.

> 특정 query가 forecasting head로 전달되는 channel weighting에 얼마나 기여하는가?

### 2.3 중요한 제한

현재 group ablation의 top-k query는 test set single-query ablation 결과로 선택되고, 같은 test set에서 group effect를 평가했다. 이는 exploratory ranking에는 유효하지만 confirmatory significance test에는 selection bias가 있다.

논문 본문에서는 group effect를 **탐색적 기능 증거**로 표현해야 한다. 최종 검증에서는 validation set으로 query를 선택하고 test set에서 한 번만 평가해야 한다.

---

## 3. 지표 정의와 해석

### 3.1 Routing concentration

#### Mean attention entropy

Query별 channel attention entropy의 평균이다.

```text
H(q) = -Σc p(q,c) log p(q,c)
```

- 낮음: 소수 채널에 집중
- 높음: 많은 채널에 분산
- 321개 채널에서 균등 attention의 최대 entropy는 약 `ln(321)=5.77`

낮은 entropy를 곧바로 나쁜 구조로 보면 안 된다. Query들이 서로 다른 중요한 채널 subset을 선택한다면 낮은 entropy는 specialization의 증거가 될 수 있다.

### 3.2 Routing diversity

#### Top-1 unique ratio

16개 query가 선택한 top-1 채널 중 고유 채널의 비율이다.

```text
unique top-1 channels / 16
```

- `1.0`: 모든 query가 서로 다른 top-1 채널 선택
- 낮음: 여러 query가 같은 top-1 채널을 반복 선택

#### Mean attention overlap

Query attention vector 간 off-diagonal cosine similarity 평균이다.

- 낮음: query별 channel distribution이 다름
- 높음: query들이 비슷한 채널들을 읽음

Top-1 unique ratio보다 전체 distribution 중복을 더 직접적으로 측정한다.

### 3.3 Post-mixer query geometry

#### Mean post-query similarity

Post-mixer query representation 간 off-diagonal cosine similarity 평균이다.

- 높음: query representation이 전반적으로 유사
- 낮음: query가 서로 다른 방향으로 분화

#### P90 post-query similarity

Query pair similarity의 90 percentile이다. 평균이 낮아도 일부 query cluster가 collapse하는 현상을 검출한다.

#### Collapse ratio

Cosine similarity가 `0.9 이상`인 query pair의 비율이다.

- 높음: 거의 동일한 query pair가 많음
- 낮음: 강한 pairwise collapse가 적음

평균이 낮고 P90 및 collapse ratio가 높다면 query들이 균일하게 분화된 것이 아니라, 서로 반대인 group과 거의 동일한 group이 함께 존재하는 양극화 구조일 수 있다.

### 3.4 Selected-channel coherence

#### Mean top-k pre-mixer similarity

각 query가 선택한 top-k 채널들이 mixer 이전 representation에서 얼마나 유사했는지 나타낸다.

- 높음: query가 이미 유사한 채널 cluster를 수집
- 낮음: query가 이질적인 채널들을 결합

이 값은 좋고 나쁨의 단독 기준이 아니다. Coherent grouping인지 task-irrelevant grouping인지는 downstream ablation과 함께 판단해야 한다.

### 3.5 Functional ablation

#### Single-query delta

한 query의 affinity를 제거했을 때 test MSE의 상대 증가율이다.

```text
100 × (MSEablated - MSEbaseline) / MSEbaseline
```

- 큼: 특정 query에 기능이 집중
- 작음: query가 중복되거나 contribution이 분산됨
- 음수: 해당 query를 제거했을 때 오히려 성능이 개선되어 harmful/redundant contribution 가능

#### Top-k group delta

Single-query delta가 큰 query 1, 2, 4개를 함께 제거한 상대 MSE 증가율이다.

#### Random-control delta

동일한 크기의 random query group을 16회 제거한 평균 상대 MSE 증가율이다.

#### Top-k enrichment

```text
top-k delta ratio - random-k mean delta ratio
```

- 큼: 중요한 기능이 특정 query subset에 집중
- 0에 가까움: top-ranked query group이 random group보다 특별하지 않음

Random repetition 수가 16으로 작고 top-k가 test set에서 선택됐으므로 enrichment는 효과 크기로 해석하며 정식 p-value로 사용하지 않는다.

---

## 4. 전체 Structure 결과

| Mode | Entropy | Top-1 unique | Attention overlap | Post mean | Post P90 | Collapse ratio | Top-k pre sim |
|---|---:|---:|---:|---:|---:|---:|---:|
| `metadata_query_gate_stats` | 4.3824 | 0.3125 | 0.4602 | 0.5420 | 0.9990 | 0.4667 | 0.8744 |
| `metadata_query_gate_text` | 2.5099 | 0.8750 | 0.0699 | 0.0368 | 0.9981 | 0.4250 | 0.6299 |
| `metadata_query_gate_text_stats_avg` | 2.6740 | 0.8125 | 0.0826 | 0.0809 | 0.9921 | 0.3833 | 0.5823 |
| `metadata_query_gate_text_stats_joint` | 2.6878 | 0.6250 | 0.1337 | 0.0698 | 0.9956 | 0.3917 | 0.7101 |
| `mixer_concat_stats` | 2.5478 | 0.3125 | 0.3221 | 0.8903 | 0.9877 | 0.4750 | 0.5327 |
| `mixer_concat_text` | 2.5541 | 1.0000 | 0.0146 | 0.1000 | 0.5386 | 0.0167 | 0.4979 |
| `mixer_concat_text_stats_avg` | 3.4332 | 1.0000 | 0.0753 | 0.1495 | 0.7856 | 0.0333 | 0.6780 |
| `mixer_concat_text_stats_joint` | 1.8072 | 0.8750 | 0.0338 | 0.5115 | 0.8508 | 0.0500 | 0.6092 |
| `mixer_stats` | 3.1879 | 0.4375 | 0.2174 | 0.8098 | 0.9875 | 0.4083 | 0.9315 |
| `mixer_text` | 2.1688 | 1.0000 | 0.0058 | 0.0521 | 0.6399 | 0.0000 | 0.8289 |
| `mixer_text_stats_avg` | 2.7232 | 0.8125 | 0.0607 | 0.2849 | 0.7912 | 0.0500 | 0.8015 |
| `mixer_text_stats_joint` | 2.0665 | 0.8125 | 0.0505 | 0.5515 | 0.8202 | 0.0250 | 0.8575 |

### 4.1 가장 명확한 routing specialization

`mixer_text`가 가장 선명하다.

- Top-1 unique ratio: `1.0`
- Attention overlap: `0.0058`
- Post-query mean similarity: `0.0521`
- Collapse ratio: `0`

이는 모든 query가 서로 다른 top channel을 선택하고, 전체 attention distribution도 거의 겹치지 않으며, post-mixer query pair 중 similarity `0.9 이상`인 pair가 없음을 의미한다.

`mixer_concat_text`도 유사한 특성을 보인다.

- Top-1 unique ratio: `1.0`
- Attention overlap: `0.0146`
- Collapse ratio: `0.0167`

### 4.2 가장 강한 routing redundancy

`mixer_concat_stats`는 모든 redundancy 지표가 같은 방향을 가리킨다.

- Top-1 unique ratio: `0.3125`
- Attention overlap: `0.3221`
- Post-query mean similarity: `0.8903`
- Collapse ratio: `0.4750`

16개 query가 소수의 top channel을 공유하며, post-mixer query의 거의 절반이 cosine `0.9 이상`인 pair를 이룬다.

`mixer_stats`도 높은 redundancy를 보인다.

- Top-1 unique ratio: `0.4375`
- Post-query mean: `0.8098`
- Collapse ratio: `0.4083`

### 4.3 Metadata gate의 두 routing regime

`metadata_query_gate_stats`는 매우 diffuse하고 중복된 routing이다.

- Entropy: `4.3824`
- Top-1 unique ratio: `0.3125`
- Attention overlap: `0.4602`

반면 text를 사용하는 metadata gate 계열은 attention overlap이 낮다.

- Text: `0.0699`
- Text+stats avg: `0.0826`
- Text+stats joint: `0.1337`

그러나 세 mode 모두 P90 `0.99 이상`, collapse ratio `0.38~0.43`이다. Attention routing은 다르게 보여도 post-mixer representation에서는 일부 query cluster가 거의 동일해지는 구조다.

---

## 5. 전체 Functional Ablation 결과

CI는 channel representation을 제거하고 mixer는 query affinity contribution을 제거하므로 서로 직접 비교하지 않는다.

### 5.1 Mixer query group ablation

| Mode | Baseline MSE | Top-1 / Random | Top-2 / Random | Top-4 / Random | Top-4 enrichment |
|---|---:|---:|---:|---:|---:|
| `metadata_query_gate_stats` | **0.4036** | 2.76 / 2.15 | 6.29 / 5.05 | 16.28 / 15.27 | 1.01 |
| `metadata_query_gate_text` | 0.4931 | 3.85 / 2.76 | 8.10 / 6.97 | 17.85 / 16.69 | 1.16 |
| `metadata_query_gate_text_stats_avg` | 0.4821 | 6.24 / 3.29 | 12.94 / 7.34 | 26.51 / 18.45 | 8.07 |
| `metadata_query_gate_text_stats_joint` | 0.4936 | 4.37 / 3.07 | 8.51 / 7.06 | 19.48 / 18.07 | 1.41 |
| `mixer_concat_stats` | 0.6473 | 4.63 / 0.59 | 8.20 / 1.34 | 13.48 / 6.98 | 6.50 |
| `mixer_concat_text` | 0.4525 | 5.42 / 3.61 | 10.30 / 7.54 | 26.70 / 21.49 | 5.22 |
| `mixer_concat_text_stats_avg` | 0.5549 | 5.75 / 2.93 | 11.73 / 7.36 | 33.72 / 15.62 | 18.10 |
| `mixer_concat_text_stats_joint` | 0.6077 | 8.94 / 1.27 | 15.24 / 7.02 | 25.89 / 13.31 | 12.59 |
| `mixer_stats` | 0.5262 | 6.86 / 3.26 | 17.15 / 5.04 | 48.18 / 16.00 | 32.18 |
| `mixer_text` | 0.4891 | **10.06 / 2.39** | **29.59 / 7.28** | **85.54 / 13.37** | **72.17** |
| `mixer_text_stats_avg` | 0.4696 | 7.71 / -1.29 | 20.25 / 4.45 | 41.85 / 12.07 | 29.78 |
| `mixer_text_stats_joint` | 0.4636 | 8.60 / 3.23 | 16.91 / 4.62 | 30.65 / 15.60 | 15.04 |

표의 `Top-k / Random`은 각각 top-k query 제거 상대 MSE 증가율과 random-k 평균 상대 MSE 증가율이다.

### 5.2 기능 집중이 가장 강한 mode

`mixer_text`의 기능 집중이 가장 강하다.

- Top-1 제거: `+10.06%`, random `+2.39%`
- Top-2 제거: `+29.59%`, random `+7.28%`
- Top-4 제거: `+85.54%`, random `+13.37%`
- Top-4 enrichment: `+72.17 percentage points`

Structure에서도 가장 낮은 overlap과 collapse를 보였으므로, routing specialization과 functional specialization이 일치하는 가장 명확한 사례다.

`mixer_stats`는 structure redundancy가 높음에도 top-4 enrichment가 `32.18`이다. 이는 전체 query geometry가 유사해도 일부 query subset이 forecasting weighting에서 핵심 역할을 가질 수 있음을 보여준다.

`mixer_text_stats_avg`도 top-4 enrichment `29.78`로 높다. Random single-query 제거 평균은 `-1.29%`로, 다수 query가 불필요하거나 약간 harmful한 반면 특정 query subset은 중요할 가능성이 있다.

### 5.3 기능이 넓게 분산된 mode

Metadata query gate의 stats, text, text+stats joint는 top-4 enrichment가 각각 `1.01`, `1.16`, `1.41`로 매우 작다.

이는 top-ranked query 네 개를 제거해도 random query 네 개를 제거한 것과 효과가 크게 다르지 않음을 의미한다.

가능한 해석은 다음과 같다.

- Query contribution이 넓게 분산됨
- 여러 query가 상호 대체 가능한 routing을 수행
- Forecasting 성능이 특정 query specialist보다 공유 channel weighting에서 발생

특히 `metadata_query_gate_stats`는 mixer 중 baseline MSE가 가장 좋지만 top-k enrichment가 가장 낮다. 이는 좋은 forecasting 성능이 강한 query specialization을 필요로 하지 않는다는 중요한 반례다.

### 5.4 CI 기준선

`ci_none`은 channel 1/2/4개 제거 시 각각 다음 상대 MSE 증가를 보인다.

- Top-1 channel: `70.04%`
- Top-2 channels: `139.49%`
- Top-4 channels: `229.95%`

이는 output channel representation 자체를 제거하는 실험이므로 mixer query ablation보다 훨씬 큰 것이 자연스럽다. CI 수치는 channel별 forecast가 해당 channel token에 강하게 의존한다는 sanity check로 사용한다.

---

## 6. 모드별 논문용 해석

### 6.1 `ci_none`

**볼 지표**

- Baseline MSE `0.3253`
- Channel top-1/2/4 ablation `70.04% / 139.49% / 229.95%`

**해석**

Channel-independent forecast head는 각 channel representation에 직접 의존한다. 이는 ablation pipeline이 실제 task-relevant unit 제거에 반응한다는 positive control이다.

Mixer query와 ablation 단위가 다르므로 query specialization 비교 표에는 포함하지 않는다.

### 6.2 `mixer_stats`

**볼 지표**

- Top-1 unique `0.4375`
- Post-query mean `0.8098`
- Collapse ratio `0.4083`
- Top-4 delta `48.18%`, random `16.00%`
- Baseline MSE `0.5262`

**해석**

Global geometry는 상당히 redundant하지만 중요한 query subset은 분명히 존재한다. 따라서 high collapse가 모든 query를 기능적으로 무의미하게 만든 것은 아니다.

이 mode는 “구조적 유사도와 기능적 중요도는 동일하지 않다”는 사례로 사용하기 적합하다.

### 6.3 `mixer_text`

**볼 지표**

- Top-1 unique `1.0`
- Attention overlap `0.0058`
- Collapse ratio `0`
- Top-4 enrichment `72.17`
- Baseline MSE `0.4891`

**해석**

가장 강한 structural/functional specialization을 보인다. Query가 서로 다른 채널 distribution을 선택하고, 중요한 query subset 제거는 random 제거보다 훨씬 큰 성능 저하를 유발한다.

하지만 baseline MSE가 최상은 아니다. Specialization은 존재하지만 좋은 forecasting을 위한 충분조건은 아니다.

### 6.4 `mixer_text_stats_avg`

**볼 지표**

- Attention overlap `0.0607`
- Collapse ratio `0.05`
- Top-1 random mean `-1.29%`
- Top-4 enrichment `29.78`
- Baseline MSE `0.4696`

**해석**

대부분 query는 약하거나 대체 가능하지만 소수 query subset은 매우 중요하다. Text와 stats 평균 결합이 sparse functional hierarchy를 만든 것으로 해석할 수 있다.

### 6.5 `mixer_text_stats_joint`

**볼 지표**

- Entropy `2.0665`
- Attention overlap `0.0505`
- Post-query mean `0.5515`
- Collapse ratio `0.025`
- Top-4 enrichment `15.04`

**해석**

Routing distribution은 분화되어 있지만 post-query 평균 similarity는 비교적 높다. 즉 서로 다른 채널을 선택한 query가 downstream representation 단계에서 일부 공통 방향으로 수렴한다.

Functional specialization은 존재하지만 text-only보다 약하다.

### 6.6 `mixer_concat_stats`

**볼 지표**

- Top-1 unique `0.3125`
- Attention overlap `0.3221`
- Post-query mean `0.8903`
- Collapse ratio `0.475`
- Baseline MSE `0.6473`
- Top-4 enrichment `6.50`

**해석**

강한 routing redundancy, post-query collapse, 낮은 forecasting 성능이 동시에 나타난다. 현재 가설을 가장 직접적으로 지지하는 mode다.

다만 top-ranked subset은 random보다 중요하므로 모든 query가 완전히 동일한 것은 아니다.

### 6.7 `mixer_concat_text`

**볼 지표**

- Top-1 unique `1.0`
- Attention overlap `0.0146`
- Collapse ratio `0.0167`
- Top-4 enrichment `5.22`
- Baseline MSE `0.4525`

**해석**

Structure는 명확히 분화되지만 functional enrichment는 상대적으로 작다. Attention pattern이 다르다는 사실만으로 각 query가 forecast에 독점적인 정보를 제공한다고 볼 수 없다.

좋은 baseline은 일부 specialist보다 여러 query의 분산된 contribution에서 나올 가능성이 있다.

### 6.8 `mixer_concat_text_stats_avg`

**볼 지표**

- Top-1 unique `1.0`
- Post-query mean `0.1495`
- Collapse ratio `0.0333`
- Top-4 enrichment `18.10`
- Baseline MSE `0.5549`

**해석**

Structural specialization과 functional concentration이 모두 존재하지만 baseline 성능은 낮다. Query 분화의 존재와 task-relevant representation 품질은 별개임을 보여준다.

### 6.9 `mixer_concat_text_stats_joint`

**볼 지표**

- Entropy `1.8072`
- Attention overlap `0.0338`
- Post-query mean `0.5115`
- Top-1 enrichment `7.67`
- Top-4 enrichment `12.59`

**해석**

일부 query가 매우 sharp한 channel routing을 수행하고 기능도 특정 subset에 집중된다. 그러나 post-query representation은 중간 수준으로 유사하며 baseline 성능도 낮다.

과도한 concentration이 존재할 가능성은 있지만 현재 자료만으로 성능 저하의 직접 원인이라고 단정할 수 없다.

### 6.10 `metadata_query_gate_stats`

**볼 지표**

- Entropy `4.3824`
- Top-1 unique `0.3125`
- Attention overlap `0.4602`
- Collapse ratio `0.4667`
- Top-4 enrichment `1.01`
- Baseline MSE `0.4036`, mixer 중 최상

**해석**

Query specialization은 가장 약한 축이지만 forecasting 성능은 가장 좋다. Stats metadata가 독립적인 query expert를 만드는 대신 여러 query에 공유되는 broad channel prior를 제공했을 가능성이 높다.

이는 “query overlap 또는 collapse가 높으면 반드시 downstream이 나쁘다”는 단순 가설을 반박하는 핵심 mode다.

### 6.11 `metadata_query_gate_text`

**볼 지표**

- Attention overlap `0.0699`
- Post-query mean `0.0368`
- P90 `0.9981`
- Collapse ratio `0.425`
- Top-4 enrichment `1.16`

**해석**

평균 similarity만 보면 query가 잘 분리된 것처럼 보이지만 P90과 collapse ratio는 일부 query cluster가 거의 동일함을 보여준다.

Functional contribution도 top query에 집중되지 않는다. 이 mode는 평균만으로 collapse를 판단하면 안 된다는 사례다.

### 6.12 `metadata_query_gate_text_stats_avg`

**볼 지표**

- Attention overlap `0.0826`
- Collapse ratio `0.3833`
- Top-4 enrichment `8.07`
- Baseline MSE `0.4821`

**해석**

Metadata gate 계열 중 functional specialization이 가장 분명하다. Text와 stats 평균 결합이 distributed gate 구조 안에서도 중요한 query subset을 형성한 것으로 보인다.

다만 post-query collapsed pair는 여전히 많다.

### 6.13 `metadata_query_gate_text_stats_joint`

**볼 지표**

- Top-1 unique `0.625`
- Attention overlap `0.1337`
- Post mean `0.0698`
- P90 `0.9956`
- Collapse ratio `0.3917`
- Top-4 enrichment `1.41`

**해석**

Attention과 post-query 평균은 어느 정도 분화되어 있지만 기능은 특정 query subset에 집중되지 않는다. Query들이 서로 다른 route를 만들더라도 forecasting channel weighting에서는 상호 대체적으로 사용될 가능성이 있다.

---

## 7. 가설 검정 결과

### 7.1 Structure metric과 baseline MSE 상관

12개 mixer mode에서 계산한 Pearson correlation은 다음과 같다.

| Structure metric | ETTm1-96 baseline MSE와 Pearson r |
|---|---:|
| Attention entropy | -0.377 |
| Top-1 unique ratio | -0.117 |
| Attention overlap | -0.012 |
| Mean post-query similarity | **0.429** |
| Collapse ratio | 0.054 |
| Mean top-k pre similarity | -0.420 |

Mode 수가 12개뿐이고 seed 반복이 없으므로 유의성 검정 결과로 해석하지 않는다.

현재 관찰에서 가장 큰 양의 연관은 post-query mean similarity와 baseline MSE 사이의 `r=0.429`다. Query representation이 전반적으로 비슷할수록 forecasting MSE가 높아지는 약한 경향은 있다.

반면 attention overlap과 collapse ratio는 baseline MSE와 거의 상관이 없다.

### 7.2 초기 가설에 대한 판정

초기 가설을 보편적인 인과 명제로 받아들이기는 어렵다.

```text
Query concentration/redundancy 증가
→ downstream 성능 저하
```

이를 지지하는 사례:

- `mixer_concat_stats`: redundancy와 collapse가 높고 baseline이 가장 나쁨
- `mixer_text`: 분화가 강하고 functional enrichment가 가장 큼

이를 반박하는 사례:

- `metadata_query_gate_stats`: redundancy와 collapse가 높지만 mixer baseline이 가장 좋음
- `mixer_concat_text_stats_avg`: 분화와 functional concentration이 있지만 baseline은 낮음

따라서 현재 결과가 지지하는 더 정확한 결론은 다음과 같다.

> Query routing structure는 mode별로 뚜렷하게 다르며, 구조적 specialization은 functional specialization과 일부 mode에서 일치한다. 그러나 specialization 정도는 forecasting 성능을 단독으로 설명하지 못하며, distributed routing도 높은 성능을 낼 수 있다.

---

## 8. 논문에 사용할 수 있는 주장

### 8.1 사용 가능한 주장

1. **Mode에 따라 query routing regime이 달라진다.**  
   Text mixer는 낮은 overlap과 높은 top-1 diversity를, stats gate는 diffuse하고 중복된 routing을 보였다.

2. **Structural specialization과 functional specialization은 구별해야 한다.**  
   `mixer_text`에서는 둘이 일치하지만 `mixer_concat_text`에서는 구조 분화 대비 top-k enrichment가 작다.

3. **중요 query subset은 random subset보다 큰 성능 영향을 줄 수 있다.**  
   특히 `mixer_text`, `mixer_stats`, `mixer_text_stats_avg`에서 top-4 제거 효과가 random control보다 크게 나타났다.

4. **Query specialization은 좋은 forecasting의 충분조건이 아니다.**  
   가장 강한 specialization을 보인 `mixer_text`가 가장 좋은 mixer baseline은 아니었다.

5. **Distributed routing도 강한 forecasting 성능을 낼 수 있다.**  
   `metadata_query_gate_stats`는 낮은 top-k enrichment에도 가장 좋은 mixer baseline을 기록했다.

6. **Post-query 평균 similarity만으로 collapse를 판단하면 안 된다.**  
   Metadata gate text 계열은 낮은 평균과 높은 P90/collapse ratio가 동시에 나타났다.

### 8.2 아직 사용하면 안 되는 주장

1. “Query가 특정 채널에 집중해서 downstream 성능이 나빠졌다.”
2. “Metadata gate stats의 query는 쓸모없다.”
3. “Query overlap이 높은 모델은 항상 성능이 낮다.”
4. “Top-k ablation이 통계적으로 유의하다.”
5. “관찰된 query structure가 전체 validation set과 다른 seed에서도 안정적이다.”
6. “Query ID가 mode 간 동일한 semantic role을 가진다.”

현재 결과는 이 문장들을 뒷받침하지 않는다.

---

## 9. Figure 및 Table 구성 제안

### Figure 1. Routing regime comparison

대표 mode 4개를 선택한다.

- `mixer_text`: 강한 specialization
- `mixer_concat_stats`: 강한 redundancy와 낮은 성능
- `metadata_query_gate_stats`: distributed routing과 높은 성능
- `metadata_query_gate_text_stats_avg`: 중간형

각 mode에 대해 다음을 나란히 배치한다.

- Query-channel attention heatmap
- Post-query cosine heatmap
- Post-query cosine histogram

### Figure 2. Top-k versus random ablation

X축은 group size `1, 2, 4`, Y축은 relative MSE increase로 둔다.

- Top-k query group
- Random group mean
- Random group standard deviation error bar

`mixer_text`, `mixer_stats`, `metadata_query_gate_stats`를 함께 표시하면 specialized, partially concentrated, distributed regime을 비교할 수 있다.

### Figure 3. Structure-performance scatter

- X축: mean post-query similarity
- Y축: ETTm1-96 baseline MSE
- Point label: mode

보조 Figure로 attention overlap 및 collapse ratio scatter를 제시해 단순 단조 관계가 없음을 보여준다.

### Table 1. Structure metrics

본 보고서 4절 표를 사용한다.

### Table 2. Functional ablation

본 보고서 5절 표를 사용하되 random mean과 함께 random standard deviation을 추가한다.

---

## 10. Publication-grade 검증을 위한 남은 실험

1. **Multi-batch structure evaluation**  
   Validation loader를 batch 여러 개로 분할해 mode별 평균과 95% confidence interval을 계산한다.

2. **Validation-selected top-k ablation**  
   Validation set에서 query ranking을 결정하고 test set에서는 선택을 고정한다.

3. **`zero_renorm` confirmatory ablation**  
   Query 제거 시 전체 affinity scale 감소를 보정하고 기존 `zero` 결과와 비교한다.

4. **Seed repetition**  
   최소 3개, 권장 5개 seed에서 structure와 ablation을 반복한다.

5. **Dataset/horizon repetition**  
   ETTm1/ETTm2와 prediction length 96/192/336에서 동일 패턴을 확인한다.

6. **Random-control confidence interval**  
   Random group 반복을 16회에서 최소 100회로 늘리고 empirical percentile 또는 bootstrap confidence interval을 보고한다.

7. **Cross-mode query matching**  
   Query ID가 아니라 affinity cosine과 Hungarian matching으로 유사 역할 query를 정렬한다.

---

## 11. 최종 결론

현재 분석은 query-to-channel routing이 mode별로 서로 다른 구조를 형성하고, 일부 mode에서 그 구조가 실제 forecasting contribution과 연결됨을 보여준다.

가장 강한 사례는 `mixer_text`다. 이 mode는 attention overlap과 post-query collapse가 가장 낮고, 중요한 query 네 개를 제거할 때 random 네 개 제거보다 상대 MSE가 `72.17 percentage points` 더 증가했다.

그러나 `metadata_query_gate_stats`는 강한 routing redundancy에도 가장 좋은 mixer baseline을 기록했다. 따라서 query specialization이 forecasting 성능의 필요조건 또는 충분조건이라는 주장은 성립하지 않는다.

논문에서 가장 방어 가능한 핵심 메시지는 다음과 같다.

> **Channel Mixer는 하나의 고정된 routing behavior를 학습하지 않는다. Metadata injection 방식에 따라 specialized, redundant, 또는 distributed query-routing regime이 형성되며, 이 regime은 query-level functional dependence를 변화시킨다. 그러나 downstream forecasting 성능은 specialization만으로 결정되지 않는다.**

