# `runs_temp` Pretrain 분석 보고서

## 1. 분석 목적

이 보고서는 `./runs_temp`에 저장된 13개의 pretraining run이 단순히 validation loss만 낮춘 것이 아니라, 실제로 **의미 있고 건강한 representation**을 학습했는지 평가하기 위해 작성되었다.

이번 평가는 두 종류의 근거를 함께 사용한다.

1. 각 run 폴더의 **가장 최근 TensorBoard event file**에서 추출한 scalar metric
2. 각 run의 **가장 최근 attention map 저장 결과** 중 `epoch_100_val` 스냅샷

핵심 목표는 “어떤 모델의 validation loss가 가장 낮은가”를 넘어서, 아래 질문에 답하는 것이다.

- prediction-target alignment가 충분히 좋은가?
- representation collapse 없이 표현이 살아 있는가?
- attention이 의미 있게 분화되어 있는가?

---

## 2. 분석 방법

### 2.1 Run 선택 기준

각 `runs_temp` 폴더에서 **가장 최근 event file 하나만** 사용했다.

이유는 일부 폴더에 재시작(run restart) 또는 누적 로그가 섞여 있을 수 있기 때문이다. 따라서 마지막 event file이 현재 상태를 가장 잘 반영한다고 보았다.

### 2.2 Attention snapshot 선택 기준

attention map이 저장된 run은 모두 아래 경로의 마지막 snapshot을 사용했다.

- `attention_maps/<tag>/epoch_100_val/`

이 값은 **최종 epoch 기준 attention 상태**를 보여준다. 따라서 `best_val_loss`가 더 이른 epoch에서 나왔다면, 최종 attention 상태와 best checkpoint 상태가 정확히 같지 않을 수 있다.

이 점은 특히 `mixer_concat_stats` 같은 run을 해석할 때 중요하다.

---

## 3. 주요 지표 해석 기준

아래 지표 정의는 `train_pretrain.py`의 실제 계산 방식에 근거한다.

### 3.1 예측 품질 관련 지표

#### `best_val_loss`
학습 중 가장 낮았던 validation loss이다.

- 낮을수록 좋다.
- 다만 이것만으로 representation 품질을 판단하면 안 된다.

#### `val_pred_loss`
validation prediction loss이다.

- 낮을수록 좋다.
- target patch를 얼마나 잘 복원 또는 예측하는지 보여준다.

#### `val_align_cosine_similarity_mean`
predicted token과 target token 사이의 cosine similarity이다.

- 높을수록 좋다.
- 대략적인 해석:
  - `0.90 이상` → 매우 좋음
  - `0.80대` → 좋음
  - `0.70대` → 보통
  - `0.50 이하` → 약함

이 값은 pretraining objective 자체가 얼마나 잘 풀렸는지 보여주는 핵심 지표다.

---

### 3.2 Representation 건강도 관련 지표

#### `val_repr_pairwise_cosine_similarity_mean`
같은 batch 안의 서로 다른 샘플 representation 사이의 평균 cosine similarity이다.

이 값은 **무조건 높다고 좋은 것이 아니다.**

- 너무 높으면 representation들이 거의 같은 방향으로 모였다는 뜻이고,
- 이는 collapse 가능성을 시사한다.

해석 기준:

- `0.99 이상` → 강한 collapse 경고
- `0.95 ~ 0.99` → 지나치게 뭉친 상태일 수 있음
- `0.85 ~ 0.95` → 비교적 건강한 편

#### `val_repr_feature_var_mean`
representation 차원별 분산의 평균이다.

- 높을수록 표현 다양성이 살아 있다고 볼 수 있다.
- 너무 낮으면 collapse를 의심해야 한다.

이번 실험에서의 해석 기준:

- `0.02 이상` → 비교적 건강
- `0.005 이하` → collapse 경고
- `0.001 수준` → 매우 위험

즉 representation cosine similarity와 feature variance는 함께 봐야 한다.

---

### 3.3 Attention map 해석 기준

이번 보고서에서는 attention artifact도 함께 사용했다.

`train_pretrain.py` 기준으로 저장되는 주요 파일은 다음과 같다.

- `channel_affinity.pt` → patch 평균된 query-to-channel attention, shape `[H, Q, C]`
- `mixer_head_XX_topk.txt` → 각 query가 주로 보는 top-k channel 목록

이번 보고서에서는 attention의 품질을 아래 기준으로 해석했다.

#### Attention entropy
attention이 channel 위에서 얼마나 퍼져 있는가를 나타낸다.

- 너무 낮으면 → 소수 channel에 과집중
- 너무 높으면 → 지나치게 평평해서 specialization이 약함

#### Max weight mean
각 query가 가장 강하게 보는 channel weight의 평균이다.

- 높을수록 더 날카로운 attention
- 너무 높으면 과집중 가능성

#### Top-1 unique ratio
서로 다른 query들이 서로 다른 top channel을 선택하는 비율이다.

- 높을수록 specialization이 잘 되어 있음
- 낮을수록 여러 query가 같은 channel만 반복해서 보는 경향

건강한 attention은 대체로 아래 특성을 가진다.

- query마다 보는 channel이 다르다.
- top-k 구조가 해석 가능하다.
- 완전히 flat하지도 않고, 한 channel에만 완전히 고정되지도 않는다.

---

## 4. 핵심 관찰 결과

### 4.1 Validation loss가 가장 낮다고 해서 representation이 가장 좋은 것은 아니다

이번 실험의 가장 중요한 결론이다.

대표적인 예는 다음과 같다.

- `pretrain_electricity_laya_mixer_concat_text`
  - `best_val_loss = 0.0383`
  - `val_repr_pairwise_cosine_similarity_mean = 0.9995`
  - `val_repr_feature_var_mean = 0.00035`

이 모델은 pretraining objective는 매우 잘 맞췄지만, representation 측면에서는 **collapse 가능성이 매우 크다**.

같은 경고는 아래 모델들에도 적용된다.

- `pretrain_electricity_laya_mixer_text`
- `pretrain_electricity_laya_mixer_text_stats_avg`

따라서 이런 모델들을 단순히 “잘 pretrained되었다”고 부르는 것은 위험하다.

---

### 4.2 가장 설득력 있는 모델은 scalar와 attention 근거가 함께 맞는 모델이었다

이번 실험에서 가장 납득 가능한 run은 다음 세 개였다.

1. `pretrain_electricity_laya_ci_none`
2. `pretrain_electricity_laya_mixer_stats`
3. `pretrain_electricity_laya_metadata_query_gate_stats`

이 모델들은 단순히 loss만 낮춘 것이 아니라,

- validation alignment가 좋고,
- representation collapse가 심하지 않으며,
- attention specialization도 비교적 자연스럽게 나타났다.

---

### 4.3 어떤 모델은 objective fit은 약하지만 representation은 더 건강했다

대표적인 예는 다음 모델이다.

- `pretrain_electricity_laya_metadata_query_gate_text_stats_joint`

이 run은 validation loss 자체는 최고 수준이 아니었지만, metadata-query-gate 계열 중 **representation diversity가 가장 건강하게 유지된 모델**이었다.

즉 pure reconstruction 관점에서는 1등이 아니지만, downstream representation transfer 관점에서는 흥미로운 후보라고 볼 수 있다.

---

## 5. 모델별 상세 평가

### 5.1 `pretrain_electricity_laya_ci_none`

**Scalar 근거**

- `best_val_loss = 0.1885`
- `val_align_cos = 0.9221`
- `val_repr_cos = 0.9541`
- `val_repr_var = 0.0169`

**Attention 근거**

- 저장된 attention-map artifact가 없음

**해석**

이 모델은 attention 증거는 없지만 scalar만 놓고 보면 가장 균형이 좋다.
validation alignment가 매우 높고, representation도 심한 collapse를 보이지 않는다.

**결론**

이번 실험에서 가장 안전하게 **잘 pretrained된 모델**이라고 부를 수 있는 기준선이다.

---

### 5.2 `pretrain_electricity_laya_mixer_stats`

**Scalar 근거**

- `best_val_loss = 0.3907`
- `val_align_cos = 0.8733`
- `val_repr_cos = 0.8505`
- `val_repr_var = 0.0684`

**Attention 근거**

- entropy ≈ `1.956`
- max weight mean ≈ `0.424`
- top-1 unique ratio = `1.0`
- query별 top-k channel 목록이 분명히 다름

**해석**

이 모델은 loss가 가장 낮지는 않지만 representation 품질이 매우 건강하다.

- representation similarity가 과도하게 높지 않고,
- variance가 충분히 살아 있으며,
- attention도 query별로 분화되어 있다.

**결론**

representation learning 관점에서는 이번 13개 중 가장 설득력 있는 모델 중 하나다.

---

### 5.3 `pretrain_electricity_laya_metadata_query_gate_stats`

**Scalar 근거**

- `best_val_loss = 0.2910`
- `val_align_cos = 0.8402`
- `val_repr_cos = 0.9697`
- `val_repr_var = 0.0237`
- `val_meta_score_delta_ratio ≈ 0.9283`

**Attention 근거**

- entropy ≈ `1.273`
- max weight mean ≈ `0.603`
- top-1 unique ratio = `0.9375`
- attention이 sharp하고 top-k 구조가 분명함

**해석**

이 모델은 metadata가 실제로 attention score를 강하게 바꾸고 있다는 점이 수치로 확인된다.
attention은 plain mixer보다 더 날카롭고, metadata가 단순 장식이 아니라 실질적으로 작동한 것으로 보인다.

representation diversity가 완벽하진 않지만 허용 가능한 수준이다.

**결론**

metadata-query-gate 계열 중 가장 잘 된 모델이다.

---

### 5.4 `pretrain_electricity_laya_mixer_concat_text`

**Scalar 근거**

- `best_val_loss = 0.0384` (전체 최저)
- `val_align_cos = 0.8015`
- `val_repr_cos = 0.9995`
- `val_repr_var = 0.00035`

**Attention 근거**

- entropy ≈ `2.416`
- max weight mean ≈ `0.328`
- top-1 unique ratio = `1.0`
- query별 top-k channel은 서로 다름

**해석**

이 모델은 “validation loss가 가장 낮은데도 representation은 가장 좋은 것이 아닐 수 있다”는 점을 가장 잘 보여준다.

objective fit은 매우 좋지만,
representation들은 거의 같은 방향으로 수렴했고, feature variance는 거의 0에 가깝다.

attention은 살아 있지만, 최종 representation은 강하게 collapse된 것으로 보인다.

**결론**

예측 objective는 매우 잘 맞췄지만, **representation 학습 모델로는 신뢰하기 어렵다**.

---

### 5.5 `pretrain_electricity_laya_mixer_text`

**Scalar 근거**

- `best_val_loss = 0.0600`
- `val_align_cos = 0.7812`
- `val_repr_cos = 0.9964`
- `val_repr_var = 0.00264`

**Attention 근거**

- entropy ≈ `2.102`
- max weight mean ≈ `0.377`
- top-1 unique ratio = `1.0`

**해석**

이 모델도 `mixer_concat_text`와 유사하다.

loss는 매우 좋고 attention도 구조가 보이지만,
representation은 상당히 강하게 뭉쳐 있다.

**결론**

task fitting은 좋지만, representation quality 모델로 보기는 어렵다.

---

### 5.6 `pretrain_electricity_laya_mixer_text_stats_avg`

**Scalar 근거**

- `best_val_loss = 0.2672`
- `val_align_cos = 0.7724`
- `val_repr_cos = 0.9950`
- `val_repr_var = 0.00359`

**Attention 근거**

- entropy ≈ `1.533`
- max weight mean ≈ `0.589`
- top-1 unique ratio = `1.0`

**해석**

이 모델 역시 representation collapse 성향이 강하다.
attention은 날카롭지만, representation variance가 너무 낮다.

**결론**

objective는 맞췄지만 representation transfer 목적에는 주의가 필요하다.

---

### 5.7 `pretrain_electricity_laya_mixer_concat_text_stats_joint`

**Scalar 근거**

- `best_val_loss = 0.1970`
- `val_align_cos = 0.8723`
- `val_repr_cos = 0.9915`
- `val_repr_var = 0.00593`

**Attention 근거**

- entropy ≈ `1.521`
- max weight mean ≈ `0.533`
- top-1 unique ratio = `1.0`

**해석**

이 모델은 text-only collapse 모델들보다 낫다.
alignment가 좋고 attention도 구조가 있지만, representation similarity가 여전히 너무 높다.

**결론**

중상위권 모델이지만 약한 collapse 위험이 남아 있다.

---

### 5.8 `pretrain_electricity_laya_mixer_concat_text_stats_avg`

**Scalar 근거**

- `best_val_loss = 0.2093`
- `val_align_cos = 0.7703`
- `val_repr_cos = 0.9671`
- `val_repr_var = 0.0334`

**Attention 근거**

- entropy ≈ `3.121` (가장 diffuse한 편)
- max weight mean ≈ `0.208`
- top-1 unique ratio = `1.0`

**해석**

이 모델은 attention이 지나치게 날카롭지 않고 비교적 넓게 퍼져 있다.
representation variance도 충분히 살아 있어 collapse는 덜하다.

alignment가 아주 최고 수준은 아니지만 전체적으로 더 균형적이다.

**결론**

plain low-loss 모델보다 더 건강한 구조를 가진, 균형적인 후보다.

---

### 5.9 `pretrain_electricity_laya_metadata_query_gate_text_stats_joint`

**Scalar 근거**

- `best_val_loss = 0.3751`
- `val_align_cos = 0.7503`
- `val_repr_cos = 0.7032`
- `val_repr_var = 0.1395`

**Attention 근거**

- entropy ≈ `1.679`
- max weight mean ≈ `0.496`
- top-1 unique ratio = `1.0`
- query별 top-k 구조가 명확히 다름

**해석**

이 모델은 metadata-query-gate 계열 중 representation diversity가 가장 건강하다.

objective fit은 최상위가 아니지만,
representation이 가장 덜 collapse되어 있고 attention specialization도 좋다.

**결론**

representation richness를 중시한다면 매우 흥미로운 후보다.

---

### 5.10 `pretrain_electricity_laya_metadata_query_gate_text`

**Scalar 근거**

- `best_val_loss = 0.4403`
- `val_align_cos = 0.7117`
- `val_repr_cos = 0.8721`
- `val_repr_var = 0.0710`

**Attention 근거**

- entropy ≈ `1.564`
- max weight mean ≈ `0.544`
- top-1 unique ratio = `0.9375`

**해석**

representation은 비교적 건강하고 attention도 구조가 있다.
하지만 validation objective 자체는 상위권 모델보다 약하다.

**결론**

무난하지만 강력한 추천 모델은 아니다.

---

### 5.11 `pretrain_electricity_laya_metadata_query_gate_text_stats_avg`

**Scalar 근거**

- `best_val_loss = 0.4737`
- `val_align_cos = 0.6694`
- `val_repr_cos = 0.9767`
- `val_repr_var = 0.0164`

**Attention 근거**

- entropy ≈ `1.713`
- max weight mean ≈ `0.538`
- top-1 unique ratio = `0.9375`

**해석**

이 모델은 objective도 아주 강하지 않고, representation도 살짝 과도하게 뭉쳐 있다.
attention 구조가 아주 나쁘진 않지만, 특별히 돋보이는 장점도 없다.

**결론**

중간 수준이지만 설득력은 약한 run이다.

---

### 5.12 `pretrain_electricity_laya_mixer_text_stats_joint`

**Scalar 근거**

- `best_val_loss = 0.3029`
- `val_align_cos = 0.4632`
- `val_repr_cos = 0.9662`
- `val_repr_var = 0.0296`

**Attention 근거**

- entropy ≈ `1.723`
- max weight mean ≈ `0.502`
- top-1 unique ratio = `1.0`

**해석**

attention 구조는 나쁘지 않고 representation도 완전히 죽진 않았다.
하지만 validation alignment가 너무 낮다.

즉 모델이 target token 정렬을 충분히 잘 못 하고 있다고 보는 편이 맞다.

**결론**

전체적으로 강한 pretrained 모델이라고 보긴 어렵다.

---

### 5.13 `pretrain_electricity_laya_mixer_concat_stats`

**Scalar 근거**

- `best_val_loss = 0.2675`
- final `val_loss = 1.0063`
- `val_align_cos = 0.7927`
- `val_repr_cos = 0.9797`
- `val_repr_var = 0.0170`

**Attention 근거**

- entropy ≈ `1.257`
- max weight mean ≈ `0.583`
- top-1 unique ratio = `1.0`

**해석**

이 모델은 해석할 때 주의가 필요하다.
best checkpoint 시점의 validation 성능은 나쁘지 않지만, 마지막 epoch 기준 validation loss가 크게 악화되어 있다.

즉 attention snapshot은 최종 상태를 반영하고 있고, best checkpoint 상태와는 차이가 있을 가능성이 높다.

**결론**

best checkpoint 기준으론 쓸 만할 수 있지만, 최종 상태는 불안정하다.

---

## 6. 최종 순위 및 추천

### 6.1 가장 신뢰할 만한 pretrained 모델

가장 설득력 있는 모델은 다음 세 개다.

1. **`pretrain_electricity_laya_ci_none`**
   - 전체 scalar 균형이 가장 좋음
   - validation alignment가 가장 강함
   - 심한 collapse 징후 없음

2. **`pretrain_electricity_laya_mixer_stats`**
   - representation 품질이 가장 건강한 축
   - attention specialization이 분명함
   - validation alignment도 강함

3. **`pretrain_electricity_laya_metadata_query_gate_stats`**
   - metadata가 실제로 작동한 흔적이 뚜렷함
   - sharp하지만 해석 가능한 attention
   - objective와 representation의 균형이 좋음

---

### 6.2 Loss는 좋지만 주의해서 봐야 하는 모델

다음 모델들은 validation loss는 매우 좋지만, representation collapse 징후가 강하다.

- `pretrain_electricity_laya_mixer_concat_text`
- `pretrain_electricity_laya_mixer_text`
- `pretrain_electricity_laya_mixer_text_stats_avg`

이 모델들은 “잘 pretrained된 representation 모델”이라고 바로 부르기보다는, 반드시 caveat와 함께 언급해야 한다.

---

### 6.3 Representation diversity 관점에서 가장 흥미로운 모델

- **`pretrain_electricity_laya_metadata_query_gate_text_stats_joint`**

이 모델은 pretraining objective를 가장 잘 맞춘 것은 아니지만, representation geometry는 가장 건강하고 풍부하다.

만약 downstream transfer나 표현 해석 가능성이 중요하다면, 가장 먼저 후속 검증해볼 만한 후보다.

---

## 7. 최종 요약

이번 분석의 핵심 결론은 다음과 같다.

> **validation loss가 낮다고 해서 반드시 좋은 pretraining은 아니다.**

몇몇 모델은 매우 좋은 loss를 기록했지만 representation collapse 징후가 강했고,
반대로 약간 더 높은 loss를 가진 모델들 중 일부는 훨씬 건강한 representation diversity와 attention specialization을 보여주었다.

따라서 실제로 **representation learner로서 잘 pretrained되었다고 볼 수 있는 모델**은 다음 세 개가 가장 강한 근거를 가진다.

- `pretrain_electricity_laya_ci_none`
- `pretrain_electricity_laya_mixer_stats`
- `pretrain_electricity_laya_metadata_query_gate_stats`

반면 representation richness 자체를 더 중요하게 볼 경우에는 다음 모델이 가장 흥미롭다.

- `pretrain_electricity_laya_metadata_query_gate_text_stats_joint`

즉 이번 실험에서는 **loss와 representation quality, 그리고 attention evidence를 반드시 함께 봐야 한다**는 점이 가장 중요하다.
