# 결과와 출처

이 저장소는 로컬에 저장된 **v3 실험 집계 결과**와 해당 수치의 시각화를 공개합니다. CSV는 2026년 8월 27일, 대표 그림은 2026년 8월 31일에 저장된 파일입니다. 정리 과정에서 학습·평가를 다시 실행하지 않았습니다.

## 대표 결과

K=14 조건의 저장된 test 결과입니다. F1은 validation set에서 선택한 임곗값을 test set에 적용한 `F1_valthr`입니다.

| 모델 | ROC-AUC | AP | F1 |
|---|---:|---:|---:|
| CatBoost | 0.8416 | 0.6294 | 0.6270 |
| LightGBM | 0.8385 | 0.6353 | 0.6272 |
| XGBoost | 0.8361 | 0.6286 | 0.6284 |
| RandomForest | 0.8318 | 0.6150 | 0.6226 |
| LSTM | 0.8296 | 0.5940 | 0.6236 |
| GRU | 0.8266 | 0.5853 | 0.6261 |
| Transformer | 0.8254 | 0.5769 | 0.6198 |
| TFT | 0.8236 | 0.5838 | 0.6219 |
| MLP-Tabular | 0.8235 | 0.5758 | 0.6215 |

출처: [ML 집계](../results/tables/tab6_ml_v3.csv), [DL 집계](../results/tables/tab6_dl_v3.csv). CatBoost의 저장된 ROC-AUC 95% bootstrap 신뢰구간은 **[0.8252, 0.8575]**입니다. CatBoost의 ROC-AUC가 가장 높지만 AP는 LightGBM, F1은 XGBoost가 더 높습니다.

## 공개 파일의 원본 대응

아래 원본 경로는 정리 전 `D:\youtube`를 기준으로 기록했습니다. 공개 사본의 CSV·PNG는 원본과 SHA-256이 일치하며, 값이나 이미지를 수정하지 않았습니다.

| 공개 파일 | 정리 전 원본 경로 |
|---|---|
| [tab6_ml_v3.csv](../results/tables/tab6_ml_v3.csv) | `D:\youtube\output\tab6_ml_v3.csv` |
| [tab6_dl_v3.csv](../results/tables/tab6_dl_v3.csv) | `D:\youtube\output\tab6_dl_v3.csv` |
| [tab7_ml_kabl_v3.csv](../results/tables/tab7_ml_kabl_v3.csv) | `D:\youtube\output\tab7_ml_kabl_v3.csv` |
| [tab8_dl_kabl_v3.csv](../results/tables/tab8_dl_kabl_v3.csv) | `D:\youtube\output\tab8_dl_kabl_v3.csv` |
| [tab9_featabl_v3.csv](../results/tables/tab9_featabl_v3.csv) | `D:\youtube\output\tab9_featabl_v3.csv` |
| [tab10_importance_v3.csv](../results/tables/tab10_importance_v3.csv) | `D:\youtube\output\tab10_importance_v3.csv` |
| [tab11_thresh_v3.csv](../results/tables/tab11_thresh_v3.csv) | `D:\youtube\output\tab11_thresh_v3.csv` |
| [fig1_model_comparison.png](../results/figures/fig1_model_comparison.png) | `D:\youtube\figures_v3\fig1_model_comparison.png` |
| [fig2_k_ablation.png](../results/figures/fig2_k_ablation.png) | `D:\youtube\figures_v3\fig2_k_ablation.png` |

원본 `9_paper_figures_v3.py`는 저장된 집계값을 코드에 명시하여 위 그림을 그립니다. 따라서 그림을 다시 생성하는 작업은 모델 평가를 다시 실행하는 것과 다릅니다.

## 해석 범위

- **예측 대상:** 전체 트렌딩 등장 횟수가 60회 이상인 장기 지속 영상입니다. 60회는 전체 17,883개 영상의 75백분위수이며, 일반적인 조회수 인기나 외부 확산을 직접 측정한 라벨은 아닙니다.
- **입력 범위:** 초기 K개 스냅샷을 사용합니다. 전체 지속 횟수에는 초기 K개도 포함되므로, 입력 관측 구간과 라벨 산정 구간이 완전히 분리됐다고 해석하지 않습니다.
- **분할과 표본:** 주 실험은 stratified random split의 학습/검증/시험 70/15/15입니다. K=5/7/10/14의 대상 영상 수는 각각 17,743/17,677/17,569/17,410개로 다릅니다. K 비교는 완전히 동일한 표본에 대한 관측 길이 효과만을 분리한 실험이 아닙니다. 원본 근거는 `output/_seq_rebuild.log`와 `output/label_metadata.csv`입니다.
- **라벨 임곗값:** p65~p85 비교에서는 양성률과 예측 과제의 정의도 바뀝니다. AUC·AP·F1을 동일한 과제의 단순 성능 순위로 해석하지 않습니다.
- **특징 중요도:** 저장된 `category_id` 중요도에는 범주 코드를 수치형으로 처리한 효과가 포함될 수 있으며, 인과적 영향이나 범주형 처리의 우월성을 뜻하지 않습니다.

## 이전 원고·추가 실험과의 구분

이전 `KCI_PAPER_DRAFT.md`와 일부 원고·그림에는 CatBoost ROC-AUC **0.8444**가 남아 있습니다. 공개 README와 위 표는 저장된 v3 집계의 **0.8416**을 사용합니다. 2026년 11월 17일 [게재 논문의 KCI 공식 초록](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART003397844)에서 CatBoost(K=14)의 **ROC-AUC 0.8416, AP 0.6294, F1 0.6270**을 확인했으며, 저장 결과와 일치합니다. 다른 모델을 포함한 전체 비교 표는 공개 논문 본문과 대조하지 않았습니다. [논문 안내](PAPER.md)

원본 시간순 분할 결과 `output/tab12_temporal_v3.csv`는 이미 생성된 표준화 배열을 다시 나누어 사용합니다. 시간순 학습 구간만으로 전처리를 독립적으로 다시 적합한 검증으로 보기 어려워 공개 대표 표에는 포함하지 않았습니다. 또한 원본 `output/paired_boot_v3.csv`의 모델 간 비교는 공통 영상 2,600개에 대한 결과로, 전체 test 표본과 동일하다고 가정하지 않습니다.

원본 데이터, 영상별 예측값·ID, 임베딩, 전처리 배열, 가중치, 원고와 심사 대응 자료는 이 결과 패키지에 포함하지 않았습니다.
