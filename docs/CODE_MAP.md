# 코드 구성과 실행 안내

이 저장소는 여러 시점의 연구 실험을 보존한 코드 아카이브입니다. 파일 번호는 연구 단계의 구분이며, 모든 파일을 번호순으로 일괄 실행하는 통합 파이프라인을 뜻하지 않습니다.

## 실행 위치와 입력 자료

Python 파일을 저장소 루트에 함께 배치했습니다. **저장소 루트를 현재 작업 디렉터리로 두고** 실행해야 합니다. 대부분의 스크립트는 `./data`, `./output`, `./models`를 사용하며, 일부는 자신의 파일 위치를 기준으로 같은 폴더를 찾습니다.

```text
repository-root/
  2_preprocess.py
  4_make_sequence.py
  ...
  data/youtube_trends/
    most_popular.csv     별도로 준비할 원본 스냅샷
    tag.csv              별도로 준비할 태그 자료
  output/                전처리 자료·시퀀스·예측·평가 결과
  models/                학습한 모델과 체크포인트
```

원본 CSV, 전처리 데이터, 개별 영상의 예측값과 모델 파일은 공개 패키지에 포함하지 않습니다. `requirements.txt`는 코드에서 확인한 직접 의존성 목록이며, 버전이 고정된 재현 환경은 아닙니다. PyTorch와 GPU/CUDA 환경은 사용할 장비에 맞춰 준비해야 합니다.

## 주요 흐름

| 단계 | 파일 | 역할 |
|---|---|---|
| 선택적 원본 점검 | [1_load_data.py](../1_load_data.py) | 두 원본 CSV를 점검하고 로드 사본을 저장합니다. 다음 전처리 단계는 원본 CSV를 직접 읽으므로 필수 선행 단계는 아닙니다. |
| KR 전처리 | [2_preprocess.py](../2_preprocess.py) | 한국 스냅샷 필터링, 특성·지속성 레이블 생성, 영상 단위 분할, train 기준 상대 조회수 계산 |
| 기존 스냅샷 재사용 | [2b_relabel_from_snapshot.py](../2b_relabel_from_snapshot.py) | 이미 보관된 `output/kr_snapshot.csv`로 레이블·특성을 다시 생성하는 대안 경로 |
| 시퀀스 구성 | [4_make_sequence.py](../4_make_sequence.py) | K=5·7·10·14의 시퀀스, train 기준 스케일러와 정적·요약 특성을 생성합니다. |
| ML 평가 | [eval_v3_main.py](../eval_v3_main.py) | 71개 특성의 K=14 모델 비교, AUC·AP·여러 F1 임곗값, 예측값 저장 |
| ML 분석 표 | [ml_tables_v3.py](../ml_tables_v3.py) | validation 임곗값 F1을 사용하는 모델 비교, K·특성군·레이블 임곗값 분석과 bootstrap CI |
| DL 학습 | [5_run_dl.py](../5_run_dl.py) | LSTM·GRU·Transformer·자체 TFT 계열 모델을 네 K 조건에서 학습하고 val/test 예측을 저장합니다. |
| DL 평가 | [eval_dl_v3.py](../eval_dl_v3.py) | 저장된 `predsv3_*_val.csv`, `predsv3_*_test.csv`를 읽어 평가 표와 CI를 생성합니다. |
| 모델 간 비교 | [paired_boot_v3.py](../paired_boot_v3.py) | K=14 예측값을 `video_id`로 맞춘 paired bootstrap 비교 |

입력 준비 후 목적에 맞는 흐름을 선택합니다. 예를 들어 원본 데이터에서 시작하는 공통 준비 단계는 다음과 같습니다. 이 명령은 실제 전처리 자료를 생성합니다.

```powershell
# 저장소 루트에서 실행
python 2_preprocess.py
python 4_make_sequence.py
```

그다음 ML 평가 또는 DL 학습 등 필요한 실험을 실행합니다. `5_run_dl.py`는 최대 16개 학습 실험을 수행합니다. 저장된 평가표를 보는 데에는 학습을 다시 실행할 필요가 없습니다.

## 다른 실험 버전과 분석 도구

| 구분 | 파일 |
|---|---|
| 초기 ML 기준선 | `3_run_all.py`, `3a_rf.py`–`3e_mlp.py`, `model_utils.py` |
| ML v2 | `3_run_ml_v2.py` |
| 초기 개별 DL 실험 | `5a_lstm.py`, `5b_gru.py`, `5c_transformer.py`, `5d_tft.py` |
| 예측·설명·기존 bootstrap | `6a_save_preds.py`–`6e_save_tft_attention.py` |
| 시간 분할·특성·K·레이블 임곗값 | `8_temporal_validation.py`, `9_ablation_baselines.py`, `10_threshold_sensitivity.py`, `12_ml_k_ablation.py`, `ml_tables_v3_temporal.py` |
| 동일 표본 K 비교·범주 처리 | `17_revision_required_experiments.py`, `check_catboost_category_handling.py` |
| 하이브리드와 추가 모델 탐색 | `18_hybrid_transformer_tabular.py`, `19_transformer_catboost_hybrid.py`, `20_revision_best_model_search.py` |
| 그림 제작 | `7_paper_figures.py`, `7b_fig5_replace.py`, `9_paper_figures_v3.py`, `9b_paper_figures_1col.py`, `11_make_kci_figures.py`, `13_make_figure_overview_images.py`–`16_make_korean_figures.py` |
| 자료·평가 확인 | `inspect_*.py`, `verify_counts_and_features.py`, `compute_table2_ap.py` |

실험 버전마다 입력 특성, 학습 분할, F1 임곗값과 출력 파일이 다릅니다. 초기 `model_utils.py` 계열은 80/20 분할을 다시 만들며, `6a_save_preds.py`의 ML 부분은 train+validation 자료로 재학습합니다. 이를 v3의 validation 임곗값 평가와 같은 설정으로 간주하지 않습니다.

`ml_tables_v3_temporal.py`는 `ml_tables_v3.py`를 import하므로 실행 시 ML 분석 표 전체 계산도 함께 시작됩니다. 여러 파일에는 모듈 최상위의 학습·저장 코드가 있어, 함수를 확인하기 위한 import만으로도 작업이 시작될 수 있습니다.

한국어 그림은 원래 Windows의 맑은 고딕 등 글꼴을 사용합니다. 다른 운영체제에서는 해당 그림 스크립트의 글꼴 설정을 준비해야 합니다. 기존 저장 그림과 코드에 직접 기록된 그래프 수치도 보존되어 있으므로, 그림 생성만으로 평가가 다시 수행되는 것은 아닙니다.

## 공개 사본에서 정리한 범위

- `1_load_data.py .py` → `1_load_data.py`
- `3.run_all.py` → `3_run_all.py`; 전처리 완료 메시지의 실행 파일명도 이에 맞췄습니다.
- 일부 파일에 고정된 로컬 프로젝트 경로를 `Path(__file__).resolve().parent` 기준으로 바꿨습니다.
- 모델 계산, 특성 정의, 실험 분할과 평가 로직은 보존했습니다.
- 논문 문서 생성, HWP/PDF 텍스트 추출·교정·렌더링 보조 스크립트는 로컬 자료에 보관했습니다.

원본 파일명, 공개 파일명, 내용 변경과 SHA-256은 [CODE_PROVENANCE.json](CODE_PROVENANCE.json)에 기록했습니다. 공개 Python 파일 51개는 import나 실행 없이 메모리에서 구문 분석·컴파일했습니다. 데이터 전처리, 학습, 추론 및 모델 성능 재현은 이번 정리에서 수행하지 않았습니다.
