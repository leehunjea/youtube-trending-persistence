# 코드 구성과 실행 안내

연구 코드는 scripts 아래에 목적별로 분류했습니다. 초기 실험은 legacy에 모았으며, 원래 파일명은 실험 기록을 따라갈 수 있도록 유지했습니다.

## 실행 위치

**저장소 루트에서 파일 경로를 지정해 실행합니다.** scripts 하위 폴더로 이동해서 실행하지 않습니다.

~~~powershell
# 원본 데이터가 준비된 저장소 루트에서 실행
python scripts/preprocessing/2_preprocess.py
python scripts/preprocessing/4_make_sequence.py
~~~

~~~text
repository-root/
  scripts/              공개 연구 코드
  data/youtube_trends/   별도로 준비할 most_popular.csv, tag.csv
  output/               전처리 자료·분할·영상별 예측·실험 출력
  models/               학습 모델·체크포인트
  results/              공개용으로 선별한 집계표와 그림
~~~

기존 코드의 ./data, ./output, ./models는 계속 저장소 루트를 기준으로 합니다. 파일 위치로 프로젝트 경로를 구하는 코드도 이 구조에 맞췄습니다. 모델 파일과 원본 CSV는 공개본에 포함하지 않습니다.

## 폴더별 목적

| 폴더 | 파일 수 | 내용 |
|---|---:|---|
| [preprocessing](../scripts/preprocessing/) | 4 | 원본 점검, KR 전처리, 라벨 생성, 초기 K개 시퀀스 구성 |
| [training](../scripts/training/) | 5 | ML v2, DL 학습, 하이브리드 모델과 추가 모델 탐색 |
| [evaluation](../scripts/evaluation/) | 11 | 모델 비교, 예측 생성, 평가표·bootstrap·설명 정보. 모델 비교용 재학습을 수행하는 파일도 포함합니다. |
| [analysis](../scripts/analysis/) | 6 | 시간 분할, 특징 제거, K·라벨 임곗값·범주 처리 분석 |
| [visualization](../scripts/visualization/) | 9 | 저장된 결과와 기존 실험 값의 그림 생성 |
| [tools](../scripts/tools/) | 5 | 데이터 수집 기간, 스냅샷, 배열 키, 특징 점검 |
| [legacy](../scripts/legacy/) | 11 | 초기 ML 기준선, 개별 DL 실험, 공통 model_utils |

## 먼저 볼 코드

| 목적 | 파일 | 필요한 자료와 동작 |
|---|---|---|
| 원본 전처리 | [2_preprocess.py](../scripts/preprocessing/2_preprocess.py) | 원본 CSV에서 KR 데이터·영상 라벨과 특징을 생성 |
| 스냅샷 재사용 | [2b_relabel_from_snapshot.py](../scripts/preprocessing/2b_relabel_from_snapshot.py) | 기존 output/kr_snapshot.csv를 사용하는 전처리 대안 |
| 초기 관측 구성 | [4_make_sequence.py](../scripts/preprocessing/4_make_sequence.py) | K=5·7·10·14, train 기준 스케일러·시퀀스·정적 특징 생성 |
| ML v3 평가 | [eval_v3_main.py](../scripts/evaluation/eval_v3_main.py) | 71개 특징으로 모델을 학습하고 K=14 결과 평가 |
| ML v3 분석 표 | [ml_tables_v3.py](../scripts/evaluation/ml_tables_v3.py) | 모델 학습, validation 임곗값 F1, K·특징·임곗값 분석과 CI |
| DL 학습 | [5_run_dl.py](../scripts/training/5_run_dl.py) | LSTM·GRU·Transformer·자체 TFT 계열 모델을 네 K 조건에서 학습 |
| DL 평가 | [eval_dl_v3.py](../scripts/evaluation/eval_dl_v3.py) | 저장된 val/test 예측에서 평가표 생성 |
| 모델 간 비교 | [paired_boot_v3.py](../scripts/evaluation/paired_boot_v3.py) | 영상 ID를 맞춰 paired bootstrap 비교 |
| 대표 그림 | [9_paper_figures_v3.py](../scripts/visualization/9_paper_figures_v3.py) | 코드에 기록된 v3 집계값을 시각화 |

모든 파일을 번호순으로 실행하는 하나의 재현 파이프라인은 아닙니다. 필요한 실험 분기를 선택하고 입력 파일을 확인하세요. 데이터 준비만으로 모든 후속 실험의 파일이 자동 생성되는 것은 아닙니다.

[1_load_data.py](../scripts/preprocessing/1_load_data.py)는 선택적 점검·내보내기 도구입니다. 현재 전처리는 원본 CSV를 직접 읽으므로 이 도구가 필수 선행 단계는 아닙니다. 이를 실행하면 대용량 raw_popular.csv와 raw_tag.csv 사본이 다시 생성될 수 있습니다.

## 초기 실험과 실행 시 주의점

- legacy/model_utils 계열은 80/20 분할을 다시 만들며, v3 실험과 설정이 다릅니다. [초기 ML 실행기](../scripts/legacy/3_run_all.py)는 같은 폴더의 3a–3e 파일을 실행합니다.
- evaluation/6a_save_preds.py는 ML 모델을 train+validation 자료로 재학습합니다. 저장된 예측을 읽기만 하는 스크립트로 취급하지 않습니다.
- evaluation/ml_tables_v3_temporal.py는 같은 폴더의 ml_tables_v3를 import하며, 이때 최상위 학습 코드도 실행됩니다.
- 여러 스크립트는 실행 시 학습과 파일 저장을 시작합니다. 전부 import하는 방법으로 동작을 확인하지 않습니다.
- 일부 그림은 기존 집계값을 코드에 직접 기록합니다. 그림을 다시 그리는 것과 모델 평가를 다시 수행하는 것은 다릅니다. 한국어 그림은 Windows 글꼴 설정도 확인해야 합니다.

[데이터·환경 안내](DATA_AND_ENVIRONMENT.md) · [결과 출처와 해석 범위](RESULTS.md)

## 파일 이동과 보존 범위

51개 Python 파일을 역할별 폴더로 이동했습니다. 프로젝트 기준 경로와 후속 실행 안내만 수정했으며 모델 계산, 특징 정의, 데이터 분할과 평가 로직은 유지했습니다. 서로 import하는 파일은 같은 폴더에 두었습니다.

원본 연구 파일명, 현재 공개 경로, 단계별 수정 내용과 SHA-256은 [CODE_PROVENANCE.json](CODE_PROVENANCE.json)에 기록했습니다. 공개본을 정리하면서 생긴 파일명 변경 두 건(1_load_data.py .py, 3.run_all.py)도 이 기록에서 확인할 수 있습니다.

구문 검사와 파일 경로·의존 파일 연결을 확인했습니다. 원본 데이터 전처리·모델 학습·추론은 다시 실행하지 않았습니다.
