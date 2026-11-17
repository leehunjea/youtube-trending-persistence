# 데이터와 실행 환경

## 데이터 배치

공개본에는 원본 CSV와 모델을 포함하지 않습니다. 전처리 단계에서 사용하는 경로는 다음과 같습니다.

~~~text
data/youtube_trends/most_popular.csv
data/youtube_trends/tag.csv
output/    전처리 결과·시퀀스·분할 인덱스·영상별 예측
models/    학습한 모델과 체크포인트
~~~

필요한 로컬 자료를 해당 경로에 준비한 뒤 **저장소 루트**에서 `python scripts/<분류>/<파일>.py` 형식으로 실행합니다. 입력 data와 출력 output/models는 scripts 하위가 아닌 저장소 최상위 경로를 사용합니다. 원본 두 CSV의 합계는 약 129.4GB입니다. 메모리와 실행 시간은 데이터 크기 및 선택한 실험에 따라 달라집니다. 공개된 집계 결과를 열어 보는 데에는 원본 데이터가 필요하지 않습니다.

## 환경 준비

requirements.txt는 코드에서 확인한 직접 의존성 목록입니다. 새 환경을 설치하거나 재현 실행을 검증한 잠금 파일은 아닙니다.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
~~~

모델 학습에는 PyTorch와 장치에 맞는 실행 환경이 필요합니다. 기존 환경 폴더는 사용자 승인 후 정리했으며, 패키지 목록과 설치 메타데이터는 로컬 감사 기록에 보존했습니다. 새 경로에서 환경을 다시 구성합니다.

아래는 정리 당시 **설치 메타데이터에서 읽은 버전**이며, 어떤 환경이 모든 결과를 생성했는지는 확정하지 않았습니다.

| 항목 | run_env | kaggle_env |
|---|---|---|
| Python | 3.12.13 | 3.13.14 |
| PyTorch | 2.12.1 | 2.6.0+cu124 |
| NumPy | 2.4.6 | 2.4.3 |
| pandas | 3.0.3 | 3.0.1 |
| scikit-learn | 1.9.0 | 1.8.0 |
| CatBoost | 1.2.10 | 1.2.10 |
| LightGBM | 4.6.0 | 4.6.0 |
| XGBoost | 3.3.0 | 3.2.0 |

전체 설치 목록은 로컬 아카이브의 감사 기록에 보관했습니다. 설치 목록은 의존성 해석·GPU 호환성·재현성을 확인했다는 뜻이 아닙니다.

## 실험 선택

1. [코드 안내](CODE_MAP.md)에서 필요한 실험 분기를 선택합니다.
2. 전처리를 시작하려면 원본 CSV를 준비하고 [전처리](../scripts/preprocessing/2_preprocess.py), [시퀀스 생성](../scripts/preprocessing/4_make_sequence.py)의 설정을 확인합니다.
3. 후속 실험은 기존 전처리 파일·모델 등 각 스크립트가 읽는 입력을 확인한 뒤 실행합니다.

선택적 점검 도구인 1_load_data.py는 output/raw_popular.csv와 output/raw_tag.csv를 추가로 내보냅니다. 현재 전처리는 이 사본이 아닌 원본 CSV를 직접 읽습니다. 로컬의 기존 내보내기 사본 두 개(약 58.7GB)는 정리했으며, 이 도구를 실행하면 다시 생성될 수 있습니다.

일부 스크립트는 실행 즉시 학습하거나 기존 출력 파일을 덮어씁니다. 특히 ml_tables_v3_temporal.py는 ml_tables_v3를 import하면서 해당 모듈의 최상위 학습 코드도 실행합니다. 전체 파일을 일괄 import하는 방식으로 검사하지 않습니다.

공개 준비에서는 Python 문법, 파일 연결, 집계 수치, 공개 범위를 확인했으며 학습·추론·원본 데이터 재처리는 실행하지 않았습니다. 연구 코드에 새 라이선스를 임의로 부여하지 않았습니다.

## English summary

Keep the original CSVs under data/youtube_trends/ and run scripts from the repository root using `python scripts/<category>/<file>.py`. Generated arrays and predictions belong in the root output/ directory; checkpoints belong in the root models/ directory. These directories are excluded from Git. Do not change into a script subdirectory before running it.

Create a fresh environment and install the direct dependencies in requirements.txt. The version table above records metadata from two former environments; it is not a tested lockfile or a claim that either environment reproduces all results. The environment directories were removed with approval after preserving package lists and installation metadata.

The optional 1_load_data.py tool exports output/raw_popular.csv and output/raw_tag.csv. Current preprocessing reads the original CSVs directly. The old exports (about 58.7GB) were removed during cleanup; running this optional tool can regenerate them.

Choose an experiment branch using the code map. Some scripts train at module scope and overwrite outputs. Importing ml_tables_v3_temporal.py also triggers the top-level training code in ml_tables_v3. Validation during packaging was static; training, inference, and full data processing were not rerun.
