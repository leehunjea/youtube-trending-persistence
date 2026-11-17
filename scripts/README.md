# Research Scripts

연구 코드 51개를 용도별로 분류했습니다. **저장소 루트에서** 파일 경로를 지정해 실행합니다.

Run scripts **from the repository root**, keeping data/, output/, and models/ at that root.

~~~powershell
python scripts/preprocessing/2_preprocess.py
python scripts/preprocessing/4_make_sequence.py
~~~

| Folder | Files | Purpose |
|---|---:|---|
| [preprocessing](preprocessing/) | 4 | 데이터 전처리와 시퀀스 구성 · Data preparation |
| [training](training/) | 5 | 모델 학습과 하이브리드 실험 · Model training |
| [evaluation](evaluation/) | 11 | 모델 비교·예측·평가 · Comparison and evaluation |
| [analysis](analysis/) | 6 | 특성·관측 길이·임곗값 분석 · Ablation and sensitivity studies |
| [visualization](visualization/) | 9 | 결과 그림 생성 · Result plotting |
| [tools](tools/) | 5 | 데이터·배열 점검 · Data checks |
| [legacy](legacy/) | 11 | 초기 ML·DL 실험 · Earlier experiments |

Some evaluation scripts also train models. These files represent several experiment branches; they are not a single pipeline to run in numerical order.

[Detailed code guide](../docs/CODE_MAP.md) · [Data and environment](../docs/DATA_AND_ENVIRONMENT.md) · [Back to overview](../README.md)
