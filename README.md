# Bitcoin Price Forecasting: Flow Matching vs Baselines

본 저장소는 비트코인 시간당 가격 예측을 위해 Flow Matching 기반 모델과 전통/딥러닝 대조군을 구현하고, 재현 가능한 실험 파이프라인을 제공합니다.

## 주요 특징
- 데이터 파이프라인: Yahoo Finance 1시간 봉(BTC-USD), 최근 720일 자동 폴백, 특징 엔지니어링(OHLCV + 기술지표), 표준화, 윈도우 생성
- 모델:
  - Baselines: Naive, SeasonalNaive(24h)
  - ML/DL: LSTM, Transformer, N-BEATS
  - Proposed: Flow Matching(TimeSeriesFlowNet + Rectified Flow-style 학습 + ODE 추론)
- 평가/시각화: MAE/RMSE/MAPE/방향정확도, 학습곡선/검증곡선, 비교 막대그래프, 예측 예시, 실측 단위 역스케일 지표 저장
- 스크립트: 데모 실행(run_demo)과 대조군 비교(run_compare)

## 프로젝트 구조
```
FlowMatching/
├── src/
│   ├── data/                     # 데이터 로딩/전처리/특징
│   ├── models/                   # 모델 구현(베이스라인/딥러닝/Flow)
│   ├── training/                 # 학습/평가 루틴
│   ├── evaluation/               # 메트릭/평가 유틸
│   └── utils/                    # IO/로깅/시각화
├── experiments/
│   └── scripts/                  # 실행 스크립트(run_demo, run_compare)
├── results/                      # 실행 결과(메트릭/플롯/체크포인트)
├── requirements.txt
└── README.md
```

## 데이터
- 소스: Yahoo Finance(`BTC-USD`), 1시간 봉
- 제약: 1시간 데이터는 최근 ~730일만 제공 → 요청 범위 실패 시 최근 720일 `period`로 자동 폴백
- 특징: 
  - 기본: `open, high, low, close, volume`
  - 파생: `returns, volatility, ma_7/25/99, rsi, macd, macd_signal, bb_upper/lower, volume_ma/std`
- 분할: 제공 기간 내 시계열 순서 유지. None 입력 시 70/15/15 비율 자동 분할
- 정규화: `StandardScaler`(train 기준). 지표 역스케일을 위해 target std 저장/활용

## 설치
```bash
python3 -m pip install -r requirements.txt
```

## 빠른 시작(데모)
최근 720일 데이터 기준으로 파이프라인 실행, 베이스라인 평가, Flow Matching 학습/평가, 결과 저장/시각화.
```bash
python3 -m experiments.scripts.run_demo
```
결과 산출물(예시): `results/demo/`
- `baselines_metrics.json`(Naive/SeasonalNaive)
- `flow_training_history.json`, `flow_model.pth`
- `flow_last_epoch_metrics_real.json`(실측 단위 MAE/RMSE)
- `train_loss.png`, `val_metrics.png`, `forecast_example.png`
- `compare_mae.png`, `compare_rmse.png`

## 대조군 비교 실행
Naive, SeasonalNaive, LSTM, Transformer, N-BEATS, Flow를 동일 파이프라인에서 비교.
```bash
python3 -m experiments.scripts.run_compare
```
출력: `results/compare/compare_metrics.json`, `compare_mae.png`, `compare_rmse.png`

## 주요 파일
- 데이터/전처리: `src/data/data_loader.py`, `src/data/feature_engineering.py`, `src/data/preprocessor.py`
- 모델:
  - Baselines: `src/models/baselines/naive.py`
  - LSTM: `src/models/deep_learning/lstm.py`
  - Transformer: `src/models/deep_learning/transformer.py`
  - N-BEATS: `src/models/deep_learning/nbeats.py`
  - Flow Matching: `src/models/flow_matching/flow_net.py`
- 학습/평가: `src/training/trainer.py`, `src/evaluation/metrics.py`, `src/evaluation/evaluator.py`
- 시각화/IO: `src/utils/visualization.py`, `src/utils/io.py`
- 실행 스크립트: `experiments/scripts/run_demo.py`, `experiments/scripts/run_compare.py`

## 재현성
- 시드 고정: 필요 시 `torch.manual_seed(42)` 추가
- 환경: `requirements.txt`
- 결과 저장: 자동으로 `results/` 아래에 메트릭/플롯/체크포인트 기록

## 참고 문헌
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Rectified Flow (Liu et al., 2023)
- Time series DL baselines (LSTM/Transformer/N-BEATS)

## 주의 및 면책
- 본 코드는 연구/실험 용도입니다. 투자 의사결정에 사용하지 마십시오.
- 데이터 제공처 제약(야후 1시간 봉 최대 ~730일)에 따른 폴백 로직을 포함합니다.

## 라이선스
- 별도 명시 없을 경우 MIT 라이선스를 권장합니다. 필요 시 업데이트하세요.
