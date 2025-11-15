# Bitcoin Price Forecasting: Flow Matching vs Baselines

본 저장소는 비트코인 시간당 가격 예측을 위해 Flow Matching 기반 모델과 전통/딥러닝 대조군을 구현하고, 재현 가능한 실험 파이프라인을 제공합니다.

## 주요 특징
- 데이터 파이프라인: Yahoo Finance 1시간 봉(BTC-USD), 최근 720일 자동 폴백, 특징 엔지니어링(OHLCV + 기술지표), 표준화, 윈도우 생성
- 모델:
  - Baselines: Naive, SeasonalNaive(24h)
  - ML/DL: LSTM, Transformer, N-BEATS
  - Generative: Transformer Diffusion(DDPM-style denoising + Transformer encoder-decoder)
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
Naive, SeasonalNaive, LSTM, Transformer, N-BEATS, Transformer Diffusion, Flow를 동일 파이프라인에서 비교.
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
  - Transformer Diffusion: `src/models/deep_learning/transformer_diffusion.py`
  - Flow Matching: `src/models/flow_matching/flow_net.py`
- 학습/평가: `src/training/trainer.py`, `src/evaluation/metrics.py`, `src/evaluation/evaluator.py`
- 시각화/IO: `src/utils/visualization.py`, `src/utils/io.py`
- 실행 스크립트: `experiments/scripts/run_demo.py`, `experiments/scripts/run_compare.py`

## 모델 상세 설명

### Transformer Diffusion (새로 추가)
**개념**: DDPM(Denoising Diffusion Probabilistic Models) 방식의 생성 모델을 시계열 예측에 적용한 트랜스포머 기반 아키텍처

**핵심 특징**:
- **Transformer Encoder**: 과거 시계열 데이터(lookback window)를 인코딩하여 컨텍스트 생성
- **Transformer Decoder**: Cross-attention을 통해 과거 데이터와 예측 시퀀스를 연결
- **Diffusion Process**:
  - 학습: 정규 분포 노이즈를 예측 타겟에 추가하고, 모델이 노이즈를 예측하도록 학습 (DDPM objective)
  - 추론: 순수 노이즈에서 시작해 50 스텝의 반복적 디노이징을 통해 최종 예측 생성
- **Timestep Conditioning**: Sinusoidal embedding + Adaptive Layer Normalization으로 디퓨전 타임스텝 조건화
- **일괄 예측**: 미래 horizon(24시간) 전체를 한 번에 생성

**장점**:
- 불확실성 모델링 가능 (생성 모델의 확률적 특성)
- 트랜스포머의 장거리 의존성 포착 능력
- 반복적 정제(iterative refinement)를 통한 고품질 예측

**단점**:
- 추론 시 여러 스텝 필요로 계산 비용 증가
- 학습이 상대적으로 느림

### Flow Matching (기존)
**개념**: Rectified Flow 스타일의 연속 정규화 플로우(Continuous Normalizing Flow)

**핵심 특징**:
- GRU 기반 컨디션 인코더로 과거 데이터 임베딩
- MLP 기반 velocity field 예측
- 선형 보간 기반 학습 목표 (노이즈 → 타겟 간 직선 경로)
- ODE solver로 추론 (10 스텝)

**장점**:
- Diffusion보다 빠른 샘플링
- 안정적인 학습

### 전통 딥러닝 모델
- **LSTM**: 순환 신경망 기반, 시계열의 순차적 패턴 학습
- **Transformer**: Self-attention 기반, 전체 시퀀스의 관계 포착
- **N-BEATS**: 해석 가능한 백캐스팅/포캐스팅 구조, 트렌드/시즌성 분해

### 베이스라인
- **Naive**: 마지막 관측값 반복
- **Seasonal Naive**: 24시간 전 값 반복 (일간 계절성 반영)

## 재현성
- 시드 고정: `set_seed(42)` 사용 (run_compare 스크립트)
- 환경: `requirements.txt`
- 결과 저장: 자동으로 `results/` 아래에 메트릭/플롯/체크포인트 기록

## 참고 문헌
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Rectified Flow (Liu et al., 2023)
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Time series DL baselines (LSTM/Transformer/N-BEATS)

## 주의 및 면책
- 본 코드는 연구/실험 용도입니다. 투자 의사결정에 사용하지 마십시오.
- 데이터 제공처 제약(야후 1시간 봉 최대 ~730일)에 따른 폴백 로직을 포함합니다.

## 라이선스
- 본 프로젝트는 MIT License를 따릅니다. 상세 내용은 `LICENSE` 파일을 참고하세요.


---

## English Overview

### What is this project?
Reproducible PyTorch pipeline for Bitcoin hourly forecasting comparing Flow Matching against classic and deep learning baselines.

### Key features
- Data pipeline: Yahoo Finance hourly BTC-USD with recent-720-day fallback, feature engineering (OHLCV + TA), standardization, sliding windows
- Models:
  - Baselines: Naive, SeasonalNaive (24h)
  - Deep Learning: LSTM, Transformer, N-BEATS
  - Generative: Transformer Diffusion (DDPM-style denoising + Transformer encoder-decoder)
  - Proposed: Flow Matching (TimeSeriesFlowNet with rectified-flow objective + ODE inference)
- Evaluation & viz: MAE/RMSE/MAPE/Directional Accuracy, training/validation curves, model comparison bars, forecast examples, inverse-scaled metrics
- Scripts: `run_demo` for a minimal end-to-end run, `run_compare` for multi-model benchmarking

### Quickstart
```bash
python3 -m pip install -r requirements.txt
python3 -m experiments.scripts.run_demo
```
Outputs (examples) under `results/demo/`: training curves, validation metrics, forecast example, model checkpoint, and JSON metrics.

### Benchmark script
```bash
python3 -m experiments.scripts.run_compare
```
Generates `results/compare/compare_metrics.json`, `compare_mae.png`, `compare_rmse.png` covering Naive, SeasonalNaive, LSTM, Transformer, N-BEATS, Transformer Diffusion, and Flow.

### Model Details

#### Transformer Diffusion (Newly Added)
**Concept**: Transformer-based architecture applying DDPM (Denoising Diffusion Probabilistic Models) to time series forecasting.

**Key Features**:
- **Transformer Encoder**: Encodes historical time series (lookback window) to create context
- **Transformer Decoder**: Connects past data and forecast sequence via cross-attention
- **Diffusion Process**:
  - Training: Adds Gaussian noise to target predictions; model learns to predict the noise (DDPM objective)
  - Inference: Starts from pure noise and generates final forecast through 50-step iterative denoising
- **Timestep Conditioning**: Uses sinusoidal embedding + adaptive layer normalization for diffusion timestep conditioning
- **Direct Multi-step Forecasting**: Generates entire future horizon (24 hours) at once

**Advantages**:
- Uncertainty modeling (probabilistic nature of generative models)
- Long-range dependency capture via Transformer architecture
- High-quality predictions through iterative refinement

**Disadvantages**:
- Higher computational cost due to multi-step inference
- Slower training compared to direct forecasters

#### Flow Matching (Existing)
**Concept**: Continuous Normalizing Flow with Rectified Flow-style objective.

**Key Features**:
- GRU-based condition encoder for history embedding
- MLP-based velocity field prediction
- Linear interpolation training objective (straight path from noise to target)
- ODE solver for inference (10 steps)

**Advantages**:
- Faster sampling than Diffusion
- Stable training

#### Traditional Deep Learning Models
- **LSTM**: Recurrent neural network for sequential pattern learning
- **Transformer**: Self-attention based, captures relationships across entire sequence
- **N-BEATS**: Interpretable backcasting/forecasting structure with trend/seasonality decomposition

#### Baselines
- **Naive**: Repeats last observed value
- **Seasonal Naive**: Repeats value from 24 hours ago (daily seasonality)

### Notes
- Yahoo hourly data is limited to the most recent ~730 days; the pipeline falls back to `period=720d` automatically.
- Inverse scaling of MAE/RMSE to price units uses the target scaler's std.
- Seed fixed at 42 in `run_compare` for reproducibility.

### References
- Flow Matching for Generative Modeling (Lipman et al., 2022)
- Rectified Flow (Liu et al., 2023)
- Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- Time series DL baselines (LSTM/Transformer/N-BEATS)

### License
This project is licensed under the MIT License. See `LICENSE` for details.
