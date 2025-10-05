# 비트코인 가격 예측: Flow Matching vs 전통 기법 비교 실험 설계

## 📋 1. 실험 개요

### 1.1 목적
- **주 목표**: Flow Matching 기반 시계열 예측이 비트코인 가격 예측에서 전통적 방법보다 우수한지 검증
- **부 목표**: 
  - 불확실성 정량화 성능 비교
  - 계산 효율성 비교
  - 극단 시장 상황에서의 robustness 비교

### 1.2 연구 질문
1. Flow Matching이 점 예측(point forecast) 정확도에서 우수한가?
2. 확률적 예측(probabilistic forecast)에서 더 나은 불확실성 추정을 제공하는가?
3. 계산 비용 대비 성능이 우수한가?
4. 시장 변동성이 클 때 더 robust한가?

---

## 📊 2. 데이터셋 구성

### 2.1 데이터 소스
```python
# 데이터 수집
출처: Yahoo Finance, Binance API, CoinGecko API
기간: 2018-01-01 ~ 2024-12-31 (7년)
빈도: 1시간 단위 (hourly)
총 샘플: ~61,000 시간
```

### 2.2 Feature 구성

#### 기본 OHLCV
- `open`: 시가
- `high`: 고가
- `low`: 저가
- `close`: 종가 (주 타겟)
- `volume`: 거래량

#### 파생 Features (Technical Indicators)
```python
features = {
    # 가격 기반
    'returns': '로그 수익률',
    'volatility': '이동 표준편차 (20시간)',
    
    # 이동평균
    'ma_7': '7시간 이동평균',
    'ma_25': '25시간 이동평균',
    'ma_99': '99시간 이동평균',
    
    # 기술 지표
    'rsi': 'Relative Strength Index',
    'macd': 'MACD',
    'macd_signal': 'MACD Signal',
    'bb_upper': '볼린저 밴드 상단',
    'bb_lower': '볼린저 밴드 하단',
    
    # 거래량 지표
    'volume_ma': '거래량 이동평균',
    'volume_std': '거래량 표준편차',
}
```

#### 외부 Features (선택적)
- Fear & Greed Index
- Google Trends (Bitcoin 검색량)
- S&P 500 수익률 (시장 심리)

### 2.3 데이터 분할

```python
# 시계열 특성 고려 (랜덤 분할 X)
train_period = '2018-01-01' ~ '2022-12-31'  # 5년 (87.5%)
val_period   = '2023-01-01' ~ '2023-06-30'  # 6개월 (8.75%)
test_period  = '2023-07-01' ~ '2024-12-31'  # 1.5년 (3.75%)

# 특별 테스트셋
crisis_test = {
    'luna_crash': '2022-05-07' ~ '2022-05-15',     # 루나 사태
    'ftx_collapse': '2022-11-06' ~ '2022-11-14',   # FTX 파산
    'high_volatility': 'volatility > 90th percentile'
}
```

### 2.4 정규화
```python
# 각 feature별 정규화 (train set 기준)
scaler = StandardScaler()  # 또는 RobustScaler

# 타겟 정규화
target_scaler = StandardScaler()
# 또는 로그 변환: log(price / price[t-1])
```

### 2.5 시퀀스 생성
```python
window_config = {
    'lookback': 168,   # 7일 (7 * 24시간)
    'horizon': 24,     # 1일 예측 (24시간)
    'stride': 1        # 슬라이딩 윈도우
}
```

---

## 🤖 3. 비교 모델 (대조군)

### 3.1 Baseline Models

#### Model 1: Naive Forecasting
```python
class NaiveForecaster:
    """
    가장 단순한 baseline
    - Persistence: y_t+h = y_t
    - Seasonal Naive: y_t+h = y_t-s (s=24 for daily seasonality)
    """
    def predict(self, last_value):
        return last_value  # 또는 last_value + drift
```

#### Model 2: ARIMA / SARIMA
```python
from statsmodels.tsa.statespace.sarimax import SARIMAX

model_config = {
    'order': (5, 1, 2),              # (p, d, q)
    'seasonal_order': (1, 0, 1, 24)  # (P, D, Q, s)
}
```

### 3.2 Machine Learning Models

#### Model 3: XGBoost
```python
import xgboost as xgb

model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=7,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror'
)

# Feature: 과거 168시간 flatten + technical indicators
```

#### Model 4: LightGBM
```python
import lightgbm as lgb

model = lgb.LGBMRegressor(
    n_estimators=1000,
    num_leaves=63,
    learning_rate=0.01,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5
)
```

### 3.3 Deep Learning Models

#### Model 5: LSTM
```python
class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, 24)  # 24시간 예측
    
    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        return self.fc(h_n[-1])
```

#### Model 6: Transformer (Temporal Fusion Transformer)
```python
from pytorch_forecasting import TemporalFusionTransformer

model_config = {
    'hidden_size': 128,
    'attention_head_size': 4,
    'dropout': 0.1,
    'hidden_continuous_size': 16,
    'output_size': 7  # quantiles
}
```

#### Model 7: N-BEATS
```python
# Neural Basis Expansion Analysis for Time Series
class NBeats(nn.Module):
    """
    구조: Stacks of blocks
    - Trend stack
    - Seasonality stack
    - Generic stack
    """
    pass
```

### 3.4 Probabilistic Models

#### Model 8: Quantile Regression (여러 백본으로)
```python
class QuantileRegression:
    """
    백본: LSTM, Transformer, XGBoost
    손실: Quantile Loss for [0.1, 0.25, 0.5, 0.75, 0.9]
    """
    def __init__(self, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9]):
        self.quantiles = quantiles
        self.models = {q: backbone() for q in quantiles}
```

#### Model 9: Bayesian Neural Network (MC Dropout)
```python
class BayesianLSTM(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.lstm = nn.LSTM(...)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(...)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        self.train()  # dropout on!
        predictions = [self.forward(x) for _ in range(n_samples)]
        return predictions
```

#### Model 10: TimeGrad (Diffusion Baseline)
```python
# 전통적 디퓨전 모델 (비교용)
class TimeGrad:
    """
    Diffusion 기반 시계열 예측
    - 100-1000 스텝
    - Backward denoising
    """
    pass
```

### 3.5 제안 모델

#### Model 11: Flow Matching (Main)
```python
class FlowMatchingForecaster:
    """
    당신의 아이디어 구현
    - Forward velocity field 학습
    - ODE 적분으로 예측
    - 1-10 스텝
    """
    pass
```

---

## 📏 4. 평가 지표

### 4.1 점 예측 정확도 (Point Forecast Metrics)

```python
metrics = {
    # 기본 지표
    'MAE': 'Mean Absolute Error',
    'RMSE': 'Root Mean Squared Error',
    'MAPE': 'Mean Absolute Percentage Error',
    
    # 금융 특화 지표
    'Directional_Accuracy': '방향 예측 정확도',
    'Sharpe_Ratio': '예측 기반 전략의 샤프 비율',
    
    # 시간대별
    'MAE_1h': '1시간 후 MAE',
    'MAE_6h': '6시간 후 MAE',
    'MAE_24h': '24시간 후 MAE',
}
```

### 4.2 확률적 예측 품질 (Probabilistic Forecast Metrics)

```python
probabilistic_metrics = {
    # Calibration
    'CRPS': 'Continuous Ranked Probability Score',
    'Pinball_Loss': '각 quantile의 평균 손실',
    'Coverage': '실제값이 예측 구간 내에 있는 비율',
    
    # Sharpness
    'Interval_Width': '예측 구간의 평균 폭',
    'Std_Prediction': '예측 분포의 표준편차',
    
    # 종합
    'Winkler_Score': 'coverage + sharpness 결합',
}
```

### 4.3 계산 효율성

```python
efficiency_metrics = {
    'Training_Time': '학습 시간 (초)',
    'Inference_Time': '1000 샘플 예측 시간 (초)',
    'Memory_Usage': '최대 GPU/CPU 메모리 (GB)',
    'Parameters': '모델 파라미터 수',
    'FLOPs': '연산량',
}
```

### 4.4 Robustness 지표

```python
robustness = {
    # 극단 상황 성능
    'High_Vol_MAE': '변동성 상위 10% 구간 MAE',
    'Crisis_MAE': '위기 구간 (Luna, FTX) MAE',
    
    # 안정성
    'Prediction_Variance': '같은 입력에 대한 예측 분산',
    'Outlier_Sensitivity': '이상치에 대한 민감도',
}
```

---

## ⚙️ 5. 실험 설정

### 5.1 하드웨어

```yaml
environment:
  GPU: NVIDIA RTX 4090 24GB (or A100)
  CPU: AMD Ryzen 9 / Intel i9
  RAM: 64GB
  Storage: 1TB SSD
```

### 5.2 소프트웨어 스택

```python
requirements = {
    'python': '3.10+',
    'pytorch': '2.1.0',
    'numpy': '1.24.0',
    'pandas': '2.0.0',
    'scikit-learn': '1.3.0',
    
    # 데이터
    'yfinance': '0.2.28',
    'ccxt': '4.0.0',  # 암호화폐 거래소 API
    'ta': '0.11.0',   # Technical Analysis
    
    # 시각화
    'matplotlib': '3.7.0',
    'seaborn': '0.12.0',
    'plotly': '5.17.0',
    
    # 실험 관리
    'wandb': '0.15.0',  # 또는 mlflow
    'hydra-core': '1.3.0',  # config 관리
}
```

### 5.3 하이퍼파라미터 탐색

```python
# Optuna 사용
import optuna

search_space = {
    # 공통
    'learning_rate': [1e-5, 1e-2],  # log scale
    'batch_size': [32, 64, 128, 256],
    'hidden_dim': [64, 128, 256, 512],
    
    # Flow Matching 전용
    'num_flow_steps': [1, 5, 10, 20, 50],
    'velocity_net_layers': [2, 3, 4, 5],
    
    # 정규화
    'dropout': [0.1, 0.5],
    'weight_decay': [1e-6, 1e-3],
}

# 각 모델당 100 trials
```

### 5.4 훈련 설정

```python
training_config = {
    'epochs': 100,
    'early_stopping_patience': 15,
    'lr_scheduler': 'CosineAnnealingWarmRestarts',
    'optimizer': 'AdamW',
    'grad_clip': 1.0,
    
    # 배치
    'batch_size': 128,
    'num_workers': 8,
    
    # 검증
    'val_check_interval': 500,  # steps
    'save_top_k': 3,
}
```

---

## 🏗️ 6. 구현 아키텍처

### 6.1 프로젝트 구조

```
bitcoin-forecasting/
├── data/
│   ├── raw/                    # 원본 데이터
│   ├── processed/              # 전처리된 데이터
│   └── features/               # Feature engineering 결과
│
├── src/
│   ├── data/
│   │   ├── data_loader.py      # 데이터 로딩
│   │   ├── preprocessor.py     # 전처리
│   │   └── feature_engineering.py
│   │
│   ├── models/
│   │   ├── baselines/
│   │   │   ├── naive.py
│   │   │   ├── arima.py
│   │   │   └── ml_models.py    # XGBoost, LightGBM
│   │   │
│   │   ├── deep_learning/
│   │   │   ├── lstm.py
│   │   │   ├── transformer.py
│   │   │   ├── nbeats.py
│   │   │   └── quantile_reg.py
│   │   │
│   │   ├── probabilistic/
│   │   │   ├── bayesian_nn.py
│   │   │   └── timegrad.py     # Diffusion baseline
│   │   │
│   │   └── flow_matching/
│   │       ├── flow_net.py     # 제안 모델
│   │       ├── velocity_field.py
│   │       └── ode_solver.py
│   │
│   ├── training/
│   │   ├── trainer.py          # 통합 학습 루프
│   │   ├── losses.py           # 손실 함수들
│   │   └── callbacks.py        # 콜백
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # 평가 지표
│   │   ├── evaluator.py        # 평가 실행
│   │   └── statistical_tests.py # 통계 검정
│   │
│   └── utils/
│       ├── config.py           # 설정 관리
│       ├── logger.py           # 로깅
│       └── visualization.py    # 시각화
│
├── experiments/
│   ├── configs/                # Hydra configs
│   ├── notebooks/              # 분석 노트북
│   └── scripts/                # 실행 스크립트
│
├── tests/                      # 단위 테스트
├── results/                    # 실험 결과
└── docs/                       # 문서
```

### 6.2 핵심 모듈 인터페이스

```python
# models/base.py
class BaseForecaster(ABC):
    @abstractmethod
    def fit(self, train_data, val_data):
        """모델 학습"""
        pass
    
    @abstractmethod
    def predict(self, x, num_samples=1):
        """예측 (확률적 샘플링 지원)"""
        pass
    
    def evaluate(self, test_data, metrics):
        """평가"""
        pass

# models/flow_matching/flow_net.py
class FlowMatchingForecaster(BaseForecaster):
    def __init__(self, config):
        self.model = TimeSeriesFlowNet(config)
        self.forecaster = FlowForecaster(self.model)
    
    def fit(self, train_data, val_data):
        # 학습 로직
        pass
    
    def predict(self, x, num_samples=10, num_steps=10):
        # Flow ODE 적분
        return self.forecaster.predict(x, num_steps, num_samples)
```

---

## 📈 7. 실험 프로토콜

### 7.1 실험 단계

#### Phase 1: 데이터 준비 및 EDA
```python
tasks = [
    '1. 데이터 수집 및 전처리',
    '2. 탐색적 데이터 분석',
    '3. Feature engineering',
    '4. 데이터 분할 및 검증',
]
```

#### Phase 2: Baseline 구축
```python
tasks = [
    '1. Naive 모델 구현 및 평가',
    '2. ARIMA 모델 튜닝',
    '3. ML 모델 (XGBoost, LightGBM) 구현',
    '4. Baseline 성능 확립',
]
```

#### Phase 3: Deep Learning 모델
```python
tasks = [
    '1. LSTM 구현 및 튜닝',
    '2. Transformer 구현',
    '3. N-BEATS 구현',
    '4. 성능 비교',
]
```

#### Phase 4: Probabilistic 모델
```python
tasks = [
    '1. Quantile Regression 구현',
    '2. Bayesian NN 구현',
    '3. TimeGrad (Diffusion baseline) 구현',
    '4. 불확실성 평가',
]
```

#### Phase 5: Flow Matching (Main)
```python
tasks = [
    '1. Flow Matching 구현',
    '2. 하이퍼파라미터 튜닝',
    '3. Ablation study',
    '4. 종합 평가',
]
```

#### Phase 6: 분석 및 문서화
```python
tasks = [
    '1. 통계적 유의성 검정',
    '2. 시각화 및 해석',
    '3. 논문/리포트 작성',
    '4. 코드 정리 및 문서화',
]
```

### 7.2 Cross-Validation 전략

```python
# Walk-Forward Validation
num_folds = 5
test_size = 365 * 24  # 1년 (시간 단위)

for i in range(num_folds):
    train_end = initial_train_end + i * test_size
    test_start = train_end
    test_end = test_start + test_size
    
    # 학습 및 평가
    model.fit(data[:train_end])
    predictions = model.predict(data[test_start:test_end])
    evaluate(predictions)
```

### 7.3 통계적 검정

```python
statistical_tests = {
    # 모델 간 비교
    'Diebold_Mariano': '예측 정확도 차이 검정',
    'Wilcoxon_Signed_Rank': '비모수 비교',
    
    # Calibration
    'Kolmogorov_Smirnov': '분포 적합도 검정',
    'Chi_Square': '범주별 coverage 검정',
}

# 유의수준
alpha = 0.05

# Bonferroni 보정 (다중 비교)
adjusted_alpha = alpha / num_comparisons
```

---

## 📊 8. 예상 결과 및 가설

### 8.1 주요 가설

**H1: 점 예측 정확도**
- Flow Matching이 LSTM, Transformer보다 MAE/RMSE가 낮을 것
- 특히 장기 예측(12-24시간)에서 우위

**H2: 불확실성 정량화**
- Flow Matching이 TimeGrad보다 CRPS가 낮고 calibration이 좋을 것
- Coverage가 명목 수준(예: 90%)에 가까울 것

**H3: 계산 효율성**
- Flow Matching이 TimeGrad보다 10-100배 빠를 것
- 추론 시간이 실시간 트레이딩에 적합할 것 (< 1초)

**H4: Robustness**
- 고변동성 구간에서 Flow Matching이 더 안정적
- 위기 상황에서도 합리적인 예측

### 8.2 예상 결과 표

| 모델 | MAE (24h) | CRPS | 추론 시간 | Coverage (90%) |
|------|-----------|------|-----------|----------------|
| Naive | 2500 | - | 0.001s | - |
| ARIMA | 2200 | - | 0.1s | - |
| XGBoost | 1800 | - | 0.01s | - |
| LSTM | 1500 | 120 | 0.05s | 85% |
| Transformer | 1400 | 110 | 0.1s | 87% |
| Quantile Reg | 1450 | 105 | 0.05s | 89% |
| Bayesian NN | 1480 | 115 | 5s | 88% |
| TimeGrad | **1300** | 95 | 50s | 91% |
| **Flow Matching** | **1280** | **90** | **0.5s** | **90%** |

*주: 가상 수치, 실험 후 업데이트*

---

## 🔬 9. Ablation Study

### 9.1 Flow Matching 변형

```python
ablation_experiments = {
    # 1. ODE 스텝 수
    'flow_steps': [1, 5, 10, 20, 50],
    
    # 2. 조건 인코더
    'condition_encoder': ['LSTM', 'Transformer', 'CNN'],
    
    # 3. Velocity Network 구조
    'velocity_net': ['MLP', 'ResNet', 'Transformer'],
    
    # 4. 초기화 방식
    'initialization': [
        'last_value',      # 마지막 값
        'gaussian',        # 가우시안 노이즈
        'learned'          # 학습 가능한 초기화
    ],
    
    # 5. 시간 임베딩
    'time_embedding': ['sinusoidal', 'learned', 'none'],
}
```

### 9.2 Feature Ablation

```python
feature_groups = {
    'baseline': ['close'],
    '+ technical': ['close', 'rsi', 'macd', 'bb'],
    '+ volume': ['close', 'rsi', 'macd', 'bb', 'volume'],
    '+ external': ['close', ..., 'fear_greed', 'trends'],
}

# 각 조합으로 학습 및 평가
```

---

## 📝 10. 문서화 및 재현성

### 10.1 실험 추적

```python
# WandB 설정
import wandb

wandb.init(
    project="bitcoin-flow-matching",
    config={
        "model": "flow_matching",
        "dataset": "bitcoin_hourly",
        "lookback": 168,
        "horizon": 24,
    }
)

# 메트릭 로깅
wandb.log({
    "train_loss": loss,
    "val_mae": mae,
    "test_crps": crps,
})

# 모델 저장
wandb.save("model.pth")
```

### 10.2 재현성 체크리스트

```python
reproducibility = {
    '✓ Random seed 고정': 'torch.manual_seed(42)',
    '✓ 데이터 버전 관리': 'DVC 또는 Git LFS',
    '✓ 환경 고정': 'requirements.txt, Dockerfile',
    '✓ 코드 버전': 'Git commit hash 기록',
    '✓ 설정 파일 저장': 'Hydra config 자동 저장',
    '✓ 하드웨어 정보': 'GPU 모델, CUDA 버전',
}
```

### 10.3 결과 보고서 구조

```markdown
# 최종 보고서 목차

1. Executive Summary
2. Introduction & Motivation
3. Related Work
4. Methodology
   4.1 Data
   4.2 Models
   4.3 Evaluation
5. Results
   5.1 Main Results
   5.2 Ablation Study
   5.3 Statistical Tests
6. Discussion
   6.1 Interpretation
   6.2 Limitations
   6.3 Future Work
7. Conclusion
8. Appendix
   8.1 Hyperparameters
   8.2 Additional Plots
   8.3 Code Repository
```

---

## 🎯 11. 마일스톤 및 타임라인

### 11.1 예상 일정 (3개월)

```gantt
Week 1-2:   데이터 수집 및 전처리
Week 3-4:   Baseline 모델 구현
Week 5-6:   Deep Learning 모델 구현
Week 7-8:   Probabilistic 모델 구현
Week 9-10:  Flow Matching 구현 및 튜닝
Week 11:    Ablation study 및 분석
Week 12:    논문/보고서 작성
```

### 11.2 체크포인트

```python
checkpoints = {
    'Week 2': '데이터 파이프라인 완성',
    'Week 4': 'Baseline 성능 확립',
    'Week 6': 'DL 모델 구현 완료',
    'Week 8': 'Probabilistic 모델 완료',
    'Week 10': 'Flow Matching 최적화 완료',
    'Week 11': '통계 분석 완료',
    'Week 12': '최종 보고서 제출',
}
```

---

## 📚 12. 참고 문헌

### 12.1 핵심 논문

```bibtex
@article{lipman2022flow,
  title={Flow Matching for Generative Modeling},
  author={Lipman, Yaron and others},
  journal={arXiv preprint arXiv:2210.02747},
  year={2022}
}

@article{rasul2021autoregressive,
  title={Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting},
  author={Rasul, Kashif and others},
  journal={ICML},
  year={2021}
}

@article{liu2023rectified,
  title={Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow},
  author={Liu, Xingchao and others},
  journal={ICLR},
  year={2023}
}
```

### 12.2 관련 자료

- PyTorch Forecasting Documentation
- Temporal Fusion Transformer Paper
- Bitcoin Price Prediction: A Survey
- Financial Time Series Forecasting: Best Practices

---

## 💡 13. 추가 고려사항

### 13.1 윤리 및 책임

```python
ethical_considerations = {
    '투자 권유 아님': '학술 연구 목적임을 명시',
    '리스크 고지': '모델의 한계 명확히 설명',
    '재현성': '코드 및 데이터 공개',
    '투명성': '실패한 실험도 보고',
}
```

### 13.2 확장 가능성

```python
future_extensions = {
    '다중 자산': '비트코인 외 알트코인 추가',
    '멀티모달': '뉴스, 소셜미디어 데이터 통합',
    '강화학습': '트레이딩 에이전트 개발',
    '실시간': '스트리밍 데이터 처리',
}
```

---

## ✅ 14. 체크리스트

### 14.1 구현 전

- [ ] 데이터 수집 API 접근 확인
- [ ] GPU 환경 설정
- [ ] Git repository 생성
- [ ] WandB 프로젝트 생성
- [ ] 문헌 조사 완료

### 14.2 구현 중

- [ ] 데이터 파이프라인 테스트
- [ ] 각 모델 단위 테스트
- [ ] 학습 안정성 확인
- [ ] 중간 결과 저장
- [ ] 코드 리뷰

### 14.3 구현 후

- [ ] 모든 평가 지표 계산
- [ ] 통계 검정 수행
- [ ] 시각화 생성
- [ ] 문서 작성
- [ ] 코드 정리 및 주석
- [ ] README 업데이트
- [ ] 결과 발표 준비

---

## 📧 연락처 및 협업

```
프로젝트 관리자: [이름]
Email: [이메일]
GitHub: [레포지토리 링크]
WandB: [프로젝트 링크]
```

---

**마지막 업데이트**: 2025-10-05
**버전**: 1.0
**상태**: 설계 완료, 구현 대기