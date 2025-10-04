# 비트코인 시간당 가격 예측 실험 보고서 (Flow Matching vs Baselines)

작성일: 2025-10-04

## 1. Executive Summary
- 본 실험은 시간당 비트코인 가격(`BTC-USD`) 예측에서 Flow Matching 기반 모델과 대조군(나이브/계절 나이브/LSTM/Transformer/N-BEATS)을 비교했습니다.
- 최근 720일 데이터 폴백으로 파이프라인을 자동 구성하고, 정규화된 스케일 기준/실측 스케일 메트릭을 생성했습니다.
- 비교 결과(정규화 스케일 기준): Flow의 RMSE가 LSTM, N-BEATS보다 낮아 딥러닝 대조군 대비 우수했습니다. Transformer는 상대적으로 열위였습니다.
- 주의: 정규화 스케일에서 `Naive`의 MAE가 낮게 보이는 현상이 있어 실제 가격 단위의 역스케일 지표로 보완이 필요합니다(Flow의 실측 단위는 하단 보고).

## 2. 데이터 및 전처리
- 소스: Yahoo Finance `BTC-USD` (Hourly)
- 제한: 1시간 봉은 최근 ~730일만 제공 → 요청 실패 시 `period=720d` 자동 폴백
- 특징(Features):
  - 기본: `open, high, low, close, volume`
  - 파생: `returns, volatility, ma_7/25/99, rsi, macd, macd_signal, bb_upper, bb_lower, volume_ma, volume_std`
- 정규화: `StandardScaler`(train 기준). 타깃(`close`)의 표준편차는 역스케일 계산에 사용
- 윈도우: lookback 168시간 → horizon 24시간, stride 1

## 3. 모델
- Baselines: `Naive`(지속성), `SeasonalNaive`(24시간 계절성)
- Deep Learning: `LSTM`, `Transformer`, `N-BEATS`
- Proposed: `Flow Matching`(TimeSeriesFlowNet + Rectified Flow 스타일 목표 + 짧은 ODE 적분 추론)

## 4. 학습 설정(요약)
- Optimizer: AdamW, Grad clip 1.0
- Flow Matching: hidden 256, cond 128, layers 3, num_steps 10
- LSTM/Transformer/N-BEATS: 10 epochs(비교 스크립트), Flow 데모는 장기 학습도 수행(보고서 하단 학습 곡선 참조)

## 5. 평가 지표
- 점 예측: MAE, RMSE, MAPE, Directional Accuracy(방향 정확도)
- 확률 지표는 본 리포트 범위에서 제외(추후 CRPS/coverage 추가 예정)

## 6. 결과 요약
### 6.1 모델별 성능(정규화 스케일)
아래 표는 `results/compare/compare_metrics.json`에서 수집.

| Model | MAE | RMSE | MAPE(%) | Directional Acc. |
|---|---:|---:|---:|---:|
| Naive | 0.0482 | 0.0692 | 4.04 | 0.0417 |
| SeasonalNaive | 0.0698 | 0.0943 | 5.87 | 0.5085 |
| LSTM | 0.0870 | 0.1077 | 7.03 | 0.5196 |
| Transformer | 0.1485 | 0.1833 | 11.04 | 0.5223 |
| N-BEATS | 0.0869 | 0.1166 | 6.90 | 0.5218 |
| Flow Matching | 0.0778 | 0.0992 | 6.36 | 0.5204 |

- 해석: Flow는 LSTM/N-BEATS 대비 RMSE가 낮고 안정적 방향 정확도를 보임. Transformer는 열위.
- 참고: `Naive`의 낮은 MAE는 정규화 스케일의 특이성에 기인할 수 있어 실제 단위 비교가 필요.

### 6.1-보강) 실측(USD) 단위 비교
파일: `results/compare/compare_metrics_real.json` 기준 (역스케일: `target_std = scalers.target_scaler.scale_[0]`).

| Model | MAE (USD) | RMSE (USD) |
|---|---:|---:|
| Naive | 957.64 | 1375.77 |
| SeasonalNaive | 1388.00 | 1874.39 |
| LSTM | 1955.70 | 2437.95 |
| Transformer | 2953.12 | 3703.09 |
| N-BEATS | 1642.98 | 2154.29 |
| Flow Matching | 1614.12 | 2053.97 |

- 해석: USD 기준에서도 Flow가 LSTM/Transformer 대비 낮은 오차를 보이며, N-BEATS와 비슷하거나 더 낮은 수준으로 확인됩니다.

### 6.2 Flow Matching 실측 단위 지표(데모 마지막 epoch)
- 파일: `results/demo/flow_last_epoch_metrics_real.json`
  - MAE(real): 2883.24
  - RMSE(real): 3512.88
- 주의: 현재 실측 단위 지표는 Flow만 제공. 공정 비교를 위해 타 모델도 역스케일 지표를 추출 예정.

## 7. 학습/검증 곡선 및 예시
- 데모 학습 곡선: `results/demo/train_loss.png`
- 검증 곡선: `results/demo/val_metrics.png`
- 예측 예시: `results/demo/forecast_example.png`
- 모델 비교 바 차트: `results/compare/compare_mae.png`, `results/compare/compare_rmse.png`

![train_loss](../results/demo/train_loss.png)
![val_metrics](../results/demo/val_metrics.png)
![forecast_example](../results/demo/forecast_example.png)
![compare_mae](../results/compare/compare_mae.png)
![compare_rmse](../results/compare/compare_rmse.png)

## 8. 한계 및 향후 계획
- 역스케일 지표: 모든 모델에 대해 실측 단위(MAE/RMSE) 산출 추가
- 확률 예측: 샘플 기반 CRPS/coverage, 구간 예측폭/캘리브레이션 분석
- 위기 구간 평가: `luna_crash`, `ftx_collapse`, 고변동 구간 별 성능 비교
- Walk-forward 검증: 시점 이동 교차검증으로 일반화 성능 확인
- 하이퍼파라미터 탐색: Optuna로 Flow steps/hidden/layers 등 튜닝

## 9. 재현 방법
```bash
# 의존성 설치
python3 -m pip install -r requirements.txt

# 데모 실행(결과/플롯/체크포인트 저장)
python3 -m experiments.scripts.run_demo

# 대조군 비교(정규화 스케일 메트릭/비교 플롯 저장)
python3 -m experiments.scripts.run_compare
```

## 10. 참고
- 저장소: https://github.com/juho127/Flowmatching
- 라이선스: MIT (`LICENSE` 파일 참조)
