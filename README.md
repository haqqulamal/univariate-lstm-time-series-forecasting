# Univariate LSTM — Time-Series Forecasting

This repository provides a clean **univariate LSTM** baseline for time-series forecasting:
- data preparation with **time-aware split** (no shuffle),
- **windowing** from sequence → supervised (look_back → horizon),
- **training** with Keras LSTM,
- **evaluation** using RMSE/MAE/MAPE/R² and diagnostic plots.

Notebook: `notebooks/lstm_univariate.ipynb`.

## Method
1. **Preprocessing:** optional resampling, missing handling, scaling (MinMax/Standard).
2. **Windowing:** convert series into supervised samples (`look_back`, `horizon`, `stride`).
3. **Model:** single/multi-layer **LSTM** with dropout and early stopping (optional).
4. **Training:** monitor validation loss; save learning curves.
5. **Evaluation:** report **RMSE/MAE/MAPE/R²**, plot ŷ vs y and residuals.

## Results (update from your notebook)
- LSTM: `RMSE = ...`, `MAE = ...`, `MAPE = ...`, `R² = ...`
- See `results/metrics.json` and `results/plots/*.png`.
