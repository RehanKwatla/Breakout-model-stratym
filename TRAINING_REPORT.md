# SP-04 Breakout Policy — Training and Evaluation Report

## Result

The existing GRU breakout policy was trained, saved, and loaded successfully. Training used CPU because CUDA was unavailable. The best validation checkpoint is epoch 8; early stopping ended training after epoch 15. Total notebook run time was about 8 minutes 22 seconds. A post-hoc check of that saved checkpoint gave 95.55% accuracy on train and 94.91% on validation proxy labels.

The final test classification accuracy was **94.76%**, but test performance was weak on the rare BUY and SELL classes, and the historical test backtest lost **17.79%** after modeled costs. This result does not support a profitability claim.

## Dataset and preparation

- Source: `indian-market-historical-ohlcv.csv`, a local consolidated Indian-market file. No external price data was downloaded.
- Full file: 7,967,459 rows, 2,704 symbols, five asset types (`stocks`, `indices`, `etfs`, `commodities`, `forex`), dates 1991-01-02 through 2026-10-25.
- Full-file scan found 39,790 rows with invalid OHLC values and no missing volumes. In the selected training universe, six invalid OHLC rows were dropped; no duplicate dates or missing volumes were found.
- Stocks: HDFCBANK, INFY, WIPRO, HINDUNILVR, RELIANCE, SBIN, TATASTEEL, SUNPHARMA. Benchmark: SENSEX.
- Cleaned selected stock rows: 61,892; cleaned SENSEX rows: 7,219. The feature panel contains 58,062 rows after the model's 210-bar warm-up.
- Prices use the file's `adj_close/close` factor to adjust open, high, low, and close. Volume remains raw. The source file was preserved.

## Model and training

The existing architecture was retained: a two-layer GRU over 53 sequential features, an 8-value static branch, and a four-action HOLD/BUY/SELL/EXIT head (53,396 trainable parameters). Preprocessing used median imputation and scaling fit on training-period rows only.

Training configuration: seed 42, batch size 512, learning rate 0.001, weight decay 0.0001, maximum 40 epochs, patience 7, gradient clipping 1.0. Early stopping ended at epoch 15; the best validation-loss checkpoint was epoch 8 (validation loss 0.1574). The best model and optimizer state were saved after each improving epoch.

Labels are **strategy-derived proxy labels**, not human trader demonstrations. `outcome_confirmed` entry labels use the next 10 bars for follow-through confirmation; those future values are excluded from inputs and purged at split boundaries. Metrics therefore measure agreement with the notebook's rule-derived labels, not human behavior.

## Chronological split

Date boundaries were calculated from post-warm-up feature rows, with strict sequence-window isolation and outcome purging. Final usable sample counts were:

| Split | Samples | Share |
|---|---:|---:|
| Train | 124,431 | 72.64% |
| Validation | 12,984 | 7.58% |
| Final test | 33,891 | 19.78% |
| **Total** | **171,306** | **100%** |

Train plus validation is 80.22% of usable samples and test is 19.78%, very close to the intended 80/20. The small difference comes from date-level boundaries, sequence windows, and purged label horizons; all splits remain chronological and non-overlapping. Configured boundaries are training through 2018-08-01, validation through 2020-12-14, and test thereafter. The resulting decision-sample date ranges are train 1997-11-03–2018-07-17, validation 2018-09-17–2020-11-26, and test 2021-01-27–2026-10-07.

The test split was excluded from fitting, early stopping, and temperature calibration. The notebook prints split label counts before fitting; those counts were not used to change the model or its settings. Test classification and backtest results were generated after training.

## Final test classification

Temperature calibration was fit on validation only. The test majority-class baseline is 51.99%.

| Metric | Test result |
|---|---:|
| Accuracy | 94.76% |
| Balanced accuracy | 91.73% |
| Macro F1 | 61.61% |
| Weighted F1 | 95.25% |
| Negative log likelihood | 0.1178 |
| Calibration ECE | 0.0050 |

| Action | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| HOLD | 95.77% | 93.12% | 94.43% | 16,158 |
| BUY | 17.48% | 100.00% | 29.75% | 72 |
| SELL | 15.50% | 77.50% | 25.83% | 40 |
| EXIT | 96.57% | 96.28% | 96.42% | 17,621 |

Confusion matrix (true rows; predicted columns in HOLD, BUY, SELL, EXIT order):

```text
             HOLD   BUY  SELL  EXIT
HOLD        15047   340   169   602
BUY             0    72     0     0
SELL            9     0    31     0
EXIT          656     0     0 16965
```

BUY/SELL support is small and precision is low, so the high overall accuracy should not be read as evidence of reliable entries.

## Historical test backtest

The balanced long/short policy's test result after the notebook's fees, slippage, and borrow assumptions was total return **-17.79%**, CAGR **-3.43%**, Sharpe **-0.94**, and maximum drawdown **-21.97%**. For context, SENSEX buy-and-hold returned **+54.24%** (Sharpe 0.61; max drawdown -16.85%), and the equal-weight selected-stock baseline returned **+80.12%**. This historical path underperformed both reference series. It is not a forecast.

## Inference and artifacts

The notebook's disk reload check passed. The standalone prediction script also loaded the saved model and ran against the local CSV: HDFCBANK as of 2026-10-23 returned HOLD with 99.995% confidence. This is an inference smoke check, not a new evaluation of strategy performance.

Run a prediction from the project directory with:

```powershell
.\.venv\Scripts\python.exe .\predict_breakout.py --data .\indian-market-historical-ohlcv.csv --symbol HDFCBANK --benchmark-symbol SENSEX --artifacts .\sp04_breakout_artifacts_local --device cpu
```

For a separate single-symbol OHLCV CSV, pass `--benchmark-data <path>` as well. Input files need a date column and Open/High/Low/Close/Volume columns; `adj_close` is used when present.

Key files:

- Model weights: `sp04_breakout_artifacts_local/breakout_policy.pt`
- Best training checkpoint: `sp04_breakout_artifacts_local/best_training_checkpoint.pt`
- Prediction script: `predict_breakout.py`
- Preprocessing: `sp04_breakout_artifacts_local/preprocessing.pkl`
- Metrics and assumptions: `sp04_breakout_artifacts_local/evaluation_metrics.json`
- Curves and diagnostic plots: `sp04_breakout_artifacts_local/figures/`
- Executed notebook: `04_breakout_policy_local_executed.ipynb`
- Full execution log: `training_execution.log`
- Debugging record: `ERROR_LOG.md`

`learning_curves.png` contains train/validation loss and validation macro-F1. The original training loop did not record accuracy each epoch, so no epoch-by-epoch accuracy curve is available. `train_val_accuracy_final.png` is a genuine post-hoc accuracy comparison of the saved best checkpoint on train and validation only; it is not presented as an epoch curve.

## Errors and corrections

See `ERROR_LOG.md`. The local CSV frame alignment issue and IPython namespace issue were corrected before successful training. Split boundaries were moved to post-warm-up feature rows to meet the intended 72/8/20 split more closely, and best-validation checkpoint saving was added to the local notebook copy. The original notebook and dataset remain unchanged.
