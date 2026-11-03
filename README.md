# SP-04 Breakout Policy

A research and paper-trading model for breakout policy actions (HOLD, BUY, SELL, EXIT). It uses the existing GRU architecture and was trained on eight Indian stocks with SENSEX as the benchmark.

## Results and limits

The final chronological test set had 33,891 samples. Classification accuracy was 94.76% and macro F1 was 61.61%. Labels are rule-derived proxy labels, not human trader actions. BUY/SELL support is small and precision is low. The historical test backtest returned -17.79% after modeled costs, so the result does not support a profitability claim.

See [TRAINING_REPORT.md](TRAINING_REPORT.md) for dataset details, exact split counts, per-class metrics, backtest results, assumptions, and limitations.

## Data

The original 941 MB CSV is intentionally excluded from this repository. To reproduce training or predict from the consolidated dataset, place a compatible local file at `indian-market-historical-ohlcv.csv`. The inference script expects columns `asset_type,date,open,high,low,close,adj_close,volume,symbol`, with selected stock records marked `stocks` and the index benchmark marked `indices`. Prices are adjusted using `adj_close/close`; volume is kept as supplied.

## Environment

Use Python 3.12 and install the pinned dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements_breakout.txt
```

## Prediction

```powershell
.\.venv\Scripts\python.exe .\predict_breakout.py --data .\indian-market-historical-ohlcv.csv --symbol HDFCBANK --benchmark-symbol SENSEX --artifacts .\sp04_breakout_artifacts_local --device cpu
```

For separate single-symbol price and benchmark CSVs, pass `--benchmark-data <path>` as well. The policy requires at least 240 historical bars and accepts a date column plus OHLCV columns.

## Reproducing training

With the local dataset in place, run `run_local_training.py` or execute `04_breakout_policy_local.ipynb`. Both use the local CSV and keep the original notebook unchanged. The run uses chronological boundaries, train-only preprocessing, validation-based early stopping, and a held-out final test period.

## Files

- `sp04_breakout_artifacts_local/breakout_policy.pt`: inference-ready weights and config
- `sp04_breakout_artifacts_local/best_training_checkpoint.pt`: best validation checkpoint with optimizer state
- `sp04_breakout_artifacts_local/preprocessing.pkl`: train-fitted preprocessing parameters
- `sp04_breakout_artifacts_local/evaluation_metrics.json`: test, validation, and backtest metrics
- `sp04_breakout_artifacts_local/figures/`: learning curves, confusion matrix, backtest, and diagnostics
- `predict_breakout.py`, `sp04_features.py`, `sp04_model.py`: inference pipeline
