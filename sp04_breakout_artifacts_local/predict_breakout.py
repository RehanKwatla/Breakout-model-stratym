"""Run the trained SP-04 policy on recent OHLCV history; no training or downloads."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "sp04_breakout_artifacts_local"
sys.path.insert(0, str(ARTIFACTS))
from sp04_model import BreakoutPolicy


def _adjusted_frame(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    cols = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    date_key = next((cols[k] for k in ("date", "datetime", "timestamp") if k in cols), None)
    if date_key is None:
        raise ValueError("CSV needs a date/Date column")
    required = ("open", "high", "low", "close", "volume")
    missing = [k for k in required if k not in cols]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    close = pd.to_numeric(df[cols["close"]], errors="coerce").to_numpy()
    if "adj_close" in cols:
        adjusted = pd.to_numeric(df[cols["adj_close"]], errors="coerce").to_numpy()
        factor = np.divide(adjusted, close, out=np.ones_like(adjusted), where=np.isfinite(close) & (close != 0))
        close_out = adjusted
    else:
        factor = np.ones(len(df), dtype=float)
        close_out = close
    out = pd.DataFrame({
        "Open": pd.to_numeric(df[cols["open"]], errors="coerce").to_numpy() * factor,
        "High": pd.to_numeric(df[cols["high"]], errors="coerce").to_numpy() * factor,
        "Low": pd.to_numeric(df[cols["low"]], errors="coerce").to_numpy() * factor,
        "Close": close_out,
        "Volume": pd.to_numeric(df[cols["volume"]], errors="coerce").to_numpy(),
    }, index=pd.to_datetime(df[date_key], errors="coerce"))
    out.index.name = "Date"
    out = out.loc[~out.index.isna()]
    out = out[~out.index.duplicated(keep="last")].sort_index()
    valid = out[["Open", "High", "Low", "Close"]].notna().all(axis=1)
    valid &= (out[["Open", "High", "Low", "Close"]] > 0).all(axis=1)
    valid &= out["High"] >= out["Low"]
    return out.loc[valid]


def _load_master(path: Path, symbol: str, benchmark_symbol: str):
    wanted = {symbol.upper(), benchmark_symbol.upper()}
    parts = {s: [] for s in wanted}
    usecols = ["asset_type", "date", "open", "high", "low", "close", "adj_close", "volume", "symbol"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=250_000,
                             dtype={"asset_type": "string", "symbol": "string"}, low_memory=False):
        chunk["symbol"] = chunk["symbol"].str.upper()
        chunk = chunk[chunk["symbol"].isin(wanted)]
        for sym, group in chunk.groupby("symbol", sort=False):
            expected_type = "indices" if sym == benchmark_symbol.upper() else "stocks"
            group = group[group["asset_type"].eq(expected_type)]
            if group.empty:
                continue
            parts[sym].append(_adjusted_frame(group))
    frames = {}
    for sym in wanted:
        if not parts[sym]:
            raise ValueError(f"{sym} ({'index' if sym == benchmark_symbol.upper() else 'stock'}) was not found in {path}")
        frames[sym] = pd.concat(parts[sym]).sort_index()
        frames[sym] = frames[sym][~frames[sym].index.duplicated(keep="last")]
    return frames[symbol.upper()], frames[benchmark_symbol.upper()]


def _load_single(path: Path) -> pd.DataFrame:
    return _adjusted_frame(pd.read_csv(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Consolidated dataset or one-symbol OHLCV CSV")
    parser.add_argument("--symbol", required=True, help="Stock symbol to score")
    parser.add_argument("--benchmark-symbol", default="SENSEX", help="Index symbol when --data is a consolidated dataset")
    parser.add_argument("--benchmark-data", help="Separate OHLCV CSV when --data is a one-symbol CSV")
    parser.add_argument("--artifacts", default=str(ARTIFACTS), help="Directory containing breakout_policy.pt")
    parser.add_argument("--position", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--unrealized-pnl", type=float, default=0.0, help="Fractional unrealized return, e.g. -0.012 for -1.2%%")
    parser.add_argument("--holding-bars", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    data_path = Path(args.data)
    columns = {str(c).strip().lower() for c in pd.read_csv(data_path, nrows=0).columns}
    if {"symbol", "asset_type", "adj_close"} <= columns:
        prices, benchmark = _load_master(data_path, args.symbol, args.benchmark_symbol)
    else:
        if not args.benchmark_data:
            parser.error("--benchmark-data is required when --data is a single-symbol OHLCV CSV")
        prices = _load_single(data_path)
        benchmark = _load_single(Path(args.benchmark_data))
    policy = BreakoutPolicy.load(args.artifacts, args.device)
    state = policy.state_from_ohlcv(prices, benchmark, position=args.position,
                                    unrealized_pnl=args.unrealized_pnl, holding_bars=args.holding_bars)
    result = policy.predict(state)
    print(json.dumps({
        "symbol": args.symbol.upper(), "as_of": str(prices.index[-1].date()),
        "action": result["action_name"], "confidence": float(result["confidence"]),
        "action_probabilities": {k: float(v) for k, v in zip(("HOLD", "BUY", "SELL", "EXIT"), result["action_probs"])},
        "valid_actions": {k: bool(v) for k, v in zip(("HOLD", "BUY", "SELL", "EXIT"), result["valid_actions"])},
        "device": args.device,
    }, indent=2))


if __name__ == "__main__":
    main()
