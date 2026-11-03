"""SP-04 Breakout Policy - shared constants and causal feature engineering.

Schema family rule: the first 30 sequence features and the 8 static features are identical to the other
policies (shared contract). SP-04 adds 23 breakout-specific sequence features that are produced by
`BreakoutFeatureAdapter` from OHLCV and passed to the policy either appended to `sequence` or as a separate
`breakout_features` array, so the shared 30-column sequence stays untouched.

BARS AVAILABLE AT DECISION TIME t (close of bar t known, order filled at the open of bar t+1):
  * every rolling range ("prior high/low") is computed on bars t-N .. t-1 ONLY (shift(1) before rolling), so
    the bar being judged is never part of its own reference range;
  * volume surge compares bar t with the mean of the PREVIOUS 20/50 bars; ATR-normalised quantities use the
    ATR known at the close of bar t-1;
  * follow-through features look only at breakouts that happened at or before bar t.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Common action space (shared by all five specialised policies)
# ----------------------------------------------------------------------------
HOLD, BUY, SELL, EXIT = 0, 1, 2, 3
ACTION_NAMES = ["HOLD", "BUY", "SELL", "EXIT"]
N_ACTIONS = 4

# Position state: +1 long, 0 flat, -1 short. One-hot order below is part of the schema.
POSITION_ORDER = (0, 1, -1)                      # one-hot index -> position value
POSITION_NAMES = ["pos_flat", "pos_long", "pos_short"]

# Trader-profile features, each in [0, 1].
PROFILE_FEATURES = ["risk_tolerance", "patience", "aggressiveness"]
DEFAULT_PROFILE = (0.5, 0.5, 0.5)

STATIC_FEATURE_NAMES = POSITION_NAMES + PROFILE_FEATURES   # shared prefix: 3 + 3 = 6 dims

# Sequential market features: exact column order of the model input (30 dims).
BASE_FEATURE_COLUMNS = [
    # returns
    "ret_1", "logret_1", "ret_5", "ret_10", "ret_20", "ret_60",
    # momentum oscillators
    "rsi_14", "macd_pct", "macd_signal_pct", "macd_hist_pct",
    # moving-average relationships
    "close_ema20", "close_ema50", "ema20_ema50", "close_sma200", "sma50_sma200",
    # trend strength
    "adx_14", "di_diff",
    # volatility
    "atr_pct", "rv_20", "rv_60",
    # volume
    "vol_chg_1", "vol_ratio_20", "vol_z_60",
    # candle shape
    "hl_range", "gap_open", "close_loc",
    # relative strength vs benchmark and market context
    "rel_strength_20", "rel_strength_60", "bench_ret_20", "bench_rv_20",
]
# SP-04 breakout additions (appended after the 30 shared features; see BreakoutFeatureAdapter)
EXTRA_FEATURE_COLUMNS = [
    "brk_up_flag_20", "brk_down_flag_20", "brk_up_flag_55", "brk_down_flag_55",
    "brk_dist_high_20", "brk_dist_low_20", "brk_dist_high_55", "brk_dist_low_55",
    "brk_range_pos_20", "brk_range_width_20", "brk_range_width_55", "brk_compression_10_50",
    "brk_atr_ratio_5_20", "brk_bar_range_atr", "brk_gap_atr", "brk_vol_surge_20", "brk_vol_surge_50",
    "brk_bars_since_up", "brk_bars_since_down", "brk_follow_up", "brk_hold_level_up",
    "brk_follow_down", "brk_hold_level_down",
]
FEATURE_COLUMNS = BASE_FEATURE_COLUMNS + EXTRA_FEATURE_COLUMNS
N_BASE_FEATURES = len(BASE_FEATURE_COLUMNS)
N_FEATURES = len(FEATURE_COLUMNS)
N_BASE_STATIC = len(STATIC_FEATURE_NAMES)
EXTRA_STATIC_NAMES = ["unrealized_pnl_scaled", "holding_bars_scaled"]
STATIC_FEATURE_NAMES = STATIC_FEATURE_NAMES + EXTRA_STATIC_NAMES
N_STATIC = len(STATIC_FEATURE_NAMES)
PNL_SCALE, PNL_CLIP = 0.10, 3.0       # unrealized_pnl_scaled = clip(pnl / 0.10, -3, 3)
HOLD_SCALE, HOLD_CAP = 20.0, 60       # holding_bars_scaled   = min(bars, 60) / 20
WARMUP_BARS = 210          # sma_200 is the longest look-back (breakout features need 61 bars)
BREAKOUT_MIN_HISTORY = 70
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


# ----------------------------------------------------------------------------
# Action validity and position transitions
# ----------------------------------------------------------------------------
def valid_action_mask(position, allow_short: bool = True) -> np.ndarray:
    """Boolean (B, 4) mask of valid actions given the current position.

    flat  -> HOLD, BUY (open long), SELL (open short, only if allow_short)
    long  -> HOLD, EXIT
    short -> HOLD, EXIT
    """
    pos = np.asarray(position).astype(int).reshape(-1)
    mask = np.zeros((pos.size, N_ACTIONS), dtype=bool)
    mask[:, HOLD] = True
    flat = pos == 0
    mask[flat, BUY] = True
    mask[flat, SELL] = bool(allow_short)
    mask[~flat, EXIT] = True
    return mask


def apply_action(position, action) -> np.ndarray:
    """Return the new position after executing (already validated) actions."""
    pos = np.asarray(position).astype(int).reshape(-1)
    act = np.asarray(action).astype(int).reshape(-1)
    new = pos.copy()
    new[(act == BUY) & (pos == 0)] = 1
    new[(act == SELL) & (pos == 0)] = -1
    new[(act == EXIT) & (pos != 0)] = 0
    return new


def profile_array(profile, batch_size: int) -> np.ndarray:
    """Normalise a profile (dict, sequence, (B,3) array or None) to float32 (B, 3)."""
    if profile is None:
        profile = DEFAULT_PROFILE
    if isinstance(profile, dict):
        profile = [profile.get(k, d) for k, d in zip(PROFILE_FEATURES, DEFAULT_PROFILE)]
    arr = np.asarray(profile, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.tile(arr[None, :], (batch_size, 1))
    if arr.shape != (batch_size, len(PROFILE_FEATURES)):
        raise ValueError(f"profile must have shape (3,) or ({batch_size}, 3); got {arr.shape}")
    if np.any(arr < 0) or np.any(arr > 1) or not np.all(np.isfinite(arr)):
        raise ValueError("profile values must be finite and within [0, 1]")
    return arr


def encode_static(position, profile, unrealized_pnl=None, holding_bars=None) -> np.ndarray:
    """Static input (B, 8) = [flat, long, short] + [risk, patience, aggressiveness] + [pnl_scaled, hold_scaled].

    unrealized_pnl : position-signed fractional P&L since entry (e.g. -0.02 = losing 2%). Ignored when flat.
    holding_bars   : bars the position has been held (entry bar = 1). Ignored when flat.
    """
    pos = np.asarray(position).astype(int).reshape(-1)
    if not np.all(np.isin(pos, POSITION_ORDER)):
        raise ValueError("position must be in {-1, 0, +1}")
    n = pos.size
    onehot = np.stack([(pos == p) for p in POSITION_ORDER], axis=1).astype(np.float32)

    def _vec(v, name):
        if v is None:
            return np.zeros(n, dtype=np.float32)
        a = np.asarray(v, dtype=np.float32).reshape(-1)
        if a.size == 1 and n > 1:
            a = np.repeat(a, n)
        if a.size != n or not np.all(np.isfinite(a)):
            raise ValueError(f"{name} must be a finite scalar or have one entry per state")
        return a

    pnl = np.clip(_vec(unrealized_pnl, "unrealized_pnl") / PNL_SCALE, -PNL_CLIP, PNL_CLIP)
    hold = np.minimum(np.maximum(_vec(holding_bars, "holding_bars"), 0.0), HOLD_CAP) / HOLD_SCALE
    flat = pos == 0
    pnl = np.where(flat, 0.0, pnl); hold = np.where(flat, 0.0, hold)
    return np.concatenate([onehot, profile_array(profile, n), pnl[:, None], hold[:, None]], axis=1).astype(np.float32)


# ----------------------------------------------------------------------------
# Indicators (manual implementations, causal)
# ----------------------------------------------------------------------------
def _pct(s: pd.Series, n: int) -> pd.Series:
    return s / s.shift(n) - 1.0


def _wilder(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    rs = _wilder(gain, n) / _wilder(loss, n)
    return 100.0 - 100.0 / (1.0 + rs)


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    return pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)


def adx_di(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14):
    """Wilder ADX with +DI / -DI."""
    up = high.diff()
    down = -low.diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    atr = _wilder(true_range(high, low, close), n)
    plus_di = 100.0 * _wilder(plus_dm, n) / atr
    minus_di = 100.0 * _wilder(minus_dm, n) / atr
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    return _wilder(dx, n), plus_di, minus_di


# ----------------------------------------------------------------------------
# Breakout features (depend on OHLCV only) and the documented adapter
# ----------------------------------------------------------------------------
def _bars_since(flag: pd.Series) -> pd.Series:
    """Bars since the last True value (0 on the bar itself); NaN before the first True."""
    n = len(flag)
    last = pd.Series(np.where(flag.to_numpy(), np.arange(n, dtype=float), np.nan), index=flag.index).ffill()
    return pd.Series(np.arange(n, dtype=float), index=flag.index) - last


def breakout_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """The 23 SP-04 breakout features (EXTRA_FEATURE_COLUMNS) from an OHLCV frame. Causal.

    Prior range boundaries use bars t-N..t-1 only. Sign conventions:
      brk_dist_high_N  = close / prior_high_N - 1      (> 0: closed above the prior N-bar high)
      brk_dist_low_N   = close / prior_low_N  - 1      (< 0: closed below the prior N-bar low)
      brk_follow_up    = close / close_at_last_up_break - 1   (0 if no up-break in the last 10 bars)
      brk_hold_level_up= close / broken_level - 1             (< 0: price fell back INSIDE the old range = failing breakout)
      brk_follow_down / brk_hold_level_down are the mirror images (hold_level_down > 0 = failing breakdown).
    """
    df = ohlcv[OHLCV_COLUMNS].astype(float)
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    vz = v.where(v > 0)
    tr = true_range(h, l, c)
    atr14_prev = _wilder(tr, 14).shift(1)
    out = pd.DataFrame(index=df.index)

    ph20, pl20 = h.shift(1).rolling(20).max(), l.shift(1).rolling(20).min()      # prior ranges EXCLUDE bar t
    ph55, pl55 = h.shift(1).rolling(55).max(), l.shift(1).rolling(55).min()
    ok20, ok55 = ph20.notna() & pl20.notna(), ph55.notna() & pl55.notna()
    up20, dn20 = (c > ph20) & ok20, (c < pl20) & ok20
    out["brk_up_flag_20"] = up20.astype(float).where(ok20)
    out["brk_down_flag_20"] = dn20.astype(float).where(ok20)
    out["brk_up_flag_55"] = ((c > ph55) & ok55).astype(float).where(ok55)
    out["brk_down_flag_55"] = ((c < pl55) & ok55).astype(float).where(ok55)
    out["brk_dist_high_20"] = c / ph20 - 1.0
    out["brk_dist_low_20"] = c / pl20 - 1.0
    out["brk_dist_high_55"] = c / ph55 - 1.0
    out["brk_dist_low_55"] = c / pl55 - 1.0
    out["brk_range_pos_20"] = (c - pl20) / (ph20 - pl20).replace(0.0, np.nan)
    out["brk_range_width_20"] = (ph20 - pl20) / ((ph20 + pl20) / 2.0)
    out["brk_range_width_55"] = (ph55 - pl55) / ((ph55 + pl55) / 2.0)
    r10 = h.shift(1).rolling(10).max() - l.shift(1).rolling(10).min()
    r50 = h.shift(1).rolling(50).max() - l.shift(1).rolling(50).min()
    out["brk_compression_10_50"] = r10 / r50.replace(0.0, np.nan)                  # < 1: recent range is compressed
    out["brk_atr_ratio_5_20"] = _wilder(tr, 5) / _wilder(tr, 20)                   # > 1: volatility expansion
    out["brk_bar_range_atr"] = (h - l) / atr14_prev
    out["brk_gap_atr"] = (o - c.shift(1)) / atr14_prev
    out["brk_vol_surge_20"] = vz / vz.shift(1).rolling(20).mean()
    out["brk_vol_surge_50"] = vz / vz.shift(1).rolling(50).mean()

    since_up, since_dn = _bars_since(up20), _bars_since(dn20)
    recent_up, recent_dn = since_up <= 10, since_dn <= 10                           # NaN (no break yet) -> False
    level_up, c_up = ph20.where(up20).ffill(), c.where(up20).ffill()
    level_dn, c_dn = pl20.where(dn20).ffill(), c.where(dn20).ffill()
    out["brk_bars_since_up"] = since_up.clip(upper=20).fillna(20.0)
    out["brk_bars_since_down"] = since_dn.clip(upper=20).fillna(20.0)
    out["brk_follow_up"] = (c / c_up - 1.0).where(recent_up, 0.0)
    out["brk_hold_level_up"] = (c / level_up - 1.0).where(recent_up, 0.0)
    out["brk_follow_down"] = (c / c_dn - 1.0).where(recent_dn, 0.0)
    out["brk_hold_level_down"] = (c / level_dn - 1.0).where(recent_dn, 0.0)
    out = out.where(ok20 & ok55 & atr14_prev.notna())                              # undefined during warm-up
    return out[EXTRA_FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)


class BreakoutFeatureAdapter:
    """Documented adapter: turns OHLCV history into the SP-04 breakout features and plugs them into a shared state.

    The shared state dictionary (sequence (L, 30), position, profile, unrealized_pnl, holding_bars) is unchanged;
    the adapter only adds the key ``breakout_features`` (L, 23). It never downloads anything: the caller supplies
    OHLCV history of at least ``min_history`` bars ending at the decision bar.
    """
    min_history = BREAKOUT_MIN_HISTORY
    columns = list(EXTRA_FEATURE_COLUMNS)

    def transform(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return breakout_features(ohlcv)

    def window(self, ohlcv: pd.DataFrame, seq_len: int) -> np.ndarray:
        """Last `seq_len` rows of the breakout features, shape (seq_len, 23), aligned with the last seq_len bars of ohlcv."""
        if len(ohlcv) < self.min_history + seq_len:
            raise ValueError(f"need at least {self.min_history + seq_len} bars of OHLCV history; got {len(ohlcv)}")
        return self.transform(ohlcv).iloc[-seq_len:].to_numpy(dtype=np.float32)

    def adapt_state(self, state: dict, ohlcv: pd.DataFrame) -> dict:
        """Return a copy of a shared-contract state with ``breakout_features`` added."""
        seq_len = np.asarray(state["sequence"]).shape[-2]
        return {**state, "breakout_features": self.window(ohlcv, seq_len)}


# ----------------------------------------------------------------------------
# Feature builder
# ----------------------------------------------------------------------------
def build_features(ohlcv: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    """Compute causal features for one symbol.

    Parameters
    ----------
    ohlcv : DataFrame indexed by date, columns Open/High/Low/Close/Volume (adjusted prices).
    benchmark : DataFrame indexed by date with at least a Close column.

    Returns a DataFrame with the same index: the OHLCV columns followed by FEATURE_COLUMNS.
    Row t only depends on rows <= t (verified by a perturbation test in the notebook).
    """
    if benchmark is None or "Close" not in benchmark:
        raise ValueError("a benchmark DataFrame with a Close column is required")
    missing = [c for c in OHLCV_COLUMNS if c not in ohlcv.columns]
    if missing:
        raise ValueError(f"ohlcv is missing columns: {missing}")

    df = ohlcv[OHLCV_COLUMNS].astype(float).copy()
    o, h, l, c, v = df["Open"], df["High"], df["Low"], df["Close"], df["Volume"]
    f = pd.DataFrame(index=df.index)

    # returns
    f["ret_1"] = _pct(c, 1)
    f["logret_1"] = np.log(c).diff()
    for n in (5, 10, 20, 60):
        f[f"ret_{n}"] = _pct(c, n)

    # momentum oscillators
    f["rsi_14"] = rsi(c, 14)
    ema12 = c.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = c.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    f["macd_pct"] = macd / c
    f["macd_signal_pct"] = macd_signal / c
    f["macd_hist_pct"] = (macd - macd_signal) / c

    # moving-average relationships
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = c.ewm(span=50, adjust=False, min_periods=50).mean()
    sma50 = c.rolling(50).mean()
    sma200 = c.rolling(200).mean()
    f["close_ema20"] = c / ema20 - 1.0
    f["close_ema50"] = c / ema50 - 1.0
    f["ema20_ema50"] = ema20 / ema50 - 1.0
    f["close_sma200"] = c / sma200 - 1.0
    f["sma50_sma200"] = sma50 / sma200 - 1.0

    # trend strength
    adx, plus_di, minus_di = adx_di(h, l, c, 14)
    f["adx_14"] = adx
    f["di_diff"] = plus_di - minus_di

    # volatility
    f["atr_pct"] = _wilder(true_range(h, l, c), 14) / c
    f["rv_20"] = f["logret_1"].rolling(20).std() * np.sqrt(252.0)
    f["rv_60"] = f["logret_1"].rolling(60).std() * np.sqrt(252.0)

    # volume (zero volume treated as missing for ratios; never fabricated)
    vz = v.where(v > 0)
    f["vol_chg_1"] = np.log(vz / vz.shift(1)).clip(-3.0, 3.0)
    f["vol_ratio_20"] = vz / vz.rolling(20).mean()
    f["vol_z_60"] = (vz - vz.rolling(60).mean()) / vz.rolling(60).std().replace(0.0, np.nan)

    # candle shape
    f["hl_range"] = (h - l) / c
    f["gap_open"] = o / c.shift(1) - 1.0
    f["close_loc"] = (c - l) / (h - l).replace(0.0, np.nan)

    # relative strength vs benchmark (past-only forward fill, at most 3 bars)
    bc = benchmark["Close"].astype(float).reindex(df.index).ffill(limit=3)
    b_ret20 = _pct(bc, 20)
    b_ret60 = _pct(bc, 60)
    f["rel_strength_20"] = f["ret_20"] - b_ret20
    f["rel_strength_60"] = f["ret_60"] - b_ret60
    f["bench_ret_20"] = b_ret20
    f["bench_rv_20"] = np.log(bc).diff().rolling(20).std() * np.sqrt(252.0)

    # ---- SP-04 breakout extras (computed by the same function the adapter exposes) -----------
    ex = breakout_features(df)
    for col in EXTRA_FEATURE_COLUMNS:
        f[col] = ex[col]

    f = f[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return pd.concat([df, f], axis=1)


def scale_features(x, median, mean, std, clip) -> np.ndarray:
    """Single implementation of the preprocessing used in training AND inference.

    Non-finite values -> training-set median, then z-score, then clip to +/- clip.
    """
    x = np.asarray(x, dtype=np.float32)
    x = np.where(np.isfinite(x), x, np.asarray(median, dtype=np.float32))
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.clip(x, -float(clip), float(clip)).astype(np.float32)


def build_feature_schema(seq_len: int) -> dict:
    """Machine-readable description of the exact model input layout."""
    return {
        "schema_version": "1.0.0",
        "policy_id": "SP-04",
        "sequence_input": {
            "name": "sequence",
            "shape": ["batch", seq_len, N_FEATURES],
            "dtype": "float32",
            "order": "oldest bar first, newest bar last (index -1 is the decision bar)",
            "features": [{"index": i, "name": n} for i, n in enumerate(FEATURE_COLUMNS)],
            "scaling": "NaN -> train median, then (x - mean) / std, then clip to +/- clip (see preprocessing.pkl)",
            "note": "Callers pass RAW (unscaled) features; the policy scales internally.",
        },
        "static_input": {
            "name": "static",
            "shape": ["batch", N_STATIC],
            "extras_note": "identical 8-dim static input as SP-02/SP-03; indices 6-7 (unrealized P&L, holding duration) default to 0 when absent",
            "dtype": "float32",
            "features": [{"index": i, "name": n} for i, n in enumerate(STATIC_FEATURE_NAMES)],
            "position_encoding": {"one_hot_order": list(POSITION_NAMES), "position_values": list(POSITION_ORDER)},
            "profile_ranges": {k: [0.0, 1.0] for k in PROFILE_FEATURES},
            "default_profile": dict(zip(PROFILE_FEATURES, DEFAULT_PROFILE)),
        },
        "output": {
            "action_probs": ["batch", N_ACTIONS],
            "action_space": {str(i): n for i, n in enumerate(ACTION_NAMES)},
            "confidence": "temperature-calibrated probability of the chosen action after masking invalid actions",
        },
        "valid_actions": {
            "flat": ["HOLD", "BUY", "SELL (only if allow_short)"],
            "long": ["HOLD", "EXIT"],
            "short": ["HOLD", "EXIT"],
        },
        "execution_convention": "decision at close of bar t; fill at open of bar t+1",
        "compatibility": {
            "family_rule": "every SP-0x policy consumes the shared base features first, then its own extras; extras DIFFER per policy, so select columns by name",
            "shared_sequence_features": list(BASE_FEATURE_COLUMNS),
            "sp04_extra_sequence_features": list(EXTRA_FEATURE_COLUMNS),
            "shared_static_features": list(STATIC_FEATURE_NAMES[:N_BASE_STATIC]),
            "sp04_extra_static_features": list(EXTRA_STATIC_NAMES),
            "state_dict_keys": ["sequence", "position", "profile", "unrealized_pnl (optional)", "holding_bars (optional)", "breakout_features (SP-04 only; from BreakoutFeatureAdapter)"],
            "adapter": {"class": "BreakoutFeatureAdapter", "input": "OHLCV history ending at the decision bar (>= seq_len + 70 bars)", "output": "breakout_features (L, 23)", "alternative": "pass sequence already extended to 53 columns"},
        },
    }
