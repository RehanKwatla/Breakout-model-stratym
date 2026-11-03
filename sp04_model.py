"""SP-04 Breakout Policy - network and inference wrapper.

Contains NO data download and NO training code. Load a saved policy with
``BreakoutPolicy.load(dir_or_pt_file, device)`` and call ``predict(state)``.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sp04_features import (
    ACTION_NAMES, DEFAULT_PROFILE, FEATURE_COLUMNS, N_ACTIONS, N_BASE_FEATURES, N_FEATURES, N_STATIC, BreakoutFeatureAdapter,
    PROFILE_FEATURES, WARMUP_BARS, build_feature_schema, build_features, encode_static,
    scale_features, valid_action_mask,
)

ARTIFACT_NAME = "breakout_policy.pt"
CONFIG_NAME = "breakout_config.json"
MASK_VALUE = -1e9


class BreakoutPolicyNet(nn.Module):
    """GRU/LSTM encoder over market features + static (position, profile) branch + action head."""

    def __init__(self, n_features: int = N_FEATURES, n_static: int = N_STATIC, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.25, static_hidden: int = 16,
                 head_hidden: int = 64, encoder: str = "gru", n_actions: int = N_ACTIONS):
        super().__init__()
        if encoder not in ("gru", "lstm"):
            raise ValueError("encoder must be 'gru' or 'lstm'")
        rnn_cls = nn.GRU if encoder == "gru" else nn.LSTM
        self.rnn = rnn_cls(n_features, hidden_size, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.static = nn.Sequential(nn.Linear(n_static, static_hidden), nn.GELU())
        self.head = nn.Sequential(
            nn.Linear(hidden_size + static_hidden, head_hidden), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(head_hidden, n_actions),
        )

    def forward(self, seq: torch.Tensor, static: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(seq)
        z = torch.cat([out[:, -1, :], self.static(static)], dim=1)
        return self.head(z)


def mask_logits(logits: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~valid, MASK_VALUE)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class BreakoutPolicy:
    """Inference wrapper: scaling + network + action masking + calibrated confidence."""

    def __init__(self, net: BreakoutPolicyNet, config: dict, preprocessing: dict, device="cpu"):
        self.device = torch.device(device)
        self.net = net.to(self.device).eval()
        self.config = _jsonable(config)
        self.pre = {
            "feature_columns": list(preprocessing["feature_columns"]),
            "median": np.asarray(preprocessing["median"], dtype=np.float32),
            "mean": np.asarray(preprocessing["mean"], dtype=np.float32),
            "std": np.asarray(preprocessing["std"], dtype=np.float32),
            "clip": float(preprocessing["clip"]),
        }
        if self.pre["feature_columns"] != list(FEATURE_COLUMNS):
            raise ValueError("feature order in preprocessing does not match sp04_features.FEATURE_COLUMNS")
        self.seq_len = int(self.config["seq_len"])
        self.allow_short = bool(self.config.get("allow_short", True))
        self.temperature = float(self.config.get("temperature", 1.0))

    # ------------------------------------------------------------------ scaling
    def scale(self, seq: np.ndarray) -> np.ndarray:
        """Raw (..., F) -> scaled float32. NaN -> train median, standardise, clip."""
        return scale_features(seq, self.pre["median"], self.pre["mean"], self.pre["std"], self.pre["clip"])

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def predict_scaled(self, seq_scaled: torch.Tensor, static: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Calibrated probabilities (B, 4) from already-scaled tensors on self.device."""
        logits = mask_logits(self.net(seq_scaled, static), valid)
        return torch.softmax(logits / self.temperature, dim=1)

    @torch.no_grad()
    def predict(self, state: dict) -> dict:
        """Predict an action for one state or a batch of states.

        state keys
        ----------
        sequence : array (L, 30) or (B, L, 30) of RAW shared features (FEATURE_COLUMNS[:30]), oldest bar first,
                   plus 'breakout_features' (L, 23) / (B, L, 23) from BreakoutFeatureAdapter; OR a single
                   (L, 53) / (B, L, 53) array already extended with the breakout columns.
                   L >= seq_len; the last seq_len bars are used.
        position : int in {-1, 0, 1} (or length-B array). Default 0 (flat).
        profile  : dict or length-3 / (B, 3) array [risk_tolerance, patience, aggressiveness]
                   in [0, 1]. Default (0.5, 0.5, 0.5).
        unrealized_pnl : (optional) position-signed fractional P&L since entry, scalar or per-state. Default 0.
        holding_bars   : (optional) bars held, entry bar = 1, scalar or per-state. Default 0.
                   Both are ignored (zeroed) when position == 0.

        Returns a dict with action_probs, action, action_name, confidence, valid_actions, entropy.
        Single-state input yields scalars / 1-D arrays; batch input yields arrays.
        """
        seq = np.asarray(state["sequence"], dtype=np.float32)
        single = seq.ndim == 2
        if single:
            seq = seq[None]
        if seq.ndim != 3:
            raise ValueError(f"sequence must have shape (L, F) or (B, L, F); got {seq.shape}")
        if seq.shape[2] == N_BASE_FEATURES:                      # shared 30-column contract + adapter output
            extra = state.get("breakout_features")
            if extra is None:
                raise ValueError(f"sequence has the shared {N_BASE_FEATURES} columns, so 'breakout_features' (L, "
                                 f"{N_FEATURES - N_BASE_FEATURES}) is required: build it with "
                                 "BreakoutFeatureAdapter().adapt_state(state, ohlcv) or policy.state_from_ohlcv(...)")
            extra = np.asarray(extra, dtype=np.float32)
            if extra.ndim == 2:
                extra = extra[None]
            if extra.shape[0] != seq.shape[0] or extra.shape[1] != seq.shape[1] or extra.shape[2] != N_FEATURES - N_BASE_FEATURES:
                raise ValueError(f"breakout_features must have shape (L, {N_FEATURES - N_BASE_FEATURES}) or (B, L, "
                                 f"{N_FEATURES - N_BASE_FEATURES}) aligned with sequence {seq.shape[:2]}; got {extra.shape}")
            seq = np.concatenate([seq, extra], axis=2)
        elif seq.shape[2] != N_FEATURES:
            raise ValueError(f"sequence must have {N_BASE_FEATURES} (shared) or {N_FEATURES} (extended) columns; got {seq.shape[2]}")
        if seq.shape[1] < self.seq_len:
            raise ValueError(f"need at least seq_len={self.seq_len} bars; got {seq.shape[1]}")
        seq = seq[:, -self.seq_len:, :]
        batch = seq.shape[0]

        position = np.asarray(state.get("position", 0)).astype(int).reshape(-1)
        if position.size == 1 and batch > 1:
            position = np.repeat(position, batch)
        if position.size != batch:
            raise ValueError("position must be scalar or have one entry per sequence")
        static = encode_static(position, state.get("profile", DEFAULT_PROFILE),
                               state.get("unrealized_pnl"), state.get("holding_bars"))
        valid = valid_action_mask(position, self.allow_short)

        seq_t = torch.from_numpy(self.scale(seq)).to(self.device)
        static_t = torch.from_numpy(static).to(self.device)
        valid_t = torch.from_numpy(valid).to(self.device)
        probs = self.predict_scaled(seq_t, static_t, valid_t).cpu().numpy()

        action = probs.argmax(axis=1)
        confidence = probs[np.arange(batch), action]
        entropy = -(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(axis=1)
        names = [ACTION_NAMES[a] for a in action]
        if single:
            return {"action_probs": probs[0], "action": int(action[0]), "action_name": names[0],
                    "confidence": float(confidence[0]), "valid_actions": valid[0], "entropy": float(entropy[0])}
        return {"action_probs": probs, "action": action, "action_name": names,
                "confidence": confidence, "valid_actions": valid, "entropy": entropy}

    def state_from_ohlcv(self, ohlcv: pd.DataFrame, benchmark: pd.DataFrame, position: int = 0, profile=None,
                         unrealized_pnl: float = 0.0, holding_bars: int = 0) -> dict:
        """Build a predict() state from OHLCV frames the caller already holds (no downloading)."""
        feats = build_features(ohlcv, benchmark)[FEATURE_COLUMNS]
        if len(feats) < WARMUP_BARS + self.seq_len:
            raise ValueError(f"need at least {WARMUP_BARS + self.seq_len} bars of history; got {len(feats)}")
        window = feats.iloc[-self.seq_len:]
        base = window.to_numpy(dtype=np.float32)[:, :N_BASE_FEATURES]            # shared 30-column sequence
        extra = window.to_numpy(dtype=np.float32)[:, N_BASE_FEATURES:]           # same values the adapter produces
        return {"sequence": base, "breakout_features": extra, "position": int(position),
                "profile": profile if profile is not None else DEFAULT_PROFILE,
                "unrealized_pnl": unrealized_pnl, "holding_bars": holding_bars}

    # ------------------------------------------------------------------ persistence
    def save(self, path) -> Path:
        """Write breakout_policy.pt, breakout_config.json, feature_schema.json, preprocessing.pkl into `path`.

        breakout_policy.pt is self-contained and authoritative; the other three are standard
        side-car copies for tooling and the Meta-RL selector.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        schema = build_feature_schema(self.seq_len)
        pre_plain = {"feature_columns": self.pre["feature_columns"], "median": self.pre["median"].tolist(),
                     "mean": self.pre["mean"].tolist(), "std": self.pre["std"].tolist(), "clip": self.pre["clip"]}
        config = dict(self.config)
        config["temperature"] = self.temperature
        bundle = {
            "format_version": 1,
            "state_dict": {k: v.detach().cpu() for k, v in self.net.state_dict().items()},
            "config": config,
            "preprocessing": pre_plain,
            "feature_schema": schema,
        }
        torch.save(bundle, out / ARTIFACT_NAME)
        (out / CONFIG_NAME).write_text(json.dumps(_jsonable(config), indent=2))
        (out / "feature_schema.json").write_text(json.dumps(_jsonable(schema), indent=2))
        with open(out / "preprocessing.pkl", "wb") as fh:
            pickle.dump({**pre_plain, "median": self.pre["median"], "mean": self.pre["mean"], "std": self.pre["std"]}, fh)
        return out / ARTIFACT_NAME

    @classmethod
    def load(cls, path, device=None) -> "BreakoutPolicy":
        """Load from a directory containing breakout_policy.pt, or from the .pt file itself."""
        p = Path(path)
        pt = p / ARTIFACT_NAME if p.is_dir() else p
        if not pt.exists():
            raise FileNotFoundError(f"policy file not found: {pt}")
        device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle = torch.load(pt, map_location=device, weights_only=True)
        config = bundle["config"]
        net = BreakoutPolicyNet(**config["model"])
        net.load_state_dict(bundle["state_dict"])
        return cls(net, config, bundle["preprocessing"], device)
