from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_DIM = 15
WINDOW_SIZE = 30


@dataclass
class WindowBundle:
    windows: np.ndarray
    labels: np.ndarray | None = None


def _parse_hex_payload(value: object) -> list[int]:
    text = str(value or "").strip().replace(" ", "")
    if text.lower() in {"", "nan", "none"}:
        return [0] * 8
    if len(text) % 2:
        text = "0" + text
    out: list[int] = []
    for idx in range(0, min(len(text), 16), 2):
        try:
            out.append(int(text[idx : idx + 2], 16))
        except ValueError:
            out.append(0)
    return (out + [0] * 8)[:8]


def read_can_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    if "arbitration_id" in df.columns and "can_id" not in df.columns:
        df["can_id"] = df["arbitration_id"]
    if "id" in df.columns and "can_id" not in df.columns:
        df["can_id"] = df["id"]
    if "data_field" in df.columns and "data_hex" not in df.columns:
        df["data_hex"] = df["data_field"]

    required = {"timestamp", "can_id", "data_hex"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required CAN columns: {sorted(missing)}")

    return df


def dataframe_to_features(df: pd.DataFrame) -> np.ndarray:
    last_ts_by_id: dict[int, float] = {}
    ema_iat_by_id: dict[int, float] = {}
    global_last_ts: float | None = None
    rows: list[list[float]] = []

    for _, row in df.iterrows():
        ts = float(row["timestamp"])
        raw_id = row["can_id"]
        can_id = int(str(raw_id), 16) if isinstance(raw_id, str) and any(ch.isalpha() for ch in raw_id) else int(raw_id)
        payload = _parse_hex_payload(row["data_hex"])
        dlc = int(row.get("dlc", len(payload)) or len(payload))

        prev_id_ts = last_ts_by_id.get(can_id, ts)
        per_id_dt = max(ts - prev_id_ts, 0.0)
        global_dt = 0.0 if global_last_ts is None else max(ts - global_last_ts, 0.0)
        old_ema = ema_iat_by_id.get(can_id, per_id_dt)
        ema_iat = 0.9 * old_ema + 0.1 * per_id_dt
        ema_iat_by_id[can_id] = ema_iat
        last_ts_by_id[can_id] = ts
        global_last_ts = ts

        freq = 1.0 / max(ema_iat, 1e-6)
        clock_skew = abs(per_id_dt - ema_iat)

        rows.append(
            [
                min(can_id / 0x7FF, 1.0),
                min(dlc / 8.0, 1.0),
                *[b / 255.0 for b in payload],
                np.log1p(ema_iat * 1000.0) / 10.0,
                min(freq / 1000.0, 1.0),
                np.log1p(per_id_dt * 1000.0) / 10.0,
                np.log1p(global_dt * 1000.0) / 10.0,
                np.log1p(clock_skew * 1000.0) / 10.0,
            ]
        )

    return np.asarray(rows, dtype=np.float32)


def make_windows(features: np.ndarray, labels: np.ndarray | None = None, window_size: int = WINDOW_SIZE, stride: int = 5) -> WindowBundle:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for start in range(0, max(len(features) - window_size + 1, 0), stride):
        end = start + window_size
        xs.append(features[start:end])
        if labels is not None:
            ys.append(int(np.max(labels[start:end]) > 0))
    if not xs:
        raise ValueError("Not enough frames to create one window")
    return WindowBundle(np.stack(xs).astype(np.float32), np.asarray(ys, dtype=np.int64) if labels is not None else None)


def load_windows(path: str | Path, window_size: int = WINDOW_SIZE, stride: int = 5) -> WindowBundle:
    df = read_can_csv(path)
    label_col = next((c for c in ("attack", "label", "attack_flag") if c in df.columns), None)
    labels = df[label_col].to_numpy(dtype=np.int64) if label_col else None
    return make_windows(dataframe_to_features(df), labels, window_size, stride)

