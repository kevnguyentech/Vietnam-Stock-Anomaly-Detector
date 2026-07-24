"""
Builds a feature matrix from raw OHLCV data for anomaly detection.

WHY THESE FEATURES:
Pump-and-dump schemes leave specific fingerprints in price/volume data:
  - Volume spikes (accumulation before the pump, distribution after)
  - Abnormal intraday range (manipulation pushes price hard intraday)
  - Return z-score (5-10% moves in one day stand out vs. normal drift)
  - Price-volume divergence (price moves up but volume doesn't, or v.v.)
  - ATR (Average True Range) measures recent volatility baseline
  - Relative volume vs. moving average (captures "unusual for this stock")

None of these features require tomorrow's data - each one uses only
information available at the close of trading day t. No lookahead.

Run:
    python src/features.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_RAW, DATA_PROCESSED, WINDOWS, ATR_WINDOW


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_index()

    # ── returns & log-returns ──────────────────────────────────────────
    df["return_1d"]     = df["close"].pct_change()
    df["log_return_1d"] = np.log(df["close"] / df["close"].shift(1))

    # ── intraday range features ────────────────────────────────────────
    df["intraday_range"]     = (df["high"] - df["low"]) / df["close"]
    df["gap_open"]           = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
    df["upper_shadow"]       = (df["high"] - df[["open","close"]].max(axis=1)) / df["close"]
    df["lower_shadow"]       = (df[["open","close"]].min(axis=1) - df["low"]) / df["close"]

    # ── Average True Range (normalised) ───────────────────────────────
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift(1)).abs(),
        (df["low"]  - df["close"].shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["atr"] = tr.shift(1).rolling(ATR_WINDOW).mean()
    df["atr_norm"] = df["atr"] / df["close"]   # as fraction of price

    # ── volume features ────────────────────────────────────────────────
    for w in WINDOWS:
        df[f"vol_ma_{w}"]     = df["volume"].shift(1).rolling(w).mean()
        df[f"rel_vol_{w}"]    = df["volume"] / df[f"vol_ma_{w}"]
        df[f"return_std_{w}"] = df["log_return_1d"].shift(1).rolling(w).std()
        df[f"return_z_{w}"]   = df["log_return_1d"] / (df[f"return_std_{w}"] + 1e-8)

    # ── price-volume divergence ────────────────────────────────────────
    # Positive: price and volume move in the same direction (healthy trend)
    # Negative: price up but volume below average — potential manipulation signal
    df["pv_divergence"] = df["return_1d"] * np.log(df["rel_vol_5"].clip(lower=1e-8))

    # ── return magnitude (unsigned) ────────────────────────────────────
    df["abs_return"] = df["return_1d"].abs()

    return df.dropna()


FEATURE_COLS = [
    "return_1d", "log_return_1d",
    "intraday_range", "gap_open", "upper_shadow", "lower_shadow",
    "atr_norm",
    "rel_vol_5", "rel_vol_10", "rel_vol_20",
    "return_z_5", "return_z_10", "return_z_20",
    "pv_divergence", "abs_return",
]


def process_ticker(csv_path: Path) -> pd.DataFrame | None:
    df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
    if len(df) < 60:
        print(f"  Skipping {csv_path.name}: too short ({len(df)} rows)")
        return None
    df = add_features(df)
    return df


def main():
    raw_csvs = [f for f in DATA_RAW.glob("*.csv") if "_anomalies" not in f.name]
    if not raw_csvs:
        sys.exit(f"No CSV files in {DATA_RAW}. Run simulate_data.py or fetch_data.py first.")

    print(f"Processing {len(raw_csvs)} tickers...")
    for csv_path in sorted(raw_csvs):
        df = process_ticker(csv_path)
        if df is None:
            continue
        out = DATA_PROCESSED / csv_path.name
        df.to_csv(out)
        print(f"  {csv_path.stem}: {len(df)} rows, {len(FEATURE_COLS)} features -> {out.name}")

    print(f"\nFeature CSVs saved to {DATA_PROCESSED}")
    print("Next: python src/train_isolation_forest.py")


if __name__ == "__main__":
    main()
