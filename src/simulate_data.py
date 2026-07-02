"""
Generates realistic synthetic OHLCV data for 5 Vietnamese stocks.

The simulation is not random noise. Each stock has:
  - A trending price path (GBM with realistic Vietnamese blue-chip params)
  - Volume that spikes on high-volatility days (realistic)
  - 3-6 injected anomaly events per stock (the ground-truth labels for
    evaluating whether the detector actually finds them):
      * Price spike: intraday range 3-5x normal, large gap
      * Volume spike: volume 8-15x average with directional price move
      * Crash: 5-10% single-day drop with high volume
    Each injected event is written to data/raw/<ticker>_anomalies.csv
    so evaluate.py can check recall — did the model flag the days
    we know are anomalous?

Run:
    python src/simulate_data.py
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_RAW, RANDOM_SEED

TICKERS = {
    "VNM_VN": {"mu": 0.08,  "sigma": 0.20, "price0": 75_000,  "avg_vol": 2_000_000},
    "VIC_VN": {"mu": 0.05,  "sigma": 0.28, "price0": 45_000,  "avg_vol": 3_500_000},
    "VHM_VN": {"mu": 0.06,  "sigma": 0.25, "price0": 35_000,  "avg_vol": 4_000_000},
    "HPG_VN": {"mu": 0.12,  "sigma": 0.32, "price0": 25_000,  "avg_vol": 8_000_000},
    "MWG_VN": {"mu": 0.03,  "sigma": 0.22, "price0": 55_000,  "avg_vol": 1_500_000},
}

N_ANOMALIES_PER_TICKER = 5


def simulate_ohlcv(ticker: str, params: dict, dates: pd.DatetimeIndex,
                    rng: np.random.Generator) -> tuple[pd.DataFrame, list[int]]:
    n = len(dates)
    mu    = params["mu"]   / 252
    sigma = params["sigma"] / np.sqrt(252)
    p0    = params["price0"]
    avg_v = params["avg_vol"]

    # Geometric Brownian Motion for close prices
    returns = rng.normal(mu, sigma, n)
    close = p0 * np.cumprod(1 + returns)

    # Intraday range: proportional to daily return magnitude + noise
    daily_range_pct = np.abs(returns) + rng.lognormal(-3.5, 0.4, n)
    high  = close * (1 + daily_range_pct / 2)
    low   = close * (1 - daily_range_pct / 2)
    open_ = low + rng.uniform(0, 1, n) * (high - low)

    # Volume correlated with volatility (realistic)
    vol_multiplier = np.exp(3 * np.abs(returns) / sigma + rng.normal(0, 0.3, n))
    vol_multiplier = np.clip(vol_multiplier, 0.2, 15)  # cap at 15x — no real trading day sees more
    volume = (avg_v * vol_multiplier).astype(int)

    # Inject known anomalies
    anomaly_days = sorted(rng.choice(range(50, n - 10), N_ANOMALIES_PER_TICKER, replace=False))
    anomaly_types = rng.choice(["price_spike", "volume_spike", "crash"], N_ANOMALIES_PER_TICKER)

    for idx, atype in zip(anomaly_days, anomaly_types):
        if atype == "price_spike":
            spike = rng.uniform(0.07, 0.12) * rng.choice([-1, 1])
            close[idx:idx+2] *= (1 + spike)
            high[idx]   = close[idx] * 1.08
            low[idx]    = close[idx] * 0.92
            volume[idx] = int(avg_v * rng.uniform(5, 10))
        elif atype == "volume_spike":
            volume[idx] = int(avg_v * rng.uniform(8, 15))
            close[idx] *= 1 + rng.uniform(0.04, 0.08) * rng.choice([-1, 1])
        elif atype == "crash":
            close[idx]  *= rng.uniform(0.88, 0.94)
            high[idx]    = close[idx-1] * 1.01
            low[idx]     = close[idx]   * 0.97
            volume[idx]  = int(avg_v * rng.uniform(6, 12))

    df = pd.DataFrame({
        "open": open_.round(0),
        "high": high.round(0),
        "low":  low.round(0),
        "close": close.round(0),
        "volume": volume,
    }, index=dates)
    df.index.name = "date"
    return df, anomaly_days


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end",   default="2024-12-31")
    parser.add_argument("--seed",  type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    # Business days only — stock markets don't trade weekends
    dates = pd.bdate_range(args.start, args.end)
    print(f"Simulating {len(TICKERS)} tickers, {len(dates)} trading days each "
          f"({args.start} to {args.end})")

    all_anomalies = {}
    for ticker, params in TICKERS.items():
        df, anomaly_days = simulate_ohlcv(ticker, params, dates, rng)
        out_csv = DATA_RAW / f"{ticker}.csv"
        df.to_csv(out_csv)

        # Save ground-truth anomaly dates for evaluation
        anom_dates = [str(dates[i].date()) for i in anomaly_days]
        anom_df = pd.DataFrame({"date": anom_dates, "ticker": ticker})
        anom_csv = DATA_RAW / f"{ticker}_anomalies.csv"
        anom_df.to_csv(anom_csv, index=False)
        all_anomalies[ticker] = anom_dates

        price_range = f"{df['close'].min():,.0f} - {df['close'].max():,.0f} VND"
        print(f"  {ticker}: {len(df)} rows, close range {price_range}, "
              f"{N_ANOMALIES_PER_TICKER} injected anomalies")

    print(f"\nDone. CSVs -> {DATA_RAW}")
    print("Next: python src/features.py")


if __name__ == "__main__":
    main()
