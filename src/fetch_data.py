"""
Downloads OHLCV price history for Vietnamese stocks from Yahoo Finance.

Vietnamese stocks listed on HOSE/HNX are available on Yahoo Finance
with the .VN suffix: VNM.VN (Vinamilk), VIC.VN (Vingroup), HPG.VN
(Hoa Phat Steel), MWG.VN (Mobile World), VHM.VN (Vinhomes).

Run:
    python src/fetch_data.py                        # default 5 tickers
    python src/fetch_data.py --tickers VNM.VN HPG.VN --start 2020-01-01
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

from config import DATA_RAW, DEFAULT_TICKERS, DEFAULT_START, DEFAULT_END


def fetch_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if df.empty:
        print(f"  WARNING: no data returned for {ticker}. "
              f"Check the ticker (use .VN suffix for HOSE stocks).")
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when downloading one ticker
    # flatten them if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    df.index.name = "date"
    df.columns = [c.lower() for c in df.columns]
    df = df.dropna()
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start",   default=DEFAULT_START)
    parser.add_argument("--end",     default=DEFAULT_END)
    args = parser.parse_args()

    print(f"Fetching {len(args.tickers)} tickers: {args.start} to {args.end}")
    for ticker in args.tickers:
        print(f"  {ticker}...", end=" ", flush=True)
        df = fetch_ticker(ticker, args.start, args.end)
        if df.empty:
            continue
        out = DATA_RAW / f"{ticker.replace('.', '_')}.csv"
        df.to_csv(out)
        print(f"{len(df)} trading days -> {out.name}")

    print(f"\nDone. CSVs saved to {DATA_RAW}")
    print("Next: python src/features.py")


if __name__ == "__main__":
    main()
