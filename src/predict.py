"""
The actual tool. Given a stock ticker and optional date range, fetches
the latest data, runs both detectors, and prints a ranked list of the
most suspicious trading days.

Usage:
    # Analyse a ticker from the simulated dataset
    python src/predict.py --ticker VNM_VN

    # Analyse a real ticker from Yahoo Finance (needs internet)
    python src/predict.py --ticker VNM.VN --live

    # Look at just the last 90 days
    python src/predict.py --ticker VNM_VN --days 90

    # Set a custom alert threshold
    python src/predict.py --ticker VNM_VN --threshold 0.6
"""
import argparse
import sys
import numpy as np
import pandas as pd
import joblib
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PROCESSED, DATA_RAW, MODELS_DIR,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS,
    ALERT_THRESHOLD, RANDOM_SEED,
)
from features import add_features, FEATURE_COLS
from train_lstm import LSTMAutoencoder


def load_if_model(ticker: str):
    path = MODELS_DIR / f"if_{ticker}.pkl"
    if not path.exists():
        yf_ticker = ticker.replace("_", ".")
        sys.exit(
            f"No trained model for '{ticker}'.\n"
            f"This tool trains one model per ticker on historical data first —\n"
            f"it can't score a stock it's never seen.\n\n"
            f"To fix:\n"
            f"  1. Add '{ticker}' to fetch_data.py or run:\n"
            f"     python fetch_data.py --tickers {yf_ticker}\n"
            f"  2. python features.py\n"
            f"  3. python train_isolation_forest.py\n"
            f"  4. python train_lstm.py\n"
            f"  5. Then: python predict.py --ticker {ticker} --live"
        )
    return joblib.load(path)


def load_lstm_model(ticker: str):
    path = MODELS_DIR / f"lstm_{ticker}.pkl"
    if not path.exists():
        return None
    bundle = joblib.load(path)
    model = LSTMAutoencoder(len(FEATURE_COLS), LSTM_HIDDEN, LSTM_LAYERS)
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return bundle, model


def score_dataframe(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Run both models on df and return df with anomaly scores added."""
    if_bundle   = load_if_model(ticker)
    lstm_result = load_lstm_model(ticker)

    X_raw = df[FEATURE_COLS].values.astype(np.float32)

    # ── Isolation Forest ──────────────────────────────────────────────
    scaler_if = if_bundle["scaler"]
    X_if      = scaler_if.transform(X_raw)
    raw_if    = if_bundle["model"].decision_function(X_if)
    score_min = if_bundle["score_min"]
    score_max = if_bundle["score_max"]
    if_scores = 1 - (raw_if - score_min) / (score_max - score_min + 1e-8)

    # ── LSTM Autoencoder ──────────────────────────────────────────────
    if lstm_result:
        bundle, model = lstm_result
        X_lstm = bundle["scaler"].transform(X_raw)
        n      = len(X_lstm)
        seqs   = np.array([X_lstm[i: i + LSTM_SEQ_LEN]
                           for i in range(n - LSTM_SEQ_LEN + 1)],
                          dtype=np.float32)
        with torch.no_grad():
            recon = model(torch.from_numpy(seqs)).numpy()
        err_per_seq = np.mean((seqs - recon) ** 2, axis=(1, 2))

        day_err   = np.zeros(n)
        day_count = np.zeros(n)
        for i, e in enumerate(err_per_seq):
            day_err[i: i + LSTM_SEQ_LEN]   += e
            day_count[i: i + LSTM_SEQ_LEN] += 1
        day_count = np.maximum(day_count, 1)
        day_err  /= day_count
        error_min = bundle["error_min"]
        error_max = bundle["error_max"]
        lstm_scores = (day_err - error_min) / (error_max - error_min + 1e-8)
        fused = (if_scores + lstm_scores) / 2
    else:
        lstm_scores = np.zeros(len(df))
        fused = if_scores

    df = df.copy()
    df["if_score"]    = if_scores
    df["lstm_score"]  = lstm_scores
    df["fused_score"] = fused
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ticker",    required=True,
                         help="Ticker name, e.g. VNM_VN (local) or VNM.VN (live)")
    parser.add_argument("--live",      action="store_true",
                         help="Fetch latest data from Yahoo Finance instead of local CSV")
    parser.add_argument("--days",      type=int, default=None,
                         help="Analyse only the last N trading days")
    parser.add_argument("--threshold", type=float, default=ALERT_THRESHOLD,
                         help=f"Alert threshold 0-1 (default {ALERT_THRESHOLD})")
    parser.add_argument("--top",       type=int, default=10,
                         help="Show top N most suspicious days (default 10)")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────
    if args.live:
        import yfinance as yf
        ticker_yf = args.ticker.replace("_", ".")
        print(f"Fetching live data for {ticker_yf} from Yahoo Finance...")
        df_raw = yf.download(ticker_yf, period="2y", progress=False, auto_adjust=True)
        if df_raw.empty:
            sys.exit(f"No data returned for {ticker_yf}.")
        if isinstance(df_raw.columns, pd.MultiIndex):
            df_raw.columns = df_raw.columns.get_level_values(0)
        df_raw.columns = [c.lower() for c in df_raw.columns]
        df_raw.index.name = "date"
        ticker_key = args.ticker.replace(".", "_")
    else:
        ticker_key = args.ticker
        csv_path = DATA_PROCESSED / f"{ticker_key}.csv"
        if not csv_path.exists():
            sys.exit(f"No processed data for '{ticker_key}'. Run features.py first.")
        df_raw = pd.read_csv(csv_path, index_col="date", parse_dates=True)

    df = add_features(df_raw) if args.live else df_raw

    if args.days:
        df = df.iloc[-args.days:]

    # ── Score ──────────────────────────────────────────────────────────
    df = score_dataframe(df, ticker_key)

    # ── Output ────────────────────────────────────────────────────────
    flagged = df[df["fused_score"] >= args.threshold]
    top     = df.nlargest(args.top, "fused_score")

    print(f"\n{'═'*60}")
    print(f"  Stock: {args.ticker}   |   {len(df)} trading days analysed")
    print(f"  Alert threshold: {args.threshold}   |   "
          f"Days flagged: {len(flagged)} ({len(flagged)/len(df)*100:.1f}%)")
    print(f"{'═'*60}")

    print(f"\nTop {args.top} most suspicious trading days:\n")
    print(f"{'Date':<12} {'Close':>9} {'Return':>8} {'Rel Vol':>8} "
          f"{'IF':>6} {'LSTM':>6} {'Fused':>6}  Alert")
    print("-" * 70)

    for date, row in top.iterrows():
        alert   = "⚠  FLAGGED" if row["fused_score"] >= args.threshold else ""
        ret_pct = row["return_1d"] * 100 if "return_1d" in row else 0
        rel_vol = row["rel_vol_5"] if "rel_vol_5" in row else 0
        print(f"{str(date.date()):<12} "
              f"{row['close']:>9,.0f} "
              f"{ret_pct:>+7.2f}% "
              f"{rel_vol:>7.1f}x "
              f"{row['if_score']:>6.3f} "
              f"{row['lstm_score']:>6.3f} "
              f"{row['fused_score']:>6.3f}  {alert}")

    if len(flagged) == 0:
        print(f"\n✓  No days exceeded the alert threshold of {args.threshold}.")
        print("   Try lowering --threshold or checking a longer period with --days.")
    else:
        print(f"\n⚠  {len(flagged)} day(s) flagged. Verify against news/announcements.")
        print("   High volume + large price move + high score = strongest signal.")


if __name__ == "__main__":
    main()
