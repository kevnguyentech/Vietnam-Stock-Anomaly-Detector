"""
Trains an Isolation Forest anomaly detector per ticker.

WHY ISOLATION FOREST:
Anomaly detection is fundamentally different from classification.
You don't have labeled "this day was a pump-and-dump" data for every
stock in history. Isolation Forest is unsupervised — it learns what
"normal" looks like and flags days that are hard to explain by that
normal pattern.

The algorithm works by random-splitting features. Normal points need
many splits to isolate (they cluster together). Anomalies — outliers
with extreme feature values — get isolated in just a few splits.
Short path length = anomalous.

WHY ONE MODEL PER TICKER:
Each stock has different baseline volatility, average volume, typical
intraday range. A model trained on HPG (steel company, massive daily
volume) would incorrectly flag normal VNM (Vinamilk, much lower volume)
days as anomalous. Training per-ticker means "normal" is relative to
that stock's own history, not the market overall.

CONTAMINATION:
I set contamination=0.03 (3%) based on the assumption that roughly
3% of trading days show unusual activity. This is adjustable in
config.py. If you're backtesting a specific stock where you know the
manipulation history, adjust accordingly.

Run:
    python src/train_isolation_forest.py
"""
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PROCESSED, MODELS_DIR, IF_CONTAMINATION,
    IF_N_ESTIMATORS, RANDOM_SEED,
)
from features import FEATURE_COLS


def train_ticker(ticker: str, df: pd.DataFrame) -> dict:
    X = df[FEATURE_COLS].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=IF_N_ESTIMATORS,
        contamination=IF_CONTAMINATION,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(X_scaled)

    # decision_function: negative = more anomalous, positive = more normal
    # We flip and normalise to 0-1 so higher = more suspicious
    raw_scores = model.decision_function(X_scaled)
    score_min = raw_scores.min()
    score_max = raw_scores.max()
    anomaly_scores = 1 - (raw_scores - score_min) / (score_max - score_min + 1e-8)
    predictions = model.predict(X_scaled)   # -1 = anomaly, +1 = normal

    n_flagged = (predictions == -1).sum()
    return {
        "model":         model,
        "scaler":        scaler,
        "feature_cols":  FEATURE_COLS,
        "score_min":     score_min,
        "score_max":     score_max,
        "anomaly_scores": anomaly_scores,
        "predictions":   predictions,
        "dates":         df.index,
        "ticker":        ticker,
        "n_flagged":     n_flagged,
    }


def main():
    csvs = [f for f in DATA_PROCESSED.glob("*.csv") if "_scored" not in f.name]
    if not csvs:
        import sys; sys.exit("No processed CSVs. Run features.py first.")
        
    print(f"Training Isolation Forest on {len(csvs)} tickers...")
    results = {}

    for csv_path in sorted(csvs):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
        result = train_ticker(ticker, df)
        results[ticker] = result

        # Save per-ticker model bundle
        bundle = {k: v for k, v in result.items()
                  if k not in ("anomaly_scores", "predictions", "dates")}
        joblib.dump(bundle, MODELS_DIR / f"if_{ticker}.pkl")

        # Save anomaly scores alongside the processed data (for evaluate.py)
        df["if_score"] = result["anomaly_scores"]
        df["if_flag"]  = (result["predictions"] == -1).astype(int)
        df.to_csv(DATA_PROCESSED / f"{ticker}_scored.csv")

        # Print top 5 flagged days
        top_days = df.nlargest(5, "if_score")[["close","volume","return_1d","if_score"]]
        top_days["return_1d"] = (top_days["return_1d"] * 100).round(2)
        top_days["if_score"]  = top_days["if_score"].round(3)
        print(f"\n{ticker}: flagged {result['n_flagged']} days "
              f"({result['n_flagged']/len(df)*100:.1f}%)")
        print(f"  Top 5 most anomalous days:")
        print(top_days.to_string(index=True))

    print(f"\nModels saved to {MODELS_DIR}")
    print("Next: python src/train_lstm.py")


if __name__ == "__main__":
    main()
