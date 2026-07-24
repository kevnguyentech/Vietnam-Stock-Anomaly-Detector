"""
LSTM Autoencoder for time-series anomaly detection.

WHY AN AUTOENCODER ON TOP OF ISOLATION FOREST:
Isolation Forest treats each day independently (no memory of sequence).
It would flag a day with extreme volume even if that volume was part of
a legitimate earnings-week trend that built up over 5 days. An LSTM
Autoencoder sees the sequence: it learns what a normal 20-day pattern
looks like end-to-end, then flags days where its reconstruction error
is high. High reconstruction error = "I couldn't reconstruct this
sequence from what I know about normal — it's unusual."

The two models are complementary:
  - Isolation Forest: excellent at single-day outliers (sudden spikes)
  - LSTM Autoencoder: better at detecting subtle sequence patterns
    (gradual manipulation, abnormal correlation between days)
  Both scores get fused in evaluate.py.

ARCHITECTURE:
  Input:  (batch, seq_len=20, n_features=15)
  Encoder: LSTM(hidden=64, layers=2) -> last hidden state -> bottleneck
  Decoder: RepeatVector(20) -> LSTM(hidden=64, layers=2) -> reconstruct
  Loss:    MSE on reconstruction

The bottleneck forces the model to compress a 20-day sequence into
a fixed-size representation. If a sequence is anomalous, the bottleneck
can't capture it well, and the reconstruction error spikes.

Run:
    python src/train_lstm.py
"""
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

from config import (
    DATA_PROCESSED, MODELS_DIR,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS,
    LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_LR,
    LSTM_THRESHOLD_PERCENTILE, RANDOM_SEED,
)
from features import FEATURE_COLS

torch.manual_seed(RANDOM_SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, n_layers: int):
        super().__init__()
        self.n_features = n_features
        self.hidden     = hidden

        self.encoder = nn.LSTM(n_features, hidden, n_layers,
                                batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        self.decoder = nn.LSTM(n_features, hidden, n_layers,
                                batch_first=True, dropout=0.1 if n_layers > 1 else 0)
        self.fc_out  = nn.Linear(hidden, n_features)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, (h, c) = self.encoder(x)

        # Decode: feed zeros as input, use encoder hidden state
        batch = x.size(0)
        seq   = x.size(1)
        dec_input = torch.zeros(batch, seq, self.n_features, device=x.device)
        out, _ = self.decoder(dec_input, (h, c))
        return self.fc_out(out)          # (batch, seq_len, n_features)


def make_sequences(X: np.ndarray, seq_len: int) -> np.ndarray:
    """Slide a window of length seq_len over the time axis."""
    seqs = [X[i: i + seq_len] for i in range(len(X) - seq_len + 1)]
    return np.array(seqs, dtype=np.float32)


def train_ticker(ticker: str, df: pd.DataFrame):
    X_raw = df[FEATURE_COLS].values.astype(np.float32)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    sequences = make_sequences(X, LSTM_SEQ_LEN)  # (N, seq_len, n_feat)
    dataset   = TensorDataset(torch.from_numpy(sequences))
    loader    = DataLoader(dataset, batch_size=LSTM_BATCH_SIZE, shuffle=True)

    model = LSTMAutoencoder(len(FEATURE_COLS), LSTM_HIDDEN, LSTM_LAYERS).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    loss_fn = nn.MSELoss()

    print(f"  Training LSTM on {ticker} ({len(sequences)} sequences, {DEVICE})...")
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss  = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch)
        if epoch % 10 == 0:
            print(f"    epoch {epoch}/{LSTM_EPOCHS}  loss={epoch_loss/len(sequences):.5f}")

    # Reconstruction error per original time step
    # Each step appears in up to seq_len sequences; average their errors
    model.eval()
    all_seqs = torch.from_numpy(sequences).to(DEVICE)
    with torch.no_grad():
        recon_all = model(all_seqs).cpu().numpy()
    errors_per_seq = np.mean((sequences - recon_all) ** 2, axis=(1, 2))

    # Map seq-level error back to day-level (average over all windows that include it)
    n = len(df)
    day_errors = np.zeros(n)
    day_counts = np.zeros(n)
    for i, err in enumerate(errors_per_seq):
        day_errors[i: i + LSTM_SEQ_LEN] += err
        day_counts[i: i + LSTM_SEQ_LEN] += 1
    day_counts = np.maximum(day_counts, 1)
    day_errors /= day_counts

    # Threshold: top percentile of reconstruction errors = anomaly
    threshold    = np.percentile(day_errors, LSTM_THRESHOLD_PERCENTILE)
    error_min    = day_errors.min()
    error_max    = day_errors.max()
    lstm_scores  = (day_errors - error_min) / (error_max - error_min + 1e-8)
    lstm_flags   = (day_errors > threshold).astype(int)

    return {
        "model":        {k: v.cpu() for k, v in model.state_dict().items()},
        "scaler":       scaler,
        "threshold":    threshold,
        "error_min":    error_min,
        "error_max":    error_max,
        "feature_cols": FEATURE_COLS,
        "lstm_scores":  lstm_scores,
        "lstm_flags":   lstm_flags,
    }


def main():
    csvs = list(DATA_PROCESSED.glob("*.csv"))
    # Skip already-scored files
    csvs = [f for f in csvs if "_scored" not in f.name]
    if not csvs:
        sys.exit("No processed CSVs. Run features.py first.")

    print(f"Training LSTM Autoencoder on {len(csvs)} tickers (device: {DEVICE})...")

    for csv_path in sorted(csvs):
        ticker = csv_path.stem
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)

        result = train_ticker(ticker, df)

        # Merge LSTM scores into the scored CSV from Isolation Forest
        scored_path = DATA_PROCESSED / f"{ticker}_scored.csv"
        if scored_path.exists():
            df_scored = pd.read_csv(scored_path, index_col="date", parse_dates=True)
        else:
            print(f"  WARNING: {scored_path.name} not found — IF scores missing. "
                  f"Run train_isolation_forest.py first for a proper fused score.")
            df_scored = df.copy()
            df_scored["if_score"] = 0.0
            df_scored["if_flag"]  = 0

        lstm_score_series = pd.Series(result["lstm_scores"], index=df.index, name="lstm_score")
        lstm_flag_series  = pd.Series(result["lstm_flags"],  index=df.index, name="lstm_flag")
        df_scored["lstm_score"] = lstm_score_series
        df_scored["lstm_flag"]  = lstm_flag_series

        # Fused score: average of IF + LSTM anomaly scores
        df_scored["fused_score"] = (df_scored["if_score"] + df_scored["lstm_score"]) / 2
        df_scored.to_csv(scored_path)

        # Save model bundle
        joblib.dump({
            "model_state": result["model"],
            "scaler":      result["scaler"],
            "threshold":   result["threshold"],
            "error_min":   result["error_min"],
            "error_max":   result["error_max"],
            "feature_cols": FEATURE_COLS,
            "lstm_hidden":  LSTM_HIDDEN,
            "lstm_layers":  LSTM_LAYERS,
        }, MODELS_DIR / f"lstm_{ticker}.pkl")

        n_flagged = int(result["lstm_flags"].sum())
        print(f"  {ticker}: {n_flagged} LSTM-flagged days, threshold={result['threshold']:.5f}")

    print(f"\nLSTM models saved to {MODELS_DIR}")
    print("Next: python src/evaluate.py")


if __name__ == "__main__":
    main()
