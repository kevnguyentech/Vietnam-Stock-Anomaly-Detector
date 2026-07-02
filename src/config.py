"""
Shared constants across all scripts.
"""
from pathlib import Path

ROOT          = Path(__file__).resolve().parent.parent
DATA_RAW      = ROOT / "data" / "raw"
DATA_PROCESSED= ROOT / "data" / "processed"
MODELS_DIR    = ROOT / "models"
OUTPUTS_DIR   = ROOT / "outputs"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, OUTPUTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── default tickers to analyse ─────────────────────────────────────────
# Top HOSE blue-chips. Yahoo Finance uses .VN suffix for Vietnamese stocks.
DEFAULT_TICKERS = ["VNM.VN", "VIC.VN", "VHM.VN", "HPG.VN", "MWG.VN"]

# ── data ───────────────────────────────────────────────────────────────
DEFAULT_START  = "2021-01-01"
DEFAULT_END    = "2024-12-31"

# ── feature engineering ────────────────────────────────────────────────
WINDOWS = [5, 10, 20]          # rolling windows (trading days)
ATR_WINDOW = 14                 # Average True Range period

# ── Isolation Forest ──────────────────────────────────────────────────
IF_CONTAMINATION = 0.03        # expect ~3% anomalous days in any stock
IF_N_ESTIMATORS  = 200
RANDOM_SEED      = 42

# ── LSTM Autoencoder ──────────────────────────────────────────────────
LSTM_SEQ_LEN     = 20          # look-back window (trading days)
LSTM_HIDDEN      = 64
LSTM_LAYERS      = 2
LSTM_EPOCHS      = 50
LSTM_BATCH_SIZE  = 32
LSTM_LR          = 1e-3
LSTM_THRESHOLD_PERCENTILE = 97 # flag top-N% reconstruction errors

# ── anomaly score threshold for CLI output ─────────────────────────────
ALERT_THRESHOLD  = 0.45         # 0-1 fused score above this = flagged

