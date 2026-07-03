# Vietnam Stock Anomaly Detector

Detects unusual trading activity in Vietnamese stocks (HOSE/HNX) using
two complementary unsupervised models: Isolation Forest for single-day
outliers and an LSTM Autoencoder for sequence-level anomalies. The goal
is flagging days that look like pump-and-dump schemes, unusual volume
spikes, or abnormal price moves before you'd notice by eye.

This is an unsupervised problem, which makes it fundamentally harder than
classification. There are no labels ("this was manipulation, this wasn't")
for most stocks in most periods. Both models learn what normal looks like
and flag deviations from it.

---

## Quick start

```bash
pip install -r requirements.txt
cd src

python simulate_data.py          # synthetic HOSE data with injected anomalies
python features.py               # build technical indicators
python train_isolation_forest.py # train IF per ticker
python train_lstm.py             # train LSTM autoencoder per ticker
python evaluate.py               # charts + recall report
python predict.py --ticker VNM_VN
```

To run on real data (needs internet):

```bash
python fetch_data.py             # downloads VNM.VN, VIC.VN, HPG.VN, MWG.VN, VHM.VN
python features.py
python train_isolation_forest.py
python train_lstm.py
python predict.py --ticker VNM_VN --live
```

---

## Two models, one fused score

**Isolation Forest** treats each trading day independently. It works by
random-splitting the feature space: normal days cluster together and need
many splits to isolate; anomalies sit far from the cluster and isolate in
very few splits. Short isolation path = high anomaly score.

Best at: sudden single-day spikes in price or volume.

**LSTM Autoencoder** sees sequences of 20 trading days. It learns to
compress and reconstruct normal 20-day patterns. When it can't reconstruct
a sequence well (high MSE), that sequence is anomalous. Best at: subtle
multi-day patterns - a gradual volume build-up before a pump, or abnormal
autocorrelation across days that IF can't see.

The final score is a simple 0-1 average of both. Days above the threshold
(default 0.45) are flagged.

---

## Features used

15 features, all computable at market close with no lookahead:

- Daily return and log-return
- Intraday range, gap open, upper/lower shadow
- ATR (Average True Range) normalized by price
- Relative volume vs. 5/10/20-day moving average
- Return z-score over 5/10/20-day windows
- Price-volume divergence (direction agreement between return and volume)
- Absolute return magnitude

All are scaled per-ticker before training. A volume 5x above VNM's average
is flagged; the same raw volume number on HPG (much higher baseline) is not.

---

## Threshold selection

Ran a sweep from 0.3 to 0.9 to find the actual recall/false-alarm tradeoff
instead of guessing:

| Threshold | Recall | Flag rate |
|-----------|--------|-----------|
| 0.30-0.45 | 100%   | 3.8-13.1% |
| 0.50      | 96%    | 2.3%      |
| 0.70      | 44%    | 0.5%      |
| 0.90      | 20%    | 0.1%      |

`ALERT_THRESHOLD = 0.45` in config.py. Catches every injected anomaly at a
3.8% flag rate, roughly 38 flagged days out of 1022. Low enough to review by
hand, high enough to not miss real events.

The LSTM and Isolation Forest catch different anomalies. On VNM_VN, the LSTM
scored one day at 1.000 (total reconstruction failure) while IF scored the
same day 0.396. Either model alone misses it. The fusion catches it.

## Bugs found and fixed

**Look-ahead leakage**: rolling averages included the current day in their
own baseline, diluting the signal on the exact days that mattered most.
Fixed by shifting the window by 1 day before computing rolling stats.

## Why one model per ticker

Each stock has different baseline volatility, average volume, and typical
intraday range. A model trained on HPG (steel company, billions in daily
volume) would flag routine MWG (electronics retail) days as anomalous -
the raw numbers look extreme in HPG's context. Per-ticker training means
"normal" is relative to that stock's own history.

---

## Project structure

```
stock-anomaly-detector/
├── src/
│   ├── config.py                  # all hyperparameters in one place
│   ├── fetch_data.py              # real Yahoo Finance download
│   ├── simulate_data.py           # synthetic HOSE data with injected events
│   ├── features.py                # 15 technical indicators, no lookahead
│   ├── train_isolation_forest.py  # IF per ticker, saves model + scores
│   ├── train_lstm.py              # LSTM Autoencoder per ticker
│   ├── evaluate.py                # anomaly charts + recall report
│   └── predict.py                 # CLI: analyse any ticker, print ranked alerts
├── data/
│   ├── raw/                       # OHLCV CSVs + ground-truth anomaly dates
│   └── processed/                 # feature matrices + anomaly scores
├── models/                        # if_<ticker>.pkl, lstm_<ticker>.pkl
├── outputs/                       # <ticker>_anomaly_chart.png, recall_report.txt
└── requirements.txt
```

---

## Limitations

**Unsupervised = no ground truth on real data.** The recall numbers above
are on synthetic data with known injection dates. On real HOSE data you
lose that benchmark. The anomaly chart becomes your primary tool: look for
flagged days that coincide with news events (earnings surprises, regulatory
actions, rumours) to validate the model is picking up real signals.

**Market microstructure.** Vietnamese stocks have floor/ceiling daily limits
(±7% on HOSE). Very large moves are limited by this rule, which means
extreme-return anomalies are already bounded. The volume and intraday-range
features matter more here than they would in a market without price limits.

**Survivorship bias.** Training on currently-listed blue-chips means the
model learns from stocks that survived. Delisted stocks - often the ones
where manipulation actually destroyed the company aren't in this dataset.
