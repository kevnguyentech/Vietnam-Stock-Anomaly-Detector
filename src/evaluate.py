"""
Evaluates both detectors and generates visualisation plots.

Outputs (in outputs/):
  <ticker>_anomaly_chart.png  - price + volume chart with flagged days marked
  <ticker>_score_history.png  - rolling anomaly score over time
  recall_report.txt           - if ground-truth anomalies exist, how many did we catch?

The recall report is the most important output: it answers "did the
model actually find the injected pump-and-dump events?" When you switch
from simulated to real data, you lose ground-truth labels, and the
anomaly chart becomes your primary tool instead.

Run:
    python src/evaluate.py
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from config import DATA_PROCESSED, DATA_RAW, OUTPUTS_DIR, ALERT_THRESHOLD


def plot_anomaly_chart(ticker: str, df: pd.DataFrame):
    """Price + volume chart with high-score days highlighted."""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9),
                                          gridspec_kw={"height_ratios": [3, 1.5, 1.5]})
    fig.suptitle(f"{ticker} — Anomaly Detection", fontsize=13, fontweight="bold")

    dates  = df.index
    flagged = df["fused_score"] >= ALERT_THRESHOLD

    # ── Price ─────────────────────────────────────────────────────────
    ax1.plot(dates, df["close"], color="#2C3E50", linewidth=0.9, label="Close price")
    ax1.scatter(dates[flagged], df.loc[flagged, "close"],
                color="#E74C3C", s=40, zorder=5, label=f"Flagged (score ≥ {ALERT_THRESHOLD})")
    ax1.set_ylabel("Close (VND)")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.tick_params(axis="x", rotation=30, labelsize=7)

    # ── Volume ────────────────────────────────────────────────────────
    colors = ["#E74C3C" if f else "#BDC3C7" for f in flagged]
    ax2.bar(dates, df["volume"], color=colors, width=1)
    ax2.set_ylabel("Volume")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.tick_params(axis="x", rotation=30, labelsize=7)

    # ── Fused anomaly score ────────────────────────────────────────────
    ax3.fill_between(dates, df["fused_score"], alpha=0.4, color="#E74C3C")
    ax3.plot(dates, df["fused_score"], color="#C0392B", linewidth=0.7)
    ax3.axhline(ALERT_THRESHOLD, color="#E74C3C", linestyle="--",
                linewidth=1, label=f"Alert threshold ({ALERT_THRESHOLD})")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Fused score")
    ax3.legend(fontsize=8)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax3.tick_params(axis="x", rotation=30, labelsize=7)

    fig.tight_layout()
    path = OUTPUTS_DIR / f"{ticker}_anomaly_chart.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def compute_recall(ticker: str, df: pd.DataFrame) -> dict | None:
    """Check how many injected anomalies the model flagged."""
    anom_csv = DATA_RAW / f"{ticker}_anomalies.csv"
    if not anom_csv.exists():
        return None

    ground_truth = pd.read_csv(anom_csv)["date"].tolist()
    flagged_dates = df.index[df["fused_score"] >= ALERT_THRESHOLD].strftime("%Y-%m-%d").tolist()

    hits = [d for d in ground_truth if d in flagged_dates]
    recall = len(hits) / len(ground_truth) if ground_truth else 0.0
    return {
        "ticker":        ticker,
        "n_injected":    len(ground_truth),
        "n_flagged":     len(flagged_dates),
        "n_hits":        len(hits),
        "recall":        recall,
        "hit_dates":     hits,
        "missed_dates":  [d for d in ground_truth if d not in hits],
    }

def threshold_sweep(scored_csvs: list) -> pd.DataFrame:
    """Test recall/flag-rate across thresholds 0.3-0.9 to justify ALERT_THRESHOLD."""
    thresholds = np.arange(0.3, 0.95, 0.05)
    rows = []

    for t in thresholds:
        total_hits, total_injected, total_flagged, total_days = 0, 0, 0, 0
        for csv_path in scored_csvs:
            ticker = csv_path.stem.replace("_scored", "")
            df = pd.read_csv(csv_path, index_col="date", parse_dates=True)
            anom_csv = DATA_RAW / f"{ticker}_anomalies.csv"
            if not anom_csv.exists():
                continue
            ground_truth = pd.read_csv(anom_csv)["date"].tolist()
            flagged_dates = df.index[df["fused_score"] >= t].strftime("%Y-%m-%d").tolist()
            hits = [d for d in ground_truth if d in flagged_dates]

            total_hits += len(hits)
            total_injected += len(ground_truth)
            total_flagged += len(flagged_dates)
            total_days += len(df)

        rows.append({
            "threshold": round(t, 2),
            "recall": total_hits / total_injected if total_injected else 0,
            "flag_rate": total_flagged / total_days if total_days else 0,
            "hits": total_hits,
            "injected": total_injected,
        })

    return pd.DataFrame(rows)

def main():
    scored_csvs = sorted(DATA_PROCESSED.glob("*_scored.csv"))
    if not scored_csvs:
        sys.exit("No scored CSVs found. Run train_isolation_forest.py and train_lstm.py first.")

    recall_rows = []
    for csv_path in scored_csvs:
        ticker = csv_path.stem.replace("_scored", "")
        df = pd.read_csv(csv_path, index_col="date", parse_dates=True)

        path = plot_anomaly_chart(ticker, df)
        print(f"{ticker}: chart -> {path.name}")

        r = compute_recall(ticker, df)
        if r:
            recall_rows.append(r)
            status = "✓" if r["recall"] >= 0.6 else "✗"
            print(f"  {status} Recall: {r['n_hits']}/{r['n_injected']} injected anomalies found "
                  f"({r['recall']*100:.0f}%)")
            if r["missed_dates"]:
                print(f"    Missed: {r['missed_dates']}")

    # Write recall report
    if recall_rows:
        lines = ["ANOMALY DETECTION RECALL REPORT", "=" * 40, ""]
        for r in recall_rows:
            lines.append(f"Ticker: {r['ticker']}")
            lines.append(f"  Injected anomalies : {r['n_injected']}")
            lines.append(f"  Flagged days total : {r['n_flagged']}")
            lines.append(f"  Hits               : {r['n_hits']}")
            lines.append(f"  Recall             : {r['recall']*100:.1f}%")
            if r["hit_dates"]:
                lines.append(f"  Found              : {', '.join(r['hit_dates'])}")
            if r["missed_dates"]:
                lines.append(f"  Missed             : {', '.join(r['missed_dates'])}")
            lines.append("")
        avg_recall = np.mean([r["recall"] for r in recall_rows])
        lines.append(f"Average recall across {len(recall_rows)} tickers: {avg_recall*100:.1f}%")
        report_path = OUTPUTS_DIR / "recall_report.txt"
        report_path.write_text("\n".join(lines))
        print(f"\nRecall report -> {report_path}")
        print(f"Average recall: {avg_recall*100:.1f}%")

    print("\nNext: python src/predict.py --ticker VNM_VN")

    sweep_df = threshold_sweep(scored_csvs)
    print(f"\n{'='*50}")
    print("THRESHOLD SWEEP (recall vs. false-alarm tradeoff)")
    print(f"{'='*50}")
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(OUTPUTS_DIR / "threshold_sweep.csv", index=False)
    print(f"\nSaved -> {OUTPUTS_DIR / 'threshold_sweep.csv'}")

if __name__ == "__main__":
    main()
