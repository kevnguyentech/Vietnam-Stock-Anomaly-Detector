"""
Regression test for the score-normalisation bug: a given day's anomaly
score must be the same regardless of what window it's scored in.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import pytest

from config import DATA_PROCESSED, MODELS_DIR
from predict import score_dataframe

TICKER = "VNM_VN"


def _pipeline_ready() -> bool:
    csv_ready = (DATA_PROCESSED / f"{TICKER}.csv").exists()
    model_ready = (MODELS_DIR / f"if_{TICKER}.pkl").exists()
    return csv_ready and model_ready


@pytest.mark.skipif(not _pipeline_ready(), reason="run the training pipeline first")
def test_score_is_stable_across_windows():
    df = pd.read_csv(DATA_PROCESSED / f"{TICKER}.csv", index_col="date", parse_dates=True)

    full_scored = score_dataframe(df, TICKER)
    tail_scored = score_dataframe(df.tail(200), TICKER)

    common_date = df.index[-1]
    full_row = full_scored.loc[common_date]
    tail_row = tail_scored.loc[common_date]

    for col in ["if_score", "lstm_score", "fused_score"]:
        assert abs(full_row[col] - tail_row[col]) < 1e-6, (
            f"{col} for {common_date.date()} depends on scoring window: "
            f"full-history={full_row[col]:.6f} vs last-200-days={tail_row[col]:.6f}"
        )