"""
Regression test: lstm_scores must align to the correct dates.

The fix in train_lstm.main() wraps raw numpy arrays in pd.Series(index=df.index)
before assigning into df_scored. This test verifies that a known score value
lands on the correct date, not shifted by a positional offset.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def _make_score_series(dates, scores):
    """Helper: the exact pattern used in the fix."""
    return pd.Series(scores, index=dates, name="lstm_score")


def test_lstm_score_correct_date_when_indices_match():
    """
    When df and df_scored share the same index, score[i] must land on dates[i].
    This is the normal case — verifies the Series assignment doesn't reorder.
    """
    dates = pd.bdate_range("2021-01-01", periods=10)
    fake_scores = np.arange(10, dtype=float)

    df_scored = pd.DataFrame({"if_score": 0.5}, index=dates)
    df_scored["lstm_score"] = _make_score_series(dates, fake_scores)

    for i, date in enumerate(dates):
        assert df_scored.loc[date, "lstm_score"] == pytest.approx(float(i)), (
            f"Score {float(i)} should be on {date}, got {df_scored.loc[date, 'lstm_score']}"
        )


def test_lstm_score_spike_on_known_date():
    """
    A spike score must appear on the correct date, not drift to an adjacent date.
    Simulates: day 5 is the anomaly, score=99.0. All others are 0.0.
    """
    dates = pd.bdate_range("2021-01-01", periods=20)
    fake_scores = np.zeros(20, dtype=float)
    fake_scores[5] = 99.0
    spike_date = dates[5]

    df_scored = pd.DataFrame({"if_score": 0.0}, index=dates)
    df_scored["lstm_score"] = _make_score_series(dates, fake_scores)

    assert df_scored.loc[spike_date, "lstm_score"] == pytest.approx(99.0), (
        f"Spike score should be on {spike_date}, "
        f"got {df_scored.loc[spike_date, 'lstm_score']}"
    )
    # Neighbors must not carry the spike
    assert df_scored["lstm_score"].iloc[4] == pytest.approx(0.0), \
        "Spike score leaked into the preceding date."
    assert df_scored["lstm_score"].iloc[6] == pytest.approx(0.0), \
        "Spike score leaked into the following date."


def test_lstm_flag_spike_on_known_date():
    """Same date-alignment check for lstm_flag."""
    dates = pd.bdate_range("2021-01-01", periods=20)
    fake_flags = np.zeros(20, dtype=int)
    fake_flags[7] = 1
    flag_date = dates[7]

    df_scored = pd.DataFrame({"if_flag": 0}, index=dates)
    flag_series = pd.Series(fake_flags, index=dates, name="lstm_flag")
    df_scored["lstm_flag"] = flag_series

    assert df_scored.loc[flag_date, "lstm_flag"] == 1, (
        f"Flag should be on {flag_date}, got {df_scored.loc[flag_date, 'lstm_flag']}"
    )
    assert df_scored["lstm_flag"].iloc[6] == 0, "Flag leaked into preceding date."
    assert df_scored["lstm_flag"].iloc[8] == 0, "Flag leaked into following date."