"""
Regression tests for lookahead leakage in features.add_features().

Every rolling window stat (vol_ma, return_std, close_ma) must use
shift(1) before .rolling() so day t cannot see its own value.
"""
import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from features import add_features, FEATURE_COLS


def _make_flat_df(n: int = 60, volume: float = 1_000_000) -> pd.DataFrame:
    """Flat price, constant volume - easy to reason about baselines."""
    dates = pd.bdate_range("2021-01-01", periods=n)
    return pd.DataFrame(
        {
            "open":   100.0,
            "high":   101.0,
            "low":    99.0,
            "close":  100.0,
            "volume": float(volume),
        },
        index=dates,
    )


def test_rel_vol_does_not_include_current_day():
    """
    rel_vol_5 for a spike day must use only the prior 5 days as baseline.

    Setup: 59 flat days at volume=1_000_000, then day 60 spikes to 10_000_000.
    If shift(1) is present: vol_ma_5 on the spike day = 1_000_000 (prior days only).
      -> rel_vol_5 = 10_000_000 / 1_000_000 = 10.0
    If shift(1) is absent: vol_ma_5 on the spike day includes the spike itself.
      -> rel_vol_5 would be pulled toward 1.0 (spike diluted into its own baseline).
    """
    df = _make_flat_df(n=60)
    df.iloc[-1, df.columns.get_loc("volume")] = 10_000_000.0

    result = add_features(df)
    spike_rel_vol = result["rel_vol_5"].iloc[-1]

    assert spike_rel_vol == pytest.approx(10.0, rel=0.05), (
        f"rel_vol_5 on spike day = {spike_rel_vol:.3f}, expected ~10.0. "
        "Likely shift(1) is missing - current day volume is leaking into its own baseline."
    )


def test_return_z_does_not_include_current_day():
    """
    return_z_5 for a spike day must use only the prior 5 days' std as baseline.

    Setup: 59 flat days (zero log-return), then one large return day.
    With shift(1): return_std_5 on spike day = std of the 5 prior flat days ≈ 0.
      -> return_z_5 will be very large (spike / near-zero std).
    Without shift(1): the spike's own return inflates the std it's divided by,
      -> return_z_5 is artificially compressed toward ~1-2.
    """
    df = _make_flat_df(n=60)
    df.iloc[-1, df.columns.get_loc("close")] = 110.0  # ~9.5% log-return

    result = add_features(df)
    spike_z = result["return_z_5"].iloc[-1]

    # Prior 5 returns are all 0; std ≈ 0; z-score must be large (capped by +1e-8 denom)
    # Even with small numerical noise, it should far exceed 3.0
    assert abs(spike_z) > 3.0, (
        f"return_z_5 on spike day = {spike_z:.3f}, expected |z| >> 3. "
        "Likely shift(1) missing from return_std window - spike is dampening its own z-score."
    )


def test_no_future_data_in_atr():
    """
    ATR on day t must use tr.shift(1) so day t's own true range is excluded.

    We check the boundary: ATR on the very last day must equal the rolling
    mean of the 14 days *before* it, not including itself.
    """
    df = _make_flat_df(n=40)
    # All TR values are identical (high - low = 2.0 for flat data)
    expected_atr = 2.0

    result = add_features(df)
    last_atr = result["atr"].iloc[-1]

    assert last_atr == pytest.approx(expected_atr, abs=0.01), (
        f"ATR on last day = {last_atr:.4f}, expected {expected_atr}. "
        "ATR may be including current day's true range in its own rolling window."
    )


def test_feature_cols_all_present():
    """All columns in FEATURE_COLS must exist in the output of add_features()."""
    df = _make_flat_df(n=60)
    result = add_features(df)
    missing = [c for c in FEATURE_COLS if c not in result.columns]
    assert missing == [], f"Missing feature columns: {missing}"


def test_no_lookahead_in_close_ma():
    """
    close_ma_5 on the last day must reflect the prior 5 closes, not the current one.
    """
    df = _make_flat_df(n=60)
    df.iloc[-1, df.columns.get_loc("close")] = 9999.0  # spike close

    result = add_features(df)
    # close_ma_5 on spike day should be ~100.0 (prior 5 flat closes), not pulled toward 9999
    last_close_ma = result["close_ma_5"].iloc[-1]

    assert last_close_ma == pytest.approx(100.0, abs=0.01), (
        f"close_ma_5 on spike day = {last_close_ma:.2f}, expected ~100.0. "
        "close_ma may be including current day's close in the rolling mean."
    )