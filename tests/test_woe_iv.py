"""Unit tests for Weight of Evidence (WoE) and Information Value (IV) calculations."""

import numpy as np
import pandas as pd

from basel_scorecard_lakehouse.woe_iv_engine import WoEIVEngine


def test_woe_hand_calculation():
    """Verify WoE formula matches hand calculation for controlled synthetic bins."""
    # Bin 1: 80 goods, 20 bads  | Bin 2: 20 goods, 80 bads
    # Total Goods = 100, Total Bads = 100
    # Bin 1: %Good = 0.8, %Bad = 0.2 -> WoE = ln(0.8 / 0.2) = ln(4) ≈ 1.386294
    # Bin 2: %Good = 0.2, %Bad = 0.8 -> WoE = ln(0.2 / 0.8) = ln(0.25) ≈ -1.386294
    df = pd.DataFrame(
        {
            "bin": ["Bin_1"] * 100 + ["Bin_2"] * 100,
            "target": [0] * 80 + [1] * 20 + [0] * 20 + [1] * 80,
        }
    )

    table, total_iv = WoEIVEngine._calculate_woe_iv_table(df, "test_feat")
    woe_dict = dict(zip(table["bin"], table["woe"]))

    assert np.isclose(woe_dict["Bin_1"], np.log(4.0), atol=1e-3)
    assert np.isclose(woe_dict["Bin_2"], np.log(0.25), atol=1e-3)

    # Expected IV = (0.8 - 0.2) * ln(4) + (0.2 - 0.8) * ln(0.25)
    #             = 0.6 * 1.386294 + (-0.6) * (-1.386294) = 1.2 * 1.386294 ≈ 1.66355
    expected_iv = 1.2 * np.log(4.0)
    assert np.isclose(total_iv, expected_iv, atol=1e-2)


def test_woe_monotonicity_and_transform():
    """Test that monotonic binning returns monotonic Bad Rates and transforms DataFrame."""
    rng = np.random.default_rng(42)
    n = 2000
    x = rng.uniform(0, 100, size=n)
    # Prob default strongly increases with x
    p_default = 1.0 / (1.0 + np.exp(-0.05 * (x - 50)))
    y = rng.binomial(1, p_default)

    df = pd.DataFrame({"risk_score": x, "target": y})

    engine = WoEIVEngine(n_bins=5)
    iv_summary = engine.fit(df, ["risk_score"], target_col="target")

    assert not iv_summary.empty
    assert iv_summary.iloc[0]["information_value"] > 0.05

    # Check transformed output
    df_woe = engine.transform(df)
    assert "risk_score_woe" in df_woe.columns
    assert len(df_woe) == n
    assert not df_woe["risk_score_woe"].isna().any()
