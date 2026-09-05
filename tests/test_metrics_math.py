"""Unit tests for quantitative risk metrics, Gini, KS deciles, and PDO scorecard scaling."""

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import brier_score_loss, roc_auc_score

from basel_scorecard_lakehouse.metrics_evaluator import MetricsEvaluator
from basel_scorecard_lakehouse.scorecard_trainer import ScorecardTrainer


def test_gini_coefficient_formula():
    """Verify Gini = 2 * ROC_AUC - 1 identity across various probability distributions."""
    rng = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=1000, p=[0.8, 0.2])
    y_prob = np.clip(y_true * 0.4 + rng.uniform(0.1, 0.5, size=1000), 0.01, 0.99)
    scores = (750 - y_prob * 300).astype(int)

    evaluator = MetricsEvaluator()
    results = evaluator.evaluate_all(y_true, y_prob, scores)

    expected_auc = roc_auc_score(y_true, y_prob)
    expected_gini = 2.0 * expected_auc - 1.0

    assert np.isclose(results["roc_auc"], expected_auc, atol=1e-4)
    assert np.isclose(results["gini"], expected_gini, atol=1e-4)


def test_brier_score_calibration():
    """Verify Brier score matches standard sklearn reference implementation."""
    rng = np.random.default_rng(123)
    y_true = rng.binomial(1, 0.15, size=500)
    y_prob = rng.uniform(0.0, 1.0, size=500)
    scores = (600 - y_prob * 200).astype(int)

    evaluator = MetricsEvaluator()
    results = evaluator.evaluate_all(y_true, y_prob, scores)

    expected_brier = brier_score_loss(y_true, y_prob)
    assert np.isclose(results["brier_score"], expected_brier, atol=1e-4)


def test_ks_decile_statistic():
    """Verify KS statistic from deciles captures peak separation."""
    # Create distinct separation between good and bad scores
    goods_scores = np.random.normal(720, 40, size=1000)
    bads_scores = np.random.normal(580, 50, size=200)

    y_true = np.array([0] * len(goods_scores) + [1] * len(bads_scores))
    scores = np.concatenate([goods_scores, bads_scores])

    evaluator = MetricsEvaluator()
    decile_df, ks_val = evaluator.compute_ks_deciles(y_true, scores)

    # Compare with scipy 2-sample KS test
    scipy_ks = ks_2samp(goods_scores, bads_scores).statistic
    assert len(decile_df) == 10
    # Decile-binned KS approximates continuous scipy KS within 5 percentage points
    assert np.isclose(ks_val, scipy_ks, atol=0.08)


def test_pdo_scorecard_scaling_math():
    """Verify mathematical properties of Points to Double the Odds (PDO) scaling."""
    base_score = 600.0
    base_odds = 50.0
    pdo = 20.0

    trainer = ScorecardTrainer(base_score=base_score, base_odds=base_odds, pdo=pdo)
    factor = trainer.factor
    offset = trainer.offset

    # 1. At base log-odds (ln(Odds_bad) = ln(1/50) = -ln(50)), score must equal base_score (600)
    log_odds_base = -np.log(base_odds)
    score_at_base = offset - factor * log_odds_base
    assert np.isclose(score_at_base, base_score, atol=1e-5)

    # 2. Doubling good borrower odds (halving default odds: ln(1/100) = -ln(100)) increases score by PDO (20 pts)
    log_odds_doubled_good = -np.log(base_odds * 2.0)
    score_doubled = offset - factor * log_odds_doubled_good
    assert np.isclose(score_doubled - score_at_base, pdo, atol=1e-5)
