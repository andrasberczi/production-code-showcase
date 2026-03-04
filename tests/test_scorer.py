import os
from pathlib import Path

import pandas as pd
import pytest

_TESTS_DATA = Path(__file__).resolve().parent / "data"
os.environ["CHURN_TRAINING_DATA_URI"] = str(_TESTS_DATA / "test_data_for_train.csv")
os.environ["CHURN_RESULTS_URI"] = str(_TESTS_DATA / "test_results_scorer.csv")

from churn_scorer.scorer import ChurnScorer


@pytest.mark.integration
def test_churn_scorer_predict_writes_results_with_churn_scores() -> None:
    results_path = _TESTS_DATA / "test_results_scorer.csv"

    df = pd.read_csv(_TESTS_DATA / "test_data_for_prediction.csv")
    uri = ChurnScorer.create().predict(df)
    out = pd.read_csv(results_path)

    assert uri == str(results_path)
    assert results_path.is_file()
    assert "Churn" in out.columns
    assert len(out) == len(df)
    assert out["Churn"].between(0.0, 1.0).all()
