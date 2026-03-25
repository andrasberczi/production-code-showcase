import os
from pathlib import Path

import pandas as pd
import pytest

_TESTS_DATA = Path(__file__).resolve().parent / "data"
os.environ["CHURN_TRAINING_DATA_URI"] = str(_TESTS_DATA / "test_data_for_train.csv")
os.environ["CHURN_RESULTS_URI"] = str(_TESTS_DATA / "test_results_scorer.csv")

from churn_scorer.scoring.service import ChurnScorer


@pytest.fixture
def prediction(tmp_path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    results_path = _TESTS_DATA / "test_results_scorer.csv"
    input_df = pd.read_csv(_TESTS_DATA / "test_data_for_prediction.csv")
    uri = ChurnScorer.create().predict(input_df)
    output_df = pd.read_csv(results_path)
    return uri, input_df, output_df


@pytest.mark.integration
def test_predict_returns_configured_uri(prediction: tuple[str, pd.DataFrame, pd.DataFrame]) -> None:
    uri, _, _ = prediction
    results_path = _TESTS_DATA / "test_results_scorer.csv"
    assert uri == str(results_path)


@pytest.mark.integration
def test_predict_writes_results_file(prediction: tuple[str, pd.DataFrame, pd.DataFrame]) -> None:
    uri, _, _ = prediction
    assert Path(uri).is_file()


@pytest.mark.integration
def test_predict_outputs_churn_probabilities(
    prediction: tuple[str, pd.DataFrame, pd.DataFrame],
) -> None:
    _, input_df, output_df = prediction
    assert "Churn" in output_df.columns
    assert len(output_df) == len(input_df)
    assert output_df["Churn"].between(0.0, 1.0).all()
