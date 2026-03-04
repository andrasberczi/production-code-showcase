import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

_TESTS_DATA = Path(__file__).resolve().parent / "data"
os.environ["CHURN_TRAINING_DATA_URI"] = str(_TESTS_DATA / "test_data_for_train.csv")
os.environ["CHURN_RESULTS_URI"] = str(_TESTS_DATA / "test_results.csv")

from churn_scorer.api import app

client = TestClient(app)


def test_health_returns_ok() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.integration
def test_predict_end_to_end_with_test_data_files() -> None:
    predict_path = _TESTS_DATA / "test_data_for_prediction.csv"
    results_path = _TESTS_DATA / "test_results.csv"

    response = client.post(
        "/predict",
        json={"table_uri": str(predict_path)},
    )
    body = response.json()
    scored = pd.read_csv(body["prediction_results_uri"])

    assert response.status_code == 200
    assert Path(body["prediction_results_uri"]).is_file()
    assert "Churn" in scored.columns
