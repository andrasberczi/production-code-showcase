# Churn scoring (showcase)

Minimal churn scoring package with a FastAPI surface and an XGBoost-backed `ChurnScorer`.

- **Python:** 3.12+ (see `pyproject.toml` / `.python-version`)
- **Setup:** `python -m venv .venv` → activate → `pip install -U pip` → `pip install -e ".[dev,api]"`
- **Checks:** `python -m black --check churn_scorer tests` → `python -m mypy churn_scorer tests` → `python -m pytest`
- **API:** `uvicorn churn_scorer.api:app --reload --host 127.0.0.1 --port 8000` then open `http://127.0.0.1:8000/docs`. You can try out the API with the following request:
  - Request body:
    ```json
    {
      "table_uri": "data/sample_input_for_prediction.csv"
    }
    ```
  - Response body should look like:
    ```json
    {
      "message": "Prediction results written at 2026-03-23 01:23:45.",
      "prediction_results_uri": "data/results.csv"
    }
