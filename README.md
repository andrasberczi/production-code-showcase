# Churn scoring (showcase)

This project showcases how production-grade Python code should be structured — clean architecture, clear separation of concerns, type safety, and easy expandability. The problem itself is made up, based on a [Kaggle competition](https://www.kaggle.com/competitions/playground-series-s6e3), so some choices are intentionally simplistic (a single XGBoost model with fixed hyperparameters, CSV-based data loading, etc.). The focus is not on the ML solution but on demonstrating how to write maintainable, well-tested code.

- **Python:** 3.12+ (see `pyproject.toml` / `.python-version`)
- **Setup:** `python -m venv .venv` → activate → `pip install -U pip` → `pip install -e ".[dev,api]"`
- **Checks:** `python -m black --check churn_scorer tests` → `python -m mypy churn_scorer tests` → `python -m pytest`
- **API:** `uvicorn churn_scorer.api.routes:app --reload --host 127.0.0.1 --port 8000` then open `http://127.0.0.1:8000/docs`. You can try out the API with the following request:
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
