import os

TARGET_COLUMN = "Churn"


def training_data_uri() -> str:
    return os.environ.get("CHURN_TRAINING_DATA_URI", "data/sample_train.csv")


def results_uri() -> str:
    return os.environ.get("CHURN_RESULTS_URI", "data/results.csv")
