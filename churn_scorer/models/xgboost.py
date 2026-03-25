from typing import Any

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

import pandas as pd

from churn_scorer.config import TARGET_COLUMN
from churn_scorer.protocols import FitPerformance

TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42

DEFAULT_XGB_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "enable_categorical": True,
    "max_depth": 2,
    "min_child_weight": 5,
    "subsample": 1,
    "colsample_bytree": 0.6,
    "gamma": 0.1,
    "reg_alpha": 10,
    "reg_lambda": 0.1,
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "random_state": 42,
}


class xgbClassifier:
    def __init__(self, **overrides: Any):
        self.xgb = XGBClassifier(**{**DEFAULT_XGB_PARAMS, **overrides})

    def fit(self, training_df: pd.DataFrame, target_column: str) -> FitPerformance:
        X = training_df.drop([target_column], axis=1)
        y = training_df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TRAIN_TEST_SPLIT_RATIO, random_state=RANDOM_STATE
        )

        self.xgb.fit(X_train, y_train)

        y_pred = self.xgb.predict_proba(X_test)[:, 1]
        performance = round(float(roc_auc_score(y_test, y_pred)), 3)

        return {"AUC": performance}

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[TARGET_COLUMN] = self.xgb.predict_proba(df)[:, 1]
        return out
