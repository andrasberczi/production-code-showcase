import numpy as np
import pandas as pd

from churn_scorer.config import TARGET_COLUMN
from churn_scorer.models.xgboost_model import xgbClassifier


def _tiny_training_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 80
    return pd.DataFrame(
        {
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
            TARGET_COLUMN: np.tile([0, 1], n // 2),
        }
    )


def test_fit_returns_auc_in_valid_range() -> None:
    df = _tiny_training_frame()
    model = xgbClassifier(n_estimators=32, max_depth=2)
    perf = model.fit(df, target_column=TARGET_COLUMN)

    assert "AUC" in perf
    assert 0.0 <= perf["AUC"] <= 1.0


def test_predict_adds_target_probabilities() -> None:
    df = _tiny_training_frame()
    model = xgbClassifier(n_estimators=32, max_depth=2)
    model.fit(df, target_column=TARGET_COLUMN)
    infer = df.drop(columns=[TARGET_COLUMN]).head(5)
    out = model.predict(infer)

    assert TARGET_COLUMN in out.columns
    assert out[TARGET_COLUMN].between(0.0, 1.0).all()
    assert len(out) == 5
    assert TARGET_COLUMN not in infer.columns
