from churn_scorer.models.xgboost_model import xgbClassifier
from churn_scorer.protocols import BinaryClassifier


def model_factory() -> BinaryClassifier:
    return xgbClassifier()
