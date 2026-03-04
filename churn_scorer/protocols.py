from typing import Callable, Protocol, TypedDict
import pandas as pd


class FitPerformance(TypedDict):
    AUC: float


class BinaryClassifier(Protocol):
    def fit(self, training_df: pd.DataFrame, target_column: str) -> FitPerformance: ...

    def predict(self, df: pd.DataFrame) -> pd.DataFrame: ...


type ClassifierFactory = Callable[[], BinaryClassifier]
