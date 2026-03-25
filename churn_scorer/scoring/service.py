from __future__ import annotations

import pandas as pd

from churn_scorer.config import TARGET_COLUMN, results_uri, training_data_uri
from churn_scorer.protocols import BinaryClassifier
from churn_scorer.pipeline.transformer import FeatureTransformer
from churn_scorer.models.factory import model_factory
from churn_scorer.pipeline.loader import load_table_from_uri


class ChurnScorer:
    def __init__(
        self,
        feature_transformer: FeatureTransformer,
        model: BinaryClassifier,
    ):
        self.feature_transformer = feature_transformer
        self.model = model

    @classmethod
    def create(cls) -> ChurnScorer:
        return ChurnScorer(
            feature_transformer=FeatureTransformer(),
            model=model_factory(),
        )

    def predict(self, df: pd.DataFrame) -> str:
        prediction_results_uri = results_uri()

        if self._is_training_needed():
            training_df = load_table_from_uri(training_data_uri())
            training_df = self.feature_transformer.target_to_binary(
                training_df, TARGET_COLUMN
            )
            training_df_with_features = self.feature_transformer.transform(training_df)
            self.model.fit(training_df_with_features, target_column=TARGET_COLUMN)

        df_with_features = self.feature_transformer.transform(df)
        prediction_df = self.model.predict(df_with_features)
        prediction_df.to_csv(prediction_results_uri, index=False)

        return prediction_results_uri

    def _is_training_needed(self) -> bool:
        # "Fake" function, in real life
        # would be some logic checking if model is outdated
        # or not existing, etc
        return True
