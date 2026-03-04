import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class FeatureTransformer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_with_features = df.copy()
        df_with_features = self._features_only(df_with_features)
        df_with_features = self._transform_features(df_with_features)
        df_with_features = self._group_numerical_features(df_with_features)

        categorical_columns = df_with_features.select_dtypes(include=["object"]).columns.tolist()

        df_with_features = self._one_hot_encode(df_with_features, categorical_columns)

        return df_with_features

    def _transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["gender"] = np.where(df["gender"] == "Male", 1, 0)
        df["Partner"] = np.where(df["Partner"] == "Yes", 1, 0)
        df["Dependents"] = np.where(df["Dependents"] == "Yes", 1, 0)
        df["PhoneService"] = np.where(df["PhoneService"] == "Yes", 1, 0)
        df["PaperlessBilling"] = np.where(df["PaperlessBilling"] == "Yes", 1, 0)
        return df

    def _group_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["tenureQuantiles"] = pd.qcut(
            df["tenure"],
            q=4,
            labels=["quantile1", "quantile2", "quantile3", "quantile4"],
        )
        df["tenureRoundedToTens"] = np.round(df["tenure"], -1)
        df["MonthlyChargesRoundedToTens"] = np.round(df["MonthlyCharges"], -1)
        df["TotalChargesRoundedToTens"] = np.round(df["TotalCharges"], -1)
        df["TotalChargesRoundedToHundreds"] = np.round(df["TotalCharges"], -2)
        df["TotalChargesRoundedToThousands"] = np.round(df["TotalCharges"], -3)
        return df

    def _one_hot_encode(self, df: pd.DataFrame, categorical_columns: list[str]) -> pd.DataFrame:
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded_columns = encoder.fit_transform(df[categorical_columns])
        one_hot_df = pd.DataFrame(one_hot_encoded_columns, columns=encoder.get_feature_names_out())
        df_encoded = pd.concat([df, one_hot_df], axis=1)
        df_encoded = df_encoded.drop(categorical_columns, axis=1)
        return df_encoded

    def _features_only(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["id"]).copy()

    def target_to_binary(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        df[target_column] = np.where(df[target_column] == "Yes", 1, 0)
        return df
