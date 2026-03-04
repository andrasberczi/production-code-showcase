from pathlib import Path

import pandas as pd
import pytest

from churn_scorer.feature_transformer import FeatureTransformer

SAMPLE_TRAIN = Path(__file__).resolve().parent / "data" / "test_data_for_train.csv"


@pytest.fixture
def transformer() -> FeatureTransformer:
    return FeatureTransformer()


def test_transform_preserves_row_count_and_drops_id(
    transformer: FeatureTransformer,
) -> None:
    df = pd.read_csv(SAMPLE_TRAIN)
    n = len(df)
    out = transformer.transform(df)
    assert len(out) == n
    assert "id" not in out.columns
