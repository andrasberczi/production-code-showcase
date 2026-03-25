import pandas as pd


def load_table_from_uri(uri: str) -> pd.DataFrame:
    df = pd.read_csv(uri)
    return df
