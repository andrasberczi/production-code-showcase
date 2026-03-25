from datetime import datetime
from churn_scorer.scoring.service import ChurnScorer
from churn_scorer.pipeline.loader import load_table_from_uri
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Churn scoring", version="0.1.0")


class PredictRequest(BaseModel):
    table_uri: str = Field(
        ...,
        description="Path or URI to a CSV of rows to score (columns must match training).",
        examples=["data/sample_input_for_prediction.csv"],
    )


class PredictResponse(BaseModel):
    message: str
    prediction_results_uri: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    df = load_table_from_uri(body.table_uri)

    scorer = ChurnScorer.create()
    prediction_results_uri = scorer.predict(df)

    return PredictResponse(
        message=f"Prediction results written at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.",
        prediction_results_uri=prediction_results_uri,
    )
