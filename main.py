from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from ml.data import apply_label, process_data
from ml.model import inference, load_model


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
ENCODER_PATH = PROJECT_ROOT / "model" / "encoder.pkl"

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


class Data(BaseModel):
    """Census record accepted by the inference endpoint."""

    model_config = ConfigDict(populate_by_name=True)

    age: int = Field(..., examples=[37])
    workclass: str = Field(..., examples=["Private"])
    fnlgt: int = Field(..., examples=[178356])
    education: str = Field(..., examples=["HS-grad"])
    education_num: int = Field(..., examples=[10], alias="education-num")
    marital_status: str = Field(
        ...,
        examples=["Married-civ-spouse"],
        alias="marital-status",
    )
    occupation: str = Field(..., examples=["Prof-specialty"])
    relationship: str = Field(..., examples=["Husband"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(..., examples=[0], alias="capital-gain")
    capital_loss: int = Field(..., examples=[0], alias="capital-loss")
    hours_per_week: int = Field(..., examples=[40], alias="hours-per-week")
    native_country: str = Field(
        ...,
        examples=["United-States"],
        alias="native-country",
    )


class Prediction(BaseModel):
    """Income prediction returned by the API."""

    result: str


encoder = load_model(ENCODER_PATH)
model = load_model(MODEL_PATH)

app = FastAPI(
    title="Census Income Classifier",
    version="1.0.0",
)


@app.get("/", response_model=str)
async def get_root():
    """Return a welcome message."""
    return "Hello from the API!"


@app.post("/data/", response_model=Prediction)
async def post_inference(data: Data):
    """Predict whether the supplied record belongs to the >50K class."""
    record = pd.DataFrame([data.model_dump(by_alias=True)])
    processed_record, _, _, _ = process_data(
        record,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
    )
    prediction = inference(model, processed_record)
    return Prediction(result=apply_label(prediction))
