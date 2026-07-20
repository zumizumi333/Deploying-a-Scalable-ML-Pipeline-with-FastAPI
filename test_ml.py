import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


CAT_FEATURES = ["workclass", "sex"]
LABEL = "salary"


@pytest.fixture(scope="module")
def sample_pipeline():
    """Return a small fitted pipeline and its source data."""
    data = pd.DataFrame(
        {
            "age": [25, 32, 38, 45, 51, 57, 63, 70],
            "workclass": [
                "Private",
                "Private",
                "State-gov",
                "State-gov",
                "Private",
                "State-gov",
                "Private",
                "State-gov",
            ],
            "sex": [
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
                "Female",
                "Male",
            ],
            "salary": [
                "<=50K",
                "<=50K",
                "<=50K",
                ">50K",
                ">50K",
                ">50K",
                ">50K",
                ">50K",
            ],
        }
    )
    X, y, encoder, label_binarizer = process_data(
        data,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    model = train_model(X, y)
    return data, X, y, encoder, label_binarizer, model


def test_process_data_reuses_encoder(sample_pipeline):
    """Inference data keeps its shape when it contains an unseen category."""
    _, X_train, _, encoder, label_binarizer, _ = sample_pipeline
    inference_data = pd.DataFrame(
        {
            "age": [40],
            "workclass": ["Never-worked"],
            "sex": ["Female"],
        }
    )

    X_inference, y, reused_encoder, reused_lb = process_data(
        inference_data,
        categorical_features=CAT_FEATURES,
        training=False,
        encoder=encoder,
    )

    assert X_inference.shape == (1, X_train.shape[1])
    assert y.size == 0
    assert np.isfinite(X_inference).all()
    assert reused_encoder is encoder
    assert reused_lb is None
    assert label_binarizer.classes_.tolist() == ["<=50K", ">50K"]


def test_train_model_and_inference(sample_pipeline):
    """Training returns the expected estimator and binary predictions."""
    _, X, y, _, _, model = sample_pipeline

    predictions = inference(model, X)

    assert isinstance(model, RandomForestClassifier)
    assert predictions.shape == y.shape
    assert set(np.unique(predictions)).issubset({0, 1})


def test_compute_model_metrics():
    """Precision, recall, and F1 match a known classification example."""
    labels = np.array([0, 1, 1, 0])
    predictions = np.array([0, 1, 0, 0])

    precision, recall, f1 = compute_model_metrics(labels, predictions)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert f1 == pytest.approx(2 / 3)


def test_save_and_load_model(sample_pipeline, tmp_path):
    """A loaded model produces the same predictions as the saved model."""
    _, X, _, _, _, model = sample_pipeline
    model_path = tmp_path / "nested" / "model.pkl"

    save_model(model, model_path)
    loaded_model = load_model(model_path)

    assert np.array_equal(inference(model, X), inference(loaded_model, X))


def test_performance_on_categorical_slice(sample_pipeline):
    """Slice evaluation returns three bounded classification metrics."""
    data, _, _, encoder, label_binarizer, model = sample_pipeline

    metrics = performance_on_categorical_slice(
        data,
        column_name="sex",
        slice_value="Female",
        categorical_features=CAT_FEATURES,
        label=LABEL,
        encoder=encoder,
        lb=label_binarizer,
        model=model,
    )

    assert len(metrics) == 3
    assert all(0.0 <= metric <= 1.0 for metric in metrics)
