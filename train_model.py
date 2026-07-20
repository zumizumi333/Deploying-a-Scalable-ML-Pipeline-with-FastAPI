from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "census.csv"
MODEL_PATH = PROJECT_ROOT / "model" / "model.pkl"
ENCODER_PATH = PROJECT_ROOT / "model" / "encoder.pkl"
LABEL_BINARIZER_PATH = PROJECT_ROOT / "model" / "label_binarizer.pkl"
SLICE_OUTPUT_PATH = PROJECT_ROOT / "slice_output.txt"
LABEL = "salary"
RANDOM_STATE = 42
TEST_SIZE = 0.2

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


def write_slice_performance(test, encoder, label_binarizer, model):
    """Write metrics for every value of each categorical feature."""
    with SLICE_OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
        for column in CAT_FEATURES:
            for slice_value in sorted(test[column].unique()):
                count = int((test[column] == slice_value).sum())
                precision, recall, f1 = performance_on_categorical_slice(
                    test,
                    column_name=column,
                    slice_value=slice_value,
                    categorical_features=CAT_FEATURES,
                    label=LABEL,
                    encoder=encoder,
                    lb=label_binarizer,
                    model=model,
                )
                print(
                    f"{column}: {slice_value}, Count: {count:,}",
                    file=output_file,
                )
                print(
                    "Precision: "
                    f"{precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}",
                    file=output_file,
                )


def main():
    """Train, evaluate, and persist the Census income classifier."""
    data = pd.read_csv(DATA_PATH)
    train, test = train_test_split(
        data,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=data[LABEL],
    )

    X_train, y_train, encoder, label_binarizer = process_data(
        train,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=True,
    )
    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=CAT_FEATURES,
        label=LABEL,
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )

    model = train_model(X_train, y_train)
    save_model(model, MODEL_PATH)
    save_model(encoder, ENCODER_PATH)
    save_model(label_binarizer, LABEL_BINARIZER_PATH)
    print(f"Model saved to {MODEL_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Encoder saved to {ENCODER_PATH.relative_to(PROJECT_ROOT)}")
    print(
        "Label binarizer saved to "
        f"{LABEL_BINARIZER_PATH.relative_to(PROJECT_ROOT)}"
    )

    saved_model = load_model(MODEL_PATH)
    predictions = inference(saved_model, X_test)
    precision, recall, f1 = compute_model_metrics(y_test, predictions)
    print(
        f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    )

    write_slice_performance(test, encoder, label_binarizer, saved_model)
    print(f"Slice metrics saved to {SLICE_OUTPUT_PATH.relative_to(PROJECT_ROOT)}")

    return precision, recall, f1


if __name__ == "__main__":
    main()
