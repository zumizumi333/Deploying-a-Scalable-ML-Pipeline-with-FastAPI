import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


def process_data(
    X,
    categorical_features=None,
    label=None,
    training=True,
    encoder=None,
    lb=None,
):
    """Process categorical features and labels for training or inference.

    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and optional label.
    categorical_features : list[str]
        Names of categorical columns.
    label : str
        Name of the label column. If omitted, an empty label array is returned.
    training : bool
        Whether to fit new encoders or reuse the supplied encoders.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted encoder required when ``training`` is False.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted label binarizer required for labeled evaluation data.

    Returns
    -------
    X : np.ndarray
        Numeric features followed by one-hot encoded categorical features.
    y : np.ndarray
        Binarized labels, or an empty array for unlabeled inference data.
    encoder : sklearn.preprocessing.OneHotEncoder
        Fitted or reused categorical encoder.
    lb : sklearn.preprocessing.LabelBinarizer
        Fitted or reused label binarizer.
    """
    categorical_features = categorical_features or []

    if label is not None:
        y = X[label]
        X = X.drop(columns=[label])
    else:
        y = np.array([])

    X_categorical = X[categorical_features].values
    X_continuous = X.drop(columns=categorical_features).values

    if training:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        X_categorical = encoder.fit_transform(X_categorical)
        if label is not None:
            lb = LabelBinarizer()
            y = lb.fit_transform(y).ravel()
    else:
        if encoder is None:
            raise ValueError("A fitted encoder is required for inference.")
        X_categorical = encoder.transform(X_categorical)
        if label is not None:
            if lb is None:
                raise ValueError(
                    "A fitted label binarizer is required for evaluation."
                )
            y = lb.transform(y).ravel()

    X = np.concatenate([X_continuous, X_categorical], axis=1)
    return X, y, encoder, lb


def apply_label(predictions):
    """Convert a single binary prediction into its salary label."""
    if predictions[0] == 1:
        return ">50K"
    if predictions[0] == 0:
        return "<=50K"
    raise ValueError(f"Unexpected binary prediction: {predictions[0]!r}")
