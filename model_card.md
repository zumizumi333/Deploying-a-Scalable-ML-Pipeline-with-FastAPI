# Model Card

## Model Details

The model is a binary income classifier implemented with scikit-learn 1.5.1.
It uses a `RandomForestClassifier` with 50 trees, `random_state=42`, and one
worker. Six numeric features are used without scaling, while eight categorical
features are converted to 102 one-hot encoded columns. The model predicts
whether the `salary` label is `<=50K` or `>50K`. No hyperparameter search was
performed.

## Intended Use

This model is intended for demonstrating a reproducible machine learning
pipeline, categorical slice evaluation, and local inference through a FastAPI
service. It can be used for educational experiments on the supplied Census
Income data. It is not intended to determine employment, credit, insurance,
benefits, eligibility, or other decisions that affect an individual.

## Training Data

The supplied `census.csv` contains 32,561 records and 14 input features from the
UCI Adult dataset. UCI states that the source records were extracted from the
1994 Census database and that the task is to predict whether annual income
exceeds $50,000. The dataset is documented at
https://archive.ics.uci.edu/dataset/2/adult and has DOI
https://doi.org/10.24432/C5XW20.

The pipeline uses a stratified 80/20 split with `random_state=42`. The training
set contains 26,048 records, including 19,775 `<=50K` labels and 6,273 `>50K`
labels. The categorical features are fitted with a one-hot encoder configured
to ignore unseen values, and the target is fitted with a label binarizer. The
source file has no standard null values, but `?` appears in `workclass`,
`occupation`, and `native-country` and is retained as a categorical value.

## Evaluation Data

The held-out evaluation set contains 6,513 records, including 4,945 `<=50K`
labels and 1,568 `>50K` labels. It is transformed with the encoder and label
binarizer fitted only on the training set. This avoids fitting preprocessing
state on evaluation data. Performance is also evaluated for every value found
in each of the eight categorical features, and those results are stored in
`slice_output.txt`.

## Metrics

Precision measures the proportion of predicted `>50K` labels that are correct.
Recall measures the proportion of actual `>50K` labels found by the model. F1
is the harmonic mean of precision and recall. On the held-out evaluation set,
the model achieves precision of 0.7285, recall of 0.6263, and F1 of 0.6735.

## Ethical Considerations

The data includes sensitive and demographic attributes such as age, race, sex,
marital status, and relationship. Patterns learned from historical Census data
may reflect social inequality, sampling limitations, and discrimination rather
than fair or causal relationships. Applying this model to people could
reproduce or amplify those patterns. The model does not establish why a person
has a particular income and should not be interpreted as a measure of ability,
merit, or future potential.

## Caveats and Recommendations

The source data is historical, the $50,000 threshold is not adjusted for
inflation, and the positive class represents only about 24 percent of the
supplied records. Missing categorical values represented by `?` are treated as
ordinary categories. The model was evaluated with one random holdout split and
was not tuned, calibrated, or externally validated. Metrics from very small
slices are unstable, and the metric implementation assigns a value of 1 when a
precision or recall denominator is zero, so apparently perfect results on tiny
slices require careful interpretation.

Before any use beyond this project, the data and income threshold should be
updated, repeated or cross-validated evaluation should be performed, and error
rates should be compared across protected groups. Probability calibration,
uncertainty reporting, human review, privacy assessment, and monitoring for
data drift would also be required for a production system.
