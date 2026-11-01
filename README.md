# Red Wine Quality Classification

## a. Problem Statement

This project compares five supervised classification algorithms for predicting whether a red wine is rated as good quality (`quality_label = 1`, original quality score at least 6) or lower quality (`quality_label = 0`, original score below 6). All classifiers use the same stratified 80/20 train/test split. A `StandardScaler` is fitted only on the training partition, and each model is evaluated on the held-out test partition using Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC).

The repository also includes an interactive Streamlit application for evaluating the saved models on labeled CSV data. It uses the saved scaler and classifiers; it does not retrain models on uploaded data.

## b. Dataset Description

The dataset is the UCI Machine Learning Repository's [Red Wine Quality dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv), containing 1,599 red wine samples. It has 11 original physicochemical measurements. Three domain-derived inputs are added:

- `total_acidity` = `fixed_acidity` + `volatile_acidity`
- `bound_sulfur_dioxide` = `total_sulfur_dioxide` − `free_sulfur_dioxide`
- `sugar_to_acidity_ratio` = `residual_sugar` / (`total_acidity` + 1e-5)

The resulting dataset has **14 input features** and one binary target, `quality_label`. The original `quality` column is removed after target creation. The cleaned dataset contains 1,599 rows with no missing values. Class counts are 744 for class 0 and 855 for class 1.

Preprocessing uses a reproducible stratified split with 1,279 training rows and 320 test rows. The exported `test_data.csv` contains the unscaled 14 test features and `quality_label`; the saved scaler was fitted using the training features only.

## c. GitHub Repository Link & Live Streamlit App Link

- **GitHub repository:** [GaneshT1133/bits-ml-assignment-2](https://github.com/GaneshT1133/bits-ml-assignment-2)
- **Live Streamlit app:** [Red Wine Quality Classifier Lab](https://bits-ml-assignment-2-32br9urkmmonneej96cxbn.streamlit.app/)

## d. Models Used & Comparison Table

The reported scores are recomputed from the saved model artifacts on the same stratified 320-row test set. AUC uses predicted probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7438 | 0.8239 | 0.7730 | 0.7368 | 0.7545 | 0.4874 |
| Decision Tree | 0.7531 | 0.7509 | 0.7614 | 0.7836 | 0.7723 | 0.5031 |
| KNN | 0.7438 | 0.8144 | 0.7697 | 0.7427 | 0.7560 | 0.4867 |
| Naive Bayes | 0.7125 | 0.7733 | 0.7484 | 0.6959 | 0.7212 | 0.4264 |
| Support Vector Machine | 0.7406 | 0.8373 | 0.7857 | 0.7076 | 0.7446 | 0.4853 |

## e. Observations Table on Model Performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Reached 74.38% accuracy and 0.8239 AUC, with a balanced precision/recall profile (0.7730 / 0.7368). It provides a strong, interpretable linear baseline. |
| Decision Tree | Had the highest accuracy (75.31%), recall (78.36%), F1 (0.7723), and MCC (0.5031) in this split. Its AUC (0.7509) is the lowest among these models, so its threshold ranking is weaker despite its best threshold-based scores. |
| KNN | Matched Logistic Regression's 74.38% accuracy and achieved 0.8144 AUC. Its precision and recall are close (0.7697 / 0.7427), resulting in a 0.7560 F1 score. |
| Naive Bayes | Scored 71.25% accuracy and 0.7212 F1, the lowest on both measures in this comparison. It is the weakest overall model on this particular split, though it still provides a useful probabilistic baseline. |
| Support Vector Machine | Achieved the best AUC (0.8373) and precision (0.7857), indicating the strongest probability ranking and positive-prediction precision here. Its recall is lower at 0.7076, and accuracy is 74.06%. |
| Overall Winner for your dataset? | **Decision Tree** is the winner if prioritizing the observed accuracy, recall, F1, and MCC. **Support Vector Machine** has the best AUC and precision. The preferred model depends on the evaluation priority; these results reflect one fixed stratified split, not a guarantee of generalization. |

---

### Run the project

Install dependencies from `requirements.txt`, then launch the Streamlit application with `streamlit run app.py`. By default, the app evaluates the repository's `test_data.csv`. You can instead upload a CSV containing the same 14 feature columns and the binary `quality_label` target.
