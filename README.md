# TrustGuard: Financial Fraud Detection

![Python](https://img.shields.io/badge/python-3.11-blue)
![LightGBM](https://img.shields.io/badge/LightGBM-4.5-brightgreen)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.5-orange)
![Metric](https://img.shields.io/badge/metric-PR--AUC-purple)

Our entry for **DLWeek Datathon 2026, Track 2 (Finance)**: estimate the probability that a transaction is fraudulent.
The final model is a **gradient-boosted LightGBM classifier with isotonic calibration**, chosen after testing
27 model variants against a deliberately shifted test set.

## Highlights

- **Imbalanced classification done properly.** Fraud is only 1.8% of transactions, so every decision is judged on
  PR-AUC (ranking of the rare fraud cases) and Brier score (probability quality), never accuracy.
- **LightGBM gradient boosting, regularised for small data.** Shallow trees (4 leaves), large minimum leaf size,
  row/column subsampling and L2 regularisation stop the model memorising the ~350 fraud cases.
- **Calibrated probabilities.** Isotonic calibration turns boosting scores into real fraud rates, so a bank can set
  its alert threshold directly from costs.
- **Rigorous validation.** 5-fold stratified cross-validation shared by every model, a train-vs-CV overfitting check,
  hyperparameter search on separate folds, and benchmarks against logistic regression, random forest, KNN and SVM.
- **Robustness under covariate shift.** The test set has twice the transaction sizes, more new devices, older accounts
  and three times the missing values. We diagnosed the shift and tested fixes (shift weighting, test-like missingness,
  monotone constraints, pseudo-labelling, feature ablation) to find the model that transfers best.

## Results

| Model | 5-fold CV PR-AUC | Public leaderboard PR-AUC |
|---|---|---|
| Random baseline | 0.018 | n/a |
| Logistic regression (ElasticNet) | 0.225 | 0.160 |
| Random forest (tuned) | 0.218 | 0.146 |
| Blend: logistic + calibrated LightGBM | **0.228** | 0.179 |
| Calibrated LightGBM, all 17 features | 0.225 | 0.189 |
| **Calibrated LightGBM, raw features (final)** | 0.215 | **0.193** |

**Key insight:** the blend won in cross-validation but not on the shifted test set. The linear model degraded most
under the shift, and our engineered ratio/threshold features transferred less well than the raw columns, so the final
model is the calibrated LightGBM on the 10 raw columns. The final submission passed the competition's sandboxed
reproducibility run.

## Approach

1. **Exploratory analysis.** Fraud concentrates in large payments from new devices (16.7% fraud), bursts of 4+
   transactions in an hour (16.1%), accounts younger than 180 days, and late-night transactions.
2. **Data processing.** Missing values are kept as information rather than dropped (rows with gaps are riskier),
   impossible values are treated as missing, and the identifier is excluded to prevent leakage.
3. **Feature engineering.** Cyclical hour encoding, spend relative to the account's recent average, activity bursts,
   account-age flags and a new-device × amount interaction. Kept for analysis; the final model uses raw columns.
4. **Modelling.** ElasticNet logistic regression baseline and LightGBM with isotonic calibration (scikit-learn
   `CalibratedClassifierCV`). Class weighting was tested and rejected: it lowered PR-AUC and inflated probabilities.
5. **Decision threshold.** For deployment we recommend the F2-optimal threshold (recall weighted over precision),
   sending flagged transactions to step-up verification such as an OTP rather than blocking them.

## Repository

| File | Description |
|---|---|
| [`TrustGuard_Fraud_Detection.ipynb`](TrustGuard_Fraud_Detection.ipynb) | Full pipeline with outputs: EDA, features, training, evaluation, overfitting check, benchmarks, threshold analysis, final model |
| [`prediction_notebook.ipynb`](prediction_notebook.ipynb) | Inference only: loads `model.pkl`, rebuilds the features and predicts (the notebook run by the competition sandbox) |
| `model.pkl` | Trained final model, saved with `joblib` as `(model, feature_list)` |
| `submission.csv` | Final test-set predictions (`id`, `prediction`) |
| `requirements.txt` | Library versions the model was trained with |

## Run it yourself

```bash
pip install -r requirements.txt
```

The competition data is not included in this repository. Place it next to the notebooks, then:

- **Retrain:** run `TrustGuard_Fraud_Detection.ipynb` top to bottom (Colab or Jupyter). It expects
  `Track 2 Training Dataset.csv` and `Track 2 Testing Dataset.csv` and writes `model.pkl` and `submission.csv`.
- **Predict only:** run `prediction_notebook.ipynb`. It reads `test.csv` (or the path in `DATATHON_INPUT_PATH`)
  and writes `predictions.csv`.

## Limitations and next steps

- Every model reached a similar ceiling (~0.22 CV PR-AUC), so further gains need richer data such as customer
  history, merchant identifiers or device fingerprints, rather than a more complex model.
- Large transactions, common in the test set, are learned from only 454 training rows; we would retrain or reweight
  as labelled data from that segment arrives.
- With only 353 fraud cases, scores vary by ±0.02-0.03 between folds, so smaller differences were treated as noise.
