
# Freight Rate Prediction

Predict the posted rate for a truck load from lane, equipment, distance, weight, and date. The development file is `train-test.csv` (1 January 2025 through 31 October 2025). Final scores are written for every load in `validation.csv`.

## Setup

From this folder:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

`train-test.csv` and `validation.csv` must sit in this folder. They are the assessment inputs and are not part of the model code.

## Run

Run these in order, from this folder:

```bash
python assignment-spotter/clean.py
python assignment-spotter/features.py
python assignment-spotter/split.py
PYTHONPATH=assignment-spotter python assignment-spotter/train.py
python predict.py
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

What each step does:

| Step | File | Output |
| --- | --- | --- |
| Clean | `assignment-spotter/clean.py` | `processed/train_clean.csv` |
| Features | `assignment-spotter/features.py` | `processed/train_features.csv` |
| Split | `assignment-spotter/split.py` | `processed/fit.csv`, `processed/check.csv` |
| Train | `assignment-spotter/train.py` | `processed/model.pkl`, check-set MAE |
| Predict | `predict.py` | `validation_predictions.csv` and filled `december-chart-inputs.csv` |
| Score | `score.py` | `scorer_results/candidate_december.png` |

## Split and model

The split is by date, because rates move over time.

- Fit: 1 January 2025 through 31 August 2025 (38,477 loads).
- Check: 1 September 2025 through 31 October 2025 (9,523 loads).

The baseline is distance times the fit-set median rate per mile for that equipment type. On the check set its MAE is $229.1. LightGBM (`500` trees, learning rate `0.05`, `31` leaves) reaches an MAE of $118.6. The saved model is retrained on all development rows, then applied to `validation.csv`.

Cleaning drops duplicate `load_id` values, treats negative weight, negative distance, and invalid coordinates as missing, and fills numeric gaps with the training median.

## Submission files

- `validation_predictions.csv` holds `load_id,predicted_rate` for all 12,000 validation loads.
- `freight-rate-report.pdf` describes the split, the check-set error, and the December chart.