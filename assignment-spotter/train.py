import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error
from columns import FEATURES


# 1. Load data
fit = pd.read_csv(
    "processed/fit.csv",
    parse_dates=["date"]
)

check = pd.read_csv(
    "processed/check.csv",
    parse_dates=["date"]
)

full = pd.read_csv(
    "processed/train_features.csv",
    parse_dates=["date"]
)


# 2. Baseline prediction ( Historical median rate-per-mile for each equipment type)

baseline_rpm = (
    (fit["posted_rate"] / fit["distance"])
    .groupby(fit["equipment_name"])
    .median()
)

baseline = (
    check["distance"]
    * check["equipment_name"].map(baseline_rpm)
)


# 3. Train LightGBM model

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

model.fit(
    fit[FEATURES],
    fit["posted_rate"]
)


# 4. Predict validation data

pred = model.predict(
    check[FEATURES]
)


# 5.  Evaluate

baseline_mae = mean_absolute_error(
    check["posted_rate"],
    baseline
)

model_mae = mean_absolute_error(
    check["posted_rate"],
    pred
)

print("Baseline MAE:", round(baseline_mae, 2))
print("Model MAE:", round(model_mae, 2))


# 6. Train final model on all development data

final = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42
)

final.fit(
    full[FEATURES],
    full["posted_rate"]
)


# 7 Save model

joblib.dump(
    final,
    "processed/model.pkl"
)

print("Saved: processed/model.pkl")