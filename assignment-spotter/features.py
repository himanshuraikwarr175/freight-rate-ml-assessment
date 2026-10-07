import pandas as pd

train = pd.read_csv(
    "processed/train_clean.csv",
    parse_dates=["date"]
)

# Date features
train["month"] = train["date"].dt.month
train["day_of_week"] = train["date"].dt.dayofweek
train["day_of_month"] = train["date"].dt.day
train["week_of_year"] = train["date"].dt.isocalendar().week.astype(int)
train["is_weekend"] = (train["day_of_week"] >= 5).astype(int)

# Lane
train["lane"] = (
    train["pickup"].astype(str) + "_" + train["delivery"].astype(str)
)

# Weight per mile
train["weight_per_mile"] = (
    train["weight"] / train["distance"].replace(0, pd.NA)
)

# Keep original equipment for baseline
train["equipment_name"] = train["equipment"]

# One-hot encode equipment
train = pd.get_dummies(
    train,
    columns=["equipment"],
    dtype=int
)

train.to_csv(
    "processed/train_features.csv",
    index=False
)

print("Feature data shape:", train.shape)
print(train.head())