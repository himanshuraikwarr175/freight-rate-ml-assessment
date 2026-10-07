import pandas as pd
from pathlib import Path


#  1. Create processed directory

Path("processed").mkdir(exist_ok=True)


# 2. Load data

train = pd.read_csv("train-test.csv",parse_dates=["date"])

print("Original shape:", train.shape)


#  3. Remove duplicate load IDs

train = train.drop_duplicates(subset=["load_id"])


# 4. Handle invalid numeric values

# Weight cannot be negative
train.loc[train["weight"] < 0, "weight"] = pd.NA

# Distance cannot be negative
train.loc[train["distance"] < 0, "distance"] = pd.NA

# Coordinates should be within valid ranges
train.loc[
    ~train["pickup_lat"].between(-90, 90),
    "pickup_lat"
] = pd.NA

train.loc[
    ~train["delivery_lat"].between(-90, 90),
    "delivery_lat"
] = pd.NA

train.loc[
    ~train["pickup_lon"].between(-180, 180),
    "pickup_lon"
] = pd.NA

train.loc[
    ~train["delivery_lon"].between(-180, 180),
    "delivery_lon"
] = pd.NA




#5. Numerical columns

numeric_columns = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight",
    "market_index",
    "quote_signal",
]

# Calculate medians
medians = {}

for column in numeric_columns:
    medians[column] = train[column].median()
    train[column] = train[column].fillna(medians[column])


#6. Categorical columns


categorical_columns = [
    "pickup",
    "delivery",
    "equipment",
]

for column in categorical_columns:
    train[column] = train[column].fillna("Unknown")


# 7. Date validation

train["date"] = pd.to_datetime(
    train["date"],
    errors="coerce"
)

print("Missing dates:", train["date"].isna().sum())



# 8. Save cleaned dataset


train.to_csv(
    "processed/train_clean.csv",
    index=False
)




# 9. Final checks


print("\nCleaned shape:", train.shape)

print("\nMissing values:")
print(train.isna().sum())


print("\nCleaning completed.")