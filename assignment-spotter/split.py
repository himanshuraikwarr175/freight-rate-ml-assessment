import pandas as pd

train = pd.read_csv(
    "processed/train_features.csv",
    parse_dates=["date"]
)

fit = train[train["date"] < "2025-09-01"].copy()
check = train[train["date"] >= "2025-09-01"].copy()

fit.to_csv("processed/fit.csv", index=False)
check.to_csv("processed/check.csv", index=False)

print("Total:", len(train))
print("Fit:", len(fit))
print("Check:", len(check))

print("\nFit dates:")
print(fit["date"].min(), "to", fit["date"].max())

print("\nCheck dates:")
print(check["date"].min(), "to", check["date"].max())