import joblib
import pandas as pd

model = joblib.load("processed/model.pkl")
FEATURES = list(model.feature_names_in_)

def prepare(df):
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["weight"] = out["weight"].fillna(31493.5)
    out["market_index"] = out["market_index"].fillna(1.0558)
    out["month"] = out["date"].dt.month
    out["day_of_week"] = out["date"].dt.dayofweek
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
    dummies = pd.get_dummies(out["equipment"])
    out["equipment_Dry_Van"] = dummies["Dry Van"].astype(int) if "Dry Van" in dummies else 0
    out["equipment_Reefer"] = dummies["Reefer"].astype(int) if "Reefer" in dummies else 0
    out["equipment_Flatbed"] = dummies["Flatbed"].astype(int) if "Flatbed" in dummies else 0
    return out

validation = prepare(pd.read_csv("validation.csv"))
pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": model.predict(validation[FEATURES]),
}).to_csv("validation_predictions.csv", index=False)

december = pd.read_csv("december-chart-inputs.csv")
december["pickup_lat"] = 36.99152
december["pickup_lon"] = -84.99876
december["delivery_lat"] = 41.31561
december["delivery_lon"] = -85.36206
december["market_index"] = 1.04402
december["quote_signal"] = 1.97252
december = prepare(december)
filled = december[["pickup", "delivery", "distance", "equipment", "weight", "date"]].copy()
filled["date"] = filled["date"].dt.strftime("%Y-%m-%d")
filled["predicted_rate"] = model.predict(december[FEATURES])
filled.to_csv("december-chart-inputs.csv", index=False)

print("validation rows", len(validation))
print("december rows", len(filled))