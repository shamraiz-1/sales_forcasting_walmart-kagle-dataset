# train_eval.py
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# ===========================
# 1. Load Data
# ===========================
train = pd.read_csv("train.csv")
features = pd.read_csv("features.csv")
stores = pd.read_csv("stores.csv")

# Merge datasets
df = train.merge(features, on=["Store", "Date"], how="left")
df = df.merge(stores, on="Store", how="left")

# Convert date
df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day"] = df["Date"].dt.day

# Sort by date
df = df.sort_values("Date")

# ===========================
# 2. Feature Engineering
# ===========================
df["Sales_lag_1"] = df.groupby("Store")["Weekly_Sales"].shift(1)
df["Sales_lag_7"] = df.groupby("Store")["Weekly_Sales"].shift(7)
df["Rolling_mean_4"] = df.groupby("Store")["Weekly_Sales"].transform(lambda x: x.rolling(4).mean())

df = df.dropna()

X = df[["Store", "Dept", "Year", "Month", "Day", "Sales_lag_1", "Sales_lag_7", "Rolling_mean_4", "Temperature", "Fuel_Price", "CPI", "Unemployment"]]
y = df["Weekly_Sales"]

# ===========================
# 3. Train / Test Split
# ===========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ===========================
# 4. Train Models
# ===========================
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

xgb_model = XGBRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# ===========================
# 5. Evaluate
# ===========================
for name, model in [("Random Forest", rf_model), ("XGBoost", xgb_model)]:
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"{name} → RMSE: {rmse:.2f}, R²: {r2:.2f}")

# ===========================
# 6. Save Models
# ===========================
os.makedirs("models", exist_ok=True)
joblib.dump(rf_model, "models/random_forest.pkl")
joblib.dump(xgb_model, "models/xgboost.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Models & scaler saved in 'models/' folder")
