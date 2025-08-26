# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# ===========================
# Load Models
# ===========================
rf_model = joblib.load("models/random_forest.pkl")
xgb_model = joblib.load("models/xgboost.pkl")
scaler = joblib.load("models/scaler.pkl")

# ===========================
# Streamlit UI
# ===========================
st.title("📊 Walmart Sales Forecasting App")

st.sidebar.header("Input Features")
store = st.sidebar.number_input("Store ID", min_value=1, value=1)
dept = st.sidebar.number_input("Department ID", min_value=1, value=1)
year = st.sidebar.number_input("Year", min_value=2010, value=2012)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=5)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=10)

lag1 = st.sidebar.number_input("Lag (t-1)", value=20000)
lag7 = st.sidebar.number_input("Lag (t-7)", value=21000)
rolling4 = st.sidebar.number_input("Rolling Mean (4 weeks)", value=20500)

temp = st.sidebar.number_input("Temperature", value=75.0)
fuel = st.sidebar.number_input("Fuel Price", value=3.0)
cpi = st.sidebar.number_input("CPI", value=200.0)
unemp = st.sidebar.number_input("Unemployment", value=7.5)

# Prepare input
X_input = pd.DataFrame([[store, dept, year, month, day, lag1, lag7, rolling4, temp, fuel, cpi, unemp]],
                       columns=["Store","Dept","Year","Month","Day","Sales_lag_1","Sales_lag_7","Rolling_mean_4","Temperature","Fuel_Price","CPI","Unemployment"])

X_input_scaled = scaler.transform(X_input)

# Predictions
rf_pred = rf_model.predict(X_input_scaled)[0]
xgb_pred = xgb_model.predict(X_input_scaled)[0]

st.subheader("🔮 Forecasted Sales")
st.write(f"**Random Forest:** ${rf_pred:,.2f}")
st.write(f"**XGBoost:** ${xgb_pred:,.2f}")

# ===========================
# Bonus: Rolling Averages & Seasonal Decomposition
# ===========================
st.subheader("📈 Bonus: Time Series Analysis")

train = pd.read_csv("train.csv")
train["Date"] = pd.to_datetime(train["Date"])
store_data = train[train["Store"] == store].groupby("Date")["Weekly_Sales"].sum()

# Rolling average
st.line_chart(store_data.rolling(4).mean(), height=250)

# Seasonal decomposition
decomp = seasonal_decompose(store_data, model="additive", period=52)
fig, axes = plt.subplots(4, 1, figsize=(10, 8))
decomp.observed.plot(ax=axes[0], title="Observed")
decomp.trend.plot(ax=axes[1], title="Trend")
decomp.seasonal.plot(ax=axes[2], title="Seasonality")
decomp.resid.plot(ax=axes[3], title="Residuals")
plt.tight_layout()
st.pyplot(fig)
