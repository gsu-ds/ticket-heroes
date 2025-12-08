from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent.parent  # /Users/.../ticket_hero_project
DATA_PATH = Path("ticket-heroes/data/processed/FINAL_market_panel.csv")

print(f"Looking for CSV at: {DATA_PATH}")
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV NOT FOUND at: {DATA_PATH}")

# ---------- Load data ----------
df = pd.read_csv(DATA_PATH)
print("Loaded data:", df.shape)
print("Columns:", list(df.columns))

# ---------- Target & Features ----------
TARGET = "avg_price"  # <- this is the correct target column

FEATURES = [
    "year",
    "rank",
    "shows",
    "shows_lag1",
    "dma_rank",
    "dma_static_strength",
    "annual_sales_avg",
    "annual_sales_current",
    "gross_growth",
    "ticket_growth",
    "sales_momentum",
    "avg_ticket_price_dma",
    "price_premium_vs_dma",
    "avg_price_change",
]

X = df[FEATURES].copy()
y = df[TARGET]

# Handle missing values (RF cannot take NaN)
X = X.fillna(X.median(numeric_only=True))

# ---------- Train / test split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------- Train Random Forest ----------
rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    n_jobs=-1,
)

rf.fit(X_train, y_train)

# ---------- Evaluate ----------
y_pred = rf.predict(X_test)

rmse = math.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test RMSE: {rmse:.3f}")
print(f"Test MAE : {mae:.3f}")
print(f"Test R²   : {r2:.3f}")

# ---------- Save model ----------
MODEL_DIR = Path(__file__).resolve().parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "rf_model.joblib"
joblib.dump(rf, MODEL_PATH)

print(f"Saved Random Forest model to: {MODEL_PATH}")
