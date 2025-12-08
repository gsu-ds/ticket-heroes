import math
from pathlib import Path

import numpy as np
import pandas as pd
import wandb

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# CONFIG

PROJECT_NAME = "ticket_prophet_models" 

DATA_PATH = Path("data/processed/FINAL_market_panel.csv")

TARGET_COL = "avg_price"

FEATURE_COLS = [
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

TEST_SIZE = 0.20
VAL_SIZE = 0.15
RANDOM_STATE = 42


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file at {path.resolve()}")
    df = pd.read_csv(path)
    return df


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    test_size: float = 0.2,
    val_size: float = 0.15,
    random_state: int = 42,
):
    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # First split train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    # Then split train vs val
    val_rel_size = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_rel_size,
        random_state=random_state,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def compute_metrics(y_true, y_pred) -> dict:
    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}


# MAIN


def main():
    # 1) Load data
    df = load_data(DATA_PATH)

    # Keep only rows where features + target are all present
    cols_needed = FEATURE_COLS + [TARGET_COL]
    missing_in_df = [c for c in cols_needed if c not in df.columns]
    if missing_in_df:
        raise ValueError(f"These required columns are missing in the data: {missing_in_df}")

    df_clean = df[cols_needed].dropna().reset_index(drop=True)

    print(f"Loaded data from {DATA_PATH}")
    print(f"Total rows (after dropping NA): {len(df_clean)}")
    print("Columns used:")
    print("  Target:", TARGET_COL)
    print("  Features:", FEATURE_COLS)

    # 2) Split into train / val / test
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        df_clean,
        target_col=TARGET_COL,
        feature_cols=FEATURE_COLS,
        test_size=TEST_SIZE,
        val_size=VAL_SIZE,
        random_state=RANDOM_STATE,
    )

    # 3) Scale features for KNN (and any others that need scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4) Setup W&B
    # Make sure you have run `wandb login` once in your environment
    wandb.login()

    # 5) Define models
    models = {
        # Baseline
        "baseline_knn": KNeighborsRegressor(
            n_neighbors=5,
            weights="distance",
            metric="minkowski",
        ),
        # Tree-based models
        "decision_tree": DecisionTreeRegressor(
            max_depth=None,
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "xgboost": XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "catboost": CatBoostRegressor(
            depth=6,
            learning_rate=0.05,
            iterations=400,
            loss_function="RMSE",
            random_seed=RANDOM_STATE,
            verbose=False,
        ),
    }

    # Which models use scaled features
    use_scaled = {
        "baseline_knn": True,
        "decision_tree": False,
        "random_forest": False,
        "xgboost": False,
        "catboost": False,
    }

    # 6) Train & evaluate each model
    for model_name, model in models.items():
        print(f"\n==============================")
        print(f"Training model: {model_name}")
        print(f"==============================")

        if use_scaled[model_name]:
            Xtr, Xv, Xte = X_train_scaled, X_val_scaled, X_test_scaled
        else:
            Xtr, Xv, Xte = X_train, X_val, X_test

        run = wandb.init(
            project=PROJECT_NAME,
            name=model_name,
            reinit=True,
            config={
                "model_name": model_name,
                "features": FEATURE_COLS,
                "target": TARGET_COL,
                "train_size": len(Xtr),
                "val_size": len(Xv),
                "test_size": len(Xte),
                "test_split": TEST_SIZE,
                "val_split": VAL_SIZE,
                "random_state": RANDOM_STATE,
            },
        )

        # Fit model
        model.fit(Xtr, y_train)

        # Predict on val & test
        y_val_pred = model.predict(Xv)
        y_test_pred = model.predict(Xte)

        # Metrics
        val_metrics = compute_metrics(y_val, y_val_pred)
        test_metrics = compute_metrics(y_test, y_test_pred)

        print(
            f"{model_name} - VAL:  RMSE={val_metrics['rmse']:.3f}, "
            f"MAE={val_metrics['mae']:.3f}, R2={val_metrics['r2']:.3f}"
        )
        print(
            f"{model_name} - TEST: RMSE={test_metrics['rmse']:.3f}, "
            f"MAE={test_metrics['mae']:.3f}, R2={test_metrics['r2']:.3f}"
        )

        # Log metrics to W&B
        wandb.log(
            {
                "val_rmse": val_metrics["rmse"],
                "val_mae": val_metrics["mae"],
                "val_r2": val_metrics["r2"],
                "test_rmse": test_metrics["rmse"],
                "test_mae": test_metrics["mae"],
                "test_r2": test_metrics["r2"],
            }
        )

        # Log feature importance for tree-based models if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            for feat, imp in zip(FEATURE_COLS, importances):
                wandb.log({f"feature_importance/{model_name}/{feat}": float(imp)})

        run.finish()


if __name__ == "__main__":
    main()
