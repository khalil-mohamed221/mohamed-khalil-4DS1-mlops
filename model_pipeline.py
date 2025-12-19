import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.exceptions import MlflowException

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from catboost import CatBoostRegressor


# ============================================================
# 1️⃣ PREPARE DATA
# ============================================================


def prepare_data():

    print("Loading dataset...")
    df = pd.read_csv("data/data.csv")

    # ---------------------------------------------
    # Clean date feature → extract year / month
    # ---------------------------------------------
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sale_year"] = df["date"].dt.year
    df["sale_month"] = df["date"].dt.month
    df.drop(columns=["date"], inplace=True)

    # ---------------------------------------------
    # Drop useless column
    # ---------------------------------------------
    if "country" in df.columns:
        df.drop(columns=["country"], inplace=True)

    # ---------------------------------------------
    # Fix yr_renovated (0 means never renovated)
    # Replace 0 with yr_built
    # ---------------------------------------------
    df["yr_renovated"] = df.apply(
        lambda row: (
            row["yr_built"] if row["yr_renovated"] == 0 else row["yr_renovated"]
        ),
        axis=1,
    )

    # ---------------------------------------------
    # HYBRID OUTLIER HANDLING (BEST OPTION)
    # ---------------------------------------------

    # 1) Remove price outliers (top 1%)
    upper_price = df["price"].quantile(0.99)
    df = df[df["price"] < upper_price]

    # 2) IQR clipping for SQFT features
    sqft_cols = ["sqft_living", "sqft_lot", "sqft_above", "sqft_basement"]

    for col in sqft_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        # Clip instead of removing rows
        df[col] = df[col].clip(lower, upper)

    print("Outliers handled (Hybrid Method).")

    # ------------------------------------------------------
    # SPLIT FEATURES (X) AND TARGET (y)
    # ------------------------------------------------------
    y = df["price"]
    X = df.drop(columns=["price"])

    # detect categorical columns (CatBoost handles them)
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    categorical_indices = [X.columns.get_loc(col) for col in categorical_cols]

    print("Categorical columns:", categorical_cols)
    print("Categorical indices:", categorical_indices)

    # ------------------------------------------------------
    # TRAIN-TEST SPLIT
    # ------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Train/Test split complete.")

    return X_train, X_test, y_train, y_test, categorical_indices


# ============================================================
# 2️⃣ TRAIN MODEL
# ============================================================


def train_model(X_train, y_train, categorical_indices):
    print("Training CatBoost model...")

    model = CatBoostRegressor(
        iterations=2000,
        depth=8,
        learning_rate=0.04,
        loss_function="RMSE",
        l2_leaf_reg=12,
        random_strength=2,
        border_count=128,
        early_stopping_rounds=100,
        verbose=200,
    )

    model.fit(X_train, y_train, cat_features=categorical_indices)

    print("Training complete.")
    return model


# ============================================================
# 3️⃣ EVALUATE MODEL
# ============================================================


def evaluate_model(model, X_test, y_test):

    print("Evaluating model...")

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse**0.5
    r2 = r2_score(y_test, predictions)

    print("MAE :", round(mae, 2))
    print("RMSE:", round(rmse, 2))
    print("R²  :", round(r2, 4))

    return mae, rmse, r2


# ============================================================
# 4️⃣ SAVE MODEL
# ============================================================


def save_model(model, path="models/house_price_model.cbm"):
    print("Saving model...")
    model.save_model(path)
    print("Model saved:", path)


# ============================================================
# 5️⃣ LOAD MODEL
# ============================================================


def load_model(path="models/house_price_model.cbm"):
    print("Loading model...")
    model = CatBoostRegressor()
    model.load_model(path)
    print("Model loaded.")
    return model


# ============================================================
# 6️⃣ RUN FULL PIPELINE WITH MLFLOW
# ============================================================


def run_pipeline_with_mlflow():
    """
    Full ML pipeline with Champion–Challenger logic.

    Behavior:
    - Always records the experiment in MLflow
    - Loads current Production model (Champion) if it exists
    - Trains a new model (Challenger)
    - Compares both on the SAME test set
    - Registers the Challenger ONLY if it is better
    - NEVER auto-promotes to Production (human approval required)
    """

    # ============================================================
    # 0️⃣ Set MLflow experiment
    # ============================================================
    mlflow.set_experiment("workshop5_experiment")

    # ============================================================
    # 1️⃣ Start MLflow run (EXPERIMENT IS ALWAYS RECORDED)
    # ============================================================
    with mlflow.start_run():

        # ============================================================
        # 2️⃣ Prepare data
        # ============================================================
        X_train, X_test, y_train, y_test, cat_idx = prepare_data()

        mlflow.log_params({"test_size": 0.2, "random_state": 42})

        # ============================================================
        # 3️⃣ Load Champion model (Production) from Registry
        # ============================================================
        champion_model = None
        champion_rmse = None

        try:
            print("🔎 Loading Production (Champion) model from registry...")
            champion_model = mlflow.pyfunc.load_model(
                "models:/house_price_model/Production"
            )
            print("✅ Champion model loaded.")
        except Exception:
            print("⚠️ No Production model found (first run scenario).")

        # ============================================================
        # 4️⃣ Evaluate Champion model (if exists)
        # ============================================================
        if champion_model is not None:
            print("📊 Evaluating Champion model...")
            champion_preds = champion_model.predict(X_test)

            champion_mse = mean_squared_error(y_test, champion_preds)
            champion_rmse = champion_mse**0.5

            print(f"🏆 Champion RMSE: {champion_rmse:.4f}")

        # ============================================================
        # 5️⃣ Train Challenger model
        # ============================================================
        print("🚀 Training Challenger model...")
        challenger_model = train_model(X_train, y_train, cat_idx)

        # Log ALL hyperparameters dynamically
        mlflow.log_params(challenger_model.get_params())

        # ============================================================
        # 6️⃣ Evaluate Challenger model
        # ============================================================
        print("📊 Evaluating Challenger model...")
        challenger_preds = challenger_model.predict(X_test)

        challenger_mse = mean_squared_error(y_test, challenger_preds)
        challenger_rmse = challenger_mse**0.5

        challenger_r2 = r2_score(y_test, challenger_preds)
        challenger_mae = mean_absolute_error(y_test, challenger_preds)

        print(f"🆕 Challenger RMSE: {challenger_rmse:.4f}")
        print(f"🆕 Challenger R²  : {challenger_r2:.4f}")
        print(f"🆕 Challenger MAE : {challenger_mae:.2f}")

        # ============================================================
        # 7️⃣ Champion vs Challenger decision
        # ============================================================
        is_better = False

        if champion_rmse is None:
            print("🏁 No Champion exists → Challenger becomes baseline.")
            is_better = True
        else:
            if challenger_rmse < champion_rmse:
                print("✅ Challenger is BETTER than Champion.")
                is_better = True
            else:
                print("❌ Challenger is NOT better than Champion.")

        # ============================================================
        # 8️⃣ Log comparison results to MLflow (AUDIT TRAIL)
        # ============================================================
        mlflow.log_metrics(
            {
                "champion_rmse": champion_rmse if champion_rmse is not None else -1,
                "challenger_rmse": challenger_rmse,
                "challenger_r2": challenger_r2,
                "challenger_mae": challenger_mae,
            }
        )

        mlflow.log_param("challenger_better_than_champion", is_better)

        # ============================================================
        # 9️⃣ Register Challenger model ONLY if it wins
        # ============================================================
        if is_better:
            print("📦 Registering Challenger model in Model Registry...")
            mlflow.sklearn.log_model(
                challenger_model,
                artifact_path="model",
                registered_model_name="house_price_model",
            )
            print("✅ Challenger registered (eligible for Staging / Promotion).")
        else:
            print("🚫 Challenger rejected. Champion remains in Production.")

        print("🎉 MLflow Champion–Challenger pipeline completed.")
