from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib


def main():
    root = Path(".")
    df = pd.read_parquet(root / "data/processed/dataset_ml.parquet")

    # ---------
    # Features (X)
    # ---------
    # Mantém só features numéricas: parâmetros dos injetores + tempo
    feature_cols = [c for c in df.columns if (
        ("_i" in c or "_j" in c or "_k1" in c or "_k2" in c or "rate" in c or "bhpmax" in c)
        and not c.startswith("_")
    )]
    feature_cols += ["t_days"]  # tempo como feature

    # garante que não entrou nada errado
    X = df[feature_cols].copy()
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        raise ValueError(f"Features não numéricas detectadas: {non_numeric}")

    # ---------
    # Targets (Y)
    # ---------
    target_cols = [c for c in df.columns if c.startswith(("WOPR_", "WWPR_", "WBHP_", "WWIR_"))]
    Y = df[target_cols].copy()

    non_numeric_y = [c for c in Y.columns if not np.issubdtype(Y[c].dtype, np.number)]
    if non_numeric_y:
        raise ValueError(f"Targets não numéricos detectados: {non_numeric_y}")

    # ---------
    # Split por caso (certo fisicamente)
    # ---------
    unique_cases = df["case_id"].unique()
    train_cases, test_cases = train_test_split(unique_cases, test_size=0.2, random_state=42)

    train_idx = df["case_id"].isin(train_cases)
    test_idx = df["case_id"].isin(test_cases)

    X_train, y_train = X.loc[train_idx], Y.loc[train_idx]
    X_test, y_test = X.loc[test_idx], Y.loc[test_idx]

    print("[INFO] Train cases:", sorted(train_cases.tolist()))
    print("[INFO] Test cases: ", sorted(test_cases.tolist()))
    print("[INFO] X_train:", X_train.shape, " y_train:", y_train.shape)
    print("[INFO] X_test: ", X_test.shape, " y_test: ", y_test.shape)

    # ---------
    # Modelo
    # ---------
    base = XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model = MultiOutputRegressor(base, n_jobs=-1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # ---------
    # Métricas agregadas
    # ---------
    mae = mean_absolute_error(y_test.values, preds)
    mse = mean_squared_error(y_test.values, preds)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_test.values, preds, multioutput="variance_weighted")

    print("\n[METRICS] MAE  (global):", float(mae))
    print("[METRICS] RMSE (global):", float(rmse))
    print("[METRICS] R2   (global):", float(r2))

    # ---------
    # Salvar
    # ---------
    out_dir = root / "models/saved_models"
    out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {"model": model, "feature_cols": feature_cols, "target_cols": target_cols},
        out_dir / "xgb_proxy_multioutput.joblib"
    )

    print("\n[OK] Saved model bundle to:", out_dir / "xgb_proxy_multioutput.joblib")


if __name__ == "__main__":
    main()