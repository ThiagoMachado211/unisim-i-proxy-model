from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.datasets.build_ml_dataset import main as run_build_ml
from src.datasets.build_tabular_dataset import main as run_build_tabular
from src.extract.build_inventory import main as run_inventory
from src.extract.extract_summary_timeseries import main as run_extract_summary
from src.extract.parse_cases_parameters import main as run_parse_cases
from src.validation.qc_and_standardize_summary import main as run_qc

FEATURE_PATTERNS = ["_i", "_j", "_k1", "_k2", "rate", "bhpmax"]
TARGET_PREFIXES = ("WOPR_", "WWPR_", "WBHP_", "WWIR_")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_feature_target_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    feature_cols = [c for c in df.columns if any(p in c for p in FEATURE_PATTERNS) and not c.startswith("_")]
    feature_cols += ["t_days"]
    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIXES)]
    return feature_cols, target_cols


def compute_global_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))
    denom = np.maximum(np.abs(y_true), 1e-9)
    mape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_pct": mape}


def compute_metrics_by_variable(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in y_true_df.columns:
        yt = y_true_df[col].values
        yp = y_pred_df[col].values
        rows.append(
            {
                "target": col,
                "MAE": float(mean_absolute_error(yt, yp)),
                "RMSE": float(np.sqrt(mean_squared_error(yt, yp))),
                "R2": float(r2_score(yt, yp)),
                "MAPE_pct": float(np.mean(np.abs(yt - yp) / np.maximum(np.abs(yt), 1e-9)) * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values("RMSE", ascending=False).reset_index(drop=True)


def compute_metrics_by_case(case_ids: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for case_id in sorted(np.unique(case_ids)):
        mask = case_ids == case_id
        rows.append({"case_id": int(case_id), **compute_global_metrics(y_true[mask], y_pred[mask])})
    return pd.DataFrame(rows).sort_values("RMSE", ascending=False).reset_index(drop=True)


def build_model(model_name: str, random_state: int = 42):
    model_name = model_name.lower()
    if model_name == "rf_tabular":
        return MultiOutputRegressor(
            RandomForestRegressor(
                n_estimators=400,
                max_depth=18,
                min_samples_leaf=1,
                random_state=random_state,
                n_jobs=-1,
            ),
            n_jobs=-1,
        )

    if model_name == "mlp_tabular":
        base = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        alpha=1e-4,
                        learning_rate="adaptive",
                        learning_rate_init=1e-3,
                        max_iter=1500,
                        early_stopping=True,
                        validation_fraction=0.1,
                        n_iter_no_change=30,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        return MultiOutputRegressor(base, n_jobs=-1)

    if model_name == "xgb_tabular":
        from xgboost import XGBRegressor

        return MultiOutputRegressor(
            XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            ),
            n_jobs=-1,
        )

    raise ValueError(f"Modelo não suportado: {model_name}")


def generate_plots(df_eval: pd.DataFrame, target_cols: list[str], case_metrics: pd.DataFrame, figures_dir: Path) -> None:
    figures_dir = ensure_dir(figures_dir)
    preferred_targets = ["WOPR_NA1A", "WWPR_NA1A", "WWIR_I1", "WBHP_NA1A"]
    chosen_targets = [c for c in preferred_targets if c in target_cols] or target_cols[:4]

    ordered_cases = case_metrics["case_id"].tolist()
    cases_to_plot = [ordered_cases[0], ordered_cases[len(ordered_cases)//2], ordered_cases[-1]] if len(ordered_cases) >= 3 else ordered_cases

    for case_id in cases_to_plot:
        dcase = df_eval[df_eval["case_id"] == case_id].sort_values("t_days")
        t = dcase["t_days"].values
        for target in chosen_targets:
            plt.figure()
            plt.plot(t, dcase[f"{target}__true"].values, label="Simulador")
            plt.plot(t, dcase[f"{target}__pred"].values, label="Proxy")
            plt.xlabel("t_days")
            plt.ylabel(target)
            plt.title(f"CASE_{int(case_id):02d} - {target}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(figures_dir / f"curve_CASE_{int(case_id):02d}_{target}.png", dpi=160)
            plt.close()

    for target in chosen_targets:
        yt = df_eval[f"{target}__true"].values
        yp = df_eval[f"{target}__pred"].values
        vmin = float(min(np.min(yt), np.min(yp)))
        vmax = float(max(np.max(yt), np.max(yp)))
        plt.figure()
        plt.scatter(yt, yp, s=8, alpha=0.5)
        plt.plot([vmin, vmax], [vmin, vmax])
        plt.xlabel("Real (Simulador)")
        plt.ylabel("Predito (Proxy)")
        plt.title(f"Scatter Real vs Predito - {target}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"scatter_{target}.png", dpi=160)
        plt.close()

    tmp = df_eval[["t_days"]].copy()
    tmp["t_bin"] = np.round(tmp["t_days"], 3)
    for target in chosen_targets:
        tmp[f"ae_{target}"] = np.abs(df_eval[f"{target}__true"].values - df_eval[f"{target}__pred"].values)
    grouped = tmp.groupby("t_bin").mean(numeric_only=True).reset_index()

    for target in chosen_targets:
        plt.figure()
        plt.plot(grouped["t_bin"].values, grouped[f"ae_{target}"].values)
        plt.xlabel("t_days")
        plt.ylabel("MAE")
        plt.title(f"MAE ao longo do tempo - {target}")
        plt.tight_layout()
        plt.savefig(figures_dir / f"mae_over_time_{target}.png", dpi=160)
        plt.close()


def train_and_validate_proxy(project_root: str | Path = ".", model_name: str = "rf_tabular", output_dir: str | Path = "outputs/rf_tabular", n_splits: int = 5, random_state: int = 42) -> dict:
    project_root = Path(project_root)
    output_dir = ensure_dir(project_root / output_dir)
    figures_dir = ensure_dir(output_dir / "figures")

    df = pd.read_parquet(project_root / "data" / "processed" / "dataset_ml.parquet")
    feature_cols, target_cols = build_feature_target_columns(df)
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    groups = df["case_id"].values

    gkf = GroupKFold(n_splits=n_splits)
    all_true, all_pred = [], []
    all_case_ids, all_t_days, all_dates = [], [], []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        case_ids_test = df.iloc[test_idx]["case_id"].values

        model = build_model(model_name=model_name, random_state=random_state)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_test)
        predict_time = time.perf_counter() - t1

        y_pred_df = pd.DataFrame(preds, columns=target_cols, index=y_test.index)
        y_true_df = y_test.copy()

        fold_rows.append({"fold": fold, "fit_time_sec": fit_time, "predict_time_sec": predict_time, **compute_global_metrics(y_true_df.values, y_pred_df.values)})

        all_true.append(y_true_df)
        all_pred.append(y_pred_df)
        all_case_ids.extend(case_ids_test.tolist())
        all_t_days.extend(df.iloc[test_idx]["t_days"].tolist())
        all_dates.extend(df.iloc[test_idx]["date"].tolist())

        joblib.dump(
            {"model": model, "feature_cols": feature_cols, "target_cols": target_cols, "fold": fold, "model_name": model_name},
            output_dir / f"model_fold_{fold}.joblib",
        )

    y_true_all = pd.concat(all_true, axis=0).reset_index(drop=True)
    y_pred_all = pd.concat(all_pred, axis=0).reset_index(drop=True)
    case_ids_all = np.array(all_case_ids)

    global_metrics = compute_global_metrics(y_true_all.values, y_pred_all.values)
    by_var = compute_metrics_by_variable(y_true_all, y_pred_all)
    by_case = compute_metrics_by_case(case_ids_all, y_true_all.values, y_pred_all.values)
    fold_metrics_df = pd.DataFrame(fold_rows)

    pd.DataFrame([global_metrics]).to_csv(output_dir / "metrics_global.csv", index=False)
    by_var.to_csv(output_dir / "metrics_by_variable.csv", index=False)
    by_case.to_csv(output_dir / "metrics_by_case.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)

    df_eval = pd.DataFrame({"case_id": case_ids_all, "date": all_dates, "t_days": all_t_days})
    for col in target_cols:
        df_eval[f"{col}__true"] = y_true_all[col].values
        df_eval[f"{col}__pred"] = y_pred_all[col].values
    df_eval.to_parquet(output_dir / "predictions.parquet", index=False)
    df_eval.to_csv(output_dir / "predictions.csv", index=False)

    generate_plots(df_eval, target_cols, by_case, figures_dir)

    metadata = {
        "model_name": model_name,
        "n_splits": n_splits,
        "random_state": random_state,
        "n_rows": int(df.shape[0]),
        "n_cases": int(df["case_id"].nunique()),
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] Proxy {model_name} salvo em: {output_dir}")
    print(global_metrics)
    return {"output_dir": str(output_dir), "metrics_global": global_metrics}


def run_full_pipeline(project_root: str | Path = ".", model_name: str = "rf_tabular", output_dir: str | Path = "outputs/rf_tabular", n_splits: int = 5, random_state: int = 42) -> dict:
    print("[Passo 1] Inventário dos casos")
    run_inventory(".")

    print("[Passo 2] Extração das séries do simulador")
    run_extract_summary()

    print("[Passo 3] QC e padronização")
    run_qc()

    print("[Passo 4] Construção do dataset tabular")
    run_build_tabular()

    print("[Passo 5] Parsing dos parâmetros dos casos")
    run_parse_cases()

    print("[Passo 6] Construção do dataset final de ML")
    run_build_ml()

    print("[Passo 7] Treino do proxy e geração das saídas")
    result = train_and_validate_proxy(project_root=project_root, model_name=model_name, output_dir=output_dir, n_splits=n_splits, random_state=random_state)

    print("\n[OK] Pipeline completa finalizada.")
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline completa: leitura dos dados do simulador, preparação do dataset, treino do proxy e geração de arquivos de saída.")
    parser.add_argument("--project-root", default=".", help="Raiz do projeto.")
    parser.add_argument("--model", default="rf_tabular", choices=["rf_tabular", "xgb_tabular", "mlp_tabular"], help="Modelo de proxy a treinar.")
    parser.add_argument("--output-dir", default=None, help="Diretório de saída. Se omitido, usa outputs/<model>.")
    parser.add_argument("--n-splits", type=int, default=5, help="Número de folds do GroupKFold.")
    parser.add_argument("--random-state", type=int, default=42, help="Random state dos modelos.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = args.output_dir or f"outputs/{args.model}"
    run_full_pipeline(project_root=args.project_root, model_name=args.model, output_dir=output_dir, n_splits=args.n_splits, random_state=args.random_state)
