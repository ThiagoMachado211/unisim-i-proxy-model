from pathlib import Path
import time
import joblib
import pandas as pd

from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

from src.pipeline.common import (
    ensure_dir,
    load_dataset_ml,
    build_feature_target_columns,
    save_metadata,
)
from src.benchmark.metrics import (
    compute_global_metrics,
    compute_metrics_by_variable,
    compute_metrics_by_case,
)


def train_xgb_proxy(project_root=".", output_dir="outputs/xgb_tabular", n_splits=5, random_state=42):
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    ensure_dir(output_dir)

    df = load_dataset_ml(project_root)
    feature_cols, target_cols = build_feature_target_columns(df)

    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    groups = df["case_id"].values

    gkf = GroupKFold(n_splits=n_splits)

    all_true = []
    all_pred = []
    all_case_ids = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        case_ids_test = df.iloc[test_idx]["case_id"].values

        base = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

        model = MultiOutputRegressor(base, n_jobs=-1)

        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds = model.predict(X_test)
        predict_time = time.perf_counter() - t1

        y_pred_df = pd.DataFrame(preds, columns=target_cols, index=y_test.index)
        y_true_df = y_test.copy()

        fold_metrics = compute_global_metrics(y_true_df.values, y_pred_df.values)
        fold_rows.append(
            {
                "fold": fold,
                "fit_time_sec": fit_time,
                "predict_time_sec": predict_time,
                **fold_metrics,
            }
        )

        all_true.append(y_true_df)
        all_pred.append(y_pred_df)
        all_case_ids.extend(case_ids_test.tolist())

        joblib.dump(
            {
                "model": model,
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "fold": fold,
            },
            output_dir / f"model_fold_{fold}.joblib",
        )

    y_true_all = pd.concat(all_true, axis=0).reset_index(drop=True)
    y_pred_all = pd.concat(all_pred, axis=0).reset_index(drop=True)
    case_ids_all = pd.Series(all_case_ids)

    global_metrics = compute_global_metrics(y_true_all.values, y_pred_all.values)
    by_var = compute_metrics_by_variable(y_true_all, y_pred_all)
    by_case = compute_metrics_by_case(case_ids_all.values, y_true_all.values, y_pred_all.values)
    fold_metrics_df = pd.DataFrame(fold_rows)

    pd.DataFrame([global_metrics]).to_csv(output_dir / "metrics_global.csv", index=False)
    by_var.to_csv(output_dir / "metrics_by_variable.csv", index=False)
    by_case.to_csv(output_dir / "metrics_by_case.csv", index=False)
    fold_metrics_df.to_csv(output_dir / "metrics_by_fold.csv", index=False)

    metadata = {
        "model_name": "xgb_tabular",
        "n_splits": n_splits,
        "random_state": random_state,
        "n_rows": int(df.shape[0]),
        "n_cases": int(df["case_id"].nunique()),
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
    }
    save_metadata(output_dir, metadata)

    print(f"[OK] XGB proxy salvo em: {output_dir}")
    print(global_metrics)

    return {
        "output_dir": str(output_dir),
        "metrics_global": global_metrics,
    }