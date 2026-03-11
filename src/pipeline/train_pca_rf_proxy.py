from pathlib import Path
import joblib
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression

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


def _prepare_curve_dataset(df):

    feature_cols, target_cols = build_feature_target_columns(df)

    X = df[feature_cols].copy()
    Y = df[target_cols].copy()

    groups = df["case_id"].values

    return X, Y, groups, feature_cols, target_cols


def train_pca_rf_proxy(project_root=".", output_dir="outputs/pca_rf", n_splits=5):

    project_root = Path(project_root)
    output_dir = project_root / output_dir
    ensure_dir(output_dir)

    df = load_dataset_ml(project_root)

    X, Y, groups, feature_cols, target_cols = _prepare_curve_dataset(df)

    gkf = GroupKFold(n_splits=n_splits)

    all_true = []
    all_pred = []
    all_case_ids = []

    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups), start=1):

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        Y_train = Y.iloc[train_idx]
        Y_test = Y.iloc[test_idx]

        case_ids_test = df.iloc[test_idx]["case_id"].values

        pca = PCA(n_components=20)

        Y_train_pca = pca.fit_transform(Y_train)

        base = RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            n_jobs=-1,
            random_state=42,
        )

        model = MultiOutputRegressor(base)

        model.fit(X_train, Y_train_pca)

        preds_pca = model.predict(X_test)

        preds = pca.inverse_transform(preds_pca)

        y_pred_df = pd.DataFrame(preds, columns=target_cols)
        y_true_df = Y_test.reset_index(drop=True)

        fold_metrics = compute_global_metrics(y_true_df.values, y_pred_df.values)

        fold_rows.append(
            {
                "fold": fold,
                **fold_metrics,
            }
        )

        all_true.append(y_true_df)
        all_pred.append(y_pred_df)
        all_case_ids.extend(case_ids_test.tolist())

        joblib.dump(
            {
                "model": model,
                "pca": pca,
                "features": feature_cols,
                "targets": target_cols,
            },
            output_dir / f"model_fold_{fold}.joblib",
        )

    y_true_all = pd.concat(all_true).reset_index(drop=True)
    y_pred_all = pd.concat(all_pred).reset_index(drop=True)

    case_ids_all = pd.Series(all_case_ids)

    global_metrics = compute_global_metrics(y_true_all.values, y_pred_all.values)

    by_var = compute_metrics_by_variable(y_true_all, y_pred_all)

    by_case = compute_metrics_by_case(
        case_ids_all.values,
        y_true_all.values,
        y_pred_all.values,
    )

    pd.DataFrame([global_metrics]).to_csv(output_dir / "metrics_global.csv", index=False)

    by_var.to_csv(output_dir / "metrics_by_variable.csv", index=False)

    by_case.to_csv(output_dir / "metrics_by_case.csv", index=False)

    pd.DataFrame(fold_rows).to_csv(output_dir / "metrics_by_fold.csv", index=False)

    metadata = {
        "model_name": "pca_rf",
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
    }

    save_metadata(output_dir, metadata)

    print("[OK] PCA_RF proxy treinado")

    return global_metrics