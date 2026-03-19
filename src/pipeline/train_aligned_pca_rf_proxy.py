from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold

from src.pipeline.common import ensure_dir, load_dataset_ml, save_metadata
from src.benchmark.metrics import (
    compute_global_metrics,
    compute_metrics_by_variable,
    compute_metrics_by_case,
)


TARGET_CURVE = "WOPR_NA1A"


def build_case_level_dataset(df, target_col=TARGET_CURVE):
    """
    Builds one curve per case.
    Rows = cases
    Columns = timesteps
    """

    base_features = [
        c for c in df.columns
        if (
            (("_i" in c) or ("_j" in c) or ("_k1" in c) or ("_k2" in c) or ("rate" in c) or ("bhpmax" in c))
            and not c.startswith("_")
            and c != "case_id"
        )
    ]

    # Seleciona as features de caso
    X_case = df.loc[:, ["case_id"] + base_features].copy()

    # Garante que não exista ambiguidade de índice
    X_case = X_case.reset_index(drop=True).rename_axis(None, axis=0)

    # Uma linha por caso, pegando a primeira ocorrência
    X_case = (
        X_case
        .groupby("case_id", as_index=False)
        .first()
        .sort_values(by="case_id")
        .reset_index(drop=True)
    )

    curves = (
        df[["case_id", "t_days", target_col]]
        .pivot_table(
            index="case_id",
            columns="t_days",
            values=target_col,
            aggfunc="mean",
        )
        .sort_index(axis=1)
        .sort_index()
    )

    t_grid = np.array(curves.columns, dtype=float)

    # Reindexa para garantir mesma ordem dos casos
    X_case = X_case.set_index("case_id").loc[curves.index].reset_index()

    return X_case, curves, t_grid, base_features


def compute_peak_index(curve_values):
    """
    Use the maximum point as alignment anchor.
    """
    return int(np.argmax(curve_values))


def shift_curve(curve, shift):
    """
    Shift curve in index space. Missing values are filled with edge values.
    Positive shift -> move curve to the right.
    Negative shift -> move curve to the left.
    """
    n = len(curve)
    shifted = np.empty_like(curve)

    if shift == 0:
        return curve.copy()

    if shift > 0:
        shifted[:shift] = curve[0]
        shifted[shift:] = curve[:-shift]
    else:
        s = abs(shift)
        shifted[-s:] = curve[-1]
        shifted[:-s] = curve[s:]

    return shifted


def align_curves_by_peak(curves_df):
    """
    Align all curves so that their peak positions match the median peak index.
    """
    curves = curves_df.values
    peak_indices = np.array([compute_peak_index(row) for row in curves], dtype=int)
    reference_peak = int(np.median(peak_indices))

    aligned = []
    shifts = []

    for row, peak_idx in zip(curves, peak_indices):
        shift = reference_peak - peak_idx
        aligned_row = shift_curve(row, shift)
        aligned.append(aligned_row)
        shifts.append(shift)

    aligned = np.array(aligned)
    shifts = np.array(shifts)

    aligned_df = pd.DataFrame(
        aligned,
        index=curves_df.index,
        columns=curves_df.columns,
    )

    return aligned_df, shifts, reference_peak


def unalign_curve(curve, shift):
    """
    Reverse the shift applied during alignment.
    """
    return shift_curve(curve, -shift)


def train_aligned_pca_rf_proxy(
    project_root=".",
    output_dir="outputs/aligned_pca_rf",
    n_splits=5,
    random_state=42,
    n_components=5,
    target_col=TARGET_CURVE,
):
    project_root = Path(project_root)
    output_dir = project_root / output_dir
    ensure_dir(output_dir)

    df = load_dataset_ml(project_root)

    X_case, curves_df, t_grid, feature_cols = build_case_level_dataset(df, target_col=target_col)

    groups = X_case["case_id"].values
    X = X_case[feature_cols].copy()
    Y = curves_df.copy()

    gkf = GroupKFold(n_splits=n_splits)

    all_true = []
    all_pred = []
    all_case_ids = []
    fold_rows = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, Y, groups=groups), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()

        Y_train = Y.iloc[train_idx].copy()
        Y_test = Y.iloc[test_idx].copy()

        case_ids_test = X_case.iloc[test_idx]["case_id"].values

        # Align only based on training reference logic
        Y_train_aligned, train_shifts, reference_peak = align_curves_by_peak(Y_train)

        # Align test curves using the same reference_peak
        test_peak_indices = np.array([compute_peak_index(row) for row in Y_test.values], dtype=int)
        test_shifts = np.array([reference_peak - p for p in test_peak_indices], dtype=int)

        Y_test_aligned_values = np.array(
            [shift_curve(row, shift) for row, shift in zip(Y_test.values, test_shifts)]
        )
        Y_test_aligned = pd.DataFrame(
            Y_test_aligned_values,
            index=Y_test.index,
            columns=Y_test.columns,
        )

        pca = PCA(n_components=n_components, random_state=random_state)
        Y_train_pca = pca.fit_transform(Y_train_aligned)

        base = RandomForestRegressor(
            n_estimators=400,
            max_depth=18,
            min_samples_leaf=1,
            random_state=random_state,
            n_jobs=-1,
        )
        model = MultiOutputRegressor(base, n_jobs=-1)

        t0 = time.perf_counter()
        model.fit(X_train, Y_train_pca)
        fit_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        preds_pca = model.predict(X_test)
        predict_time = time.perf_counter() - t1

        preds_aligned = pca.inverse_transform(preds_pca)

        # Undo alignment for each predicted curve
        preds_unaligned = np.array(
            [unalign_curve(curve, shift) for curve, shift in zip(preds_aligned, test_shifts)]
        )

        y_pred_df = pd.DataFrame(
            preds_unaligned,
            columns=Y.columns,
            index=Y_test.index,
        )
        y_true_df = Y_test.copy()

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
                "pca": pca,
                "feature_cols": feature_cols,
                "curve_columns": list(Y.columns),
                "reference_peak": reference_peak,
                "target_col": target_col,
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
        "model_name": "aligned_pca_rf",
        "target_curve": target_col,
        "n_splits": n_splits,
        "random_state": random_state,
        "n_cases": int(X_case.shape[0]),
        "n_features": len(feature_cols),
        "n_timesteps": int(Y.shape[1]),
        "n_components": n_components,
        "alignment_method": "peak_index_alignment",
    }
    save_metadata(output_dir, metadata)

    print(f"[OK] aligned_pca_rf salvo em: {output_dir}")
    print(global_metrics)

    return {
        "output_dir": str(output_dir),
        "metrics_global": global_metrics,
    }