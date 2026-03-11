import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def compute_global_metrics(y_true, y_pred):
    """
    Compute overall metrics using all targets together.
    """

    mae = float(mean_absolute_error(y_true, y_pred))

    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))

    r2 = float(
        r2_score(
            y_true,
            y_pred,
            multioutput="variance_weighted",
        )
    )

    denom = np.maximum(np.abs(y_true), 1e-9)
    mape = float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE_pct": mape,
    }


def compute_metrics_by_variable(y_true_df, y_pred_df):
    """
    Compute metrics separately for each output variable.
    """

    rows = []

    for col in y_true_df.columns:

        yt = y_true_df[col].values
        yp = y_pred_df[col].values

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp))

        denom = np.maximum(np.abs(yt), 1e-9)
        mape = float(np.mean(np.abs(yt - yp) / denom) * 100.0)

        rows.append(
            {
                "target": col,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
                "MAPE_pct": mape,
            }
        )

    df = pd.DataFrame(rows)

    return df.sort_values("RMSE", ascending=False).reset_index(drop=True)


def compute_metrics_by_case(case_ids, y_true, y_pred):
    """
    Compute metrics separately for each simulation case.
    """

    rows = []

    unique_cases = sorted(np.unique(case_ids))

    for case_id in unique_cases:

        mask = case_ids == case_id

        yt = y_true[mask]
        yp = y_pred[mask]

        m = compute_global_metrics(yt, yp)

        rows.append(
            {
                "case_id": int(case_id),
                **m,
            }
        )

    df = pd.DataFrame(rows)

    return df.sort_values("RMSE", ascending=False).reset_index(drop=True)