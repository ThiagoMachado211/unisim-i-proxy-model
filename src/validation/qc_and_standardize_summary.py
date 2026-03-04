from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

EXPECTED_WELLS = ["NA1A", "NA2", "NA3D", "RJS19", "I1", "I2", "I3", "I4", "I5", "I6"]
EXPECTED_VARS  = ["WOPR", "WWPR", "WBHP", "WWIR"]  # por enquanto, exatamente o que você extraiu

NONNEG_VARS = {"WOPR", "WWPR", "WWIR"}  # variáveis que devem ser >= 0

def main(project_root: str = "."):
    root = Path(project_root).resolve()
    in_path  = root / "data" / "extracted" / "summary_timeseries.parquet"
    out_dir  = root / "data" / "processed"
    rep_dir  = root / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    # Basic type cleanup
    df["case_id"] = df["case_id"].astype(int)
    df["date"] = pd.to_datetime(df["date"])

    # --- QC: coverage
    n_cases = df["case_id"].nunique()
    wells_found = sorted(df.loc[df["entity_type"] == "WELL", "entity_name"].unique().tolist())
    vars_found  = sorted(df["variable"].unique().tolist())

    missing_wells = sorted(list(set(EXPECTED_WELLS) - set(wells_found)))
    missing_vars  = sorted(list(set(EXPECTED_VARS)  - set(vars_found)))

    # --- QC: duplicates in (case, date, entity, var)
    dup_mask = df.duplicated(subset=["case_id", "date", "entity_type", "entity_name", "variable"], keep=False)
    n_dups = int(dup_mask.sum())

    # --- QC: time ordering per group (report only)
    # We'll compute simple stats about time deltas
    df_sorted = df.sort_values(["case_id", "entity_type", "entity_name", "variable", "date"]).copy()
    df_sorted["dt_days"] = df_sorted.groupby(
        ["case_id", "entity_type", "entity_name", "variable"]
    )["date"].diff().dt.total_seconds() / 86400.0

    dt_stats = (
        df_sorted["dt_days"]
        .dropna()
        .describe(percentiles=[0.5, 0.9, 0.99])
        .to_dict()
    )

    # --- QC: non-negative vars
    neg_mask = df_sorted["variable"].isin(NONNEG_VARS) & (df_sorted["value"] < 0)
    n_negative = int(neg_mask.sum())

    # Standardization (lightweight, keeps your original times)
    # Add normalized time per case (0..1), useful for generic proxies
    tmin = df_sorted.groupby("case_id")["t_days"].transform("min")
    tmax = df_sorted.groupby("case_id")["t_days"].transform("max")
    denom = (tmax - tmin).replace(0, 1.0)
    df_sorted["t_norm"] = (df_sorted["t_days"] - tmin) / denom

    # If there are duplicates, we can aggregate by mean (safe default)
    if n_dups > 0:
        df_std = (
            df_sorted
            .groupby(["case_id", "date", "entity_type", "entity_name", "variable"], as_index=False)
            .agg({"value": "mean", "t_days": "mean", "t_norm": "mean"})
        )
    else:
        df_std = df_sorted.drop(columns=["dt_days"])

    # Clamp negative values for non-negative vars (optional; here we only REPORT; do not modify by default)
    # If you want to clamp, uncomment:
    # df_std.loc[df_std["variable"].isin(NONNEG_VARS) & (df_std["value"] < 0), "value"] = 0.0

    out_path = out_dir / "summary_timeseries_std.parquet"
    df_std.to_parquet(out_path, index=False)

    qc = {
        "input_file": str(in_path),
        "output_file": str(out_path),
        "n_rows_in": int(df.shape[0]),
        "n_rows_out": int(df_std.shape[0]),
        "n_cases": int(n_cases),
        "wells_found": wells_found,
        "vars_found": vars_found,
        "missing_wells": missing_wells,
        "missing_vars": missing_vars,
        "n_duplicates": n_dups,
        "n_negative_nonneg_vars": n_negative,
        "dt_days_stats": dt_stats,
        "note": "No resampling performed; time grid preserved. Next step can resample to a common grid if desired."
    }

    rep_path = rep_dir / "qc_summary.json"
    rep_path.write_text(json.dumps(qc, indent=2), encoding="utf-8")

    print(f"[OK] Saved standardized: {out_path}")
    print(f"[OK] Saved QC report:    {rep_path}")
    print(json.dumps({k: qc[k] for k in ['n_cases','n_duplicates','n_negative_nonneg_vars','missing_wells','missing_vars']}, indent=2))

if __name__ == "__main__":
    main(".")