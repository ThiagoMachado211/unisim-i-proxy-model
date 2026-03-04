from __future__ import annotations
from pathlib import Path
import pandas as pd

def main(project_root: str = "."):
    root = Path(project_root).resolve()

    in_path = root / "data" / "processed" / "summary_timeseries_std.parquet"
    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(in_path)

    # Mantém apenas WELL (por enquanto)
    dfw = df[df["entity_type"] == "WELL"].copy()

    # Pivot para wide: colunas = VAR_WELL
    dfw["col"] = dfw["variable"] + "_" + dfw["entity_name"]

    # Se existir mais de um registro por (case_id, t_days, col), agregamos por mean (safe)
    wide = (
        dfw.pivot_table(
            index=["case_id", "date", "t_days", "t_norm"],
            columns="col",
            values="value",
            aggfunc="mean"
        )
        .reset_index()
    )

    # Achata o MultiIndex das colunas (pivot_table cria Index)
    wide.columns = [c if isinstance(c, str) else c[1] for c in wide.columns]

    out_path = out_dir / "dataset_tabular.parquet"
    wide.to_parquet(out_path, index=False)

    print(f"[OK] Saved: {out_path}")
    print("[INFO] Shape:", wide.shape)
    print("[INFO] Example columns:", wide.columns[:15].tolist())

if __name__ == "__main__":
    main(".")