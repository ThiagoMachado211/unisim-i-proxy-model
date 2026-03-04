from pathlib import Path
import pandas as pd

def main(project_root="."):

    root = Path(project_root)

    tabular_path = root / "data/processed/dataset_tabular.parquet"
    cases_path = root / "data/processed/cases.parquet"

    tabular = pd.read_parquet(tabular_path)
    cases = pd.read_parquet(cases_path)
    cases = cases.drop(columns=[c for c in cases.columns if c.startswith("_")], errors="ignore")
    
    # merge pelos casos
    df = tabular.merge(cases, on="case_id", how="left")

    out_path = root / "data/processed/dataset_ml.parquet"

    df.to_parquet(out_path, index=False)

    print("[OK] Saved:", out_path)
    print("[INFO] Shape:", df.shape)

    print("\nExample columns:\n")
    print(df.columns[:20])


if __name__ == "__main__":
    main()