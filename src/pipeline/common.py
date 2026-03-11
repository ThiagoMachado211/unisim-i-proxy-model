from pathlib import Path
import json
import pandas as pd


FEATURE_PATTERNS = ["_i", "_j", "_k1", "_k2", "rate", "bhpmax"]
TARGET_PREFIXES = ("WOPR_", "WWPR_", "WBHP_", "WWIR_")


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def build_feature_target_columns(df: pd.DataFrame):
    feature_cols = [
        c for c in df.columns
        if any(p in c for p in FEATURE_PATTERNS) and not c.startswith("_")
    ]
    feature_cols += ["t_days"]

    target_cols = [c for c in df.columns if c.startswith(TARGET_PREFIXES)]
    return feature_cols, target_cols


def load_dataset_ml(project_root="."):
    project_root = Path(project_root)
    dataset_path = project_root / "data" / "processed" / "dataset_ml.parquet"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em: {dataset_path}\n"
            "Execute a pipeline de preparação antes do treino."
        )

    return pd.read_parquet(dataset_path)


def save_metadata(output_dir, metadata: dict):
    output_dir = Path(output_dir)
    ensure_dir(output_dir)

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )