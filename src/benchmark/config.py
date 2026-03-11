from pathlib import Path

PROJECT_ROOT = Path(".")

PATHS = {
    "dataset_ml": PROJECT_ROOT / "data" / "processed" / "dataset_ml.parquet",
    "tabular_benchmark": PROJECT_ROOT / "data" / "benchmark" / "tabular" / "benchmark_tabular.parquet",
    "benchmark_metrics": PROJECT_ROOT / "reports" / "metrics" / "benchmark",
    "benchmark_figures": PROJECT_ROOT / "reports" / "figures" / "benchmark",
    "benchmark_models": PROJECT_ROOT / "models" / "benchmark",
}

FEATURE_PATTERNS = [
    "_i",
    "_j",
    "_k1",
    "_k2",
    "rate",
    "bhpmax",
]

TARGET_PREFIXES = (
    "WOPR_",
    "WWPR_",
    "WBHP_",
    "WWIR_",
)

BENCHMARK_CONFIG = {
    "random_state": 42,
    "cv_strategy": "group_kfold",
    "n_splits": 5,
    "models": [
        "rf_tabular",
        "xgb_tabular",
    ],
}