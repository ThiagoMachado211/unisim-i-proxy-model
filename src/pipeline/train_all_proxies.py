from src.pipeline.train_rf_proxy import train_rf_proxy
from src.pipeline.train_xgb_proxy import train_xgb_proxy
from src.pipeline.train_pca_rf_proxy import train_pca_rf_proxy
from src.pipeline.train_pca_linear_proxy import train_pca_linear_proxy
from src.pipeline.train_mlp_proxy import train_mlp_proxy
from src.pipeline.train_aligned_pca_rf_proxy import train_aligned_pca_rf_proxy


def train_all_proxies(project_root=".", output_root="outputs", n_splits=5, random_state=42):

    results = {}

    results["mlp_tabular"] = train_mlp_proxy(
        project_root=project_root,
        output_dir=f"{output_root}/mlp_tabular",
        n_splits=n_splits,
        random_state=random_state,
    )

    print("\n[OK] Todos os proxys foram treinados.\n")

    for model_name, metrics in results.items():
        print(model_name, "→", metrics)

    return results




"""
    results["rf_tabular"] = train_rf_proxy(
        project_root=project_root,
        output_dir=f"{output_root}/rf_tabular",
        n_splits=n_splits,
        random_state=random_state,
    )

    results["xgb_tabular"] = train_xgb_proxy(
        project_root=project_root,
        output_dir=f"{output_root}/xgb_tabular",
        n_splits=n_splits,
        random_state=random_state,
    ) 

    results["pca_rf"] = train_pca_rf_proxy(
        project_root=project_root,
        output_dir=f"{output_root}/pca_rf",
        n_splits=n_splits,
    )

    results["pca_linear"] = train_pca_linear_proxy(
        project_root=project_root,
        output_dir=f"{output_root}/pca_linear",
        n_splits=n_splits,
    )    
"""