from src.extract.build_inventory import main as run_inventory
from src.extract.extract_summary_timeseries import main as run_extract_summary
from src.validation.qc_and_standardize_summary import main as run_qc
from src.datasets.build_tabular_dataset import main as run_build_tabular
from src.extract.parse_cases_parameters import main as run_parse_cases
from src.datasets.build_ml_dataset import main as run_build_ml

from src.pipeline.train_all_proxies import train_all_proxies


def main():

    print("[STEP 1] Inventário dos casos")
    run_inventory()

    print("[STEP 2] Extração das séries do simulador")
    run_extract_summary()

    print("[STEP 3] QC e padronização")
    run_qc()

    print("[STEP 4] Construção do dataset tabular")
    run_build_tabular()

    print("[STEP 5] Parsing dos parâmetros dos casos")
    run_parse_cases()

    print("[STEP 6] Construção do dataset final de ML")
    run_build_ml()

    print("[STEP 7] Treino dos proxys")

    train_all_proxies(
        project_root=".",
        output_root="outputs",
        n_splits=5,
        random_state=42,
    )

    print("\n[OK] Pipeline completa finalizada.")


if __name__ == "__main__":
    main()