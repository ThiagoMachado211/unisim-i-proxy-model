from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _pick_targets(target_cols: list[str]) -> dict[str, list[str]]:
    """
    Escolhe um subconjunto 'padrão' de variáveis para plotar.
    Ajuste aqui se quiser outras.
    """
    picks = {
        # produtores
        "WOPR_NA1A": None,
        "WWPR_NA1A": None,
        "WBHP_NA1A": None,
        "WOPR_NA2": None,
        "WWPR_NA2": None,
        "WBHP_NA2": None,
        # injetores
        "WWIR_I1": None,
        "WBHP_I1": None,
    }
    chosen = [c for c in picks.keys() if c in target_cols]

    # Agrupa por "família" (WOPR/WWPR/WWIR/WBHP) para facilitar métricas
    families: dict[str, list[str]] = {}
    for c in chosen:
        fam = c.split("_", 1)[0]  # WOPR, WWPR, WBHP, WWIR
        families.setdefault(fam, []).append(c)

    return {"chosen": chosen, "families": families}


def main(project_root: str = ".") -> None:
    root = Path(project_root).resolve()

    data_path = root / "data" / "processed" / "dataset_ml.parquet"
    model_path = root / "models" / "saved_models" / "xgb_proxy_multioutput.joblib"

    fig_dir = root / "reports" / "figures"
    met_dir = root / "reports" / "metrics"
    _safe_mkdir(fig_dir)
    _safe_mkdir(met_dir)

    df = pd.read_parquet(data_path)

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle["feature_cols"]
    target_cols = bundle["target_cols"]

    # Features/Targets
    X = df[feature_cols].copy()
    Y = df[target_cols].copy()

    # Sanity: garantir numérico
    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            raise ValueError(f"Feature não numérica: {c} ({X[c].dtype})")
    for c in Y.columns:
        if not np.issubdtype(Y[c].dtype, np.number):
            raise ValueError(f"Target não numérico: {c} ({Y[c].dtype})")

    # Predições (no dataset inteiro)
    preds = model.predict(X)
    P = pd.DataFrame(preds, columns=target_cols)
    P["case_id"] = df["case_id"].values
    P["t_days"] = df["t_days"].values
    P["t_norm"] = df["t_norm"].values
    P["date"] = df["date"].values

    # ---- MÉTRICAS GLOBAIS (tudo junto)
    y_true = Y.values
    y_pred = preds

    mae_global = float(mean_absolute_error(y_true, y_pred))
    mse_global = float(mean_squared_error(y_true, y_pred))
    rmse_global = float(np.sqrt(mse_global))
    r2_global = float(r2_score(y_true, y_pred, multioutput="variance_weighted"))

    global_metrics = pd.DataFrame([{
        "MAE_global": mae_global,
        "RMSE_global": rmse_global,
        "R2_global": r2_global,
        "n_rows": int(df.shape[0]),
        "n_targets": int(len(target_cols)),
        "n_cases": int(df["case_id"].nunique()),
    }])
    global_metrics.to_csv(met_dir / "metrics_global.csv", index=False)

    # ---- MÉTRICAS POR VARIÁVEL
    rows_var = []
    for j, col in enumerate(target_cols):
        yt = Y[col].values
        yp = P[col].values
        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        # R2 por variável
        r2 = float(r2_score(yt, yp))
        # MAPE (cuidado com zeros)
        denom = np.maximum(np.abs(yt), 1e-9)
        mape = float(np.mean(np.abs(yt - yp) / denom) * 100.0)
        rows_var.append({"target": col, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE_%": mape})

    df_var = pd.DataFrame(rows_var).sort_values("RMSE", ascending=False)
    df_var.to_csv(met_dir / "metrics_by_variable.csv", index=False)

    # ---- MÉTRICAS POR CASO (global por caso)
    rows_case = []
    for case_id in sorted(df["case_id"].unique()):
        idx = (df["case_id"].values == case_id)
        yt = y_true[idx, :]
        yp = y_pred[idx, :]

        mae = float(mean_absolute_error(yt, yp))
        rmse = float(np.sqrt(mean_squared_error(yt, yp)))
        r2 = float(r2_score(yt, yp, multioutput="variance_weighted"))

        rows_case.append({"case_id": int(case_id), "MAE": mae, "RMSE": rmse, "R2": r2})

    df_case = pd.DataFrame(rows_case).sort_values("RMSE", ascending=False)
    df_case.to_csv(met_dir / "metrics_by_case.csv", index=False)

    # ---- ESCOLHER VARIÁVEIS PARA PLOTS PADRÃO
    picks = _pick_targets(target_cols)
    chosen = picks["chosen"]

    # (A) Curvas: Simulador vs Proxy para 3 casos: pior RMSE, mediano, melhor
    # Escolha por RMSE (top, meio, bottom)
    case_sorted = df_case["case_id"].tolist()
    if len(case_sorted) >= 3:
        cases_to_plot = [case_sorted[0], case_sorted[len(case_sorted)//2], case_sorted[-1]]
    else:
        cases_to_plot = case_sorted

    for case_id in cases_to_plot:
        dcase = df[df["case_id"] == case_id].copy()
        pcase = P[P["case_id"] == case_id].copy()

        # ordenar por tempo
        dcase = dcase.sort_values("t_days")
        pcase = pcase.sort_values("t_days")

        t = dcase["t_days"].values

        for col in chosen:
            y = dcase[col].values
            yh = pcase[col].values

            plt.figure()
            plt.plot(t, y, label="Simulador")
            plt.plot(t, yh, label="Proxy")
            plt.xlabel("t_days")
            plt.ylabel(col)
            plt.title(f"CASE_{int(case_id):02d} - {col}")
            plt.legend()
            plt.tight_layout()

            out = fig_dir / f"curve_CASE_{int(case_id):02d}_{col}.png"
            plt.savefig(out, dpi=160)
            plt.close()

    # (B) Scatter Real vs Predito (para variáveis escolhidas)
    for col in chosen:
        yt = Y[col].values
        yp = P[col].values

        # linha y=x no intervalo observado
        vmin = float(min(np.min(yt), np.min(yp)))
        vmax = float(max(np.max(yt), np.max(yp)))

        plt.figure()
        plt.scatter(yt, yp, s=8, alpha=0.5)
        plt.plot([vmin, vmax], [vmin, vmax])
        plt.xlabel("Real (Simulador)")
        plt.ylabel("Predito (Proxy)")
        plt.title(f"Scatter Real vs Predito - {col}")
        plt.tight_layout()

        out = fig_dir / f"scatter_{col}.png"
        plt.savefig(out, dpi=160)
        plt.close()

    # (C) Erro ao longo do tempo (MAE por timestep, agregando casos)
    # Vamos agrupar por t_days (atenção: t_days pode não ser exatamente igual em float; usamos arredondamento)
    tmp = pd.DataFrame({"t_days": df["t_days"].values})
    tmp["t_bin"] = np.round(tmp["t_days"], 3)  # bins estáveis
    for col in chosen:
        tmp[f"ae_{col}"] = np.abs(Y[col].values - P[col].values)

    # MAE por bin de tempo (média dos erros absolutos)
    group = tmp.groupby("t_bin")
    mae_t = group[[c for c in tmp.columns if c.startswith("ae_")]].mean().reset_index()

    for col in chosen:
        ae_col = f"ae_{col}"
        plt.figure()
        plt.plot(mae_t["t_bin"].values, mae_t[ae_col].values)
        plt.xlabel("t_days (binned)")
        plt.ylabel("MAE")
        plt.title(f"MAE ao longo do tempo - {col}")
        plt.tight_layout()
        out = fig_dir / f"mae_over_time_{col}.png"
        plt.savefig(out, dpi=160)
        plt.close()

    # Salvar um resumo curto
    summary_txt = (
        "Validation artifacts generated:\n"
        f"- {met_dir / 'metrics_global.csv'}\n"
        f"- {met_dir / 'metrics_by_variable.csv'}\n"
        f"- {met_dir / 'metrics_by_case.csv'}\n"
        f"- Curves/scatters/MAE-over-time in {fig_dir}\n"
    )
    (met_dir / "validation_artifacts.txt").write_text(summary_txt, encoding="utf-8")

    print("[OK] Validation complete.")
    print(f"[OK] Metrics: {met_dir}")
    print(f"[OK] Figures: {fig_dir}")
    print(f"[INFO] Global: MAE={mae_global:.4f} RMSE={rmse_global:.4f} R2={r2_global:.4f}")


if __name__ == "__main__":
    main(".")