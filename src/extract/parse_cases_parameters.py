from __future__ import annotations

import re
from pathlib import Path
import pandas as pd

INJECTORS = ["I1", "I2", "I3", "I4", "I5", "I6"]

# CASE_01 ... CASE_20
CASE_HEADER_RE = re.compile(r"^\s*CASE_(\d{2})\s*$", re.MULTILINE)

# Exemplo de linha:
# I1: (i,j)=(36,16), k=6-8, RATE=37.5 m3/d, BHPmax=400 barsa
INJ_LINE_RE = re.compile(
    r"^\s*(I[1-6])\s*:\s*"
    r"\(i\s*,\s*j\)\s*=\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*"
    r"k\s*=\s*(\d+)\s*-\s*(\d+)\s*,\s*"
    r"RATE\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*m3/d\s*,\s*"
    r"BHPmax\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*barsa\s*$",
    re.MULTILINE | re.IGNORECASE
)

def split_cases(text: str) -> list[tuple[int, str]]:
    """Return list of (case_id, case_block_text)."""
    matches = list(CASE_HEADER_RE.finditer(text))
    if not matches:
        return []

    blocks = []
    for idx, m in enumerate(matches):
        case_id = int(m.group(1))
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        block = text[start:end]
        blocks.append((case_id, block))
    return blocks

def parse_case_block(case_id: int, block: str) -> dict:
    """Parse one CASE block into a flat dict of injector parameters."""
    row: dict = {"case_id": case_id}

    inj_found = {inj: False for inj in INJECTORS}

    for m in INJ_LINE_RE.finditer(block):
        inj, i, j, k1, k2, rate, bhp = m.groups()
        inj_found[inj] = True

        row[f"{inj}_i"] = int(i)
        row[f"{inj}_j"] = int(j)
        row[f"{inj}_k1"] = int(k1)
        row[f"{inj}_k2"] = int(k2)
        row[f"{inj}_rate_m3d"] = float(rate)
        row[f"{inj}_bhpmax_barsa"] = float(bhp)

    # opcional: garantir que todos os injetores existam no bloco
    missing = [inj for inj, ok in inj_found.items() if not ok]
    if missing:
        row["_missing_injectors"] = ",".join(missing)
    else:
        row["_missing_injectors"] = ""

    return row

def main(project_root: str = "."):
    root = Path(project_root).resolve()

    # ajuste aqui se seu README estiver em outro lugar
    readme_path = root / "data" / "raw" / "README.txt"
    if not readme_path.exists():
        # fallback: caso esteja na raiz
        readme_path = root / "README.txt"

    text = readme_path.read_text(encoding="utf-8", errors="replace")

    cases = split_cases(text)
    if not cases:
        raise RuntimeError(
            f"Não encontrei blocos CASE_XX no arquivo: {readme_path}\n"
            "Verifique se os cabeçalhos estão no formato 'CASE_01' etc."
        )

    rows = [parse_case_block(cid, block) for cid, block in cases]
    df = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)

    out_dir = root / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cases.parquet"
    df.to_parquet(out_path, index=False)

    print(f"[OK] Read:  {readme_path}")
    print(f"[OK] Saved: {out_path}")
    print("[INFO] Shape:", df.shape)

    # sanity check rápido
    missing_any = df[df["_missing_injectors"] != ""]
    if not missing_any.empty:
        print("[WARN] Alguns casos estão com injetores faltando no parse:")
        print(missing_any[["case_id", "_missing_injectors"]].to_string(index=False))
    else:
        print("[OK] Todos os casos têm I1..I6 parseados.")

if __name__ == "__main__":
    main(".")