import re
import json
from pathlib import Path
import pandas as pd

CASE_RE = re.compile(r"CASE_(\d{2})$")

KEY_EXTS = [
    "DATA", "SMSPEC", "UNSMRY", "UNRST",
    "EGRID", "INIT", "PRT", "RSM", "RSSPEC",
]

REQUIRED = {"DATA", "SMSPEC", "UNSMRY", "UNRST"}

def find_one_by_ext(case_dir: Path, ext: str):
    """Return first file matching *.EXT (case-insensitive) or None."""
    matches = sorted(case_dir.glob(f"*.{ext}"))
    if matches:
        return matches[0]
    # try lowercase variants just in case
    matches = sorted(case_dir.glob(f"*.{ext.lower()}"))
    return matches[0] if matches else None

def main(project_root: str = "."):
    root = Path(project_root).resolve()
    raw_dir = root / "data" / "raw"
    out_dir = root / "data" / "extracted"
    rep_dir = root / "reports" / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)
    rep_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in sorted(raw_dir.iterdir()):
        if not p.is_dir():
            continue
        m = CASE_RE.search(p.name)
        if not m:
            continue
        case_id = int(m.group(1))

        found = {}
        for ext in KEY_EXTS:
            f = find_one_by_ext(p, ext)
            found[ext] = str(f) if f else ""

        present = {k for k, v in found.items() if v}
        ok = REQUIRED.issubset(present)
        missing = sorted(list(REQUIRED - present))

        rows.append({
            "case_id": case_id,
            "case_folder": str(p),
            "ok": ok,
            "missing_required": ",".join(missing),
            **{f"file_{ext}": found[ext] for ext in KEY_EXTS}
        })

    df = pd.DataFrame(rows).sort_values("case_id")
    inv_path = out_dir / "cases_inventory.csv"
    df.to_csv(inv_path, index=False)

    summary = {
        "n_cases_found": int(df.shape[0]),
        "n_ok": int(df["ok"].sum()) if not df.empty else 0,
        "n_not_ok": int((~df["ok"]).sum()) if not df.empty else 0,
        "required": sorted(list(REQUIRED)),
        "raw_dir": str(raw_dir),
        "inventory_file": str(inv_path),
    }
    (rep_dir / "inventory_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"[OK] Inventory saved to: {inv_path}")
    print(f"[OK] Summary saved to: {rep_dir / 'inventory_summary.json'}")
    print(summary)

if __name__ == "__main__":
    main(".")