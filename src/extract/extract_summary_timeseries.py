from pathlib import Path
import pandas as pd
import re

from resdata.summary import Summary

CASE_RE = re.compile(r"CASE_(\d{2})$")

PRODUCERS = ["NA1A", "NA2", "NA3D", "RJS19"]
INJECTORS = ["I1", "I2", "I3", "I4", "I5", "I6"]

WELL_VARS = ["WOPR", "WWPR", "WBHP", "WOPT", "WWPT", "WWIR", "WWIT"]
FIELD_VARS = ["FOPT", "FWPT", "FWIT"]

def find_unsmry(case_dir: Path):

    matches = list(case_dir.glob("*.UNSMRY"))

    if not matches:
        raise RuntimeError(f"No UNSMRY file in {case_dir}")

    return matches[0]

def extract_case(case_dir: Path, case_id: int):

    unsmry = find_unsmry(case_dir)

    smry = Summary(str(unsmry))

    dates = pd.to_datetime(smry.dates)

    rows = []

    wells = PRODUCERS + INJECTORS

    for well in wells:

        for var in WELL_VARS:

            key = f"{var}:{well}"

            if key not in smry.keys():
                continue

            values = smry.numpy_vector(key)

            for d, v in zip(dates, values):

                rows.append({
                    "case_id": case_id,
                    "date": d,
                    "entity_type": "WELL",
                    "entity_name": well,
                    "variable": var,
                    "value": float(v)
                })

    for var in FIELD_VARS:

        if var not in smry.keys():
            continue

        values = smry.numpy_vector(var)

        for d, v in zip(dates, values):

            rows.append({
                "case_id": case_id,
                "date": d,
                "entity_type": "FIELD",
                "entity_name": "FIELD",
                "variable": var,
                "value": float(v)
            })

    return pd.DataFrame(rows)


def main():

    root = Path(".")
    raw_dir = root / "data" / "raw"
    out_dir = root / "data" / "extracted"

    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []

    for case_dir in sorted(raw_dir.iterdir()):

        if not case_dir.is_dir():
            continue

        m = CASE_RE.search(case_dir.name)

        if not m:
            continue

        case_id = int(m.group(1))

        print(f"[INFO] Extracting CASE_{case_id:02d}")

        df_case = extract_case(case_dir, case_id)

        dfs.append(df_case)

    df = pd.concat(dfs, ignore_index=True)

    df["t_days"] = df.groupby("case_id")["date"].transform(
        lambda s: (s - s.min()).dt.total_seconds() / 86400
    )

    out_path = out_dir / "summary_timeseries.parquet"

    df.to_parquet(out_path, index=False)

    print(f"[OK] Saved: {out_path}")


if __name__ == "__main__":
    main()